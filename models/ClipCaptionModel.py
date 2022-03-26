import torch
import torch.nn as nn
from torch.nn import functional as nnf
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os, pdb
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel
from itertools import chain
from einops.layers.torch import Rearrange
import numpy as np

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x

class Mixer(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0., depth=1):
        super().__init__()

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, num_patch, token_dim, channel_dim))

    def forward(self, x):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        return x

class TokenLearnerModuleV11(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.

    Instead of using 4 conv. layers with small channels to implement spatial
    attention, this version uses 2 grouped conv. layers with more channels. It
    also uses softmax instead of sigmoid. We confirmed that this version works
    better when having limited training data, such as training with ImageNet1K
    from scratch.

    Attributes:
      num_tokens: Number of tokens.
      dropout_rate: Dropout rate.
    """

    def __init__(self, feat_dim, in_channels, num_tokens, num_groups, dropout_rate=0.):
        """Applies learnable tokenization to the 2D inputs.

        Args:
          inputs: Inputs of shape `[bs, h, w, c]`.

        Returns:
          Output of shape `[bs, n_token, c]`.
        """
        super(TokenLearnerModuleV11, self).__init__()
        self.att_proj = nn.Sequential(nn.Linear(feat_dim, in_channels),nn.ReLU(),nn.Dropout(dropout_rate),\
                                      nn.Linear(in_channels, in_channels),nn.Dropout(dropout_rate))
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.num_groups = num_groups  # in_channels and out_channels must both be divisible by groups
        self.norm = nn.LayerNorm(self.in_channels)  # Operates on the last axis (c) of the input data.

        self.attention_maps = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, groups=self.num_groups, bias=False),
            nn.Conv2d(self.in_channels, self.num_tokens, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
        )
        # self.feat_conv = nn.Conv2d(
        #     self.in_channels, self.in_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, groups=self.num_groups, bias=False)
        # self.feat_ff = nn.Sequential(nn.Linear(in_channels, 14*in_channels),nn.ReLU(),nn.Dropout(dropout_rate),\
        #                         nn.Linear(14*in_channels, in_channels),nn.Dropout(dropout_rate))
        # self.feat_ff = nn.Sequential(nn.Linear(in_channels, 8*in_channels),nn.ReLU(),nn.Dropout(dropout_rate),\
        #                             nn.Linear(8*in_channels, 4*in_channels),nn.ReLU(),nn.Dropout(dropout_rate),\
        #                         nn.Linear(4*in_channels, in_channels),nn.Dropout(dropout_rate))
        self.feat_ff = MixerBlock(in_channels, self.num_tokens+1, (self.num_tokens+1)*4, in_channels*8, dropout = 0.1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        inputs = self.att_proj(inputs)
        cls_emb = inputs[:,:1,:]
        inputs = inputs[:,1:,:]
        h= int(inputs.size(1)**0.5)
        w = inputs.size(1)//h
        inputs = inputs.view(inputs.size(0),h, w, inputs.size(-1))
        feature_shape = inputs.shape  # Shape:  [bs, h, w, c]

        selected = inputs
        selected = self.norm(selected)
        selected = selected.permute(0, 3, 1, 2)  # Shape:  [bs, c, h, w]
        selected = self.attention_maps(selected)  # Shape: [bs, n_token, h, w].
        selected = selected.permute(0, 2, 3, 1)  # Shape: [bs, h, w, n_token].
        selected = selected.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2],
                                 -1)  # Shape: [bs, h*w, n_token].
        selected = selected.permute(0, 2, 1)  # Shape: [bs, n_token, h*w].
        selected = nnf.softmax(selected, dim=-1)

        # Reshape the input to align it with the output of the conv block.
        feat = inputs
        feat = feat.permute(0, 3, 1, 2)   # Shape:  [bs, c, h, w]
        # feat = self.feat_conv(feat)      # Shape: [bs, c, h, w].
        feat = feat.permute(0, 2, 3, 1)   # Shape: [bs, h, w, c].
        feat = feat.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)  # Shape: [bs, h*w, c].

        # Produced the attended inputs.
        #np.save('./attn_vis.npy', selected.cpu().numpy())
        #pdb.set_trace()
        outputs = torch.einsum("...si,...id->...sd",  selected, feat)  # (B, n_token, c)
        outputs = self.dropout(outputs)

        outputs = torch.cat([cls_emb,outputs],dim=1)
        outputs = self.feat_ff(outputs)
        return outputs

class MLP(nn.Module):

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.ReLU, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
            layers.append(nn.Dropout(p=dropout))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.2):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0.2, act=nnf.relu,
                norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x




class Transformer(nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x


class TransformerMapper(nn.Module):

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.prefix_length = prefix_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x):
        # x = x[:,:,:]
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, -self.prefix_length:]
        return out



class TransformerMapperV12(nn.Module):

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapperV12, self).__init__()
        self.clip_length = clip_length
        self.prefix_length = prefix_length
        self.cross_att = Transformer(dim_embedding, 8, 2, dim_ref=dim_clip)
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x):
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        out = self.cross_att(prefix,x)
        out = self.transformer(out)
        return out


class ClipCaptionModel(AttModel):

    def __init__(self, opt, prefix_length: int = 10, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8):
        super(ClipCaptionModel, self).__init__(opt)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.opt = opt
        self.clip_length = opt.clip_length
        self.prefix_length = opt.prefix_length
        self.prefix_size = opt.fc_feat_size
        self.gpt = GPT2LMHeadModel.from_pretrained(opt.gpt_type)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if opt.mapping_type == 'mlp':
            self.clip_project = MLP((self.prefix_size, (self.gpt_embedding_size * self.prefix_length) // 2,
                                     self.gpt_embedding_size * self.prefix_length))
        elif opt.mapping_type == 'TokenLearner':
            self.clip_project = TokenLearnerModuleV11(feat_dim=opt.att_feat_size, in_channels=self.gpt_embedding_size, num_tokens=self.prefix_length-1, num_groups=4, dropout_rate=0.1)
        elif opt.mapping_type == 'transformer':
            self.clip_project = TransformerMapper(self.prefix_size, self.gpt_embedding_size, self.prefix_length,
                                                                     self.clip_length, num_layers)
        elif opt.mapping_type == 'transformerV12':
            self.clip_project = TransformerMapperV12(opt.att_feat_size, self.gpt_embedding_size, self.prefix_length,
                                                                     self.clip_length, num_layers)
        else:
            raise Exception("mapping type not supported: {}".format(opt.mapping_type)) 

        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        delattr(self, 'att_embed')
        delattr(self, 'logit')
        del self.ctx2att


    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def _forward(self, fc_feats: torch.Tensor, att_feats: torch.Tensor, tokens: torch.Tensor,  mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        if 'TokenLearner' in self.opt.mapping_type or  self.opt.mapping_type == 'transformerV11' or  self.opt.mapping_type == 'transformerV12' or  self.opt.mapping_type == 'transformerV22':
            prefix = att_feats
        else:
            prefix = fc_feats
        tokens = tokens[:,:-1]
        mask = mask[:,:-1]
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        outputs = nnf.log_softmax(out.logits,dim=-1)
        return outputs

    def init_hidden(self, bsz):
        return None

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        if 'TokenLearner' in self.opt.mapping_type or  self.opt.mapping_type == 'transformerV11' or  self.opt.mapping_type == 'transformerV12' or  self.opt.mapping_type == 'transformerV22':
            prefix = att_feats
        else:
            prefix = fc_feats
        prefix_embed = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        return fc_feats[...,:1], att_feats[...,:1], prefix_embed, att_masks

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = nnf.log_softmax(output, dim=1)

        return logprobs, state

    def core(self, it, fc_feats_ph, att_feats_ph, prefix_emb, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if state is None:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        embedding_text = self.gpt.transformer.wte(ys)
        embedding_cat = torch.cat((prefix_emb, embedding_text), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat).logits
        return out[:, -1], [ys.unsqueeze(0)]




class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self
