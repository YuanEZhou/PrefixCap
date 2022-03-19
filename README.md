# PrefixCap
## Requirements
- Python 3.8
- Pytorch 1.7.1
- You can simply install requirements by executing ```pip install -r requirements.txt```


## Prepare Data
1. Please use **git clone --recurse-submodules** to clone this repository and remember to follow initialization steps in coco-caption/README.md. Then download the [Flickr30k reference file](https://mailhfuteducn-my.sharepoint.com/:u:/g/personal/ye_zhou_mail_hfut_edu_cn/EdS09HVkYUlIoj9Al65kUtsBOVeYM9wsb3OJHitiBnnGGA?e=TljNdU) and put it under 'coco-caption/annotations/'.
2. Download the preprocessd dataset from this [link](https://mailhfuteducn-my.sharepoint.com/:u:/g/personal/ye_zhou_mail_hfut_edu_cn/ERMYZRiY-_NMpOzDRwX9f4oBMfCZOrtCA1vwh-MAVaxQjw?e=woB7hu) and extract it to 'data/'.
3. Download the MSCOCO images from [link](https://cocodataset.org/#download). We need 2014 training images and 2014 val. images. You should unzip and put the train2014/ and val2014/ in the same directory, such as 'data/coco_images'.  Download the Flickr30K images from [link](http://shannon.cs.illinois.edu/DenotationGraph/) and extract it to a directory, such as  'data/flickr30k_images'.
4. Please download the `clip-vit-base-patch16` (`clip-vit-large-patch14`) model files from [link](https://huggingface.co/openai) and put them under 'checkpoint/clip-vit-base-patch16'('checkpoint/clip-vit-large-patch14').
5. Please extract required image features by executing      
   a)```python  scripts/feat_extractor.py  --dataset mscoco  --input_json    data/dataset_coco.json   --output_dir   data/clip-vit-large-patch14    --images_root  data/coco_images  --model_root   checkpoint/clip-vit-large-patch14```         
   b)```python  scripts/feat_extractor.py  --dataset mscoco  --input_json    data/dataset_coco.json   --output_dir   data/clip-vit-base-patch16-224    --images_root  data/coco_images  --model_root   checkpoint/clip-vit-base-patch16```        
   c)```python  scripts/feat_extractor.py  --dataset flickr30k  --input_json    data/dataset_flickr30k.json   --output_dir   data/clip-vit-large-patch14-flickr30k    --images_root  data/flickr30k_images  --model_root   checkpoint/clip-vit-large-patch14```                
   d)```python  scripts/feat_extractor.py  --dataset flickr30k  --input_json    data/dataset_flickr30k.json   --output_dir   data/clip-vit-base-patch16-flickr30k    --images_root  data/flickr30k_images  --model_root   checkpoint/clip-vit-base-patch16```  

7. Download part checkpoints of our models from [here]() and extract them to 'save/new/'. Additionally, download the [shell script](https://mailhfuteducn-my.sharepoint.com/:u:/g/personal/ye_zhou_mail_hfut_edu_cn/ETbNUbIfs2pFh5kqwUuY59QBrT4Poq8z-FRXDrRkQQe9pg?e=INwm7l) and put it under 'save/new/'.

## Offline Evaluation
For example, to reproduce the results of PrefixCap-TSTM model when only freezing CLIP-ViT and using self-critical training on Karpathy test split, just run

```
python  eval.py  --model  save/new/nsc-ClipCaption-TokenLearner-gpt2-clip-vit-large-patch14/model-best.pth   --infos_path  save/new/nsc-ClipCaption-TokenLearner-gpt2-clip-vit-large-patch14/infos_nsc-ClipCaption-TokenLearner-gpt2-clip-vit-large-patch14-best.pkl      --beam_size   3   --id  nsc-ClipCaption-TokenLearner-gpt2-clip-vit-large-patch14   --split test
```
To reproduce the results of PrefixCap-TSTM model when freezing CLIP-ViT and GPT2 and using self-critical training on Karpathy test split, just run
```
python  eval.py  --model  save/new/nsc-ClipCaptionPrefix-TokenLearner-gpt2-clip-vit-large-patch14/model-best.pth   --infos_path  save/new/nsc-ClipCaptionPrefix-TokenLearner-gpt2-clip-vit-large-patch14/infos_nsc-ClipCaptionPrefix-TokenLearner-gpt2-clip-vit-large-patch14-best.pkl      --beam_size   3   --id  nsc-ClipCaptionPrefix-TokenLearner-gpt2-clip-vit-large-patch14   --split test
```

You can also eval other saved models in a similar way.

## Training
   - **Model trained with freezing CLIP-ViT and GPT2**
      - PrefixCap-TSTM
         1.  In the cross-entropy  training stage, such as using clip-vit-large-patch14 feature, one GPU with 12G memory is ok,  jsut run 
         ```
         python  train.py   --gpt_type  gpt2    --caption_model   ClipCaptionPrefix   --group   0   --mapping_type  TokenLearner   --noamopt --noamopt_warmup 5000   --seq_per_img 5 --batch_size 8 --beam_size 1  --scheduled_sampling_start 0  --save_checkpoint_every 5000  --max_epochs 10     --checkpoint_path   save/new/ClipCaptionPrefix-TokenLearner-gpt2-clip-vit-large-patch14    --id  ClipCaptionPrefix-TokenLearner-gpt2-clip-vit-large-patch14   --dataset  coco   --input_json  data/cocotalk_clip_prefix.json      --input_fc_dir    data/clip-vit-large-patch14/clip-vit-large-patch14_fc    --input_att_dir   data/clip-vit-large-patch14/clip-vit-large-patch14_att      --input_box_dir    data/clip-vit-large-patch14/clip-vit-large-patch14_box    --input_label_h5    data/cocotalk_clip_prefix_label.h5    --cached_tokens    coco-train-clip-prefix-idxs
         ```
         2. Then in the self-critical training stage, you need four GPUs with 12G memory each, please copy the above pretrained model first

         ```
         cd save/new
         ./copy_model.sh  ClipCaptionPrefix-TokenLearner-gpt2-clip-vit-large-patch14    nsc-ClipCaptionPrefix-TokenLearner-gpt2-clip-vit-large-patch14
         cd ../../
         ``` 
         and then run
         ```
         python  train.py   --self_critical_after  9   --max_length   20   --gpt_type   gpt2   --caption_model   ClipCaptionPrefix  --group   0   --mapping_type  TokenLearner    --seq_per_img 5 --batch_size 8 --beam_size 1  --learning_rate 1e-5    --save_checkpoint_every 5000  --max_epochs 20     --start_from    save/new/nsc-ClipCaptionPrefix-TokenLearner-gpt2-clip-vit-large-patch14     --checkpoint_path   save/new/nsc-ClipCaptionPrefix-TokenLearner-gpt2-clip-vit-large-patch14  --id  nsc-ClipCaptionPrefix-TokenLearner-gpt2-clip-vit-large-patch14    --dataset  coco   --input_json  data/cocotalk_clip_prefix.json      --input_fc_dir    data/clip-vit-large-patch14/clip-vit-large-patch14_fc    --input_att_dir   data/clip-vit-large-patch14/clip-vit-large-patch14_att      --input_box_dir    data/clip-vit-large-patch14/clip-vit-large-patch14_box    --input_label_h5    data/cocotalk_clip_prefix_label.h5    --cached_tokens    coco-train-clip-prefix-idxs
         ```
   - **Model trained with only freezing CLIP-ViT**
      - PrefixCap-TSTM
         1.  In the cross-entropy  training stage, such as using clip-vit-large-patch14 feature, one GPU with 12G memory is ok,  jsut run 
         ```
         python  train.py   --gpt_type  gpt2    --caption_model   ClipCaption   --group   1   --mapping_type  TokenLearner   --noamopt --noamopt_warmup 5000   --seq_per_img 5 --batch_size 8 --beam_size 1  --scheduled_sampling_start 0  --save_checkpoint_every 5000  --max_epochs 10     --checkpoint_path   save/new/ClipCaption-TokenLearner-gpt2-clip-vit-large-patch14    --id  ClipCaption-TokenLearner-gpt2-clip-vit-large-patch14   --dataset  coco   --input_json  data/cocotalk_clip_prefix.json      --input_fc_dir    data/clip-vit-large-patch14/clip-vit-large-patch14_fc    --input_att_dir   data/clip-vit-large-patch14/clip-vit-large-patch14_att      --input_box_dir    data/clip-vit-large-patch14/clip-vit-large-patch14_box    --input_label_h5    data/cocotalk_clip_prefix_label.h5    --cached_tokens    coco-train-clip-prefix-idxs
         ```
         2. Then in the self-critical training stage, you need four GPUs with 12G memory each, please copy the above pretrained model first

         ```
         cd save/new
         ./copy_model.sh  ClipCaption-TokenLearner-gpt2-clip-vit-large-patch14    nsc-ClipCaption-TokenLearner-gpt2-clip-vit-large-patch14
         cd ../../
         ``` 
         and then run
         ```
         python  train.py   --self_critical_after  9   --max_length   20   --gpt_type   gpt2   --caption_model   ClipCaption  --group   1   --mapping_type  TokenLearner    --seq_per_img 5 --batch_size 8 --beam_size 1  --learning_rate 1e-5    --save_checkpoint_every 5000  --max_epochs 20     --start_from    save/new/nsc-ClipCaption-TokenLearner-gpt2-clip-vit-large-patch14     --checkpoint_path   save/new/nsc-ClipCaption-TokenLearner-gpt2-clip-vit-large-patch14  --id  nsc-ClipCaption-TokenLearner-gpt2-clip-vit-large-patch14    --dataset  coco   --input_json  data/cocotalk_clip_prefix.json      --input_fc_dir    data/clip-vit-large-patch14/clip-vit-large-patch14_fc    --input_att_dir   data/clip-vit-large-patch14/clip-vit-large-patch14_att      --input_box_dir    data/clip-vit-large-patch14/clip-vit-large-patch14_box    --input_label_h5    data/cocotalk_clip_prefix_label.h5    --cached_tokens    coco-train-clip-prefix-idxs
         ```
## Citation

```

```

## Acknowledgements
This repository is built upon [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch). Thanks for the released  code.
