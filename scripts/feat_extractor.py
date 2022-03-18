from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, pdb
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io  as  io

from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPVisionConfig, CLIPConfig
from transformers import ViTFeatureExtractor, ViTModel


def main(params):
  # clip vit model
  device = torch.device('cuda:0')
  configuration = CLIPConfig.from_pretrained(os.path.join(params['model_root'], 'config.json'))
  model = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path=params['model_root'], config=configuration.vision_config)
  model.to(device)
  processor = CLIPProcessor.from_pretrained(params['model_root'])
  model.eval()


  imgs = json.load(open(params['input_json'], 'r'))
  imgs = imgs['images']
  N = len(imgs)

  seed(123) # make reproducible
  if not os.path.isdir(params['output_dir']):
    os.mkdir(params['output_dir'])

  dir_fc = os.path.join(params['output_dir'], params['output_dir']+'_fc')
  dir_att = os.path.join(params['output_dir'], params['output_dir']+'_att')
  if not os.path.isdir(dir_fc):
    os.mkdir(dir_fc)
  if not os.path.isdir(dir_att):
    os.mkdir(dir_att)

  for i,img in enumerate(imgs):
    # load the image
    if params['dataset'] == 'mscoco':
      image = Image.open(os.path.join(params['images_root'], img['filepath'], img['filename']))
    elif params['dataset'] == 'flickr30k':
      image = Image.open(os.path.join(params['images_root'], img['filename']))

    if image.mode != 'RGB':
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
      outputs = model(**inputs)
      last_hidden_state = outputs.last_hidden_state.squeeze()
      pooled_output = outputs.pooler_output # pooled CLS states
    # write to pkl
    if params['dataset'] == 'mscoco':
      np.save(os.path.join(dir_fc, str(img['cocoid'])), pooled_output.cpu().float().numpy())
      np.savez_compressed(os.path.join(dir_att, str(img['cocoid'])), feat=last_hidden_state.cpu().float().numpy())
    elif params['dataset'] == 'flickr30k':
      np.save(os.path.join(dir_fc, img['filename'][:-4]), pooled_output.cpu().float().numpy())
      np.savez_compressed(os.path.join(dir_att, img['filename'][:-4]), feat=last_hidden_state.cpu().float().numpy())      

    if i % 1000 == 0:
      print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
  print('wrote ', params['output_dir'])

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--dataset', default='mscoco', help='mscoco|flickr30k')
  parser.add_argument('--input_json', default='data/dataset_coco.json', help='input json file to process into hdf5')
  parser.add_argument('--output_dir', default='data/clip-vit-large-patch14', help='feature output directory.')

  # options
  parser.add_argument('--images_root', default='data/coco_images', help='root location in which images are stored.')
  parser.add_argument('--model_root', default='checkpoint/clip-vit-large-patch14', type=str, help='model root. Please download the corresponding files from https://huggingface.co/openai')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)