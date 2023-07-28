"""python img_caption.py --file_fn dataset/evermotion_dataset/out_list.txt --out_json_fn out.json --parallel
"""
import torch
import torch.multiprocessing as mp
from models.blip import blip_decoder
import argparse
import os
import json
import time
import datetime

from utils.captioning import parallel_caption_images, caption_images


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_fn', type=str, default=None)
    argparser.add_argument('--file_fn', type=str, help="either file with images or single image")
    argparser.add_argument('--out_json_fn', type=str, help="output file name in json format")
    argparser.add_argument('--parallel', action='store_true', help="parallelize the captioning process")
    return argparser.parse_args()


def get_file_names(file_fn: str):
    file_extension = os.path.splitext(file_fn)[-1]
    if file_extension in ['.txt']:
        with open(file_fn, 'r') as f:
            file_names = f.readlines()
        file_names = [x.strip() for x in file_names]
        return file_names
    elif file_extension in ['.jpg', '.png', '.jpeg']:
        return [file_fn]
    else:
        raise ValueError(f'file extension {file_extension} not supported')


def main(cfg):
    image_size = 384
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    img_fns = get_file_names(cfg.file_fn)
    print(f"total number of images: {len(img_fns)}")
    print("starting the captioning process...")
    
    if cfg.parallel:
        s_time = time.time()
        imgs_captions_list = parallel_caption_images(img_fns, image_size, model, device)
    else:
        s_time = time.time()
        imgs_captions_list = caption_images(img_fns, model, image_size, device)
    print("It took to process the data: ", str(datetime.timedelta(seconds=time.time()-s_time)))
    

    with open(cfg.out_json_fn, 'w') as file:
        for entry in imgs_captions_list:
            json.dump(entry, file)
            file.write('\n')


if __name__ == "__main__":
    cfg = args()
    if cfg.parallel:
        mp.set_start_method('spawn')
    main(cfg)