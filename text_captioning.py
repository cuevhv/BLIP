import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import argparse
import os
import warnings
import json

warnings.filterwarnings('ignore')


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_txt', type=str, required=True,
                        help="either file with images or single image")
    parser.add_argument('--out_json_fn', type=str, default="out.json",
                        help="output file name in json format")
    parser.add_argument('-b', '--bath_size', type=int, default=1,
                        help="output file name in json format")
    return parser.parse_args()


def batch_images(tokenizer, queries, images):
    inputs = []
    for i, image in enumerate(images):
        if type(queries) is str:
            query = queries
        else:
            query = queries[i]
        inputs.append(model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image]))
    return inputs


def recur_move_to(item, tgt, criterion_func):
    """Taken from https://github.com/THUDM/CogVLM/issues/143#issuecomment-1835389727"""
    if criterion_func(item):
        device_copy = item.to(tgt)
        return device_copy
    elif isinstance(item, list):
        return [recur_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item

def collate_fn(features, tokenizer) -> dict:
    """Taken from https://github.com/THUDM/CogVLM/issues/143#issuecomment-1835389727"""
    images = [feature.pop('images') for feature in features]
    tokenizer.padding_side = 'left'
    padded_features = tokenizer.pad(features)
    inputs = {**padded_features, 'images': images}
    return inputs


def create_model():
    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True, load_in_8bit=True
    ).eval()
    return model, tokenizer


"/home/hcuevas/Desktop/01186.jpg"

if __name__ == '__main__':
    args = parser()
    model, tokenizer = create_model()

    # chat example
    query = ("Describe the person in the scene, what the person is wearing. "
            "If there is a green screen, don't describe the green screen."
            "If there is a head band, don't descbre the head band.")

    if args.img_txt.endswith((".png", ".jpg", ".jpeg")):
        images_fns = [args.img_txt]
    else:
        with open(args.img_txt, 'r') as f:
            images_fns = f.readlines()
        images_fns = [x.strip() for x in images_fns]

    ouput_data = {"image": [], "text": []}

    for i in range(0, len(images_fns), args.bath_size):
        images = images_fns[i:i+args.bath_size]
        images = [Image.open(image).convert('RGB') for image in images]

        gen_kwargs = {"max_length": 2048, "do_sample": True,
                    "top_p": 0.4, "top_k": 5, "temperature": 0.8,
                    #   "num_beams": 10, "early_stopping": True,
                    }

        input_batch = collate_fn(batch_images(tokenizer, query, images), tokenizer)
        input_batch = recur_move_to(input_batch, 'cuda', lambda x: isinstance(x, torch.Tensor))
        input_batch = recur_move_to(input_batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

        with torch.no_grad():
            outputs = model.generate(**input_batch, **gen_kwargs)
            outputs = outputs[:, input_batch['input_ids'].shape[1]:]
            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        ouput_data["image"].extend(images_fns[i:i+args.bath_size])
        ouput_data["text"].extend(answers)


    with open(args.out_json_fn, 'w') as file:
        for i in range(len(ouput_data["image"])):
            json.dump({"image": ouput_data["image"][i],
                       "text": ouput_data["text"][i],
                       "conditioning_image": ouput_data["image"][i].replace("png_seqs",
                                                                            "body_correspondences_masked_seqs")},
                        file)
            file.write('\n')