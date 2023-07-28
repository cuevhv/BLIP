from PIL import Image
import torch
from torchvision import transforms
from multiprocessing.pool import Pool
from torchvision.transforms.functional import InterpolationMode


def load_demo_image(img_fn: str, image_size: int, device: str = "cpu"):
    raw_image = Image.open(img_fn).convert('RGB')   
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


def caption_image(img_fn, image_size, model, device):
    # report a message
    image = load_demo_image(img_fn, image_size=image_size, device=device)
    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        print('caption: '+caption[0])
    torch.cuda.empty_cache()
    return {"source": img_fn, "prompt": caption[0]}


def parallel_caption_images(img_fns, image_size, model, device):
    n_images = len(img_fns)
    pool_items = [(img_fns[i], image_size, model, device) for i in range(n_images)]
    with Pool() as pool:
        imgs_captions_list = pool.starmap(caption_image, pool_items)
        # print(imgs_captions_list)
    return imgs_captions_list


def caption_images(img_fns, model, image_size, device):
    imgs_captions_list = []
    for img_fn in img_fns:
        img_caption_dict = caption_image(img_fn, image_size, model, device)
        imgs_captions_list.append(img_caption_dict)
    
    return imgs_captions_list
