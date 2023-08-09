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


def caption_image(idx, img_fn, image_size, model, processor, device):
    # report a message
    with torch.no_grad():
        # beam search
        if processor is None:
            image = load_demo_image(img_fn, image_size=image_size, device=device)
            caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        else:
            image = Image.open(img_fn).convert('RGB')
            inputs = processor(image, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        with open(f'logs/done/{idx}.txt', 'w') as f:
            f.write(f"")
    torch.cuda.empty_cache()
    return {"target": img_fn, "prompt": caption[0]}


def parallel_caption_images(img_fns, image_size, model, processor, device):
    n_images = len(img_fns)
    print("number of images: ", n_images)
    pool_items = [(i, img_fns[i], image_size, model, processor, device) for i in range(n_images)]
    with Pool() as pool:
        imgs_captions_list = pool.starmap(caption_image, pool_items)
        # print(imgs_captions_list)
    return imgs_captions_list


def caption_images(img_fns, model, image_size, processor, device):
    imgs_captions_list = []
    for idx, img_fn in enumerate(img_fns):
        img_caption_dict = caption_image(idx, img_fn, image_size, model, processor, device)
        imgs_captions_list.append(img_caption_dict)

    return imgs_captions_list
