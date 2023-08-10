from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import cv2
from PIL import Image
import time

precision = torch.float16
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=precision)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

image = Image.open("/home/hcuevas/Documents/work/gen_bedlam/datasets/20230804_1_3000_hdri/png_untar/20230804_1_3000_hdri_png.0/20230804_1_3000_hdri/png/seq_000001/seq_000001_0010.png").convert('RGB')
image = [image, image, image]
s_time = time.time()
inputs = processor(image, return_tensors="pt").to(device, precision)
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)#[0].strip()
    print(generated_text)
print("It took to process the data: ", str(time.time()-s_time))