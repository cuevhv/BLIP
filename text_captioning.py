import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import warnings
warnings.filterwarnings('ignore')


tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-chat-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True, load_in_8bit=True
).eval()


# chat example

query = ("Describe the person in the scene, what the person is wearing. "
        "If there is a green screen, don't describe the green screen."
        "If there is a head band, don't descbre the head band.")
print(query)
image = Image.open("/home/hcuevas/Desktop/01186.jpg").convert('RGB')
image.save("tmp.png")
inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": True,
              "top_p": 0.4, "top_k": 5, "temperature": 0.8,
            #   "num_beams": 10, "early_stopping": True,
              }

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
