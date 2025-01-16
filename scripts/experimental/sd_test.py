from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
from PIL import Image
import torch
import os

seed_parent = os.path.join(r"D:\utils\ComfyUI_windows_portable\ComfyUI\output", "TEST")
seed_path =  os.path.join(seed_parent, "seed.png")



# pipeline = StableDiffusionPipeline.from_pretrained(
# "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
# )

# # Running the inference on GPU with cuda enabled
# pipeline = pipeline.to('cuda')

# prompt = "In the style of Pixar film, create a character in his mid 20s with dark clothes and a mean expression standing in the streets of a futuristic city"

# image = pipeline(prompt=prompt, height=512, width=512).images[0]

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "In the style of realistic contemporary art, a man in his late 30s with sharp features, piercing gray eyes, and a faint scar on his left cheek. He wears a dark, high-collared jacket with subtle metallic accents, standing in a dimly lit room illuminated by the faint glow of a nearby screen."
image = pipeline_text2image(prompt=prompt, width=1024, height=1024).images[0]
new_image = image. resize((768, 768))
new_image.save(seed_path)
print("Generated seed image")