from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline, DiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils import load_image
from PIL import Image
import torch
import os
base_model_path = "/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0"
controlnet_path = "/mnt/nfs/file_server/public/lipengxiang/sdxl_depth"

# prepare model
controlnet = ControlNetModel.from_pretrained(controlnet_path,).half().to('cuda')
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16
).to('cuda')
print(controlnet.device)
print(pipeline.device)

# prepare input
data_dir= r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test'
depth_mode = '.depth'   # r'-dpt-hybrid-midas'
image_id = '000000000285'
# image_path = fr'{data_dir}/{image_id}.jpg'
txt_path = fr'{data_dir}/{image_id}.txt'
with open(txt_path, 'r')as f:
    prompt1 = f.readline().strip()
depth_path = rf'{data_dir}/{image_id}{depth_mode}.png'
depth_img = Image.open(depth_path)
# prompt1 = "a stop sign on the street"
negative_prompt = 'worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, ' \
                 'extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, ' \
                 'monochrome, horror, geometry, mutation, disgusting'
print(f'image mode:\t{depth_img.mode}')

# processing
images = pipeline(
    prompt1,
    num_inference_steps=20,
    num_images_per_prompt=4,
    generator=None,
    image=depth_img,
    controlnet_conditioning_scale=0.5,
    guidance_scale=5,
    original_size=(1024, 1024),
    negative_prompt=negative_prompt
).images

for i, image in enumerate(images):
    image.save(rf'/home/mingjiahui/data/{i}--{image_id}-debug_depth_v2.jpg')
