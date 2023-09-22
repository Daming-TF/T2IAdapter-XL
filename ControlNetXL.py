from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image
import numpy as np
model_path = r'/mnt/nfs/file_server/public/mingjiahui/models/Intel--dpt-hybrid-midas'
depth_estimator = DPTForDepthEstimation.from_pretrained(model_path).to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained(model_path)
def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline, DiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils import load_image
import torch
import os
base_model_path = "/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0"
controlnet_path = "/mnt/nfs/file_server/public/lipengxiang/sdxl_depth"

controlnet = ControlNetModel.from_pretrained(controlnet_path,).half()
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

image_path = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test/000000000285.jpg'
depth_img = get_depth_map(Image.open(image_path))
depth_img.save(r'/home/mingjiahui/data/285_debug_depth.jpg')
exit(0)
# prompt1 = "a stop sign on the street"
with open(image_path.replace('.jpg', '.txt'), 'r')as f:
    prompt1 = f.readline().strip()
negative_prompt = ""
image = pipeline(
    prompt1, num_inference_steps=20, num_images_per_prompt=4, generator=None, image=depth_img, controlnet_conditioning_scale=0.5, guidance_scale=5, original_size=(1024, 1024)
).images
image.save(r'/home/mingjiahui/data/285_debug.jpg')
