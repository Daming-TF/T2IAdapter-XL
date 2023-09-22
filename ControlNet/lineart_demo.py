from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline, DiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch
import os
import sys
base_model_path = "/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0"
controlnet_path = "/mnt/nfs/file_server/public/lipengxiang/train_512_xl_12m_data/checkpoint-10000/controlnet/"      # r'/mnt/nfs/file_server/public/lipengxiang/train_512_xl_12m_data'

# prepare XL model
controlnet = ControlNetModel.from_pretrained(controlnet_path,).half().to('cuda')
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16
).to('cuda')
print(controlnet.device)
print(pipeline.device)

# prepare lineart model
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from condition_extractor.lineart_multi import LineartDetector
model = LineartDetector()

from torchvision import transforms
image_transforms = transforms.Compose([
        transforms.Resize(size=1024, interpolation=3),
        transforms.CenterCrop(size=1024),
        # transforms.ToTensor(),
    ])

# prepare input
data_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2'
save_dir = r'/home/mingjiahui/data/controlnet'
os.makedirs(save_dir, exist_ok=True)

# get path
image_path = r'/mnt/nfs/file_server/public/lipengxiang/improved_aesthetics_6plus_out/00000/000000222.jpg'
txt_path = image_path.replace('.jpg', '.txt')
cond_path = r'/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/lineart_align_controlnet/00000/000000222.png'

# get cond img
if os.path.exists(cond_path):
    cond_img = Image.open(cond_path)
    cond_img = image_transforms(cond_img)
else:
    img = Image.open(image_path).convert('RGB')
    np_image = np.array(image_transforms(img)) / 255.0
    cond_img = model(np_image, coarse=False)
cond_img.save(os.path.join(save_dir, 'debug_linear.jpg'))

# get prompt
# prompt1 = 'Scenic View Of Lake Against Sky During Winter'
prompt1 = 'snow mountains, snow field, water, reflections'
# prompt1 = 'From the Angle of shooting from the lake to the mountain, ' \
#           'the lake is full of stones but there is a clear reflection of the snow mountain, ' \
#           'the vast snow, ' \
#           'the snow mountain, ' \
#           'the forest'
# with open(txt_path, 'r')as f:
#     prompt1 = f.readline().strip()
#     print(f'Prompt:\t{prompt1}')

# negative_prompt = 'extra digit, fewer digits, cropped, worst quality, low quality'
# negative_prompt = 'worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, ' \
#                   'extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, '
negative_prompt = 'black and white'

# processing
images = pipeline(
    prompt1,
    num_inference_steps=20,
    num_images_per_prompt=4,
    generator=torch.manual_seed(42),
    image=cond_img,
    controlnet_conditioning_scale=0.5,
    guidance_scale=8,
    # original_size=(1024, 1024),
    negative_prompt=negative_prompt
).images

for i, image in enumerate(images):
    image.save(os.path.join(save_dir, f'debug_res_{i}.jpg'))

# print(negative_prompt)
