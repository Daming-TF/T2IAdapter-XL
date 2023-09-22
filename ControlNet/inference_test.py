from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline, DiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils import load_image
from PIL import Image
import torch
import os
base_model_path = "/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0"
controlnet_path = "/mnt/nfs/file_server/public/lipengxiang/sdxl_depth"      # r'/mnt/nfs/file_server/public/lipengxiang/train_512_xl_12m_data'

# prepare model
controlnet = ControlNetModel.from_pretrained(controlnet_path,).half().to('cuda')
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16
).to('cuda')
print(controlnet.device)
print(pipeline.device)

from torchvision import transforms
image_transforms = transforms.Compose([
        transforms.Resize(size=1024, interpolation=3),
        transforms.CenterCrop(size=1024),
        # transforms.ToTensor(),
    ])

# prepare input
data_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test'
save_dir = r'/home/mingjiahui/data/controlnet'
os.makedirs(save_dir, exist_ok=True)
depth_mode = r'-dpt-hybrid-midas'       # '.depth'   # r'-dpt-hybrid-midas'
image_ids = [name.split('.')[0] for name in os.listdir(data_dir) if name.endswith('.jpg')]
for image_id in image_ids:
    # image_path = fr'{data_dir}/{image_id}.jpg'
    txt_path = fr'{data_dir}/{image_id}.txt'
    with open(txt_path, 'r')as f:
        prompt1 = f.readline().strip()
    depth_path = rf'{data_dir}/{image_id}{depth_mode}.png'
    depth_img = Image.open(depth_path)
    depth_img = image_transforms(depth_img)
    depth_img.save(os.path.join(save_dir, f'{image_id}-depth.jpg'))

    # prompt1 = "a stop sign on the street"
    negative_prompt = 'extra digit, fewer digits, cropped, worst quality, low quality'
    # negative_prompt = 'worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, ' \
    #                   'extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, '

    print('no negative prompt')
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
        # negative_prompt=negative_prompt
    ).images

    for i, image in enumerate(images):
        image.save(os.path.join(save_dir, f'{i}--{image_id}{depth_mode}-debug.jpg'))

    # print(negative_prompt)
