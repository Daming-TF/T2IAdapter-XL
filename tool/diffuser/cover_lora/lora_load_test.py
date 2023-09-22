# import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
#
# model_base = "/mnt/nfs/file_server/public/mingjiahui/models/runwayml--stable-diffusion-v1-5/"
#
# print('loading sd')
# pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, use_safetensors=True)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#
# print('loading lora')
# pipe.unet.load_attn_procs(r'/mnt/nfs/file_server/public/mingjiahui/models/lora/Paper_Cutout.safetensors')
# pipe.to("cuda")
#
# image = pipe(
#     "A pokemon with blue eyes.", num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}
# ).images[0]
#
# image = pipe("A pokemon with blue eyes.", num_inference_steps=25, guidance_scale=7.5).images[0]
# image.save("blue_pokemon.png")


import diffusers


def model_setup(LoRA_name='Paper_Cutout.safetensors'):
    MODEL_NAME = "/mnt/nfs/file_server/public/liujia/Models/StableDiffusion/diffusers/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a/"
    # print(MODEL_NAME)
    global pipe
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            safety_checker=None,
            torch_dtype=torch.float16,
            ).to('cuda')
    pipe.load_lora_weights('/mnt/nfs/file_server/public/mingjiahui/models/lora/', weight_name=LoRA_name)
    # # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.safety_checker = lambda images, clip_input: (images, False)
    # pipe.scheduler.set_timesteps(50)
    print('Finished setup')
    return pipe


import torch
import cv2
from PIL import Image
import numpy as np

scales = np.arange(0, 1.1, 0.1).tolist()
generator = torch.Generator('cuda').manual_seed(42)
pipe = model_setup()

result = None
for scale in scales:
    
    image = pipe(
                prompt='1old lady,seekoo_gds,best quality,sharp,',
                guidance_scale=7.5,
                num_inference_steps=20,
                generator=generator,
                cross_attention_kwargs={"scale": scale}
            ).images[0]
    result = cv2.hconcat([result, np.array(image)]) if result is not None else np.array(image)
        
Image.fromarray(result).save("./output/debug.png")

# generator = torch.Generator('cuda').manual_seed(42)
# image = pipe(
#     prompt='1old lady,seekoo_gds',
#     guidance_scale=7.5,
#     num_inference_steps=20,
#     generator=generator,
#     # cross_attention_kwargs={"scale": scale},
#     ).images[0]
# image.save(r'./output/debug.jpg')