import argparse
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from PIL import Image
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from IPAdapter.ip_adapter.ldm import UNet2DConditionModelV1, UNet2DConditionModelV1_1
from IPAdapter.ip_adapter.ldm import StableDiffusionPipelineV1, StableDiffusionPipelineV1_1, StableDiffusionPipelineV2


def load_model(
        base_model_path, 
        image_encoder_path, 
        ip_ckpt,
        vae_model_path=None, 
        controlnet_model_path=None, 
        lora_name=None,
        device='cuda', 
        unet_load=False
        ):
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # load vae
    print(f'loading vae...... ==> {vae_model_path}')
    vae = AutoencoderKL.from_pretrained(
        vae_model_path,
        # use_safetensors=True,
        low_cpu_mem_usage=False,
        device_map=None,
    ).to(dtype=torch.float16)
    # vae = StableDiffusionPipeline.from_single_file(
    #     args.vae_model_path,
    #     torch_dtype=torch.float16,
    # )

    # load my define unet
    print(f'loading unet...... ==> {base_model_path}')
    if unet_load is True:
        unet = UNet2DConditionModelV1_1.from_pretrained(
            base_model_path,
            subfolder='unet',
        ).to(dtype=torch.float16)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder='unet',
        ).to(dtype=torch.float16)

    # load SD pipeline
    if controlnet_model_path is None:
        print(f'loading sd...... ==> {base_model_path}')
        pipe = StableDiffusionPipelineV1_1.from_pretrained(
            base_model_path,
            use_safetensors=True,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            unet=unet,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )
    else:
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
        from IPAdapter.ip_adapter.controlnet import StableDiffusionControlNetPipelineV1_1 \
            as StableDiffusionControlNetPipeline, ControlNetModelV1 as ControlNetModel
        # from IPAdapter.ip_adapter.controlnet import StableDiffusionControlNetPipelineV1_1 as StableDiffusionControlNetPipeline
        # from IPAdapter.ip_adapter.controlnet import ControlNetModelV1_1 as ControlNetModel
        print(f'loading controlnet...... ==> {controlnet_model_path}')
        controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
        print(f'loading sd...... ==> {base_model_path}')
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            unet=unet,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )
    
    # load lora
    if lora_name is not None:
        print(f'loading the lora......  ==> {lora_name}')
        pipe.load_lora_weights('/mnt/nfs/file_server/public/lichanglin/stable-diffusion-api/models/LyCORIS/', weight_name=lora_name)

    # load ip-adapter
    print(f'loading ipadapter ..... ==> {image_encoder_path}')
    from IPAdapter import IPAdapterPlus, IPAdapter
    from IPAdapter import IPAdapterV1 as IPAdapterPlus
    # from IPAdapter.ip_adapter.ip_adapter_backup import IPAdapterPlus, IPAdapter
    if 'ip-adapter_sd15' in os.path.basename(image_encoder_path):
        ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device, num_tokens=4)
    else:
        ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

    return ip_model


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    # grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_model_path',
        type=str,
        default='runwayml/stable-diffusion-v1-5'
    )
    parser.add_argument(
        '--vae_model_path',
        type=str,
        default='stabilityai/sd-vae-ft-mse'
    )
    parser.add_argument(
        '--image_encoder_path',
        type=str,
        default='models/image_encoder/',
    )
    parser.add_argument(
        '--ip_ckpt',
        type=str,
        default=r'models/ip-adapter-plus-face_sd15.bin'
    )
    parser.add_argument(
        '--controlnet_model_path',
        type=str,
    )
    parser.add_argument(
        '--img_prompt',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--cond_img',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--lora_model_path',
        type=str,
        default=None,
    )
    return parser.parse_args()



