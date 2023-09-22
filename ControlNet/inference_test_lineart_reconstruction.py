from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline, DiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils import load_image
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import os
import sys


def main(args):
    # init
    base_model_path = args.model_id
    controlnet_path = args.controlnet_id      # r'/mnt/nfs/file_server/public/lipengxiang/train_512_xl_12m_data'
    cond_mode = r'-orisize'  # '.depth'   # r'-dpt-hybrid-midas'
    save_dir = args.output
    os.makedirs(save_dir, exist_ok=True)

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

    from tool.inference_sample import get_prompt, get_cond_image, get_img_paths

    # prepare input
    image_paths = get_img_paths(args.input)

    for image_path in image_paths:
        image_id = os.path.basename(image_path).split('.')[0]
        # get prompt
        prompt1 = get_prompt(args.prompt, prompt_path=image_path.replace('.jpg', '.txt'))
        # get cond
        cond_img = get_cond_image(
            model,
            image_path,
            cond_input=args.cond,
            save_dir=args.output,
            color_inversion=args.color_inversion,
            resolution=args.resolution,
        )
        if isinstance(cond_img, torch.Tensor):
            to_pil = transforms.ToPILImage()
            cond_img = to_pil(cond_img.squeeze()).convert('RGB')

        # prompt1 = "a stop sign on the street"
        # negative_prompt = 'black and white'
        # negative_prompt = 'worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, ' \
        #                   'extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, '
        negative_prompt = ''
        print(f'test: {args.seed}')
        print(f'prompt:{prompt1}')

        # processing
        images = pipeline(
            prompt1,
            num_inference_steps=args.step,      # 20
            num_images_per_prompt=args.batch_size,
            generator=torch.manual_seed(args.seed),        # 42
            image=cond_img,
            controlnet_conditioning_scale=args.scale,
            guidance_scale=8.0,
            original_size=(args.resolution, args.resolution),
            negative_prompt=negative_prompt,
            control_guidance_end=0.5
        ).images

        for i, image in enumerate(images):
            print(f"result has save in ==> {os.path.join(save_dir, f'{image_id}-{i}.jpg')}")
            image.save(os.path.join(save_dir, f'{image_id}-{i}.jpg'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id',
        type=str,
        default="/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0",
    )
    parser.add_argument(
        '--controlnet_id',
        type=str,
        default="/mnt/nfs/file_server/public/lipengxiang/train_512_xl_12m_data/checkpoint-10000/controlnet/"
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=1024
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--step',
        type=int,
        default=20
    )
    parser.add_argument(
        '--cond',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
    )
    # TODO: sketch image putin linart controlnet
    parser.add_argument(
        '--color_inversion',
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    main(args)
