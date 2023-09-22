from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline, DiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch
import os
import sys


def main(args):
    base_model_path = args.model_id
    controlnet_path = args.controlnet_id      # r'/mnt/nfs/file_server/public/lipengxiang/train_512_xl_12m_data'
    cond_mode = r'-orisize'  # '.depth'   # r'-dpt-hybrid-midas'

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
            transforms.Resize(size=args.resolution, interpolation=3),
            transforms.CenterCrop(size=args.resolution),
        ])

    # prepare input
    if os.path.isdir(args.input):
        data_dir = args.input
        image_paths = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith('.jpg')]
    elif os.path.isfile(args.input) and args.input.endswith('.jpg'):
        data_dir = os.path.dirname(args.input)
        image_paths = [args.input]
    else:
        ValueError("the args of '--input' is not validity, please check it")
        exit(1)
    save_dir = args.output
    os.makedirs(save_dir, exist_ok=True)

    for image_path in image_paths:
        image_id = os.path.basename(image_path).split('.')[0]
        # get prompt
        txt_path = image_path.replace('.jpg', '.txt')
        # txt_path = fr'{data_dir}/{image_id}.txt'
        if args.prompt is not None:
            prompt1 = args.prompt
        elif os.path.exists(txt_path):
            with open(txt_path, 'r')as f:
                prompt1 = f.readline().strip()
        else:
            ValueError("We can't find any describe prompt")
            exit(1)
        # get cond
        cond_path = rf'{data_dir}/{image_id}{cond_mode}.png'

        if os.path.exists(cond_path):
            cond_img = Image.open(cond_path)
            cond_img = image_transforms(cond_img)
        else:
            img = Image.open(os.path.join(data_dir, f'{image_id}.jpg')).convert('RGB')
            np_image = np.array(image_transforms(img)) / 255.0
            cond_img = model(np_image, coarse=False)
        cond_img.save(os.path.join(save_dir, f'{image_id}-cond.jpg'))

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
            controlnet_conditioning_scale=0.5,
            guidance_scale=8.0,
            original_size=(args.resolution, args.resolution),
            negative_prompt=negative_prompt,
            control_guidance_end = 0.5
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
    args = parser.parse_args()
    main(args)
