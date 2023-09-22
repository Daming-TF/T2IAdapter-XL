
"""
    TODO: Verify the impact of different locations where the adapter joins
"""
from omegaconf import OmegaConf
import torch
import os
import cv2
import datetime
from huggingface_hub import hf_hub_url
import subprocess
import shlex
import copy
from basicsr.utils import tensor2img

from Adapter.Sampling import diffusion_inference
from configs.utils import instantiate_from_config
from Adapter.inference_base import get_base_argument_parser
from Adapter.extra_condition.api import get_cond_model, ExtraCondition
from Adapter.extra_condition import api
# add
from tool.model_util import load_sdxl_adapter_chaeckpoit
from tool.inference_sample import get_cond_image, get_img_paths, get_prompt

import numpy as np
import random
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse


def set_parser():
    # config
    parser = get_base_argument_parser()
    parser.add_argument(
        '--model_id',
        type=str,
        default='/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/',        # "stabilityai/stable-diffusion-xl-base-1.0",
        help='huggingface url to stable diffusion model',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/inference/Adapter-XL-lineart.yaml',
        help='config path to T2I-Adapter',
    )
    parser.add_argument(
        '--path_source',
        type=str,
        default='examples/dog.png',
        help='config path to the source image',
    )
    parser.add_argument(
        '--in_type',
        type=str,
        default='image',
        help='config path to the source image',
    )

    # my add
    parser.add_argument(
            '--input',
            type=str,
            required=True,
        )
    parser.add_argument(
        '--output',
        type=str,
        default=r'/home/mingjiahui/data/T2IAdapter',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=1024,
    )
    parser.add_argument(
        '--additional_scale',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--cond',
        type=str,
        default=None,
    )
    # TODO: sketch cond in lineart adapter reslut
    parser.add_argument(
        '--color_inversion',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--inversion_ratio',
        type=float,
        default=1,
    )

    global_opt = parser.parse_args()
    global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return global_opt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_gray_distributed(image, save_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    histogram = image.histogram()
    print(f'image info:\t{type(image)}\t{image.size}\t{image.mode}')
    # histogram = cv2.calcHist([image], [0], None, [256],[0, 256])
    # histogram = np.squeeze(histogram)
    # plt.figure(figsize=(8, 6))
    # plt.bar(np.arange(256), histogram, color='gray', alpha=0.7)
    plt.bar(range(256), histogram[:256], color='gray', alpha=0.7)
    plt.xlabel('gray value')
    plt.ylabel('num')
    plt.title('gray distributed')
    plt.xlim(0, 255)
    plt.savefig(save_path)


if __name__ == '__main__':
    # init
    global_opt = set_parser()
    config = OmegaConf.load(global_opt.config)

    os.makedirs(global_opt.output, exist_ok=True)
    cond_name = config.model.params.adapter_config.name
    root_results = os.path.join(global_opt.output, cond_name)
    os.makedirs(root_results, exist_ok=True)

    # Adapter creation
    adapter_config = config.model.params.adapter_config
    adapter = instantiate_from_config(adapter_config).cuda()
    adapter.load_state_dict(
        load_sdxl_adapter_chaeckpoit(config.model.params.adapter_config.pretrained)[2]
    )

    # diffusion sampler creation
    sampler = diffusion_inference(global_opt.model_id)

    # condition extractor
    from condition_extractor import LineartDetector
    cond_model = LineartDetector()

    # get test image paths
    img_paths = get_img_paths(global_opt.input)

    print(f'Image path:\t{img_paths}')
    print(f'cond:\t{global_opt.cond}')
    for index, img_path in enumerate(img_paths):
        # get cond image    priority: input > exists > cond model
        cond = get_cond_image(
            cond_model,
            img_path,
            global_opt.cond,
            save_dir=root_results,
            color_inversion=global_opt.color_inversion,
            resolution=global_opt.resolution,
            inversion_ratio=global_opt.inversion_ratio,
        )

        # get prompt    priority: input > exists
        prompt = get_prompt(global_opt.prompt, prompt_path=img_path.replace('.jpg', '.txt'))

        # get negative prompt
        global_opt.neg_prompt = ''

        # running
        print(
            f"********************* Reconstruction **************************\n"
            f"Index:\t{index}"
            f"Seed:\t{global_opt.seed}\n"
            f"prompt:\t{prompt}\n"
            f"negative prompt:\t{global_opt.neg_prompt}\n"
            f"***********************************************"
        )
        print(f'additional_scale:{global_opt.additional_scale}')
        print(f'inversion_ratio:{global_opt.inversion_ratio}')
        with torch.no_grad():
            adapter_features = adapter(cond)
            outputs = sampler.inference(
                prompt=prompt,
                prompt_n=global_opt.neg_prompt,
                steps=global_opt.steps,
                adapter_features=copy.deepcopy(adapter_features),
                guidance_scale=global_opt.scale,
                size=(cond.shape[-2], cond.shape[-1]),
                seed=global_opt.seed,
                batch_size=global_opt.batch_size,
                # additional_scale=global_opt.additional_scale,       # 1.0
            )

        for i, output in enumerate(outputs):
            seed = np.random.get_state()[1][0]
            save_path = os.path.join(root_results, f"{os.path.basename(img_path).split('.')[0]}_seed{seed}_image_{i}.png")
            print(save_path)
            cv2.imwrite(save_path, output)
            # Image.fromarray(output).save(save_path)