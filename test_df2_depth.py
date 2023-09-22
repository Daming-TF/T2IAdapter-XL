
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
    default='configs/inference/Adapter-XL-depth.yaml',
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
# parser.add_argument(
#     '--seed',
#     type=int,
#     default=None,
# )
global_opt = parser.parse_args()
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


import numpy as np
import random
from PIL import Image
from torchvision import transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    config = OmegaConf.load(global_opt.config)
    # Adapter creation
    cond_name = config.model.params.adapter_config.name
    adapter_config = config.model.params.adapter_config
    adapter = instantiate_from_config(adapter_config).cuda()
    adapter.load_state_dict(
        load_sdxl_adapter_chaeckpoit(config.model.params.adapter_config.pretrained)[2]
    )

    # diffusion sampler creation
    sampler = diffusion_inference(global_opt.model_id)

    # # diffusion generation
    # cond_model = get_cond_model(global_opt, getattr(ExtraCondition, cond_name))
    # process_cond_module = getattr(api, f'get_cond_{cond_name}')
    # cond = process_cond_module(
    #     global_opt,
    #     global_opt.path_source,
    #     cond_inp_type=global_opt.in_type,
    #     cond_model=cond_model
    # )
    image_transforms = transforms.Compose([
        transforms.Resize(size=1024, interpolation=3),
        transforms.CenterCrop(size=1024),
        transforms.ToTensor(),
    ])
    depth_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test'
    save_dir = '/home/mingjiahui/data/T2IAdapter'
    os.makedirs(save_dir, exist_ok=True)
    depth_paths = [os.path.join(depth_dir, name) for name in os.listdir(depth_dir) if '-dpt-hybrid-midas.png' in name]
    for depth_path in depth_paths:

        # depth_path = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test/000000000785.depth.png'
        depth = Image.open(depth_path)
        cond = image_transforms(depth).to('cuda').unsqueeze(0)
        with open(depth_path.replace('-dpt-hybrid-midas.png', '.txt')) as f:
            global_opt.prompt = f.readline().strip()
            print(global_opt.prompt)

        print(
            f"***********************************************\n"
            f"Seed:\t{global_opt.seed}\n"
            f"prompt:\t{global_opt.prompt}\n"
            f"negative prompt:\t{global_opt.neg_prompt}\n"
            f"***********************************************"
        )
        with torch.no_grad():
            adapter_features = adapter(cond)
            result = sampler.inference(
                prompt=global_opt.prompt,
                # prompt_n=global_opt.neg_prompt,
                steps=global_opt.steps,
                adapter_features=copy.deepcopy(adapter_features),
                guidance_scale=global_opt.scale,
                size=(cond.shape[-2], cond.shape[-1]),
                seed=global_opt.seed,
            )

        # save results
        root_results = os.path.join(save_dir, cond_name)
        if not os.path.exists(root_results):
            os.makedirs(root_results)
        now = datetime.datetime.now()
        formatted_date = now.strftime("%Y-%m-%d")
        formatted_time = now.strftime("%H:%M:%S")
        im_cond = tensor2img(cond)
        # cv2.imwrite(os.path.join(root_results, formatted_date+'-'+formatted_time+'_image.png'), result)
        # cv2.imwrite(os.path.join(root_results, formatted_date+'-'+formatted_time+'_condition.png'), im_cond)
        seed = np.random.get_state()[1][0]
        save_path = os.path.join(root_results, f"{os.path.basename(depth_path).split('.')[0]}_seed{seed}_image.png")
        print(save_path)
        cv2.imwrite(save_path, result)
