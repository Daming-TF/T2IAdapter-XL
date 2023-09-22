from omegaconf import OmegaConf
import torch
import os
import sys
import cv2
import datetime
from huggingface_hub import hf_hub_url
import subprocess
import shlex
import copy
import numpy as np
from PIL import Image
from basicsr.utils import tensor2img
from torchvision import transforms

current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))

from tool.model_util import load_sdxl_adapter_chaeckpoit
from Adapter.Sampling import diffusion_inference
from configs.utils import instantiate_from_config
from Adapter.inference_base import get_base_argument_parser
from Adapter.extra_condition.api import get_cond_model, ExtraCondition
from Adapter.extra_condition import api
from tool.inference_sample import get_cond_image

urls = {
    'TencentARC/T2I-Adapter':[
        'models_XL/adapter-xl-canny.pth', 'models_XL/adapter-xl-sketch.pth',
        'models_XL/adapter-xl-openpose.pth', 'third-party-models/body_pose_model.pth',
        'third-party-models/table5_pidinet.pth'
    ]
}

# if os.path.exists('checkpoints') == False:
#     os.mkdir('checkpoints')
# for repo in urls:
#     files = urls[repo]
#     for file in files:
#         url = hf_hub_url(repo, file)
#         name_ckp = url.split('/')[-1]
#         save_path = os.path.join('checkpoints',name_ckp)
#         if os.path.exists(save_path) == False:
#             subprocess.run(shlex.split(f'wget {url} -O {save_path}'))

# config
parser = get_base_argument_parser()
parser.add_argument(
    '--model_id',
    type=str,
    default="stabilityai/stable-diffusion-xl-base-1.0",
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

# add
parser.add_argument(
    '--input',
    type=str,
    required=True
)
parser.add_argument(
    '--output',
    type=str,
    required=True,
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=4,
)
parser.add_argument(
    '--cond',
    type=str,
    default=None,
)

global_opt = parser.parse_args()
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == '__main__':
    # init
    config = OmegaConf.load(global_opt.config)
    os.makedirs(global_opt.output, exist_ok=True)
    cond_name = config.model.params.adapter_config.name
    root_results = os.path.join(global_opt.output, cond_name)
    os.makedirs(root_results, exist_ok=True)

    # Adapter creation
    adapter_config = config.model.params.adapter_config
    adapter = instantiate_from_config(adapter_config).cuda()
    adapter.load_state_dict(torch.load(config.model.params.adapter_config.pretrained))
    # adapter.load_state_dict(
    #     load_sdxl_adapter_chaeckpoit(config.model.params.adapter_config.pretrained)[2]
    # )
    cond_model = get_cond_model(global_opt, getattr(ExtraCondition, cond_name))
    process_cond_module = getattr(api, f'get_cond_{cond_name}')

    # diffusion sampler creation
    sampler = diffusion_inference(global_opt.model_id)
    
    # # diffusion generation
    if os.path.isdir(global_opt.input):
        img_paths = [os.path.join(global_opt.input, name) for name in os.listdir(global_opt.input) if name.endswith('.jpg')]
        prompts = []
        for img_path in img_paths:
            prompt_path = img_path.replace('.jpg', '.txt')
            with open(prompt_path, 'r')as f:
                prompts.append(f.readline().strip())
    elif os.path.exists(global_opt.input) and global_opt.input.endswith('.jpg'):
        img_paths = [global_opt.input]
        prompts = [global_opt.prompt]
    else:
        ValueError(f"input is not legitimacy ==> {global_opt.input}")
        exit(1)

    for index, (img_path, prompt) in enumerate(zip(img_paths, prompts)):
        # cond = process_cond_module(
        #     global_opt,
        #     img_path,
        #     cond_inp_type = global_opt.in_type,
        #     cond_model = cond_model
        # )
        cond = get_cond_image(None, img_path, global_opt.cond, root_results)
        to_pil = transforms.ToPILImage()
        cond = to_pil(cond.squeeze())
        height, width = np.asarray(cond).shape
        inverted_img = np.ones((height, width), np.uint8) * 255
        # inverted_img[cond.cpu().numpy() > 128] = 0
        inverted_img[np.asarray(cond) > 128] = 0
        cond = Image.fromarray(inverted_img)
        # cond.save(os.path.join(root_results, 'debug.jpg'))
        image_transforms = transforms.Compose([
            transforms.Resize(size=1024, interpolation=3),
            transforms.CenterCrop(size=1024),
            transforms.ToTensor(),
        ])
        cond = image_transforms(cond).to('cuda').unsqueeze(0)

        with torch.no_grad():
            adapter_features = adapter(cond)
            print(
                f"***********************************************\n"
                f"Index:\t{index}"
                f"Seed:\t{global_opt.seed}\n"
                f"prompt:\t{prompt}\n"
                f"negative prompt:\t{global_opt.neg_prompt}\n"
                f"***********************************************"
            )
            results = sampler.inference(
                prompt = prompt,
                prompt_n = global_opt.neg_prompt,
                steps = global_opt.steps,
                adapter_features = copy.deepcopy(adapter_features),
                guidance_scale = global_opt.scale,
                size = (cond.shape[-2], cond.shape[-1]),
                seed= global_opt.seed,
                batch_size=4
            )

        # save results
        im_cond = tensor2img(cond)
        img_id = os.path.basename(img_path).split('.')[0]
        cv2.imwrite(os.path.join(root_results, img_id + '_condition.png'), im_cond)

        for i, result in enumerate(results):
            # now = datetime.datetime.now()
            # formatted_date = now.strftime("%Y-%m-%d")
            # formatted_time = now.strftime("%H:%M:%S")
            cv2.imwrite(os.path.join(root_results, img_id + f'_{i}.png'), result)

