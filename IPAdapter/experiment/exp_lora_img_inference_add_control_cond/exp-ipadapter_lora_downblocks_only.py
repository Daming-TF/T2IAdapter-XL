import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import sys
import cv2
from controlnet_aux import OpenposeDetector
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
sys.path.append(r'/home/mingjiahui/project/T2IAdapter-XL')
from IPAdapter.util import load_model, set_parser


def main(args):
    # init
    image_dir = r'/home/mingjiahui/data/ipadapter/test_data/pose_lora/lora'
    save_dir = r'./output/ipadapter/exp_lora'
    os.makedirs(save_dir, exist_ok=True)

    # prepare
    # 1.image paths
    image_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir)][:1]
    # 2.scales
    scales = np.arange(0, 1.1, 0.1).tolist()
    # scales = [1]
    # 3.cond img
    lora_name = ['Paper_Cutout.safetensors', 'Chinese_Aesthetic_Illustration.safetensors'][0]
    lora_id = lora_name.split('.')[0]
    save_dir = os.path.join(save_dir, lora_id)
    os.makedirs(save_dir, exist_ok=True)

    # load ip model
    print(r'loading model......')
    ip_model = load_model(
        base_model_path=args.base_model_path,
        image_encoder_path=args.image_encoder_path,
        ip_ckpt=args.ip_ckpt,
        vae_model_path=args.vae_model_path,
        unet_load=True, # load unetV1_1: encoder hidden states support input lists
        # controlnet_model_path=args.controlnet_model_path,
        lora_name=lora_name
    )

    # process
    for index, image_path in enumerate(image_paths):
        image_id = os.path.basename(image_path).split('.')[0]
        save_dir_ = os.path.join(save_dir, image_id)
        os.makedirs(save_dir_, exist_ok=True)
        image = Image.open(image_path).resize((512, 512))

        result = None
        for scale in scales:
            outputs = ip_model.generate(
                pil_image=image,
                num_samples=1,
                num_inference_steps=20,
                seed=42,
                guidance_scale=7,
                prompt='1old lady',
                scale=scale,
                cross_attention_kwargs={
                        'down_blocks': 'txt_img', 
                        'mid_block': 'txt_img', 
                        'up_blocks': 'txt_img',
                        'scale': 0.5
                        }
            )

            out_put = np.array(outputs[0])
            result = cv2.hconcat([result, out_put]) if result is not None else out_put

        Image.fromarray(result).save(os.path.join(save_dir_, f'{image_id}-lora{0.5}.jpg'))


if __name__ == '__main__':
    args = set_parser()

    main(args)
