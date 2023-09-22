import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import sys
import cv2
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
sys.path.append(r'/home/mingjiahui/project/T2I-Adapter-XL')
from IPAdapter.util import load_model, set_parser


def main(args):
    # init
    image_dir = r'/home/mingjiahui/data/ipadapter/test_data/single_image_inference'
    save_dir = r'/home/mingjiahui/data/ipadapter/exp_add_diff_denoising'
    os.makedirs(save_dir, exist_ok=True)

    # prepare
    image_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir)]
    scales = np.arange(0, 1.1, 0.1).tolist()
    cond_tau_list = np.arange(0, 0.6, 0.1).tolist()
    prompts = [
        '1chicken',
        # '2wolvesa,laying on top of a rock',
        '1old lady',
        # 'A little girl, flying a kit',
        'a white unicorn with wings',
        # 'two carved pumpkins with faces carved into them',
    ]

    # load model
    print(r'loading model......')
    ip_model = load_model(
        base_model_path=args.base_model_path,
        image_encoder_path=args.image_encoder_path,
        ip_ckpt=args.ip_ckpt,
        vae_model_path=args.vae_model_path,
        unet_load=True
    )

    # process
    for index, image_path in enumerate(image_paths):
        image_id = os.path.basename(image_path).split('.')[0]
        save_dir_ = os.path.join(save_dir, image_id)
        os.makedirs(save_dir_, exist_ok=True)
        image = Image.open(image_path).resize((512, 512))
        for prompt in prompts:
            result = None
            for cond_tau in cond_tau_list:
                h_concat = None
                for scale in scales:
                    outputs = ip_model.generate(
                        pil_image=image,
                        num_samples=1,
                        num_inference_steps=20,
                        seed=42,
                        guidance_scale=7,
                        prompt=prompt,
                        scale=scale,
                        cross_attention_kwargs={
                            'down_blocks': 'txt_img',
                            'mid_block': 'txt_img',
                            'up_blocks': 'txt_img',
                        },
                        cond_tau=cond_tau
                    )

                    out_put = np.array(outputs[0])
                    h_concat = cv2.hconcat([h_concat, out_put]) if h_concat is not None else out_put
                result = cv2.vconcat([result, h_concat]) if result is not None else h_concat
            Image.fromarray(result).save(os.path.join(save_dir_, f'{prompt}.jpg'))


if __name__ == '__main__':
    args = set_parser()

    main(args)
