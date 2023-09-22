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
    image_dir = r'/home/mingjiahui/data/ipadapter/test_data/multi_image_inference'
    save_dir = r'/home/mingjiahui/data/ipadapter/exp_multi_inference'
    os.makedirs(save_dir, exist_ok=True)

    # get inference image
    images = [Image.open(os.path.join(image_dir, name)).resize((512, 512)) for name in os.listdir(image_dir)[:2]]

    # scale = [0.25, 0.25, 0.25, 0.25]

    # feature mode
    prompts = [
        '1chick',
        # '2wolvesa,laying on top of a rock',
        # '1old lady',
        # 'A little girl, flying a kit',
        # 'a white unicorn with wings',
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

    for prompt in prompts:
        result = None
        for index, image in enumerate(images):
            image.save(os.path.join(save_dir, f'debug{index}.jpg'))
            # process
            outputs = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=20, seed=42,
                                       guidance_scale=7, prompt=prompt, scale=1)

            out_put = np.array(outputs[0])
            result = cv2.hconcat([result, out_put]) if result is not None else out_put

        # outputs = ip_model.generate(pil_images=images, num_samples=1, num_inference_steps=20, seed=42,
        #                            guidance_scale=7, prompt=prompt, scale=scale, feature_mode='avg_feature')
        #
        # out_put = np.array(outputs[0])
        # result = cv2.hconcat([result, out_put]) if result is not None else out_put
        Image.fromarray(result).save(os.path.join(save_dir, f'{prompt}.jpg'))


if __name__ == '__main__':
    args = set_parser()

    main(args)
