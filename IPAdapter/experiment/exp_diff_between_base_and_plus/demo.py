import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import sys
import cv2
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
sys.path.append(r'/home/mingjiahui/project/T2I-Adapter-XL')
from IPAdapter.util import load_model, image_grid, set_parser


def main(args):
    # init
    print(r'loading model......')
    ip_model = load_model(
        base_model_path=args.base_model_path,
        image_encoder_path=args.image_encoder_path,
        ip_ckpt=args.ip_ckpt,
        vae_model_path=args.vae_model_path,
    )
    save_dir = r'/home/mingjiahui/data/ipadapter/exp_add_feature_to_diff_position'
    os.makedirs(save_dir, exist_ok=True)

    # get x
    scales = np.arange(0, 1.1, 0.1).tolist()
    print(scales)

    # get y
    image_dir = r'/home/mingjiahui/data/ipadapter/test_data/paper_cutout'
    images_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir)]

    result = None
    for img_path in tqdm(images_paths):
        h_output = None
        for scale in scales:
            image = Image.open(img_path)
            image.resize((512, 512))

            # process
            images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=20, seed=42,
                                       guidance_scale=7, prompt="1chick", scale=scale)

            out_put = np.array(images[0])
            h_output = cv2.hconcat([h_output, out_put]) if h_output is not None else out_put

        result = cv2.vconcat([result, h_output]) if result is not None else h_output

    Image.fromarray(result).save(os.path.join(save_dir, 'add_to_downblocks.jpg'))


if __name__ == '__main__':
    args = set_parser()

    main(args)
