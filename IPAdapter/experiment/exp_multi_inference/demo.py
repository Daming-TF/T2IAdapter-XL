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
    image_dir = r'/home/mingjiahui/data/ipadapter/test_data/multi_image_inference'
    save_dir = r'/home/mingjiahui/data/ipadapter/exp_multi_inference'
    os.makedirs(save_dir, exist_ok=True)

    # get inference image
    images = [Image.open(os.path.join(image_dir, name)).resize((512, 512)) for name in os.listdir(image_dir)[:2]]

    # get x
    scales = np.arange(0, 1.1, 0.1).tolist()
    print(scales)

    # feature mode
    feature_mode = ['simple', 'avg_embeds', 'token_concat']
    prompts = [
        '1chick',
        '2wolvesa,laying on top of a rock',
        '1old lady',
        'A little girl, flying a kit',
        'a white unicorn with wings',
        'two carved pumpkins with faces carved into them',
    ]

    # load model
    print(r'loading model......')
    ip_model = load_model(
        base_model_path=args.base_model_path,
        image_encoder_path=args.image_encoder_path,
        ip_ckpt=args.ip_ckpt,
        vae_model_path=args.vae_model_path,
    )

    for prompt in prompts:
        result = None
        for scale in scales:
            v_output = None
            # for image in images:
            #     # process
            #     outputs = ip_model.generate(pil_images=image, num_samples=1, num_inference_steps=20, seed=42,
            #                                guidance_scale=7, prompt=prompt, scale=scale, feature_mode='simple')
            #
            #     out_put = np.array(outputs[0])
            #     v_output = cv2.vconcat([v_output, out_put]) if v_output is not None else out_put
            #
            feature_mode.remove('simple') if 'simple' in feature_mode else None

            for mode in feature_mode:
                outputs = ip_model.generate(pil_images=images, num_samples=1, num_inference_steps=20, seed=42,
                                           guidance_scale=7, prompt=prompt, scale=scale, feature_mode=mode)

                out_put = np.array(outputs[0])
                v_output = cv2.vconcat([v_output, out_put]) if v_output is not None else out_put

            result = cv2.hconcat([result, v_output]) if result is not None else v_output

        Image.fromarray(result).save(os.path.join(save_dir, f'{prompt}-exp_multi_inference.jpg'))


if __name__ == '__main__':
    args = set_parser()

    main(args)
