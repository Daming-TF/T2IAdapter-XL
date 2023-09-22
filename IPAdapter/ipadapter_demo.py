from PIL import Image
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from IPAdapter.util import load_model, image_grid, set_parser


def main(args):
    print(r'loading model......')
    ip_model = load_model(
        base_model_path=args.base_model_path,
        image_encoder_path=args.image_encoder_path,
        ip_ckpt=args.ip_ckpt,
        vae_model_path=args.vae_model_path,
        unet_load=True
    )

    # read image prompt (face, here we use a ai-generation face)
    image = Image.open("/home/mingjiahui/data/ipadapter/test_data/paper_cutout/paper_cutout_0.png")    # ai_face.png， statue.png
    image.save(r'./output/img.jpg')
    image.resize((512, 512))

    # process
    # images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42, scale=0)
    # best quality, high quality, wearing sunglasses in a garden
    images = ip_model.generate(pil_images=image, num_samples=1, num_inference_steps=20, seed=42,
                               guidance_scale=7, prompt="1chick", scale=0.5)

    grid = image_grid(images, 1, 1)
    save_path = r'./output/debug.jpg'
    print(f'save to ==> {save_path}')
    grid.save(save_path)


if __name__ == '__main__':
    args = set_parser()
    main(args)
