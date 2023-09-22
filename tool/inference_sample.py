import torch
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
from basicsr.utils import tensor2img
from ldm.models.diffusion.ddim import DDIMSampler


# def inference_sample(val_dataloader, model, model_ad, device, opt):
#     samples = []
#     for index, data in enumerate(val_dataloader):
#         res = []
#         with torch.no_grad():
#             # get sample img
#             sampler = DDIMSampler(model.module)
#             c = model.module.get_learned_conditioning(data['txt'])
#             features_adapter = model_ad(data['depth'].to(device))
#             shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
#
#             samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
#                                              conditioning=c,
#                                              batch_size=opt.n_samples,
#                                              shape=shape,
#                                              verbose=False,
#                                              unconditional_guidance_scale=opt.scale,
#                                              unconditional_conditioning=model.module.get_learned_conditioning(
#                                                  opt.n_samples * [""]),
#                                              eta=opt.ddim_eta,
#                                              x_T=None,
#                                              features_adapter=features_adapter)
#             x_samples_ddim = model.module.decode_first_stage(samples_ddim)
#             x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
#             x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
#
#             for id_sample, x_sample in enumerate(x_samples_ddim):
#                 # get sample img
#                 x_sample = 255. * x_sample
#                 sample_img = x_sample.astype(np.uint8)
#                 sample_img = cv2.putText(sample_img.copy(), data['txt'][0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                   (0, 255, 0), 2)[:, :, ::-1]
#                 sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
#                 res.append(sample_img)
#
#         # get original img
#         ori_img = data['jpg'] if data['jpg'].size()[0] == 1 else data['jpg'][:-1]
#         ori_img = ori_img.squeeze(axis=0)
#         ori_img = tensor2img(ori_img, rgb2bgr=False)
#
#         # get depth map
#         depth_img = data['depth'] if data['depth'].size()[0] == 1 else data['depth'][:-1]
#         depth_img = depth_img.squeeze(axis=0)
#         depth_img = tensor2img(depth_img, rgb2bgr=False)
#
#         # concat_img = np.hstack((ori_img, depth_img)+tuple(res))
#         samples.append((ori_img, depth_img, sample_img))
#
#     return samples


def get_img_paths(input):
    if os.path.isdir(input):
        img_paths = [os.path.join(input, name) for name in os.listdir(input) if '.jpg' in name]
    elif os.path.isfile(input) and input.endswith('.jpg'):
        img_paths = [input]
    else:
        print("ERROR: the args of '--input' is not validity, please check it")
        exit(1)

    return img_paths


def get_cond_image(cond_model, img_path, cond_input, save_dir,
                   color_inversion=False, resolution=1024, inversion_ratio=1):
    image_transforms = transforms.Compose([
        transforms.Resize(size=resolution, interpolation=3),
        transforms.CenterCrop(size=resolution),
        transforms.ToTensor(),
    ])

    cond_path = img_path.replace('.jpg', '-orisize.png')
    print(f'args.cond:\t{cond_input}')
    if cond_input is not None:
        if os.path.isfile(cond_input) and os.path.basename(cond_input).split('.')[1] in ['jpg', 'png']:
            cond_path = cond_input
            cond = Image.open(cond_input).convert('L')
        elif os.path.isdir(cond_input):
            cond_path = os.path.join(cond_input, os.path.basename(img_path).split('.')[0] + '.png')
            cond = Image.open(cond_path).convert('L')
        else:
            print("Error: '--cond' is not validity, please check it again!")
            exit(1)
    elif os.path.exists(cond_path):
        cond = Image.open(cond_path)
    else:
        img = Image.open(img_path).convert('L')
        np_image = np.array(img) / 255.0
        cond = cond_model(np_image, coarse=False)

    # TODO: color inverted
    if color_inversion:
        # min_gray_value = min(cond.getdata())
        # max_gray_value = max(cond.getdata())
        # print(f'test2:{min_gray_value}~{max_gray_value}')
        # plot_gray_distributed(deepcopy(cond), os.path.join(root_results, 'plot_1.jpg'))
        cond.save(os.path.join(save_dir, os.path.basename(cond_path).split('.')[0] + '-ori.jpg'))
        height, width = np.asarray(cond).shape
        inverted_image = np.ones((height, width), np.uint8) * 255
        inverted_image[np.asarray(cond) == 255] = 255 - int(255 * inversion_ratio)
        # plot_gray_distributed(deepcopy(inverted_image), os.path.join(root_results, 'plot_2.jpg'))
        cond = Image.fromarray(inverted_image)
        # min_gray_value = min(inverted_image.getdata())
        # max_gray_value = max(inverted_image.getdata())
        # print(f'test2:{min_gray_value}~{max_gray_value}')
        # cond.save(os.path.join(root_results, 'debug1.jpg'))
    cond = image_transforms(cond).to('cuda').unsqueeze(0)

    # im_cond = Image.fromarray(tensor2img(cond))
    to_pil = transforms.ToPILImage()
    to_pil(cond.squeeze()).save(os.path.join(save_dir, os.path.basename(cond_path).split('.')[0] + '.jpg'))

    return cond


def get_prompt(prompt_input, prompt_path):
    if prompt_input is not None:
        prompt = prompt_input
    elif os.path.exists(prompt_path):
        with open(prompt_path, 'r', ) as f:
            prompt = f.readline().strip()
    else:
        print('Error: prompt is not exist, please check it!')
        exit(1)

    return prompt
