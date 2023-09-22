# From https://github.com/carolineec/informative-drawings
# MIT License

import os
import torch
import numpy as np
from PIL import Image
import argparse
from torchvision import transforms
import multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        norm_layer = nn.InstanceNorm2d
        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features)
                      ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()
        norm_layer = nn.InstanceNorm2d
        # Initial convolution block
        model0 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, 64, 7),
                  norm_layer(64),
                  nn.ReLU(inplace=True)]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                       norm_layer(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                       norm_layer(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class LineartDetector:
    def __init__(self):
        self.model = self.load_model('sk_model.pth')
        self.model_coarse = self.load_model('sk_model2.pth')

    def load_model(self, name):
        modelpath = os.path.join("/mnt/nfs/file_server/public/mingjiahui/models/sketch/", name)
        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
        model.eval()
        model = model.cuda()
        return model

    def __call__(self, input_image, coarse):
        model = self.model_coarse if coarse else self.model

        # verify validity
        assert input_image.shape[-1] == 3       # RGB
        with torch.no_grad():
            if isinstance(input_image, np.ndarray):
                image = torch.from_numpy(input_image).float().cuda()
            else:
                image = input_image.float()     # uint8 => float32

            if image.ndim == 3:
                image = image.unsqueeze(0).permute(0, 3, 1, 2)
            elif image.ndim == 4:
                image = image.permute(0, 3, 1, 2)
            outputs = model(image)

        # transformer output side by side
        res = []
        for output in torch.clamp(outputs, 0, 1):
            output = transforms.ToPILImage()(output.cpu())
            res.append(output)

        return res


def process_lineart(list_per_process, args, gpu_id):
    # # TODO: debug
    # if gpu_id != 0:
    #     exit(0)

    torch.cuda.set_device(gpu_id)
    test_sample = list_per_process[0]
    if os.path.isdir(test_sample):
        images_per_process = []
        for img_dir in tqdm(list_per_process):
            img_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.endswith('.jpg')]
            images_per_process += img_paths
    elif os.path.isfile(test_sample) and test_sample.endswith('.jpg'):
        images_per_process = list_per_process
    else:
        ValueError("funcation of 'list_per_process' must input Uion[image dirs list / image paths list] !!")
        exit(1)

    _images_per_process = images_per_process[args.start_index*args.batch_size:]
    print(f'Test:\t{int(len(images_per_process)/args.batch_size)+1} '
          f'==> {int(len(_images_per_process)/args.batch_size)+1}')
    # exit(0)

    # # Number of doka statistical datasets
    # print(f'{gpu_id}:\t{len(images_per_process)}')
    # exit(0)

    # prepare data
    from .dataset import CondExtractorDataset
    test_dataset = CondExtractorDataset(
        _images_per_process,
        resolution=args.resolution,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=None
    )

    # preprocess
    os.makedirs(args.output_path, exist_ok=True)
    model = LineartDetector()

    # Run
    with torch.no_grad():
        print('running .....')
        print(f'images result will save in ==> {args.output_path}')
        progress_bar = tqdm(test_loader)
        for index, data in enumerate(progress_bar):
            # if index < args.start_index:
            #     print(f'{index} < {args.start_index}')
            #     continue
            images = data['data'].to('cuda')
            paths = data['path']
            error = data['error']

            if True in error:
                print(f'something error happening,so throw all the batch pic')
                continue

            # skip
            all_finish = True
            for path in paths:
                cond_dir = os.path.join(args.output_path, os.path.basename(os.path.dirname(path)))
                os.makedirs(cond_dir, exist_ok=True)
                cond_path = os.path.join(
                    cond_dir, os.path.splitext(os.path.basename(path))[0] + '.png'
                )
                if not os.path.exists(cond_path):
                    all_finish = False
                    break
            if all_finish:
                # print(cond_path)
                print(f'{gpu_id}:\tskipping...... {index+args.start_index}|{len(test_loader)+args.start_index}')
                continue

            lineart_images = model(images, coarse=False)
            # print('saving the image......')
            for i, lineart_image in enumerate(lineart_images):
                save_dir = os.path.join(args.output_path, os.path.basename(os.path.dirname(paths[i])))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(
                    save_dir, os.path.splitext(os.path.basename(paths[i]))[0] + '.png'
                )
                lineart_image.save(save_path)

    # model = LineartDetector()
    # image_path = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test/000000000285.jpg'
    # np_image = np.array(Image.open(image_path)) / 255.0
    # print(np_image.ndim)
    # input = np.repeat(np_image[np.newaxis, ...], 4, axis=0)
    # print(input.ndim)
    # lineart_images = model(input, coarse=False)
    #
    # for i, lineart_image in enumerate(lineart_images):
    #     lineart_image.save(f'/home/mingjiahui/data/{i}_debug.png')




