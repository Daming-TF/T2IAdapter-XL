# From https://github.com/carolineec/informative-drawings
# MIT License

import os
import cv2
import torch
import numpy as np

import torch.nn as nn
norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

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
        # if isinstance(input_image, list):
        assert input_image.shape[-1] == 3       # RGB
        image = input_image
        with torch.no_grad():
            image = torch.from_numpy(image).float().cuda()
            if image.ndim == 3:
                image = image.unsqueeze(0).permute(0, 3, 1, 2)
            elif image.ndim == 4:
                image = image.permute(0, 3, 1, 2)
            outputs = model(image)

        # res = torch.clamp(res, 0, 1).squeeze().cpu()
        # print(res.shape)
        # print(res.dtype)
        # exit(0)
        # res = transforms.ToPILImage()(res)
        res = []
        for output in torch.clamp(outputs, 0, 1):
            print(output.shape)
            output = transforms.ToPILImage()(output.cpu())
            res.append(output)

        return res


if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    model = LineartDetector()

    # # base
    # image_path = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test/000000000285.jpg'
    # np_image = np.array(Image.open(image_path)) / 255.0
    # print(np_image.ndim)
    # input = np.repeat(np_image[np.newaxis, ...], 4, axis=0)
    # print(input.ndim)
    # lineart_images = model(input, coarse=False)
    #
    # for i, lineart_image in enumerate(lineart_images):
    #     lineart_image.save(f'/home/mingjiahui/data/{i}_debug.png')

    # inference test
    image_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test_v2'
    # save_dir = r'../../data/inference_test'
    # os.makedirs(save_dir, exist_ok=True)

    image_paths = [os.path.join(image_dir, name)for name in os.listdir(image_dir) if name.endswith('.jpg')]
    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')
        np_image = np.array(img) / 255.0
        lineart_images = model(np_image, coarse=False)

        for lineart_image in lineart_images:
            save_path = os.path.join(image_dir, f"{os.path.basename(image_path).split('.')[0]}-orisize.png")
            print(save_path)
            lineart_image.save(save_path)



