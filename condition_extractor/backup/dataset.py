import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CondExtractorDataset(Dataset):
    def __init__(self, input_path):
        self.img_paths: [list] = list(self._get_img_path(input_path))
        print(f"ToTal num : {len(self.img_paths)}")
        self.transform = transforms.Compose([
            transforms.Resize(size=1024, interpolation=3),
            transforms.CenterCrop(size=1024), ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = Image.open(path)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = self.transform(image)
        return {
            'data': np.array(image),
            'path': path,
        }

    def _get_img_path(self, input_path):
        res = []
        img_dirs = [os.path.join(input_path, name) for name in os.listdir(input_path)
                    if os.path.isdir(os.path.join(input_path, name)) and int(name) < 1]
        for img_dir in tqdm(img_dirs):
            img_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.endswith('.jpg')]
            res += img_paths
        return res

    # def _mkdirdirs(self, save_path):
    #     print("prepare dirs ......")
    #     res = []
    #     for img_path in self.img_paths:
    #         filename = os.path.basename(os.path.dirname(img_path))
    #         save_dir = os.path.join(save_path, filename)
    #         if save_dir in res:
    #             continue
    #         res.append(save_dir)
    #
    #     for output_path in tqdm(res):
    #         os.makedirs(output_path, exist_ok=True)

    # def get_data(self):
    #     img_paths = []
    #     txt_path = './img_paths.txt'
    #
    #     print('get the data paths ......')
    #     if os.path.isfile(txt_path):
    #         with open(txt_path, 'r')as file:
    #             lines_list = file.readlines()
    #             for line in lines_list:
    #                 info = line.strip()
    #                 img_paths.append(info)
    #
    #     else:
    #         for input_path in tqdm(self.input_paths):
    #             image_names = [os.path.join(input_path, name) for name in os.listdir(input_path) if name.endswith('.jpg')]
    #             img_paths += image_names
    #
    #         print('writing the img path into .txt ......')
    #         with open(txt_path, 'w') as file:
    #             for info in tqdm(img_paths):
    #                 file.write(info + '\n')
    #     return img_paths