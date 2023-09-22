import json
import cv2
import os
from tqdm import tqdm
from PIL import Image


class DepthDataset():
    def __init__(self, meta_file, transforms):
        super(DepthDataset, self).__init__()

        self.files = []
        print('loading the data......')
        with open(meta_file, 'r') as f:
            data = json.load(f)
            for key in tqdm(list(data.keys())):
                img_path = key
                depth_img_path = data[key]['depth']
                txt_path = data[key]['txt']

                self.files.append(
                    {
                        'img_path': img_path,
                        'depth_path': depth_img_path,
                        'txt_path': txt_path,
                        'img_id': os.path.dirname(key),
                    })
        self.transforms = transforms

    def __getitem__(self, idx):
        file = self.files[idx]

        # print(f"Test: {file['img_path']}")
        # print(f"Type: {type(file['img_path'])}")

        # img
        im = Image.open(file['img_path']).convert("RGB")
        im = self.transforms(im)
        # im = img2tensor(np.array(im), bgr2rgb=True, float32=True) / 255.

        # depth
        depth = Image.open(file['depth_path']).convert('L')
        depth = self.transforms(depth)     # [:,:,0]
        # depth = img2tensor(np.array(depth), bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.

        # txt
        with open(file['txt_path'], 'r') as fs:
            sentence = fs.readline().strip()

        # img id
        id = file['img_id']

        return {'jpg': im, 'depth': depth, 'txt': sentence, 'im_name': id}

    def __len__(self):
        return len(self.files)
