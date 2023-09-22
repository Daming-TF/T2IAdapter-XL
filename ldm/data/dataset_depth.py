import json
import cv2
import os
from basicsr.utils import img2tensor


class DepthDataset():
    def __init__(self, meta_file):
        super(DepthDataset, self).__init__()

        self.files = []
        with open(meta_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()

                # original code
                # img_path = os.path.join(img_dir, line)
                # depth_img_path = img_path.rsplit('.', 1)[0] + '.depth.png'
                # txt_path = img_path.rsplit('.', 1)[0] + '.txt'

                # my add
                img_path = line
                depth_dir, img_name = os.path.split(img_path.replace('image', 'depth'))
                depth_suffix = r'midas_v21_small_256'
                depth_img_path = os.path.join(depth_dir, os.path.splitext(img_name)[0]+'-'+depth_suffix+'.png')
                txt_path = img_path.replace('.jpg', '.txt')

                self.files.append(
                    {
                        'img_path': img_path,
                        'depth_img_path': depth_img_path,
                        'txt_path': txt_path,
                        'im_name': line,
                    })

    def __getitem__(self, idx):
        file = self.files[idx]

        im_name = file['im_name']

        im = cv2.imread(file['img_path'])
        im = cv2.resize(im, (512, 512))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        depth = cv2.imread(file['depth_img_path'])  # [:,:,0]
        depth = cv2.resize(depth, (512, 512))
        depth = img2tensor(depth, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.

        with open(file['txt_path'], 'r') as fs:
            sentence = fs.readline().strip()

        return {'im': im, 'depth': depth, 'sentence': sentence, 'im_name': im_name}

    def __len__(self):
        return len(self.files)
