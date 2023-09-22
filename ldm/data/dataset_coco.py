import json
import cv2
import os
from basicsr.utils import img2tensor

import warnings


class dataset_coco_mask_color():
    def __init__(self, path_json, root_path_im, root_path_mask, image_size, logger, debug=False):
        super(dataset_coco_mask_color, self).__init__()
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        data = data['annotations']
        self.files = []
        self.root_path_im = root_path_im
        self.root_path_mask = root_path_mask
        for file in data:
            name = "%012d.png" % file['image_id']
            img_path = os.path.join(self.root_path_im, name.replace('.png', '.jpg'))
            if not os.path.exists(img_path):
                warnings.warn(f"Image '{img_path}' not found.", UserWarning)
                logger.warning("Image '%s' not found.", img_path)
                continue
            self.files.append({'name': name, 'sentence': file['caption']})

        self.files = self.files[:60] if debug else self.files
        print(f'ToTal num:{len(self.files)}')

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']
        # print(os.path.join(self.root_path_im, name))
        img_path = os.path.join(self.root_path_im, name.replace('.png', '.jpg'))
        im = cv2.imread(img_path)
        im = cv2.resize(im, (512, 512))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        mask = cv2.imread(os.path.join(self.root_path_mask, name))  # [:,:,0]
        mask = cv2.resize(mask, (512, 512))
        mask = img2tensor(mask, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.

        sentence = file['sentence']
        return {'im': im, 'mask': mask, 'sentence': sentence, 'im_name': name}

    def __len__(self):
        return len(self.files)
