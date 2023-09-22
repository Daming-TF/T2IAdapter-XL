import os
import sys
from PIL import Image
import numpy as np
import argparse
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from tqdm import tqdm
import multiprocessing
from torchvision import transforms

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_dir))
# from condition_extractor.dataset import CondExtractorDataset


class DPTModel:
    def __init__(self, ):
        self._load_model()

    def _load_model(self, ):
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to('cuda')
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    def __call__(self, image):
        # print(image.shape)      # {batch_size, 1024, 1024, 3}
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depths = outputs.predicted_depth      # {batch_size, 384, 384}

            predictions = torch.nn.functional.interpolate(
                predicted_depths.unsqueeze(1),
                # size=image.size[::-1],
                size=tuple(image[0].shape[:2]),
                mode="bicubic",
                align_corners=False,
            )

        print('prapare prediction......')
        depths = []
        for i, prediction in tqdm(enumerate(predictions)):
            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth = Image.fromarray(formatted)
            depths.append(depth)

        return depths


def process_depth(images_per_process, args, gpu_id, model_type="dpt-hybrid-midas"):
    torch.cuda.set_device(gpu_id)
    # accelerator = Accelerator()

    # prepare data
    # test_dataset = CondExtractorDataset(args.input_path)
    from .dataset import CondExtractorDataset
    test_dataset = CondExtractorDataset(images_per_process)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16)

    # preprocess
    # device = accelerator.device
    os.makedirs(args.output_path, exist_ok=True)
    model = DPTModel()

    # Send everything through `accelerator.prepare`
    # test_loader, model = accelerator.prepare(test_loader, model)

    # Run
    with torch.no_grad():
        print('running .....')
        for data in tqdm(test_loader):
            images = data['data'].to('cuda')
            paths = data['path']
            depth_imgs = model(images)
            print('saving the image......')
            for i, depth_img in tqdm(enumerate(depth_imgs)):
                save_dir = os.path.join(args.output_path, os.path.basename(os.path.dirname(paths[i])))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(
                    save_dir, os.path.splitext(os.path.basename(paths[i]))[0] + '-' + model_type + '.png'
                )
                depth_img.save(save_path)
