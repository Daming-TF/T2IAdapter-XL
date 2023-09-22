from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import logging


class CondExtractorDataset(Dataset):
    def __init__(self, img_paths, resolution):
        self.resolution = resolution
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            transforms.Resize(size=self.resolution, interpolation=3),
            transforms.CenterCrop(size=self.resolution), ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        try:
            image = np.array(self.transform(image))
            return {
                'data': image,
                'path': path,
                'error': False,
            }
        except OSError as e:
            info = f"OSError for sample at index {idx}: {e}"
            logging.error(info)
            print(info)
            print(path)
            return {
                'data': np.random.randint(0, 256, size=(self.resolution, self.resolution, 3), dtype=np.uint8),
                'path': path,
                'error': True,
            }
