import os
import pandas as pd
from torchvision.io import read_image
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps


class ImageDataset(data.Dataset):
    """Custom Dataset class for Deja-Vu

    This class reads in images as Tensors, so applying a torchvision.transforms.ToTensor() will
    result in unexpected results.
    """
    def __init__(self, labels_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = Image.open(img_path)
        image = image.convert('RGB')
        image = np.array(image)
        if image.shape[2] == 4:
            image = image[..., :3]
        label = self.img_labels.iloc[idx, 2]
        name = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label, name
