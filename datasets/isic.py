from torchvision.datasets.folder import DatasetFolder, default_loader
from PIL import Image
import os
import numpy as np
import pandas as pd


class ISICDataset(DatasetFolder):
    def __init__(self, root:str, train:bool, transform=None, target_transform=None):
        self.train = train
        self.img_path = os.path.join(root, "ISIC_2019_Training_Input")
        data = np.array(pd.read_csv(os.path.join(root, "ISIC_2019_Training_GroundTruth.csv")))
        paths = [os.path.join(self.img_path, str(label) + ".jpg") for label in data[:, 0]]
        self.targets = [np.argmax(t) for t in data[:, 1:]]

        self.transform = transform
        self.target_transform = target_transform

        # TODO: sync this with literature
        if self.train:
            # Get all but every 10th element
            self.targets = np.array(self.targets)[np.mod(np.arange(len(self.targets)), 10) != 0]
            paths = np.array(paths)[np.mod(np.arange(len(paths)), 10) != 0]
        else:
            # Get every 10th element
            self.targets = np.array(self.targets)[::10]
            paths = np.array(paths)[::10]

        self.samples = [s for s in zip(paths, self.targets)]
        self.loader = default_loader

    """
    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, str(self.targets[index, 0]) + ".jpg")
        img = Image.open(img_name)
        targets = self.targets[index, 1:]
        targets = np.array(targets, dtype=int)[:8]  # remove "rest" class

        if self.target_transform is not None:
            targets = self.target_transform(targets)
        if self.transform is not None:
            img = self.transform(img)

        return img, targets
    """

    def __len__(self):
        return len(self.targets)
