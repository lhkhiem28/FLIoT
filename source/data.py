
import os, sys
from libs import *

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data_dir, 
        num_channels = 3, 
    ):
        self.image_files = glob.glob(data_dir + "*/*")
        self.transform = A.Compose(
            [
                A.Normalize(
                    mean = tuple([0.5]*num_channels), std = tuple([0.5]*num_channels), 
                ), 
                AT.ToTensorV2(), 
            ]
        )

    def __len__(self, 
    ):
        return len(self.image_files)

    def __getitem__(self, 
        index, 
    ):
        image_file = self.image_files[index]
        image = np.load(image_file)
        image = self.transform(image = image)["image"]

        label = int(image_file.split("/")[-2])

        return image, label