import os, sys
from libs import *

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data_dir, 
    ):
        self.image_paths = glob.glob(data_dir + "*/*")
        self.transform = A.Compose(
            [
                A.Resize(
                    height = 32, width = 32, 
                ), 
                A.Normalize(), AT.ToTensorV2(), 
            ]
        )

    def __len__(self, 
    ):
        return len(self.image_paths)

    def __getitem__(self, 
        index, 
    ):
        image_path = self.image_paths[index]
        label = int(image_path.split("/")[-2])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(
            image, 
            code = cv2.COLOR_BGR2RGB, 
        )
        image = self.transform(image = image)["image"]

        return image, label