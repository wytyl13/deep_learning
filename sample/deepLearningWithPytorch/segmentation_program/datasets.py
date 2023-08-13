from torchvision import transforms
from torch.utils.data import Dataset
import os
import torch
import numpy as np


from utils import scale_image_P
from utils import scale_image_RGB
from utils import scale_image_mask
from utils import VOC

# we will define the resize method, because original resize will result to the image distortion.
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class MyDataset(Dataset):
    def __init__(self, path, num_classes = 21) -> None:
        super(MyDataset, self).__init__()
        self.path = path
        self.num_classes = num_classes
        self.name = [filename for filename in os.listdir(os.path.join(path, 'SegmentationClass')) if self.is_image(filename)]
    

    def __len__(self):
        return len(self.name)


    def __getitem__(self, index):
        # return the image name. *.png
        segment_name = self.name[index]
        # relative path for each image
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        if VOC:
            image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))
        else:
            image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'png'))
        # notice, if your segmentation image is gray, you should load used single channel.
        # or you should use RGB channel to load it.
        segment_image = torch.Tensor(np.array(scale_image_P(segment_path)))
        if VOC:
            segment_image = np.array(scale_image_mask(segment_path))
            segment_image[segment_image > self.num_classes] = 0
        image = scale_image_RGB(image_path)
        return transform(image), torch.Tensor(segment_image)

    def is_image(self, filename):
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        return any(filename.lower().endswith(ext) for ext in extensions)