import torch
from torch import nn, optim
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import tqdm
import numpy as np


from utils import scale_image_P
from utils import scale_image_RGB
from model_classic_unet import UNet
from torchvision.utils import save_image
from utils import CLASS_NAMES
from utils import WEIGHT_PATH
from utils import DATA_PATH1
from utils import DATA_PATH2
from utils import SAVE_TRAINING_IMAGE
from utils import DEVICE



# we will define the resize method, because original resize will result to the image distortion.
transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path) -> None:
        super(MyDataset, self).__init__()
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
    

    def __len__(self):
        return len(self.name)


    def __getitem__(self, index):
        # return the image name. *.png
        segment_name = self.name[index]
        # relative path for each image
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'png'))
        # notice, if your segmentation image is gray, you should load used single channel.
        # or you should use RGB channel to load it.
        segment_image = scale_image_P(segment_path)
        image = scale_image_RGB(image_path)
        return transform(image), torch.Tensor(np.array(segment_image))


if __name__ == "__main__":
    # notice the num_class is equal to your class number + backgound.
    num_classes = len(CLASS_NAMES)
    data_loader = DataLoader(MyDataset(DATA_PATH2), batch_size=1, shuffle=True)
    net = UNet(num_classes).to(DEVICE)
    if os.path.exists(WEIGHT_PATH):
        net.load_state_dict(torch.load(WEIGHT_PATH))
        print("successful load weight file")
    else:
        print("failer to load weight file")
    
    optimizer = optim.Adam(net.parameters())
    # notice the crossentropyloss function has involved onehot code
    # for the param.
    loss_function = nn.CrossEntropyLoss()
    epoch = 1
    while True:
        # tqdm is one expansion progress bar.
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(DEVICE), segment_image.to(DEVICE)
            out_image = net(image)
            # notice the out_image is num_classes*width*height
            # and the segment_image is gray image, so it is 1*width*height.
            # so if you used the simple loss function, it will be error.
            # but we have used the crossentropyloss function, it will
            # execute the one hot code first.
            train_loss = loss_function(out_image, segment_image.long())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print(f"{epoch}-{i}-train_loss ===>> {train_loss.item()}")
            
            

            # contact three image. orginal image, segment_image, and predict image.
            # notice, if you have generated one single channel image, and
            # the pixel is generated based on the label index. you should
            # multi 255 for each pixel. or you will not fail to show the image.
            # notice, if your segmentation image and predict image are all
            # gray image, but your train image is color, you can contact them.
            # you can just contact these two gray image.
            image = torch.stack([
                torch.unsqueeze(segment_image[0], 0) * 255, 
                torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255
                ], dim=0)
            save_image(image, f'{SAVE_TRAINING_IMAGE}/{i}.png')
        if epoch % 20 == 0:
            torch.save(net.state_dict(), WEIGHT_PATH)
        epoch += 1




























