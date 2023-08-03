from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from torchvision import transforms

if __name__ == "__main__":
    writer = SummaryWriter("logs")
    image_path = "c:/users/80521/desktop/gyy.png"
    image_PIL = Image.open(image_path)
    image_array = np.array(image_PIL)
    # you can set each image or figure what you want to show in tensorboard.
    # the first param is title, the second is the content what you want to show
    # the third param is the step.
    # writer.add_image("test", image_array, 1, dataformats='HWC')
    # writer.close()

    # compose test, resize and totensor
    # transform = transforms.Compose([
    #     transforms.Resize(512),
    #     transforms.ToTensor()
    # ])
    # image = transform(image_PIL)
    # writer.add_image("resize", image, 1)
    # writer.close()

    # random crop test.
    transform_randomCrop = transforms.Compose([
        transforms.RandomCrop(512),
        transforms.ToTensor()
    ])
    for i in range(10):
        image_crop = transform_randomCrop(image_PIL)
        writer.add_image("RandomCrop", image_crop, i)
    # notice you should close the writer for each times.
    writer.close()