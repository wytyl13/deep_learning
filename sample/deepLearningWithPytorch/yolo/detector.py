import torch.nn as nn
import torch
from PIL import Image, ImageDraw
import os

from net import Yolov3
from utils import WEIGHT_PATH
import config
from utils import nms
from utils import CLASS_DICT
from utils import make_image_data
from utils import transform
from utils import IMAGE_DIR_PATH




class Detector(nn.Module):
    def __init__(self) -> None:
        super(Detector, self).__init__()

        self.net = Yolov3()
        self.net.load_state_dict(torch.load(WEIGHT_PATH))
        self.eval()

    def forward(self, input, thresh_value, anchors, case):
        out_13, out_26, out_52 = self.net(input)
        idxs_13, vecs_13 = self._filter(out_13, thresh_value)
        boxes_13 = self._parse(idxs_13, vecs_13, config.DATA_WIDTH / 13, anchors[13], case)

        idxs_26, vecs_26 = self._filter(out_26, thresh_value)
        boxes_26 = self._parse(idxs_26, vecs_26, config.DATA_WIDTH / 26, anchors[26], case)

        idxs_52, vecs_52 = self._filter(out_52, thresh_value)
        boxes_52 = self._parse(idxs_52, vecs_52, config.DATA_WIDTH / 52, anchors[52], case)
        boxes = torch.cat([boxes_13, boxes_26, boxes_52], dim=0)
        boxes=nms(boxes, 0.5, mode='inter')
        return boxes


    def _filter(self, out, thresh_value):
        # transform from (N, channels, width, height) to (N, width, height, channels)
        out = out.permute(0, 2, 3, 1)
        # reshape to (N, width, height, 3, channels/3)
        out = out.reshape(out.size(0), out.size(1), out.size(2), 3, -1)

        # filter the data based on the confidence
        # the last dimension for out, the first index of the last dimension for out
        # is confidence, less than thresh_value set to false, or set to true.
        mask = torch.sigmoid(out[..., 0]) > thresh_value
        # idxs shape is
        idxs = mask.nonzero()
        vecs = out[mask]
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors, case):
        anchors = torch.tensor(anchors)
        n = idxs[:, 0]
        a = idxs[:, 3]
        cy = (idxs[:, 1].float() + vecs[:, 2]) * t / case
        cx = (idxs[:, 2].float() + vecs[:, 2]) * t / case

        w = anchors[a, 0] * torch.exp(vecs[:, 3]) / case
        h = anchors[a, 1] * torch.exp(vecs[:, 4]) / case
        p = vecs[:, 0]
        class_p = vecs[:, 5:]
        class_p=torch.softmax(class_p,dim=1)
        class_index = torch.argmax(class_p, dim=1)
        return torch.stack([n.float(), torch.sigmoid(p),cx, cy, w, h,class_index], dim=1)




if __name__ == "__main__":
    detector = Detector()
    for i in os.listdir(IMAGE_DIR_PATH):
        image_file_path = os.path.join(IMAGE_DIR_PATH, i)
        img=Image.open(image_file_path)
        _img = make_image_data(image_file_path)
        w, h = _img.size[0], _img.size[1]
        case = 416 / w
        # print(case)
        _img = _img.resize((416, 416))  # 此处要等比缩放
        _img_data = transform(_img)
        _img_data=torch.unsqueeze(_img_data,dim=0)
        # print(_img_data.shape)
        result=detector(_img_data, 0.2, config.ANCHORS,case)
        draw=ImageDraw.Draw(img)
        for rst in result:
            if len(rst)==0:
                continue
            else:
                # rst=rst[0]
                x1,y1,x2,y2=rst[2]-0.5*rst[4],rst[3]-0.5*rst[5],rst[2]+0.5*rst[4],rst[3]+0.5*rst[5]
                print(f'置信度：{str(rst[1].item())[:4]} 坐标点：{x1,y1,x2,y2} 类别：{CLASS_DICT[int(rst[6].item())]}')
                draw.text((x1,y1),CLASS_DICT[int(rst[6].item())]+str(rst[1].item())[:4])
                draw.rectangle((x1,y1,x2,y2),width=1,outline='red')
        img.show()
