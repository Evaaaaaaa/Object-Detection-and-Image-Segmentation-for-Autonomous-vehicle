import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2


class Warping:
    def __init__(self):
        def turn60(xy):
            x, y = xy
            x -= 400
            y -= 400
            x, y = 0.5 * x + 0.866 * y, -0.866 * x + 0.5 * y
            x += 400
            y += 400
            return [x, y]

        def turnN60(array, n):
            array = list(array)
            for i in range(n):
                array = [turn60(xy) for xy in array]
            return np.array(array, dtype="float32")

        src_h = 128
        src_w = 306
        x = 600
        src = np.float32([[0, src_h], [src_w-0, src_h], [0, 0], [src_w, 0]])
        dst = [None] * 6
        M = [None] * 6

        # Right
        dst[1] = np.float32([[400, 360], [400, 440], [400+x, 400-x * 0.55], [400+x, 400+x * 0.55]])
        # Upper Right
        dst[0] = turnN60(dst[1], 1)
        # Upper Left
        dst[3] = turnN60(dst[1], 2)
        # Left
        dst[4] = turnN60(dst[1], 3)
        # Lower Left
        dst[5] = turnN60(dst[1], 4)
        # Lower Right
        dst[2] = turnN60(dst[1], 5)

        self.M = [cv2.getPerspectiveTransform(src, d) for d in dst]

    def getBirdsEye(self, img_batch):
        output = []
        for b in range(len(img_batch)):
            views = img_batch[b]
            res = []
            for direction in range(6):
                view = views[direction].permute(1, 2, 0).cpu().numpy()
                view_bottom = view[128:, :,]
                warped_img = cv2.warpPerspective(view_bottom, self.M[direction], (800, 800), flags=cv2.INTER_LINEAR)
                warped_img[warped_img == 0] = 'nan'
                res.append(warped_img)

            bird_eye = np.nanmean(res, axis=0)
            bird_eye = np.where(np.isnan(bird_eye), 0, bird_eye)
            output.append(torch.tensor(bird_eye).permute(2, 0, 1))
        return torch.stack(output)

class CoordinateTransform:
    def __init__(self):
        return
    def transform_bounding_box(self, target):
        res = []
        for b_i, target_batch in enumerate(target):
            boxes = target_batch['bounding_box']
            labels = target_batch['category']
            num_of_boxes = boxes.shape[0]  # boxes: N x 2 x 4
            output = torch.zeros((num_of_boxes, 6))
            for i, box in enumerate(boxes):
                tmp = torch.zeros(2, 4)
                tmp[0:] = 10 * box[0:] + 400
                tmp[1:] = -10 * box[1:] + 400
                x1 = (tmp[0][2] + tmp[0][3]) / 2 / 800
                x2 = (tmp[0][0] + tmp[0][1]) / 2 / 800
                y1 = (tmp[1][0] + tmp[1][2]) / 2 / 800
                y2 = (tmp[1][3] + tmp[1][1]) / 2 / 800
                output[i][0] = b_i
                output[i][2:6] = torch.Tensor([(x1+x2)/2, (y1+y2)/2, abs(x2-x1), abs(y2-y1)])
                output[i][1] = 1 if x1 < x2 else 0
                output = output.float()
            res.append(output)
        return torch.cat(res, 0)
    
    def transform_back_bounding_box(self, outputs):
        res = []
        if outputs is None or len(outputs) == 0:
            return torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]).unsqueeze(0)
        for output in outputs:
            _x1, _y1, _x2, _y2 = output[:4] # in 800 x 800 scale
            turn = output[-1]
            x1, y1 = _x2, _y1
            x2, y2 = _x2, _y2
            x3, y3 = _x1, _y1
            x4, y4 = _x1, _y2
            if turn == 0:
                x1, x3 = x3, x1
                x2, x4 = x4, x2
            tmp = torch.tensor([[x1, x2, x3, x4], [y1, y2, y3, y4]])
            tmp[0] = (tmp[0] - 400) / 10
            tmp[1] = (tmp[1] - 400) / (-10)
            res.append(tmp)
        return torch.stack(res)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
    
