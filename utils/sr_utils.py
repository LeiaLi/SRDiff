import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.nn as nn


class ImgMerger:
    def __init__(self, eval_fn):
        self.eval_fn = eval_fn
        self.loc2imgs = {}
        self.max_x = 0
        self.max_y = 0
        self.clear()

    def clear(self):
        self.loc2imgs = {}
        self.max_x = 0
        self.max_y = 0

    def push(self, imgs, loc, loc_bdr):
        """

        Args:
            imgs: each of img is [C, H, W] np.array, range: [0, 255]
            loc: string, e.g., 0_0, 0_1 ...
        """
        self.max_x, self.max_y = loc_bdr
        x, y = loc
        self.loc2imgs[f'{x},{y}'] = imgs
        if len(self.loc2imgs) == self.max_x * self.max_y:
            return self.compute()

    def compute(self):
        img_inputs = []
        for i in range(len(self.loc2imgs['0,0'])):
            img_full = []
            for x in range(self.max_x):
                imgx = []
                for y in range(self.max_y):
                    imgx.append(self.loc2imgs[f'{x},{y}'][i])
                img_full.append(np.concatenate(imgx, 2))
            img_inputs.append(np.concatenate(img_full, 1))
        self.clear()
        return self.eval_fn(*img_inputs)


##########
# SSIM
##########
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        img1 = img1 * 0.5 + 0.5
        img2 = img2 * 0.5 + 0.5
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485 - 1, 0.456 - 1, 0.406 - 1]).view(1, 3, 1, 1)
            # mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229 * 2, 0.224 * 2, 0.225 * 2]).view(1, 3, 1, 1)
            # std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        loss_network = VGGFeatureExtractor()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        if next(self.loss_network.parameters()).device != high_resolution.device:
            self.loss_network.to(high_resolution.device)
            self.loss_network.eval()
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss
