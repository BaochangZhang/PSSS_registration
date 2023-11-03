import torch
import torch.nn as nn
from pathlib import Path


class L2_PoseLoss(nn.Module):
    """
    A class to represent camera pose loss
    """

    def __init__(self,):
        """
        :param learnable: Ture/False
        """
        super(L2_PoseLoss, self).__init__()

    def forward(self, est_pose, gt_pose):
        """
        Forward pass
        :param est_pose: (torch.Tensor) batch of estimated poses, a Nx7 tensor
        :param gt_pose: (torch.Tensor) batch of ground_truth poses, a Nx7 tensor
        :return: camera pose loss
        """
        # Rotation loss
        rot_loss = torch.norm(gt_pose[:, 0:3] - est_pose[:, 0:3], dim=1, p=2).mean()
        # Translation loss
        trans_loss = torch.norm(gt_pose[:, 3:] - est_pose[:, 3:], dim=1, p=2).mean()
        return trans_loss + rot_loss



class ZNCC(nn.Module):
    """
    Compute the zero normalized cross-correlation between two images with the same shape.
    """

    def __init__(self, scale=1.0):
        super(ZNCC, self).__init__()
        self.InstanceNorm = nn.InstanceNorm2d(1)
        self.scale = scale

    def forward(self, img1, img2):
        assert img1.shape == img2.shape
        _, _, h, w = img1.shape
        img1 = self.InstanceNorm(img1)
        img2 = self.InstanceNorm(img2)
        score = torch.einsum("b...,b...->b", img1, img2)
        score /= h * w
        loss = (1.0 - score).mean()
        return self.scale * loss


class Sobel(torch.nn.Module):
    def __init__(self, sigma=None, eps=1e-10):
        super().__init__()

        self.sigma = sigma
        self.eps = eps
        kernel_X = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        kernel_X = torch.nn.Parameter(kernel_X, requires_grad=False)
        kernel_Y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        kernel_Y = torch.nn.Parameter(kernel_Y, requires_grad=False)

        self.SobelX = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.SobelX.weight = kernel_X
        self.SobelY = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.SobelY.weight = kernel_Y

    def forward(self, x):
        if self.sigma is not None:
            x = gaussian_blur(x, kernel_size=[5, 5], sigma=self.sigma)
        Sx = self.SobelX(x) + self.eps
        Sy = self.SobelY(x) + self.eps
        return Sx, Sy


class GradZNCC(nn.Module):
    """Compute Normalized Cross Correlation between the image gradients of two batches of images."""

    def __init__(self, scale=1.0, sigma=None):
        super(GradZNCC, self).__init__()
        self.InstanceNorm = nn.InstanceNorm2d(1)
        self.sobel = Sobel(sigma)
        self.scale = scale
        self.ncc = ZNCC()

    def forward(self, img1, img2):
        assert img1.shape == img2.shape
        _, _, h, w = img1.shape
        grad1_x, grad1_y = self.sobel(img1)
        grad2_x, grad2_y = self.sobel(img2)

        ncc_x = self.ncc(grad1_x, grad2_x)
        ncc_y = self.ncc(grad1_y, grad2_y)

        return self.scale * (ncc_x + ncc_y)/2.0

