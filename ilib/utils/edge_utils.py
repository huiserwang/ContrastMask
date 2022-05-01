import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt
from vlkit.dense import (
    seg2edge as vlseg2edge,
    sobel, flux2angle, dense2flux, quantize_angle
    )

import matplotlib
matplotlib.use("agg")
import matplotlib.pylab as plt

def dice_loss_func(edge, target):
    """ Dice loss
        Thanks to: https://github.com/hustvl/BMaskR-CNN
    """
    smooth = 1.
    n = edge.size(0)
    iflat = edge.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

def edge_loss_func(edge, edge_targets):
    bce_loss = F.binary_cross_entropy_with_logits(edge, edge_targets)
    dice_loss = dice_loss_func(torch.sigmoid(edge), edge_targets)
    return bce_loss + dice_loss

def edge_loss(edge, edge_targets):
    """
    edge: [N 1 H W] tensor
    edge_targets: [N H W] tensor
    """
    if edge.size(0) == 0:
        loss_edge = edge.sum()
    else:
        weights = edge_targets.sum() / edge_targets.numel()
        weights = torch.where(edge_targets.to(dtype=bool), 1-weights, weights)
        edge = edge.squeeze(dim=1)
        loss_edge = F.binary_cross_entropy_with_logits(edge, edge_targets, weights)
    return loss_edge


def seg2edge(seg):
    bs = seg.size(0)
    segnp = seg.cpu().numpy()
    edge = np.zeros_like(segnp)
    for i in range(bs):
        edge[i] = vlseg2edge(segnp[i])
    return torch.tensor(edge).to(seg)

def mask2edge(seg):
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=seg.device).reshape(1, 1, 3, 3).requires_grad_(False)
    edge_targets = F.conv2d(seg.unsqueeze(1), laplacian_kernel, padding=1)
    edge_targets = edge_targets.clamp(min=0)
    edge_targets[edge_targets > 0.1] = 1
    edge_targets[edge_targets <= 0.1] = 0
    return edge_targets


def get_orient_gt(seg, edge):
    """
    seg: [N H W] tensor
    edge: [N 1 H W] tensor
    """
    N, H, W = seg.shape

    bs = seg.size(0)
    seg = seg.cpu()
    edge = edge.cpu()

    qangle = torch.zeros([N, H, W], dtype=torch.uint8)

    for i in range(bs):

        seg1 = seg[i]
        edge1 = edge[i]
        dist1 = distance_transform_edt(1 - edge1.numpy())
        flux = dense2flux(dist1)
        angle1 = flux2angle(flux)
        qangle1 = quantize_angle(angle1, num_bins=8)

        debug = False        
        if debug:
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])
            axes[0, 0].imshow(edge1.numpy())
            axes[0, 1].imshow(seg1.numpy())
            axes[0, 2].imshow(dist1)

            sobelx = cv2.Sobel(dist1, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(dist1, cv2.CV_64F, 0, 1, ksize=3)
            axes[0, 3].imshow(sobelx * (sobelx > 0))
            axes[0, 3].set_title("SobelX+")

            axes[0, 4].imshow(-sobelx * (sobelx < 0))
            axes[0, 4].set_title("SobelX-")

            axes[0, 5].imshow(sobely * (sobely > 0))
            axes[0, 5].set_title("SobelY+")

            axes[0, 6].imshow(-sobely * (sobely < 0))
            axes[0, 6].set_title("SobelY-")

            for idx in range(8):
                axes[1, idx].imshow(qangle1 == idx)
            plt.savefig("qangle.png")
            plt.close(fig)
        qangle[i] = torch.from_numpy(qangle1)
    return qangle
