#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# ------------------------------------------------------------
# Modifications for GUAC-3DGS:
# Copyright (C) 2025, Sergio Ortiz (IAUNI)
# Adaptive Uncertainty Weighting Framework
# URL: https://github.com/sergio100899/guac-3dgs
#
# This work extends Gaussian Splatting with multi-level uncertainty-driven
# dynamic loss weighting for improved photometric and geometric consistency.
# ------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.image_utils import sobel_edges

try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01**2
C2 = 0.03**2


class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None


def l1_loss(network_output, gt):
    """
    L1 Loss extendida: devuelve mean, var y opcionalmente el mapa de error.
    """
    error_map = torch.abs(network_output - gt)
    mu = error_map.mean()
    var = error_map.var(unbiased=False)

    return mu, var


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def edge_loss(network_output: torch.tensor, gt: torch.tensor):
    gt_edge = sobel_edges(gt)
    nt_edge = sobel_edges(network_output)
    error_map = torch.abs(nt_edge - gt_edge)
    mu = error_map.mean()
    var = error_map.var(unbiased=False)

    return mu, var


def compute_depth_loss_from_normalized_gradients(
    rendered_depth, gt_depth, gt_image, alpha=1.0
):
    gt_depth_detached = gt_depth.detach()

    # Align means and standard deviations
    mean_gt = gt_depth_detached.mean()
    std_gt = gt_depth_detached.std()

    mean_rendered = rendered_depth.mean()
    std_rendered = rendered_depth.std()

    aligned_depth = (rendered_depth - mean_rendered) / (
        std_rendered + 1e-6
    ) * std_gt + mean_gt

    # Edges and error
    gt_edges = sobel_edges(gt_image)
    depth_diff = torch.abs(gt_depth_detached - aligned_depth)

    # Weighted loss
    gradient_loss_map = torch.exp(-gt_edges * alpha) * torch.log(1 + depth_diff)
    mu = gradient_loss_map.mean()
    var = gradient_loss_map.var(unbiased=False)

    return mu, var


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
