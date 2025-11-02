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
from PIL import Image
from transformers import pipeline
import torchvision.transforms as transforms

pipe = None


def get_depth_pipe():
    global pipe
    if pipe is None:
        print("[INFO] Loading depth estimation model (depth-anything)...")
        pipe = pipeline(
            task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf"
        )
    return pipe


def release_depth_model():
    global pipe
    if pipe is not None:
        try:
            del pipe
            pipe = None
            torch.cuda.empty_cache()
            print("[INFO] Depth model released and CUDA cache emptied.")
        except Exception as e:
            print(f"[ERROR] The depth model could not be released: {e}")


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def sobel_edges(img: torch.tensor) -> torch.tensor:
    if img.ndim == 3:
        img = img.unsqueeze(0)

    if img.shape[1] != 1:
        img = img[:, 0:1] * 0.2989 + img[:, 1:2] * 0.5870 + img[:, 2:3] * 0.1140

    sobel_x = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32, device=img.device
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32, device=img.device
    ).unsqueeze(0)

    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    edges = torch.sqrt(grad_x**2 + grad_y**2)

    edges_norm = edges / (edges.max() + 1e-8)
    return edges_norm


def depth_inference(img: torch.tensor) -> torch.tensor:
    img = (img * 255).byte()
    img = img.permute(1, 2, 0)
    imagen_pil = Image.fromarray(img.cpu().numpy())

    depth = get_depth_pipe()(imagen_pil)["depth"]
    depth_tensor = transforms.ToTensor()(depth)

    return depth_tensor
