#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import rgb_to_grayscale

def XDoG_filter(image,
                kernel_size=0,
                sigma=.4,
                k_sigma=1.6,
                epsilon=0.01,
                psi=1e9,
                p=19, 
                gray=True):
    """XDoG(Extended Difference of Gaussians)

    Args:
        image: Image
        kernel_size: Gaussian Blur Kernel Size
        sigma: sigma for small Gaussian filter
        k_sigma: large/small for sigma Gaussian filter
        eps: threshold value between dark and bright
        psi: soft threshold
        gamma: scale parameter for DoG signal to make sharp

    Returns:
        Image after applying the XDoG.
    """
    if gray: 
        image = rgb_to_grayscale(image)
    tau = p/(1+p)
    dog = DoG_filter(image, kernel_size, sigma, k_sigma, tau)
    dog /= dog.max()
    e = 1 + torch.tanh(psi * (dog - epsilon))
    e[e >= 1] = 1
    return e.to(torch.uint8) * 255

def DoG_filter(image, kernel_size=0, sigma=1.4, k_sigma=1.6, gamma=1):
    """DoG(Difference of Gaussians)
    Args:
        image: Image
        kernel_size: Gaussian Blur Kernel Size
        sigma: sigma for small Gaussian filter
        k_sigma: large/small for sigma Gaussian filter
        gamma: scale parameter for DoG signal to make sharp

    Returns:
        Image after applying the DoG.
    """
    g1_f = GaussianBlur( (kernel_size if kernel_size > 0 
                          else estimateKenelSize(sigma, image.dtype)),
                        sigma)
    g2_f = GaussianBlur( (kernel_size if kernel_size > 0
                          else estimateKenelSize(sigma*k_sigma, image.dtype)), 
                        sigma*k_sigma)
    return g1_f(image) - gamma * g2_f(image)

# automatic detection of kernel size from sigma
# https://github.com/opencv/opencv/blob/850be1e0874a4881b94392773d2f1702344658ac/modules/imgproc/src/smooth.dispatch.cpp#L287
def estimateKenelSize(sigma,dtype=torch.uint8):
    return int(torch.round(sigma*(3 if dtype==torch.uint8 else 4)*2 + 1))|1
