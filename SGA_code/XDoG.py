#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

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
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    tau = p/(1+p)
    dog = DoG_filter(image, kernel_size, sigma, k_sigma, tau)
    dog /= dog.max()
    e = 1 + np.tanh(psi * (dog - epsilon))
    e[e >= 1] = 1
    return e.astype(np.uint8) * 255

def DoG_filter(image, kernel_size=0, sigma=1.4, k_sigma=1.6, tau=1):
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
    g1_f = cv2.GaussianBlur(image,(kernel_size,kernel_size),sigma)
    g2_f = cv2.GaussianBlur(image,(kernel_size,kernel_size),sigma*k_sigma)
    return g1_f - tau * g2_f
