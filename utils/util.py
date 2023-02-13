import glob
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import scipy
import os
from typing import List
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms




class UnNorm(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        len_4_flag = False
        if len(tensor.size()) == 4:
            len_4_flag = True
            tensor_temp = tensor.squeeze().clone()
        else:
            tensor_temp = tensor.clone()
        for t, m, s in zip(tensor_temp, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        if len_4_flag:
            tensor_temp = tensor_temp.unsqueeze(0)
        return tensor_temp

