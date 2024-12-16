
import numpy as np
import torch

def MinMax(image):
    image = image - np.mean(image)  # 零均值
    if np.max(image) - np.min(image) != 0:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # 归一化

    return image

