import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy
#from ..test_graspnet import *
from time import time
from torch.ao.quantization.observer import MinMaxObserver 
from torch.ao.quantization.qconfig import QConfig
from ..dataset.graspnet_dataset import GraspnetPointDataset
from random import randrange
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import statistics 
import math
from datetime import datetime
from .background_clipping import *
from .subdivided_cell import *
from scipy.ndimage import convolve

from ..train_utils import *

def fill_zeros(img):
    kernel_size = 15
    required_for_fill = int(kernel_size * kernel_size * 0.3)

    img = img.astype(float)
    valid = (img != 0).astype(float)
    kernel = np.ones((kernel_size, kernel_size), dtype=float)

    neighbor_sum = convolve(img, kernel, mode='nearest')
    neighbor_count = convolve(valid, kernel, mode='nearest')

    result = img.copy()

    zero_mask = (img == 0) & (neighbor_count >= required_for_fill)
    result[zero_mask] = (
        neighbor_sum[zero_mask] /
        neighbor_count[zero_mask]
    )

    print("zeros", np.count_nonzero(zero_mask))

    return result

def remove_0s(depth):
    while np.any(depth == 0):
        new_depth = fill_zeros(depth)
        if np.array_equal(new_depth, depth):
            break
        depth = new_depth

        # plt.subplot(221)
        # plt.imshow(depth)
        # plt.tight_layout()
        # plt.show()
    
    return depth

# def extract_edges(depth):
#     depth = cv2.blur(depth, (5, 5))
#     depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     canny_edges = cv2.Canny(depth_normalized, threshold1 = 100, threshold2 = 300)  
#     mask = canny_edges > 0

#     return mask

def get_clutter_map(depth):
    clutter = cv2.blur(depth, (5, 5))
    sobelx = cv2.Sobel(clutter,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(clutter,cv2.CV_64F,0,1,ksize=5)
    clutter = cv2.magnitude(sobelx, sobely)  
    return clutter

def get_clutter_value_from_map(map):
    return np.sum(map)

def get_clutter_metric_in_cells(cells, rgb, bg_clipper):
    depth = bg_clipper.replace_depth_error_with_table()
    clutter_map = get_clutter_map(depth)
    visualization = np.zeros(depth.shape)

    clutters = []
    for cell in cells:
        clutter_map_sub = cell.original_bb.clip_img(clutter_map)
        clutter_in_cell = get_clutter_value_from_map(clutter_map_sub)
        visualization[cell.original_bb.y_min:cell.original_bb.y_max, cell.original_bb.x_min:cell.original_bb.x_max] = clutter_in_cell
        clutters.append(clutter_in_cell)

    # plt.subplot(221)
    # plt.imshow(rgb)
    # plt.subplot(222)
    # plt.imshow(depth)
    # plt.subplot(223)
    # plt.imshow(clutter_map)
    # plt.subplot(224)
    # plt.imshow(visualization)
    # plt.tight_layout()
    # plt.show()

    return clutters



def clutter_map_test(rgb, bg_clipper):
    depth = bgClipper.replace_depth_error_with_table()

    # depth_filtered = remove_0s(depth)
    clutter = get_clutter_map(depth)
    
    plt.subplot(221)
    plt.imshow(rgb)
    plt.subplot(222)
    plt.imshow(depth)
    plt.subplot(223)
    plt.imshow(clutter)
    plt.tight_layout()
    plt.show()

    # scharrx = cv2.Scharr(depth, cv2.CV_64F, 1, 0)
    # scharry = cv2.Scharr(depth, cv2.CV_64F, 0, 1)
    # scharr_edges = cv2 . magnitude ( scharrx , scharrx )  

    return clutter