import math
from typing import List

import numpy as np
import torch
from numba import njit
from torchvision.transforms.functional import gaussian_blur

from ..dataset.collision_detector import ModelFreeCollisionDetector

from .config import get_camera_intrinsic
from .grasp import RectGrasp, RectGraspGroup
from .pc_dataset_tools import select_2d_center
from .utils import angle_distance, euclid_distance, rotation_distance

eps = 1e-6

def collision_detect(points_all: torch.Tensor, pred_gg, mode='regnet'):
    # collison detect
    cloud = points_all[:, :3].clone()
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01, mode=mode)
    no_collision_mask = mfcdetector.detect(pred_gg, approach_dist=0.05)
    collision_free_gg = pred_gg[no_collision_mask]
    return collision_free_gg, no_collision_mask
