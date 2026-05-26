import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy as np
#from ..test_graspnet import *
from time import time
from torch.ao.quantization.observer import MinMaxObserver 
from torch.ao.quantization.qconfig import QConfig
# from ..dataset.graspnet_dataset import GraspnetPointDataset
from random import randrange
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import statistics 
import math
from datetime import datetime
from ..dataset.config import get_camera_intrinsic
from random import sample
from skimage import measure, morphology
from pathlib import Path

# from ..train_utils import *

class BackgroundClipper:
    def __init__(self, 
                depth_map, 
                rgb,
                group_size = 50, 
                proximity = 0.005, 
                clip_distance = 0.005, 
                max_residuals = 0.1, 
                ahead_point_score_ratio = 0.5,
                allowed_null_depth_ratio = 0.1,
                scene = -1,
                view = -1):
        self.width = depth_map.shape[1]
        self.height = depth_map.shape[0]
        self.group_size = group_size
        self.proximity = proximity
        self.clip_distance = clip_distance
        self.max_residuals = max_residuals
        self.ahead_point_score_ratio = ahead_point_score_ratio
        self.allowed_null_depth_ratio = allowed_null_depth_ratio
        self.rgb = rgb
        self.scene = scene
        self.view = view

        self.depth_map = depth_map

        #self.depth_map = cv2.GaussianBlur(depth_map, (5, 5), sigmaX=1.0)
        # self.depth_map = cv2.medianBlur(depth_map.astype(np.float32), 5)
        # self.depth_map = cv2.bilateralFilter(
        #     depth_map.astype(np.float32),
        #     d=9,
        #     sigmaColor=50,
        #     sigmaSpace=50
        # )

        # self.depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        # self.depth_map = cv2.GaussianBlur(depth_map, (7,7), 1.5)

        self.point_map = self.get_point_map()
        self.foreground_mask = self.get_foreground_mask()

    def get_point_map(self):
        self.zs = self.depth_map / 1000.0
        self.zs = self.zs.T

        # get intrinsics
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # cal x, y
        ymap, xmap = np.meshgrid(np.arange(self.height), np.arange(self.width))
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        self.points_x = torch.from_numpy(points_x).float()
        self.points_y = torch.from_numpy(points_y).float()

        cur_xs = self.points_x * self.zs
        cur_ys = self.points_y * self.zs

        points = np.stack(
            (cur_xs, cur_ys, self.zs),
            axis=-1
        )
            
        return points.transpose(1, 0, 2)

    def to_3d_coords(self, x, y):
        z = self.zs[x][y]
        return [self.points_x[x][y] * z, self.points_y[x][y] * z, z]

    def clip_empty_border(self, rgb, depth):
        foreground_mask = get_foreground_mask(depth)
        return foreground_mask
        x_min = 100000
        y_min = 100000
        x_max = 0
        y_max = 0

        for y in range(foreground_mask.shape[0]):
            for x in range(foreground_mask.shape[1]):
                if foreground_mask[y][x]:
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

        return rgb[y_min:y_max, x_min:x_max]

    def fit_plane(self, points):
        """
        Fit a plane z = ax + by + c to a set of 3D points using least squares.

        Parameters:
            points: (n,3) array-like of (x,y,z)

        Returns:
            a, b, c (plane parameters)
        """
        points = np.asarray(points)
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]

        # Design matrix
        A = np.c_[X, Y, np.ones(len(points))]

        # Solve least squares
        if (len(points) == 3):
            if np.linalg.matrix_rank(A) < 3:
                return None

            a, b, c = np.linalg.solve(A, Z)
        else:
            coeffs, residuals, _, _ = np.linalg.lstsq(A, Z, rcond=None)
            a, b, c = coeffs

            if residuals > self.max_residuals:
                return None

        return [a, b, c]
        
    def get_plane_distance_map(self, plane):
        point_plane_distances = plane[0] * self.point_map[:, :, 0] + plane[1] * self.point_map[:, :, 1] - self.point_map[:, :, 2] + plane[2]
        point_plane_distances /= math.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + 1)
        return point_plane_distances

    def score_plane(self, plane):
        point_plane_distances = self.get_plane_distance_map(plane)

        points_through_bitmap = (np.abs(point_plane_distances) < self.proximity) & (self.depth_map != 0)
        points_ahead_bitmap = (point_plane_distances >= self.clip_distance) & (self.depth_map != 0)

        score = np.count_nonzero(points_through_bitmap) + np.count_nonzero(points_ahead_bitmap) * self.ahead_point_score_ratio

        return score

    save_dir = Path("/mnt/c/Users/tymek/Desktop/MaRBLe/debug")
    def show_imgs(self, imgs):
        for i in range(len(imgs)):
            plt.subplot(221 + i)
            plt.imshow(imgs[i])
        plt.tight_layout()
        plt.savefig(self.save_dir / f"scene_{self.scene}_view_{self.view}.png")
        # plt.show()

    def generate_planes_random(self):
        planes = []
        for _ in range(100):
            points = [ self.point_map[randrange(self.height), randrange(self.width)],
                self.point_map[randrange(self.height), randrange(self.width)],
                self.point_map[randrange(self.height), randrange(self.width)]]

            plane = self.fit_plane(points)

            if plane != None:
                planes.append(plane)

        return planes

    def generate_planes_tiling(self):
        planes = []
        for y in range(0, self.depth_map.shape[0], self.group_size):
            for x in range(0, self.depth_map.shape[1], self.group_size):
                if y + self.group_size <= self.depth_map.shape[0] and x + self.group_size <= self.depth_map.shape[1]:
                    region = self.point_map[x:(x + self.group_size), y:(y + self.group_size)]
                    points = region.reshape(-1, 3)
                    size_before = len(points)
                    points = points[points[:, 2] != 0]

                    # Only proceed if less than 10% is 0 (sensor glitch)
                    if len(points) - size_before < self.allowed_null_depth_ratio * size_before:
                        plane = self.fit_plane(points)

                        if plane != None:
                            planes.append(plane)
        return planes


    def get_foreground_mask(self, debug = False):
        planes = self.generate_planes_random()
        
        self.best_plane = None
        best_score = 0
        for plane in planes:
            score = self.score_plane(plane)

            if score > best_score:
                best_score = score
                self.best_plane = plane

        point_plane_distances = self.get_plane_distance_map(self.best_plane)
        foreground_mask = (point_plane_distances >= self.clip_distance) & (self.depth_map != 0)
        foreground_mask = morphology.remove_small_objects(
            foreground_mask.astype(bool),
            min_size=1000
        ).astype(int)

        # Debug
        if debug:
            points_through_bitmap = (np.abs(point_plane_distances) < self.proximity) & (self.depth_map != 0)
            self.show_imgs([self.depth_map, self.rgb, points_through_bitmap, foreground_mask])
        
        # return np.ones(foreground_mask.shape)

        return foreground_mask

    def replace_depth_error_with_table(self):
        table = -self.best_plane[2] / ((np.array(self.points_x) * self.best_plane[0]) + (np.array(self.points_y) * self.best_plane[1]) - 1)
        table = table.T

        new_depth = np.array(self.depth_map, copy=True)

        error_map = (self.point_map[:, :, 2] >= table) | (self.point_map[:, :, 2] <= 0)
        new_depth[error_map] = table[error_map] * 1000

        # plt.subplot(221)
        # plt.imshow(self.point_map[:, :, 2])
        # plt.subplot(222)
        # plt.imshow(table)
        # plt.subplot(223)
        # plt.imshow(error_map)
        # plt.tight_layout()
        # plt.show()
        return new_depth
