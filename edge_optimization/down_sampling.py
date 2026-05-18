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

from ..train_utils import *

class DividedAnchorNet(nn.Module):
    def __init__(self, anchornet):
        super().__init__()

        self.anchornet = anchornet
        if isinstance(anchornet, torch.nn.DataParallel):
            self.anchornet = anchornet.module

        self.anchornet = self.anchornet.cpu()
    
    def tensor_to_visualization(self, tensor):
        if tensor.shape[1] == 4:
            # RGBD image
            rgb = np.array(tensor).squeeze().transpose(2, 1, 0)
            rgb = rgb[:,:,[1, 2, 3]]
        else:
            # heatmap
            rgb = np.array(tensor).squeeze().transpose(1, 0)
        return rgb

    def visualize_tensor(self, tensor):
        if tensor.shape[1] == 4:
            # RGBD image
            rgb = np.array(tensor).squeeze().transpose(2, 1, 0)
            rgb = rgb[:,:,[1, 2, 3]]
        else:
            # heatmap
            rgb = np.array(tensor).squeeze().transpose(1, 0)

        print(rgb.shape)
        plt.subplot(221)
        plt.imshow(rgb)
        plt.tight_layout()
        plt.show()

    times = []

    def forward(self, x):
        #target_size = 100
        original_shape = np.array([x.shape[2], x.shape[3]])

        partition = np.array([8, 4])# np.rint(original_shape / target_size) 
        scalar = 0
        padding = np.array([40, 45]) 
        partitions_shape = np.ceil(original_shape / partition)

        cells = subdivide(x, partition, partitions_shape, padding)

        # self.visualize_tensor(x)

        xs = []

        start = time()

        for cell in cells:
            xs.append(self.anchornet(cell))

        end = time()

        x_celled = reconstruct(xs, partition, padding)
        x_original = self.anchornet(x)

        self.times.append(end - start)
        print("time average", np.average(np.array(self.times)))

        img_0 = self.tensor_to_visualization(x_original[0])
        img_1 = self.tensor_to_visualization(x_celled[0])

        plt.subplot(221)
        plt.imshow(img_0)
        plt.subplot(222)
        plt.imshow(img_1)
        plt.tight_layout()
        plt.show()

        return x_celled

def get_random_scene():
    scene = randrange(190)
    view = randrange(256)
    sceneIds = list([scene])
    test_dataset = GraspnetPointDataset(args.all_points_num,
                                    args.dataset_path,
                                    args.scene_path,
                                    sceneIds,
                                    sigma=args.sigma,
                                    ratio=args.ratio,
                                    anchor_k=args.anchor_k,
                                    anchor_z=args.anchor_z,
                                    anchor_w=args.anchor_w,
                                    grasp_count=args.grasp_count,
                                    output_size=(args.input_w,
                                                    args.input_h),
                                    random_rotate=False,
                                    random_zoom=False)

    SCENE_LIST = test_dataset.scene_list()
    test_data = DataLoader(test_dataset,
                           batch_size=1,
                           pin_memory=True,
                           num_workers=1)
    test_data.dataset.unaug()
    test_data.dataset.eval()

    print("returning scene", scene, "view", view)
    return test_dataset[view]


def extract_edges(depth):
    depth = cv2.blur(depth, (5, 5))
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canny_edges = cv2.Canny(depth_normalized, threshold1 = 100, threshold2 = 300)  
    mask = canny_edges > 0

    return mask

def clip_empty_border(rgb, depth):
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

def fit_plane(points):
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
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = coeffs

    return [a, b, c]

def fit_plane_range(depth_map, x_min, y_min, size):
    points = []

    for y in range(y_min, y_min + size):
        for x in range(x_min, x_min + size):
            points.append([x, y, depth_map[y][x]])

    plane = fit_plane(points)
    return plane
    
def plane_distance(plane_a, plane_b):
    max_distance = 0

    for measure_point in [(0, 0), (0, 720), (1280, 0), (1280, 720)]:
        depth_a = plane_a[0] * measure_point[0] + plane_a[1] * measure_point[1] + plane_a[2]
        depth_b = plane_b[0] * measure_point[0] + plane_b[1] * measure_point[1] + plane_b[2]

        max_distance = max(max_distance, abs(depth_a - depth_b))

    return max_distance

def point_plane_distance(plane, point):
    distance = plane[0] * point[0] + plane[1] * point[1] - point[2] + plane[2]
    distance /= math.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + 1)
    return distance

def get_foreground_mask(depth_map, group_size = 50, proximity = 50, clip_distance = 20):
    planes = []

    for y in range(0, depth_map.shape[0], group_size):
        for x in range(0, depth_map.shape[1], group_size):
            if y + group_size <= depth_map.shape[0] and x + group_size <= depth_map.shape[1]:
                plane_here = fit_plane_range(depth_map, x, y, group_size)
                planes.append(plane_here)
                # allocated_group = allocate_to_group(groups, plane_here, proximity)
                # plane_image[y:y + group_size, x:x + group_size] = allocated_group * 10
    
    best_planes = []
    for plane_a in planes:
        matching_planes = []
        for plane_b in planes:
            if plane_distance(plane_a, plane_b) <= proximity:
                matching_planes.append(plane_b)
        
        if len(matching_planes) > len(best_planes):
            best_planes = matching_planes
    
    best_planes = np.array(best_planes)

    a = np.average(best_planes[:, 0])
    b = np.average(best_planes[:, 1])
    c = np.average(best_planes[:, 2])

    plane_image = np.zeros(depth_map.shape)
    for y in range(depth_map.shape[0]):
        for x in range(depth_map.shape[1]):
            plane_image[y][x] = point_plane_distance([a, b, c], [x, y, depth_map[y][x]]) >= clip_distance and depth_map[y][x] != 0

    return plane_image


def calculate_gradient(depth):
    sobelx = cv2.Sobel(depth,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(depth,cv2.CV_64F,0,1,ksize=5)
    sobel_magnitude = cv2.magnitude(sobelx, sobely)  

    # scharrx = cv2.Scharr(depth, cv2.CV_64F, 1, 0)
    # scharry = cv2.Scharr(depth, cv2.CV_64F, 0, 1)
    # scharr_edges = cv2 . magnitude ( scharrx , scharrx )  

    return sobel_magnitude

def remove_padding(cell, cell_x, cell_y, partition, partitions_size, padding):

    #print("padding is", padding)

    padding_left, padding_right, padding_top, padding_bot = get_padding(cell_x, cell_y, partition, partitions_size, padding)

    return cell[:, :, padding_left : cell.shape[2] - padding_right, padding_top : cell.shape[3] - padding_bot]



def reconstruct(cell_outputs, partition, padding):
    output = []

    for i in range(6):
        desired_dim = torch.Size([1, 6, 80, 45])
        if i == 0:
            desired_dim =  torch.Size([1, 1, 640, 360])

        desired_partition_dim = np.ceil(np.array([desired_dim[2], desired_dim[3]]) / partition)

        padding_here = padding * (np.array([desired_dim[2], desired_dim[3]]) / np.array([640, 360]))

        j = 0
        channel_output = None

        for cell_x in range(int(partition[0])):
            column_output = None

            for cell_y in range(int(partition[1])):
                cell = cell_outputs[j][i]
                cell_output = remove_padding(
                    cell, 
                    cell_x,
                    cell_y,
                    partition,
                    desired_partition_dim,
                    padding_here
                )


                if column_output == None:
                    column_output = cell_output
                else:
                    column_output = torch.cat([column_output, cell_output], dim=3)

                j += 1

            if channel_output == None:
                channel_output = column_output
            else:
                channel_output = torch.cat([channel_output, column_output], dim=2)

        if channel_output.shape != desired_dim:
            print("Reshaping an output from ", channel_output.shape, "to", desired_dim, "!!!")
            channel_output = F.interpolate(channel_output, torch.Size([desired_dim[2], desired_dim[3]]))
        output.append(channel_output)

    return output

def clamp(number, lower, upper):
    return int(min(max(number, lower), upper))

def get_corners_with_padding(cell_x, cell_y, partition, partitions_size, padding):

    total_width = partition[0] * partitions_size[0]
    total_height = partition[1] * partitions_size[1]

    #print("test", cell_x , partitions_size[0], padding[0], total_width)

    start_x = clamp(int(cell_x * partitions_size[0]) - padding[0], 0, total_width)
    end_x = clamp(int((cell_x + 1) * (partitions_size[0])) + padding[0], 0, total_width)
    start_y = clamp(int(cell_y * partitions_size[1]) - padding[1], 0, total_height)
    end_y = clamp(int((cell_y + 1) * partitions_size[1]) + padding[1], 0, total_height)
    
    return (start_x, end_x, start_y, end_y)

def get_padding(cell_x, cell_y, partition, partitions_size, padding):
    start_x, end_x, start_y, end_y = get_corners_with_padding(cell_x, cell_y, partition, partitions_size, padding)

    padding_left = int(cell_x * partitions_size[0]) - start_x
    padding_right = end_x - int((cell_x + 1) * (partitions_size[0])) 
    padding_top = int(cell_y * partitions_size[1]) - start_y
    padding_bot = end_y - int((cell_y + 1) * partitions_size[1])

    #print("padding:", cell_x, cell_y, end_x, start_x, padding_left, padding_right, padding_top, padding_bot)

    return (padding_left, padding_right, padding_top, padding_bot)

def subdivide(image, partition, partitions_size, padding):

    cells = []
    for cell_x in range(int(partition[0])):
        for cell_y in range(int(partition[1])):
            start_x, end_x, start_y, end_y = get_corners_with_padding(cell_x, cell_y, partition, partitions_size, padding)
            #print("start x ", start_x, "end x", end_x)
            cells.append(image[:, :, start_x:end_x, start_y:end_y])
            #print("cell size ", cells[len(cells) - 1].shape)


    return cells

def show_image(rgb, depth, mask = None):
    #resized_rgb = Image.fromarray((x * 255.0).astype(np.uint8))
    #resized_rgb = np.array(
     #   resized_rgb.resize((args.input_w, args.input_h))) / 255.0
    plt.subplot(221)
    plt.imshow(rgb)
    plt.subplot(222)
    plt.imshow(depth)
    if len(mask) != 0:
        print(np.min(mask), np.max(mask))


        plt.subplot(223)
        plt.imshow(mask)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    (anchor_data, rgb, depth, grasppaths) = get_random_scene()

    rgb = np.array(rgb).squeeze().transpose(2, 1, 0)
    depth = np.array(depth).squeeze().T

    mask = clip_empty_border(rgb, depth)
    #subdivide(depth, 128)
    show_image(rgb, depth, mask)