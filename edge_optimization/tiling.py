import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy
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
from .background_clipping import *
from .subdivided_cell import *
from .clutter_metric import *

# from ..train_utils import *

class DividedAnchorNet(nn.Module):
    def __init__(self, anchornet, partition):
        super().__init__()

        self.partition = np.array(partition)
        self.anchornet = anchornet
        if isinstance(anchornet, torch.nn.DataParallel):
            self.anchornet = anchornet.module

        self.anchornet = self.anchornet.cpu()
        self.clutter_debug = []
        self.times = []
        self.times_original = []
    
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

    

    def forward(self, x, depth):
        depth = np.array(depth).squeeze().T
        depth = cv2.resize(depth, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)
        # depth = np.array(x[0, 0].transpose(1, 0))
        rgb = np.array(x[0, 1:4].transpose(2, 0))

        # print(depth.shape)
        # print(rgb.shape)

        #target_size = 100
        original_shape = np.array([x.shape[2], x.shape[3]])

        #partition = np.array([1, 1])# np.array([8, 4])# np.rint(original_shape / target_size) 
        padding = np.array([0, 0]) 
        partitions_shape = np.ceil(original_shape / self.partition)

        # self.visualize_tensor(depth_img)


        bgClipper = BackgroundClipper(depth, rgb)
        foreground_mask = bgClipper.foreground_mask

        # self.visualize_tensor(foreground_mask)
        cells = subdivide(x, self.partition, partitions_shape, padding, foreground_mask)

        # clutter_metrics = get_clutter_metric_in_cells(cells, rgb, bgClipper)
        # self.clutter_debug.append(clutter_metrics[0])
        # print("clutters: ", self.clutter_debug)

        xs = []

        start_original = time()

        x_original = self.anchornet(x)

        end_original = time()

        start = time()

        for cell in cells:
            if cell.model_input != None:
                output = self.anchornet(cell.model_input)
                xs.append(output)
            else:
                xs.append(None)

        end = time()



        self.times.append(end - start)
        self.times_original.append(end_original - start_original)
        # print("tiling time avg.", np.average(np.array(self.times)), "original avg.", np.average(np.array(self.times_original)))

        # x_celled = reconstruct(xs, cells, self.partition, padding)

        # img_0 = self.tensor_to_visualization(x_original[0])
        # img_1 = self.tensor_to_visualization(x_celled[0])

        # plt.subplot(221)
        # plt.imshow(img_0)
        # plt.subplot(222)
        # plt.imshow(img_1)
        # plt.tight_layout()
        # plt.show()

        return x_original

scene_num = 0
view_num = 0

def get_sample(args, scene, view):
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
    return test_dataset[view], scene, view

def get_scenes_in_order(args):
    global scene_num
    global view_num
    scene = scene_num
    view = view_num

    view_num += 1
    if view_num == 256:
        scene_num += 1
        view_num = 0

    return get_sample(args, scene, view)

def get_random_scene(args):
    scene = randrange(190)
    view = randrange(256)

    return get_sample(args, scene, view)

def subdivide(image, partition, partitions_size, padding, foreground_mask):
    cells = []
    for cell_x in range(int(partition[0])):
        for cell_y in range(int(partition[1])):
            cells.append(Cell([cell_x, cell_y], partition, partitions_size, padding, image, foreground_mask, 1))

    return cells

def reconstruct(model_outputs, cells, partition, padding):
    output = []

    for i in range(6):
        desired_dim = torch.Size([1, 6, 80, 45])
        if i == 0:
            desired_dim = torch.Size([1, 1, 640, 360])
        if i == 5:
            desired_dim = torch.Size([1, 32, 80, 45])

        cell_i = 0
        j = 0
        channel_output = None

        for cell_x in range(int(partition[0])):
            column_output = None

            for cell_y in range(int(partition[1])):
                to_rescale = None
                if model_outputs[cell_i] != None:
                    to_rescale = model_outputs[cell_i][i]

                cell_output = cells[cell_i].fit_output_to_size(to_rescale, desired_dim, i == 0)

                if column_output is None:
                    column_output = cell_output
                else:
                    column_output = torch.cat([column_output, cell_output], dim=3)

                j += 1
                cell_i += 1

            if channel_output == None:
                channel_output = column_output
            else:
                channel_output = torch.cat([channel_output, column_output], dim=2)

        if channel_output.shape != desired_dim:
            print("ERROR!!! Reshaping an output from ", channel_output.shape, "to", desired_dim, "!!!")
            channel_output = F.interpolate(channel_output, torch.Size([desired_dim[2], desired_dim[3]]))
        output.append(channel_output)

    return output

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

def test_clutter_metric(args):
    random.seed(time())

    while True:
        (anchor_data, rgb, depth, grasppaths), scene, view = get_random_scene(args)
        rgb = np.array(rgb).squeeze().transpose(2, 1, 0)
        depth = np.array(depth).squeeze().T

        depth = cv2.resize(depth, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)

        bgClipper = BackgroundClipper(depth, rgb)

        partition = np.array([8, 4])# np.array([8, 4])# np.rint(original_shape / target_size) 
        padding = np.array([20, 20]) 
        original_shape = np.array([640, 360])
        partitions_shape = np.ceil(original_shape / partition)

        # self.visualize_tensor(depth_img)


        bgClipper = BackgroundClipper(depth, rgb)
        foreground_mask = bgClipper.foreground_mask

        # self.visualize_tensor(foreground_mask)
        x = torch.empty((1, 4, 640, 360))
        cells = subdivide(x, partition, partitions_shape, padding, foreground_mask)

        get_clutter_metric_in_cells(cells, rgb, bgClipper)

def test_clipping(args):
    random.seed(time())

    while True:
        (anchor_data, rgb, depth, grasppaths), scene, view = get_random_scene(args)

        rgb = np.array(rgb).squeeze().transpose(2, 1, 0)
        depth = np.array(depth).squeeze().T

        mask = BackgroundClipper(cv2.resize(depth, dsize=(640, 360), interpolation=cv2.INTER_CUBIC), rgb, scene = scene, view = view).foreground_mask
        #subdivide(depth, 128)
        # show_image(rgb, depth, mask)