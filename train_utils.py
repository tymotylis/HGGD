import argparse
import datetime
import json
import logging
import os
import random
import sys

import torch.nn as nn
import numpy as np
import tensorboardX
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm

from .customgraspnetAPI import Grasp, GraspGroup
from .models.anchornet import AnchorGraspNet
from .models.localgraspnet import PointMultiGraspNet
from .dataset.pc_dataset_tools import (get_ori_grasp_label, feature_fusion, data_process, get_center_group_label)
from .dataset.evaluation import (anchor_output_process, calculate_6d_match,
                                calculate_coverage, calculate_iou_match,
                                detect_2d_grasp,
                                detect_6d_grasp_multi)
from .dataset.evaluation_coll import collision_detect
from .dataset.grasp import RectGraspGroup
from .models.losses import compute_anchor_loss, compute_multicls_loss
from .edge_optimization.tiling import *


eval_scale = np.linspace(0.2, 1, 5)


def log_acc_str(name, T, F):
    T, F = int(T), int(F)
    if T + F == 0:
        return f'{name} 0/0 = 0'
    return f'{name} {T}/{T + F} = {T / (T + F):.3f}'


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    # Network
    # 2d
    parser.add_argument('--resume', type=str, default=None, help='Model path')
    parser.add_argument('--input-h',
                        type=int,
                        default=360,
                        help='Input image size for the network')
    parser.add_argument('--input-w',
                        type=int,
                        default=640,
                        help='Input image size for the network')
    parser.add_argument('--use-depth',
                        type=int,
                        default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb',
                        type=int,
                        default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--iou-threshold',
                        type=float,
                        default=0.25,
                        help='Threshold for IOU matching')

    # pc
    parser.add_argument(
        '--center-num',
        type=int,
        default=128,
        help='choose how many centers from 2d predicted heatmap')
    parser.add_argument('--group-num',
                        type=int,
                        default=512,
                        help='point num around one center of ball query')
    parser.add_argument('--anchor-num',
                        type=int,
                        default=7,
                        help='anchor num for gamma and beta')
    parser.add_argument('--local-grasp-num',
                        type=int,
                        default=500,
                        help='number of local grasps in local pointcloud')

    # Dataset
    parser.add_argument('--scene-l',
                        type=int,
                        default=0,
                        help='Scene id left range')
    parser.add_argument('--scene-r',
                        type=int,
                        default=100,
                        help='Scene id right range')
    parser.add_argument('--dataset-path',
                        type=str,
                        default=None,
                        help='Path to grasp dataset')
    parser.add_argument('--scene-path',
                        type=str,
                        default=None,
                        help='Path to scene dataset')
    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='Checkpoint path to load')
    parser.add_argument('--num-workers',
                        type=int,
                        default=4,
                        help='Dataset workers')

    # Anchor
    parser.add_argument('--ratio',
                        type=int,
                        default=8,
                        help='Down sample ratio')
    parser.add_argument('--grid-size',
                        type=int,
                        default=8,
                        help='2D center select grid size')
    parser.add_argument(
        '--anchor-k',
        type=int,
        default=6,
        help='The number of oriented anchor boxes with different angles')
    parser.add_argument('--anchor-w',
                        type=float,
                        default=50.0,
                        help='The default width of the anchor boxes')
    parser.add_argument('--anchor-z',
                        type=float,
                        default=20.0,
                        help='The default z of the anchor boxes')
    parser.add_argument('--grasp-count',
                        type=int,
                        default=5000,
                        help='The default grasp count of one image')
    parser.add_argument('--sigma',
                        type=int,
                        default=10,
                        help='Gaussian kernel sigma')
    parser.add_argument('--loc-a',
                        type=float,
                        default=1,
                        help='loss coef for loc map')
    parser.add_argument('--reg-b',
                        type=float,
                        default=1,
                        help='loss coef for regress')
    parser.add_argument('--cls-c',
                        type=float,
                        default=1,
                        help='loss coef for classify')
    parser.add_argument('--offset-d',
                        type=float,
                        default=5,
                        help='loss coef for grasp 3d offset')

    # Training
    parser.add_argument('--all-points-num',
                        type=int,
                        default=25600,
                        help='downsample scene points')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--shift-epoch',
                        type=int,
                        default=5,
                        help='Epoch num for anchor shifting')
    parser.add_argument(
        '--pre-epochs',
        type=int,
        default=-1,
        help='Pre training 2d epochs, will be 0 if joint-trainning')
    parser.add_argument('--joint-trainning',
                        action='store_true',
                        help='Whether to train 2d and 6d net together')
    parser.add_argument('--epochs',
                        type=int,
                        default=15,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--optim',
                        type=str,
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optmizer for the training. (adam, adamw or SGD)')
    parser.add_argument(
        '--step-cnt',
        type=int,
        default=1,
        help='Network batch step cnt (batch_size * step_cnt == real_batch_size)'
    )
    parser.add_argument('--noise', type=float, default=0.0, help='Depth noise')

    # grasp detection
    parser.add_argument('--heatmap-thres',
                        type=float,
                        default=0.01,
                        help='2D grasp generation heatmap_thres')
    parser.add_argument('--local-k',
                        type=int,
                        default=3,
                        help='Local anchor top-k selection')
    parser.add_argument('--local-thres',
                        type=float,
                        default=0.01,
                        help='6D grasp generation local multi_cls score thres')
    parser.add_argument('--rotation-num',
                        type=int,
                        default=1,
                        help='Local rotation num for 2D grasp')
    parser.add_argument('--top-num',
                        type=float,
                        default=1.0,
                        help='Grasp Detect Ratio Number')

    # Quantization
    parser.add_argument('--q_anchornet_type', type=str, default="None", help='None | 8-bit | 4-bit | QAT')
    parser.add_argument('--q_anchornet_scales', type=str, default="Affine", help='Symmetric | Affine')

    parser.add_argument('--q_localnet_type', type=str, default="None", help='None | Normal | Optimized | QAT')
    # parser.add_argument('--q_localnet_scales', type=str, default="Per-tensor", help='Per-tensor | Per-channel')

    parser.add_argument('--qat_epochs', type=int, default=10)
    parser.add_argument('--callibration_samples', type=int, default=256)

    # Logging etc.
    parser.add_argument('--description',
                        type=str,
                        default='',
                        help='Training description')
    parser.add_argument('--save-freq',
                        type=int,
                        default=1,
                        help='Model save frequency')
    parser.add_argument('--logdir',
                        type=str,
                        default='./logs/',
                        help='Log directory')
    parser.add_argument('--random-seed',
                        type=int,
                        default=123,
                        help='Random seed')

    args = parser.parse_args()
    if args.joint_trainning:
        args.pre_epochs = 0
        print('Joint Trainning for the whole network')
    return args


def prepare_torch_and_logger(args, mode='train'):
    # multiprocess
    # mp.set_start_method('spawn')
    # set torch and gpu setting
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        raise RuntimeError('CUDA not available')

    # random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set-up output directories
    net_desc = datetime.now().strftime('%y%m%d_%H%M%S')
    net_desc = net_desc + '_' + args.description
    if mode == 'test':
        net_desc = 'test' + net_desc

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename='{0}/{1}.log'.format(save_folder, 'log'),
        format=
        '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    return tb, save_folder


def get_optimizer(args, params):
    # get optimizer
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    elif args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(
            args.optim))
    return optimizer


def print_model(args, input_channels, model, save_folder):
    summary(model, (input_channels, args.input_w, args.input_h), device='cpu')
    with open(os.path.join(save_folder, 'arch.txt'), 'w') as f:
        sys.stdout = f
        summary(model, (input_channels, args.input_w, args.input_h),
                device='cpu')
        sys.stdout = sys.__stdout__


def log_match_result(results, dis_criterion, rot_criterion):
    for scale_factor in eval_scale:
        # get threshold from criterion and factor
        thre_dis = dis_criterion * scale_factor
        thre_rot = rot_criterion * scale_factor

        t_trans, f_trans = results[f'trans_{thre_dis}']
        t_rot, f_rot = results[f'rot_{thre_rot}']
        t_grasp, f_grasp = results[f'grasp_{scale_factor}']

        t_str = log_acc_str(f'trans_{thre_dis:.2f}', t_trans, f_trans)
        r_str = log_acc_str(f'rot_{thre_rot:.2f}', t_rot, f_rot)
        g_str = log_acc_str(f'grasp_{scale_factor:.2f}', t_grasp, f_grasp)

        logging.info(f'{t_str}  {r_str}  {g_str}')


def log_and_save(args,
                 tb,
                 results,
                 epoch,
                 anchornet,
                 localnet,
                 optimizer,
                 anchors,
                 save_folder,
                 mode='regnet'):
    # Log validation results to tensorboard
    # loss
    tb.add_scalar('val_loss/loss', results['loss'], epoch)
    tb.add_scalar('val_loss/anchor_loss', results['anchor_loss'], epoch)
    for n, l in results['losses'].items():
        tb.add_scalar('val_loss/' + n, l, epoch)

    logging.info('Validation Loss:')
    logging.info(f'test loss: {results["loss"]:.3f}')
    logging.info(f'anchor loss: {results["anchor_loss"]:.3f}')
    if 'loc_map_loss' in results['losses']:
        logging.info(
            f'loc: {results["losses"]["loc_map_loss"]:.3f}, reg: {results["losses"]["reg_loss"]:.3f}, cls: {results["losses"]["cls_loss"]:.3f}'
        )
    if epoch >= args.pre_epochs:
        tb.add_scalar('val_loss/multi_cls_loss', results['multi_cls_loss'],
                      epoch)
        tb.add_scalar('val_loss/offset_loss', results['offset_loss'], epoch)
        logging.info(f'multicls_loss: {results["multi_cls_loss"]:.3f}')
        logging.info(f'offset_loss: {results["offset_loss"]:.3f}')

    # coverage
    if epoch >= args.pre_epochs:
        cover_cnt = results['cover_cnt']
        label_cnt = results['label_cnt']
        tb.add_scalar('coverage', cover_cnt / label_cnt, epoch)
        logging.info(
            f'coverage rate: {cover_cnt} / {label_cnt} = {cover_cnt / label_cnt:.3f}'
        )

    # 2d iou
    if results['total'] > 0:
        iou = results['correct'] / results['total']
        tb.add_scalar('IOU', iou, epoch)
        logging.info(f'2d iou: {iou:.2f}')

    # regnet validation
    if epoch >= args.pre_epochs:
        if mode == 'regnet':
            view_num = results['grasp_nocoll_view_num']
            vgr = results['vgr']
            score = results['score']
            if view_num > 0:
                tb.add_scalar('collision_free_ratio', vgr / view_num, epoch)
                tb.add_scalar('score', score / view_num, epoch)

                logging.info('REGNet validation:')
                logging.info(f'vgr: {vgr} / {view_num} = {vgr / view_num:.3f}')
                logging.info(
                    f'score: {score:.3f} / {view_num} = {score / view_num:.3f}'
                )
            else:
                logging.info('No collision-free grasp')
        elif mode == 'graspnet':
            logging.info('please run test_graspnet.py for graspnet result')

    anchornet_module = anchornet
    if isinstance(anchornet_module, torch.nn.DataParallel):
        anchornet_module = anchornet_module.module

    localnet_module = localnet
    if isinstance(localnet_module, torch.nn.DataParallel):
        localnet_module = localnet_module.module

    # Save best performing network
    if epoch % args.save_freq == 0 and optimizer is not None:
        if epoch < args.pre_epochs:
            torch.save(
                {
                    'anchor': anchornet_module.state_dict(),
                    'local': localnet_module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'gamma': anchors['gamma'],
                    'beta': anchors['beta']
                }, os.path.join(save_folder, f'epoch_{epoch}_iou_{iou:.3f}'))
        elif mode == 'regnet':
            torch.save(
                {
                    'anchor': anchornet_module.state_dict(),
                    'local': localnet_module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'gamma': anchors['gamma'],
                    'beta': anchors['beta']
                },
                os.path.join(
                    save_folder,
                    f'epoch_{epoch}_score_{score / view_num:.3f}_cover_{cover_cnt / label_cnt:.3f}'
                ))
        elif mode == 'graspnet':
            torch.save(
                {
                    'anchor': anchornet_module.state_dict(),
                    'local': localnet_module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'gamma': anchors['gamma'],
                    'beta': anchors['beta']
                },
                os.path.join(
                    save_folder,
                    f'epoch_{epoch}_iou_{iou:.3f}_cover_{cover_cnt / label_cnt:.3f}'
                ))


def log_test_result(args, results, epoch, mode='regnet'):
    # Log validation results to tensorboard
    # loss
    logging.info('Test Loss:')
    logging.info(f'test loss: {results["loss"]:.3f}')
    logging.info(f'anchor loss: {results["anchor_loss"]:.3f}')
    logging.info(
        f'loc: {results["losses"]["loc_map_loss"]:.3f}, reg: {results["losses"]["reg_loss"]:.3f}, cls: {results["losses"]["cls_loss"]:.3f}'
    )
    if epoch >= args.pre_epochs:
        logging.info(f'multicls_loss: {results["multi_cls_loss"]:.3f}')

    # coverage
    if epoch >= args.pre_epochs:
        cover_cnt = results['cover_cnt']
        label_cnt = results['label_cnt']
        logging.info(
            f'coverage rate: {cover_cnt} / {label_cnt} = {cover_cnt / label_cnt:.3f}'
        )

    # 2d iou
    iou = results['correct'] / (results['correct'] + results['failed'])
    logging.info(f'2d iou: {iou:.2f}')

    # regnet validation
    if epoch >= args.pre_epochs:
        if mode == 'regnet':
            view_num = results['grasp_nocoll_view_num']
            vgr = results['vgr']
            score = results['score']
            if view_num > 0:
                logging.info('REGNet validation:')
                logging.info(f'vgr: {vgr} / {view_num} = {vgr / view_num:.3f}')
                logging.info(
                    f'score: {score:.3f} / {view_num} = {score / view_num:.3f}'
                )
            else:
                logging.info('No collision-free grasp')
        elif mode == 'graspnet':
            logging.info('please run test_graspnet.py for graspnet result')


def log_anchor_loss(epoch, batch_idx, loss, anchor_loss, anchor_losses,
                    batch_cnt):
    logging.info('Epoch: {}, Batch: {}, total_loss: {:0.4f}'.format(
        epoch, batch_idx, loss / batch_cnt))
    logging.info('anchor_loss: {:0.4f}'.format(anchor_loss / batch_cnt))
    logging.info(
        'loc_map_loss: {:0.4f}, reg_loss: {:0.4f}, cls_loss: {:0.4f}'.format(
            anchor_losses['loc_map_loss'] / batch_cnt,
            anchor_losses['reg_loss'] / batch_cnt,
            anchor_losses['cls_loss'] / batch_cnt))


def dump_grasp(epoch, batch_idx, pred_gg, scene_list, dump_dir='./pred'):
    gg = GraspGroup()
    for g in pred_gg:
        g = Grasp(1, g.width, 0.02, 0.02, g.rotation.reshape(9, ),
                  g.translation, -1)
        gg.add(g)

    # save grasps
    save_dir = os.path.join(dump_dir, f'epoch_{epoch}')
    save_dir = os.path.join(save_dir, scene_list[batch_idx])
    save_dir = os.path.join(save_dir, 'realsense')
    save_path = os.path.join(save_dir, str(batch_idx % 256).zfill(4) + '.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gg.save_npy(save_path)

dis_criterion = 0.05
rot_criterion = 0.25

def validate(epoch, anchornet: nn.Module, localnet: nn.Module,
             val_data: torch.utils.data.DataLoader, anchors: dict, args, use_cuda = True, quantized_mode = False, end_early = -1):
    device = 'cuda' if use_cuda else 'cpu'
    fixed_center_num = 48

    # network eval mode
    anchornet.eval()
    localnet.eval()
    # stop rot and zoom for validation
    val_data.dataset.eval()

    results = {
        'correct': 0,
        'total': 0,
        'loss': 0,
        'losses': {},
        'multi_cls_loss': 0,
        'offset_loss': 0,
        'offset_loss': 0,
        'anchor_loss': 0,
        'cover_cnt': 0,
        'label_cnt': 0
    }
    valid_center_num, total_center_num = 0, 0
    for scale_factor in eval_scale:
        thre_dis = dis_criterion * scale_factor
        thre_rot = rot_criterion * scale_factor
        results[f'grasp_{scale_factor}'] = np.zeros((2, ))
        results[f'trans_{thre_dis}'] = np.zeros((2, ))
        results[f'rot_{thre_rot}'] = np.zeros((2, ))

    # stop rot and zoom for validation
    batch_idx = -1
    with torch.no_grad():
        for anchor_data, rgb, depth, grasppaths in tqdm(val_data,
                                                        desc=f'Valid_{epoch}',
                                                        ncols=80):
            end_early -= 1
            if end_early == 0:
                break

            batch_idx += 1
            # get scene points
            points, _, _ = val_data.dataset.helper.to_scene_points(
                rgb.cuda() if use_cuda else rgb, depth.cuda() if use_cuda else depth, include_rgb=False, use_cuda=use_cuda)
            # get xyz maps
            xyzs = val_data.dataset.helper.to_xyz_maps(depth.cuda() if use_cuda else depth, use_cuda)
            # get labels
            gg_ori_labels = get_ori_grasp_label(grasppaths)
            all_grasp_labels = []
            for grasppath in grasppaths:
                all_grasp_labels.append(np.load(grasppath))

            # 2d prediction
            x, y, _, _, _ = anchor_data

            if use_cuda:
                x = x.cuda()
            target = [(yy.cuda() if use_cuda else yy) for yy in y]

            if quantized_mode:
                x = x.cpu()

            if isinstance(anchornet, DividedAnchorNet):
                outputs = anchornet(x.cpu(), depth.cpu())
                x = x.cuda()
                depth = depth.cuda()
            else:
                outputs = anchornet(x)
            
            pred_2d = (outputs[0].cuda(), outputs[1].cuda(), outputs[2].cuda(), outputs[3].cuda(), outputs[4].cuda())
            perpoint_features = outputs[5].cuda()

            loc_map, cls_mask, theta_offset, depth_offset, width_offset = \
                anchor_output_process(*pred_2d, sigma=args.sigma)

            # detect 2d grasp (x, y, theta)
            rect_gg = detect_2d_grasp(loc_map,
                                      cls_mask,
                                      theta_offset,
                                      depth_offset,
                                      width_offset,
                                      ratio=args.ratio,
                                      anchor_k=args.anchor_k,
                                      anchor_w=args.anchor_w,
                                      anchor_z=args.anchor_z,
                                      mask_thre=args.heatmap_thres,
                                      center_num=fixed_center_num,
                                      grid_size=args.grid_size,
                                      grasp_nms=args.grid_size)

            # cal loss
            anchor_lossd = compute_anchor_loss(pred_2d,
                                               target,
                                               loc_a=args.loc_a,
                                               reg_b=args.reg_b,
                                               cls_c=args.cls_c)
            anchor_losses = anchor_lossd['losses']
            anchor_loss = anchor_lossd['loss']

            # convert back to np.array
            # rot should be 0, zoom should be 1
            idx = anchor_data[2].numpy().squeeze()
            rot = anchor_data[3].numpy().squeeze()
            zoom_factor = anchor_data[4].numpy().squeeze()

            # 2d bbox validation
            grasp_label = val_data.dataset.load_grasp_labels(idx)
            gt_rect_gg = RectGraspGroup()
            gt_rect_gg.load_from_dict(grasp_label)
            gt_bbs = val_data.dataset.get_gtbb(gt_rect_gg, rot, zoom_factor)

            # cal 2d iou
            s = calculate_iou_match(rect_gg[0:1], gt_bbs, thre=0.25)
            if s:
                results['correct'] += 1
            results['total'] += 1

            multi_cls_loss = 0
            offset_loss = 0
            if epoch >= args.pre_epochs:
                # check 2d result
                if rect_gg.size == 0:
                    print('No 2d grasp found')
                    continue

                # feature fusion using knn and max pooling
                points_all = feature_fusion(points, perpoint_features, xyzs)
                rect_ggs = [rect_gg]
                pc_group, valid_local_centers = data_process(
                    points_all,
                    depth.cuda() if use_cuda else depth,
                    rect_ggs,
                    args.center_num,
                    args.group_num, (args.input_w, args.input_h),
                    is_training=False,
                    use_cuda=use_cuda)
                rect_gg = rect_ggs[0]  # maybe modify in data process
                # batch_size == 1 when valid
                points_all = points_all.squeeze()

                # check pc_group
                if pc_group.shape[0] == 0:
                    print('No partial point clouds')
                    continue

                # get 2d grasp info (not grasp itself) for trainning
                grasp_info = np.zeros((0, 3), dtype=np.float32)
                g_thetas = rect_gg.thetas[None]
                g_ws = rect_gg.widths[None]
                g_ds = rect_gg.depths[None]
                cur_info = np.vstack([g_thetas, g_ws, g_ds])
                grasp_info = np.vstack([grasp_info, cur_info.T])
                grasp_info = torch.from_numpy(grasp_info).to(
                    dtype=torch.float32, device=device)

                # get gamma and beta classification result
                # padding for benchmark
                zero_pad_num = fixed_center_num - pc_group.shape[0]
                pc_group = torch.concat([
                    pc_group,
                    torch.zeros(zero_pad_num,
                                pc_group.shape[1],
                                pc_group.shape[2],
                                device=device)
                ])
                grasp_info = torch.concat([
                    grasp_info,
                    torch.zeros(zero_pad_num,
                                grasp_info.shape[1],
                                device=device)
                ])

                if quantized_mode:
                    pc_group = pc_group.cpu()
                    grasp_info = grasp_info.cpu()

                localnet_output = localnet([pc_group, grasp_info])

                pred_view = localnet_output[1].cuda()
                offset = localnet_output[2].cuda()
                pc_group = pc_group.cuda()
                grasp_info = grasp_info.cuda()

                valid_num = fixed_center_num - zero_pad_num
                pc_group = pc_group[:valid_num]
                pred_view = pred_view[:valid_num]
                offset = offset[:valid_num]

                # detect 6d grasp from 2d output and 6d output
                pred_grasp, pred_rect_gg = detect_6d_grasp_multi(
                    rect_gg,
                    pred_view,
                    offset,
                    valid_local_centers, (args.input_w, args.input_h),
                    anchors,
                    k=args.local_k)
                pred_grasp = torch.from_numpy(pred_grasp).to(
                    device=device, dtype=torch.float32)

                # get nearest grasp labels
                gg_labels, _ = get_center_group_label(valid_local_centers,
                                                      all_grasp_labels,
                                                      args.local_grasp_num)
                # get center valid stats
                total_center_num += len(gg_labels)
                for gg in gg_labels:
                    valid_center_num += len(gg) > 0
                # get loss
                multi_cls_loss, offset_loss = compute_multicls_loss(
                    pred_view, offset, gg_labels, grasp_info, anchors, args)

                # collision detect
                pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group()
                pred_gg, valid_mask = collision_detect(points_all,
                                                       pred_grasp_from_rect,
                                                       mode='graspnet')
                pred_grasp = pred_grasp[valid_mask]

                # cal distance to evaluate grasp quality
                # multi scale thresold
                gg_ori_labels = get_ori_grasp_label(grasppaths)
                for scale_factor in eval_scale:
                    thre_dis = dis_criterion * scale_factor
                    thre_rot = rot_criterion * scale_factor
                    r_g, r_d, r_r = calculate_6d_match(pred_grasp,
                                                       gg_ori_labels,
                                                       threshold_dis=thre_dis,
                                                       threshold_rot=thre_rot)

                    results[f'grasp_{scale_factor}'] += r_g
                    results[f'trans_{thre_dis}'] += r_d
                    results[f'rot_{thre_rot}'] += r_r

                # cal coverage rate
                cover_cnt = calculate_coverage(pred_grasp, gg_ori_labels)
                results['cover_cnt'] += cover_cnt
                results['label_cnt'] += len(gg_ori_labels)

            # tensorboard record
            results['loss'] += anchor_loss.item() + multi_cls_loss.item(
            ) + offset_loss.item()
            results['anchor_loss'] += anchor_loss.item()
            if epoch >= args.pre_epochs:
                results['multi_cls_loss'] += multi_cls_loss.item()
                results['offset_loss'] += offset_loss.item()
            for ln, l in anchor_losses.items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

    # center stat
    if total_center_num > 0:
        logging.info(
            f'valid center == {valid_center_num / total_center_num:.2f}')

    # loss stat
    batch_idx += 1
    results['loss'] /= batch_idx
    results['anchor_loss'] /= batch_idx
    results['multi_cls_loss'] /= batch_idx
    results['offset_loss'] /= batch_idx
    for ln, l in anchor_losses.items():
        results['losses'][ln] /= batch_idx
    return results
