import itertools
import logging
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import copy
from matplotlib import pyplot as plt
from PIL import Image
from torchsummary import summary
from tqdm import tqdm

from .dataset.evaluation import (anchor_output_process, calculate_6d_match,
                                calculate_coverage, calculate_iou_match,
                                collision_detect, detect_2d_grasp,
                                detect_6d_grasp_multi)
from .dataset.grasp import RectGraspGroup
from .dataset.graspnet_dataset import GraspnetPointDataset
from .dataset.pc_dataset_tools import (data_process, feature_fusion,
                                      get_center_group_label,
                                      get_ori_grasp_label)
from .dataset.utils import shift_anchors
from .models.anchornet import AnchorGraspNet, BNMomentumScheduler
from .models.localgraspnet import PointMultiGraspNet
from .models.losses import compute_anchor_loss, compute_multicls_loss
from .train_utils import *
from .edge_optimization.quantization import *
from .edge_optimization.tiling import *

dis_criterion = 0.05
rot_criterion = 0.25

def train(epoch, anchornet: nn.Module, localnet: nn.Module,
          train_data: torch.utils.data.DataLoader, optimizer: optim.AdamW,
          anchors: dict, args):
    """train one epoch.

    Args:
        epoch (int): epoch idx
        anchornet (nn.Module): anchornet (GHM)
        localnet (nn.Module): localnet (NMG)
        train_data (torch.utils.data.DataLoader): trian dataset
        optimizer (optim.AdamW): optimizer
        anchors (dict): local rotation anchors for gamma and beta
        args (args): args
    """
    results = {
        'loss': 0,
        'losses': {},
        'multi_cls_loss': 0,
        'offset_loss': 0,
        'anchor_loss': 0
    }
    valid_center_num, total_center_num = 0, 0

    optimizer.zero_grad()
    anchornet.train()
    localnet.train()

    if args.joint_trainning:
        train_data.dataset.unaug()
    else:
        if epoch >= args.pre_epochs:
            logging.info('Attention: freeze anchor net!')
            anchornet.eval()
            for para in anchornet.parameters():
                para.requires_grad_(False)
            train_data.dataset.unaug()
        else:
            # extra aug for 2d net
            logging.info('Extra augmentation for 2d network trainning!')
            train_data.dataset.setaug()

    # rot and zoom for trainning
    train_data.dataset.train()

    # log loss stat
    start = time()
    batch_idx = -1
    sum_local_loss = 0
    sum_offset_loss = 0
    sum_anchor_loss = 0
    sum_anchor_loss_d = {'loc_map_loss': 0, 'reg_loss': 0, 'cls_loss': 0}

    # for anchor shift
    cur_labels = torch.zeros((0, 8), dtype=torch.float32)

    data_start = time()
    data_time = 0
    for anchor_data, rgbs, depths, grasppaths in tqdm(train_data,
                                                      desc=f'Train_{epoch}',
                                                      ncols=80):
        if len(rgbs) < args.batch_size:
            continue
        data_time += time() - data_start
        batch_idx += 1

        # get scene points
        points, _, _ = train_data.dataset.helper.to_scene_points(
            rgbs.cuda(), depths.cuda(), include_rgb=False)
        # get xyz maps
        xyzs = train_data.dataset.helper.to_xyz_maps(depths.cuda())
        # get labels
        all_grasp_labels = []
        for grasppath in grasppaths:
            all_grasp_labels.append(np.load(grasppath))

        # train anchornet first
        x, y, _, _, _ = anchor_data
        x = x.cuda(non_blocking=True)
        target = [yy.cuda(non_blocking=True) for yy in y]
        outputs = anchornet(x)
        pred_2d = (outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
        perpoint_features = outputs[5] 

        # cal anchor loss
        anchor_lossd = compute_anchor_loss(pred_2d,
                                           target,
                                           loc_a=args.loc_a,
                                           reg_b=args.reg_b,
                                           cls_c=args.cls_c)
        anchor_losses = anchor_lossd['losses']
        anchor_loss = anchor_lossd['loss']

        # get loss stat
        if args.joint_trainning or epoch < args.pre_epochs:
            loss = anchor_loss
        else:
            loss = 0

        if epoch >= args.pre_epochs:
            # detect 2d grasp center
            loc_maps, theta_cls, theta_offset, depth_offset, width_offset = \
                    anchor_output_process(*pred_2d, sigma=args.sigma)

            # detect 2d grasp (x, y, theta)
            rect_ggs = []
            for i in range(args.batch_size):
                rect_gg = detect_2d_grasp(loc_maps[i],
                                          theta_cls[i],
                                          theta_offset[i],
                                          depth_offset[i],
                                          width_offset[i],
                                          ratio=args.ratio,
                                          anchor_k=args.anchor_k,
                                          anchor_w=args.anchor_w,
                                          anchor_z=args.anchor_z,
                                          mask_thre=0,
                                          center_num=args.center_num,
                                          grid_size=args.grid_size,
                                          grasp_nms=args.grid_size)
                rect_ggs.append(rect_gg)

            if len(rect_ggs) == 0:
                print('No 2d grasp found')
                continue

            # using 2d grasp to crop point cloud
            points_all = feature_fusion(points, perpoint_features, xyzs)

            # crop local pcs
            pc_group, valid_local_centers = data_process(
                points_all,
                depths.cuda(),
                rect_ggs,
                args.center_num,
                args.group_num, (args.input_w, args.input_h),
                is_training=False)

            # get 2d grasp info (not grasp itself) for trainning
            grasp_info = np.zeros((0, 3), dtype=np.float32)
            for i in range(args.batch_size):
                g_thetas = rect_ggs[i].thetas[None]
                g_ws = rect_ggs[i].widths[None]
                g_ds = rect_ggs[i].depths[None]
                cur_info = np.vstack([g_thetas, g_ws, g_ds])
                grasp_info = np.vstack([grasp_info, cur_info.T])
            grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32,
                                                         device='cuda')

            # check pc_group
            if pc_group.shape[0] == 0:
                print('No partial point clouds')
                continue

            # local net
            localnet_output = localnet([pc_group, grasp_info])
            pred_view = localnet_output[1]
            offset = localnet_output[2]

            # get nearest grasp labels
            gg_labels, total_labels = get_center_group_label(
                valid_local_centers, all_grasp_labels, args.local_grasp_num)

            # get center valid stats
            total_center_num += len(gg_labels)
            for gg in gg_labels:
                valid_center_num += len(gg) > 0

            # shift anchors only for first serveral epochs
            if epoch < args.shift_epoch:
                cur_labels = torch.cat([cur_labels, total_labels.cpu()], 0)
                if len(cur_labels) > 1e6:
                    shift_start = time()
                    old_gammas = anchors['gamma'].clone()
                    old_betas = anchors['beta'].clone()
                    anchors = shift_anchors(cur_labels, anchors)
                    # get shift error
                    error = (old_gammas - anchors['gamma']).abs().sum()
                    error += (old_betas - anchors['beta']).abs().sum()
                    logging.info(f'shift error == {error:.5f}')
                    logging.info(f'shift time == {time() - shift_start:.3f}')
                    cur_labels = torch.zeros((0, 8), dtype=torch.float32)
                    # stop when stable
                    # if error < 1e-2:
                    #     shift_epoch = 0

            # get loss
            multi_cls_loss, offset_loss = compute_multicls_loss(
                pred_view, offset, gg_labels, grasp_info, anchors, args)
            loss += multi_cls_loss + offset_loss

        # backward every step
        loss.backward()

        # step sum loss
        if batch_idx > 0 and batch_idx % args.step_cnt == 0:
            nn.utils.clip_grad.clip_grad_value_(anchornet.parameters(), 1)
            nn.utils.clip_grad.clip_grad_value_(localnet.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

        # get accumulation loss (for log_batch_cnt)
        sum_anchor_loss += anchor_loss
        if epoch >= args.pre_epochs:
            sum_local_loss += multi_cls_loss
            sum_offset_loss += offset_loss
        for key in anchor_losses:
            sum_anchor_loss_d[key] += anchor_losses[key]

        log_batch_cnt = 800 // args.batch_size
        if batch_idx > 0 and batch_idx % log_batch_cnt == 0:
            print('\n')
            logging.info(
                f'{log_batch_cnt} batches using time: {time() - start:.2f} s  data time: {data_time:.2f} s'
            )
            for para in optimizer.param_groups:
                cur_lr = para['lr']
                break
            logging.info(f'current lr: {cur_lr:.7f}')
            data_time = 0
            start = time()
            # print loss stat
            log_anchor_loss(epoch, batch_idx,
                            sum_anchor_loss + sum_local_loss + sum_offset_loss,
                            sum_anchor_loss, sum_anchor_loss_d, log_batch_cnt)
            if epoch >= args.pre_epochs:
                logging.info(
                    f'multi_cls_loss: {sum_local_loss / log_batch_cnt:.4f}')
                logging.info(
                    f'offset_loss: {sum_offset_loss / log_batch_cnt:.4f}')
                logging.info(
                    f'valid center == {valid_center_num / total_center_num:.2f}'
                )
            # reset loss stat
            valid_center_num, total_center_num = 0, 0
            sum_local_loss = 0
            sum_offset_loss = 0
            sum_anchor_loss = 0
            sum_anchor_loss_d = {
                'loc_map_loss': 0,
                'reg_loss': 0,
                'cls_loss': 0
            }

        # train result update
        results['loss'] += anchor_loss.item()
        if epoch >= args.pre_epochs:
            results['loss'] += multi_cls_loss.item() + offset_loss.item()
        results['anchor_loss'] += anchor_loss.item()
        for key, value in anchor_losses.items():
            if key not in results['losses']:
                results['losses'][key] = 0
            results['losses'][key] += value.item()
        if epoch >= args.pre_epochs:
            results['multi_cls_loss'] += multi_cls_loss.item()
            results['offset_loss'] += offset_loss.item()

        data_start = time()

    # loss stat
    batch_idx += 1
    results['loss'] /= batch_idx
    results['anchor_loss'] /= batch_idx
    for key in results['losses']:
        results['losses'][key] /= batch_idx
    if epoch >= args.pre_epochs:
        results['multi_cls_loss'] /= batch_idx
        results['offset_loss'] /= batch_idx
    return results

def training_loop(args, anchornet, localnet, start_epoch, end_epoch, anchors):
    # prepare for trainning
    tb, save_folder = prepare_torch_and_logger(args)

    logging.info('Loading Dataset...')
    sceneIds = list(range(args.scene_l, args.scene_r))
    Dataset = GraspnetPointDataset(args.all_points_num,
                                   args.dataset_path,
                                   args.scene_path,
                                   sceneIds,
                                   noise=args.noise,
                                   sigma=args.sigma,
                                   ratio=args.ratio,
                                   anchor_k=args.anchor_k,
                                   anchor_z=args.anchor_z,
                                   anchor_w=args.anchor_w,
                                   grasp_count=args.grasp_count,
                                   output_size=(args.input_w, args.input_h),
                                   random_rotate=False,
                                   random_zoom=False)


    train_data = torch.utils.data.DataLoader(Dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            shuffle=True,
                                            pin_memory=True)
    
    logging.info('Training size: {}'.format(len(Dataset)))

    
    # Load Dataset
    val_list = list(range(100, 101))
    Val_Dataset = GraspnetPointDataset(args.all_points_num,
                                       args.dataset_path,
                                       args.scene_path,
                                       val_list,
                                       noise=args.noise,
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

    logging.info('Validation size: {}'.format(len(Val_Dataset)))


    val_data = torch.utils.data.DataLoader(Val_Dataset,
                                           batch_size=1,
                                           pin_memory=True)

    # set optimizer
    params = itertools.chain(anchornet.parameters(), localnet.parameters())
    optimizer = get_optimizer(args, params)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

    # Decay Batchnorm momentum from 0.5 to 0.999
    # note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MAX = 0.001
    bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * 0.5**
                             (int(it / 2)), BN_MOMENTUM_MAX)
    bnm_scheduler = BNMomentumScheduler(anchornet,
                                        bn_lambda=bn_lbmd,
                                        last_epoch=-1)
    
    for epoch in range(start_epoch, end_epoch):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, anchornet, localnet, train_data,
                              optimizer, anchors, args)
        scheduler.step()
        bnm_scheduler.step()

        # Log training losses to tensorboard
        tb.add_scalar('train_loss/loss', train_results['loss'], epoch)
        tb.add_scalar('train_loss/anchor_loss', train_results['anchor_loss'],
                      epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)
        if epoch >= args.pre_epochs:
            tb.add_scalar('train_loss/multi_cls_loss',
                          train_results['multi_cls_loss'], epoch)
            tb.add_scalar('train_loss/offset_loss',
                          train_results['offset_loss'], epoch)

        # Run Validation
        logging.info('Validating...')
        val_results = validate(epoch, anchornet, localnet, val_data, anchors,
                               args)

        if epoch >= args.pre_epochs:
            log_match_result(val_results, dis_criterion, rot_criterion)

        log_and_save(args,
                     tb,
                     val_results,
                     epoch,
                     anchornet,
                     localnet,
                     optimizer,
                     anchors,
                     save_folder,
                     mode='graspnet')
        
    return tb, val_data, optimizer, save_folder

def run():
    args = parse_args()

    # load the network
    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    anchornet = AnchorGraspNet(ratio=args.ratio,
                               in_dim=input_channels,
                               anchor_k=args.anchor_k)
    localnet = PointMultiGraspNet(3, args.anchor_num**2)

    # load checkpoint
    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1).cuda()
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
    start_epoch = 0
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        if 'gamma' in ckpt and len(ckpt['gamma']) == args.anchor_num:
            anchors['gamma'] = ckpt['gamma']
            anchors['beta'] = ckpt['beta']
            logging.info('Using saved anchors')
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch']
        else:
            print("No Epoch value in the checkpoint, assuming all epochs are finished.")
            start_epoch = args.epochs


        anchornet.load_state_dict(ckpt['anchor'])
        ckpt['local'] = remap_checkpoint(ckpt['local'])
        localnet.load_state_dict(ckpt['local'])

    # get model architecture
    # print_model(args, input_channels, anchornet, save_folder)

    # multi gpu
    anchornet = nn.parallel.DataParallel(anchornet).cuda()
    localnet = nn.parallel.DataParallel(localnet).cuda()
    logging.info('Done')

    epochs = args.epochs
    if start_epoch != 0:
        epochs = 0

    tb, val_data, optimizer, save_folder = training_loop(args, anchornet, localnet, 0, epochs, anchors)
        
    # Quantization

    #sensitivity_analysis(args, val_data, anchors, tb, optimizer, save_folder)
    print("Post Training Quantization...")
    localnet = localnet.cuda()
    (anchornet, localnet) = Q_callibration_and_training(anchornet, localnet, val_data, anchors, args)

    #anchornet = DividedAnchorNet(anchornet)

    logging.info('Post-Quantization Validation...')
    val_results = validate(args.epochs + 1, anchornet, localnet, val_data, anchors, args, quantized_mode=True)
    log_match_result(val_results, dis_criterion, rot_criterion)
    log_and_save(args,
                    tb,
                    val_results,
                    args.epochs + 1,
                    anchornet,
                    localnet,
                    optimizer,
                    anchors,
                    save_folder,
                    mode='graspnet')




if __name__ == '__main__':
    run()
