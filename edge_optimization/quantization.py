import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy
from time import time
from torch.ao.quantization.observer import MinMaxObserver 
from torch.ao.quantization.qconfig import QConfig

# These are needed for generating checkpoints
# from ..dataset.graspnet_dataset import GraspnetPointDataset
# from ..train_graspnet import training_loop
# from ..train_utils import *

from ..models.anchornet import AnchorGraspNet
from ..models.localgraspnet import PointMultiGraspNet

import re
import functools



# class QuantStub(nn.Module):
#     r"""Quantize stub module, before calibration, this is same as an observer,
#     it will be swapped as `nnq.Quantize` in `convert`.

#     Args:
#         qconfig: quantization configuration for the tensor,
#             if qconfig is not provided, we will get qconfig from parent modules
#     """

#     def __init__(self, qconfig: QConfig | None = None):
#         super().__init__()
#         if qconfig:
#             self.qconfig = qconfig

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x


# class DeQuantStub(nn.Module):
#     r"""Dequantize stub module, before calibration, this is same as identity,
#     this will be swapped as `nnq.DeQuantize` in `convert`.

#     Args:
#         qconfig: quantization configuration for the tensor,
#             if qconfig is not provided, we will get qconfig from parent modules
#     """

#     def __init__(self, qconfig: Any | None = None):
#         super().__init__()
#         if qconfig:
#             self.qconfig = qconfig

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x

class GrabbingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.grab = None
    
    def forward(self, x):
        self.grab = x
        return x

class QuantStubbed(nn.Module):
    def __init__(self, model, input_count, output_count):
        super().__init__()

        self.input_count = input_count
        self.output_count = output_count
        self.model = model

        self.quant_stubs = nn.ModuleList([
            torch.quantization.QuantStub() for _ in range(input_count)
        ])

        self.dequant_stubs = nn.ModuleList([
            torch.quantization.DeQuantStub() for _ in range(output_count)
        ])

    def forward(self, x):
        if len(x) == 1:
            x = [x]

        for i in range(self.input_count):
            x[i] = self.quant_stubs[i](x[i])
        
        if len(x) == 1:
            x = self.model(x[0])
        else:
            x = self.model(x)

        for i in range(self.output_count):
            x[i] = self.dequant_stubs[i](x[i])
        return x

def is_quantized(dictionary):
    quantized = False
    for k, v in dictionary.items():
        if (type(v) is torch.Tensor and v.is_quantized):
            quantized = True
    
    return quantized

def rename(checkpoint, old, new):
    if old in checkpoint:
        checkpoint[new] = checkpoint.pop(old)


def remap_anchornet(checkpoint):
    new_sd = {}

    for k, v in checkpoint.items():
        if "model.trconv." in k:
            parts = k.split(".")

            for branch in range(4):
                new_parts = parts.copy()
                new_parts = new_parts[:2] + [str(branch)] + new_parts[2:]
                new_k = ".".join(new_parts)

                new_sd[new_k] = v.clone() if hasattr(v, "clone") else v
        elif "trconv." in k:
            parts = k.split(".")

            for branch in range(4):
                new_parts = parts.copy()
                new_parts = new_parts[:1] + [str(branch)] + new_parts[1:]
                new_k = ".".join(new_parts)

                new_sd[new_k] = v.clone() if hasattr(v, "clone") else v
        else:
            new_sd[k] = v

    return new_sd

# ChatGPT generated for convenience 
# I had to change the structure of localnet, so that layers are fusable
# This broke the checpoint, hence this remapping:
def remap_checkpoint(checkpoint):
    for i in [1, 2, 3]:
        rename(checkpoint, f"pointnet.stn.conv{i}.weight", f"pointnet.stn.seq{i}.0.weight")
        rename(checkpoint, f"pointnet.stn.conv{i}.bias",   f"pointnet.stn.seq{i}.0.bias")

        rename(checkpoint, f"pointnet.stn.bn{i}.weight", f"pointnet.stn.seq{i}.1.weight")
        rename(checkpoint, f"pointnet.stn.bn{i}.bias",   f"pointnet.stn.seq{i}.1.bias")
        rename(checkpoint, f"pointnet.stn.bn{i}.running_mean", f"pointnet.stn.seq{i}.1.running_mean")
        rename(checkpoint, f"pointnet.stn.bn{i}.running_var",  f"pointnet.stn.seq{i}.1.running_var")
        rename(checkpoint, f"pointnet.stn.bn{i}.num_batches_tracked", f"pointnet.stn.seq{i}.1.num_batches_tracked")

    rename(checkpoint, "pointnet.stn.fc1.weight", "pointnet.stn.seq4.0.weight")
    rename(checkpoint, "pointnet.stn.fc1.bias",   "pointnet.stn.seq4.0.bias")
    rename(checkpoint, "pointnet.stn.bn4.weight", "pointnet.stn.seq4.1.weight")
    rename(checkpoint, "pointnet.stn.bn4.bias",   "pointnet.stn.seq4.1.bias")
    rename(checkpoint, "pointnet.stn.bn4.running_mean", "pointnet.stn.seq4.1.running_mean")
    rename(checkpoint, "pointnet.stn.bn4.running_var",  "pointnet.stn.seq4.1.running_var")
    rename(checkpoint, "pointnet.stn.bn4.num_batches_tracked", "pointnet.stn.seq4.1.num_batches_tracked")

    rename(checkpoint, "pointnet.stn.fc2.weight", "pointnet.stn.seq5.0.weight")
    rename(checkpoint, "pointnet.stn.fc2.bias",   "pointnet.stn.seq5.0.bias")
    rename(checkpoint, "pointnet.stn.bn5.weight", "pointnet.stn.seq5.1.weight")
    rename(checkpoint, "pointnet.stn.bn5.bias",   "pointnet.stn.seq5.1.bias")
    rename(checkpoint, "pointnet.stn.bn5.running_mean", "pointnet.stn.seq5.1.running_mean")
    rename(checkpoint, "pointnet.stn.bn5.running_var",  "pointnet.stn.seq5.1.running_var")
    rename(checkpoint, "pointnet.stn.bn5.num_batches_tracked", "pointnet.stn.seq5.1.num_batches_tracked")

    for i in [1, 2, 3]:
        rename(checkpoint, f"pointnet.conv{i}.weight", f"pointnet.seq{i}.0.weight")
        rename(checkpoint, f"pointnet.conv{i}.bias",   f"pointnet.seq{i}.0.bias")

        rename(checkpoint, f"pointnet.bn{i}.weight", f"pointnet.seq{i}.1.weight")
        rename(checkpoint, f"pointnet.bn{i}.bias",   f"pointnet.seq{i}.1.bias")
        rename(checkpoint, f"pointnet.bn{i}.running_mean", f"pointnet.seq{i}.1.running_mean")
        rename(checkpoint, f"pointnet.bn{i}.running_var",  f"pointnet.seq{i}.1.running_var")
        rename(checkpoint, f"pointnet.bn{i}.num_batches_tracked", f"pointnet.seq{i}.1.num_batches_tracked")

    return checkpoint


def fuse_all(model):
    for name, module in model.named_children():
        # Recursively fuse children
        fuse_all(module)

        # Try common patterns
        if isinstance(module, torch.nn.Sequential):
            modules = list(module._modules.keys())

            i = 0
            while i < len(modules):
                for j in range(len(modules), i + 1, -1):
                    fused = True
                    fusion_str = str(module[i:j])

                    # supa dirty fusion
                    try:
                        torch.ao.quantization.fuse_modules(module, [modules[i:j]], inplace=True)
                    except AssertionError as error:
                        fused = False
                    
                    if fused:
                        #print("Fused: ", fusion_str)
                        i = j - 1
                        break
                
                i += 1


def load_qconfig(model, q_type, q_scales):   
    torch.backends.quantized.engine = "qnnpack"
    if isinstance(model, AnchorGraspNet):
        if q_scales == "Symmetric":
            activation_scales = torch.per_tensor_symmetric
            activation_dtype = torch.qint8
        elif q_scales == "Affine":
            activation_scales = torch.per_tensor_affine
            activation_dtype = torch.quint8
        else:
            raise ValueError('Error! Anchornet quantization scales argument is incorrect!')
        
        if q_type == "8-bit":
            return torch.ao.quantization.QConfig(
                activation = lambda **kwargs: torch.ao.quantization.observer.HistogramObserver(reduce_range=False, dtype=activation_dtype, qscheme=activation_scales), 
                weight = lambda **kwargs: torch.ao.quantization.observer.MinMaxObserver(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)) 
        elif q_type == "4-bit":
            if q_scales == "Symmetric":
                raise ValueError('Error! Unsupported anchronet quantization argument combination!')

            return torch.ao.quantization.QConfig(
                activation = lambda **kwargs: torch.ao.quantization.observer.HistogramObserver(reduce_range=False, dtype=torch.quint8, qscheme=torch.per_tensor_affine, quant_min=0, quant_max=15), 
                weight = lambda **kwargs: torch.ao.quantization.observer.MinMaxObserver(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8, quant_min = -8, quant_max = 7))
        elif q_type == "QAT":
            return torch.quantization.get_default_qat_qconfig("qnnpack")
        else:
            raise ValueError('Error! Incorrect anchornet quantization type argument!')
    elif isinstance(model, PointMultiGraspNet):
        activation_scheme = torch.per_tensor_affine
        weight_scheme = torch.per_tensor_symmetric

        if q_type == "Normal" or q_type == "Optimized":
            return torch.ao.quantization.QConfig(
                activation = lambda **kwargs: torch.ao.quantization.observer.HistogramObserver(reduce_range=False, dtype=torch.quint8, qscheme=activation_scheme), 
                weight = lambda **kwargs: torch.ao.quantization.observer.MinMaxObserver(qscheme=weight_scheme, dtype=torch.qint8)) 
        elif q_type == "QAT":
            return torch.ao.quantization.QConfig(
                activation=functools.partial(
                    torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize,
                    observer=torch.ao.quantization.observer.MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    reduce_range=False,
                    qscheme=activation_scheme
                ),
                weight=functools.partial(
                    torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize,
                    observer=torch.ao.quantization.observer.MovingAverageMinMaxObserver,
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=weight_scheme,
                ),
            )
        else:
            raise ValueError('Error! Incorrect localnet quantization type argument!')
    else:
        raise Exception("Unexpeccted model passed for quantization!")
   
def get_module_and_its_input(model, module_name):
    localnet_copy = copy.deepcopy(model)
    (parent, child, child_name) = get_parent_child(localnet_copy, "model." + module_name)
    grabber = GrabbingModule()
    child_copy = copy.deepcopy(child)
    setattr(parent, child_name, torch.nn.Sequential(grabber, child))

    localnet_copy([torch.zeros([48, 512, 35], dtype=torch.float32), torch.zeros([48, 3], dtype=torch.float32)])
    #zeroscale = grabber.grab.zero_scale
    return (grabber.grab, child_copy)

def validate_model(args, anchornet, localnet, val_data, anchors, tb, optimizer, save_folder):
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

def sensitivity_analysis(args, val_data, anchors, tb, optimizer, save_folder):
    original_checkpoint = torch.load('./HGGD_realsense_checkpoint')
    quantized_checkpoint = torch.load('./both_quantized_fused_qnnpack')

    _, original_localnet = load_models(original_checkpoint, args)
    anchornet, quantized_localnet = load_models(quantized_checkpoint, args)
    original_localnet = original_localnet.cpu()
    fuse_all(original_localnet)

    ignored_types = [torch.nn.Identity, torch.nn.ReLU, torch.ao.nn.quantized.modules.linear.LinearPackedParams, torch.ao.nn.quantized.Quantize, torch.ao.nn.quantized.DeQuantize, torch.nn.modules.container.ModuleList]
    for name, module in quantized_localnet.named_modules(): 
        name = name[len("model."):]
        if name != "" and not type(module) in ignored_types:
            logging.info(f'quantizing only: {name}')
            (module_input, module) = get_module_and_its_input(quantized_localnet, name)

            localnet_copy = copy.deepcopy(original_localnet)
            (parent, child, child_name) = get_parent_child(localnet_copy, name)
            setattr(parent, child_name, torch.nn.Sequential(
                torch.ao.nn.quantized.Quantize(scale = module_input.q_scale(), zero_point = module_input.q_zero_point(), dtype=torch.quint8), 
                module, 
                torch.ao.nn.quantized.DeQuantize()))

            validate_model(args, anchornet, localnet_copy, val_data, anchors, tb, optimizer, save_folder)

    #     print("Post Training Quantization...")
    # for name, module in localnet.module.named_modules(): 
    #     localnet_copy = copy.deepcopy(localnet)
    #     anchornet_copy = copy.deepcopy(anchornet)

    #     if name != "":
    #         logging.info(f'Qauntizing only: {name}')
    #         (anchornet_copy, localnet_copy) = PTQ(anchornet_copy, localnet_copy, val_data, anchors, args, module_to_wrap = name)

    #         # check_point = torch.load(args.checkpoint_path)
    #         # reduced_mode = 16
    #         # inference(anchornet, localnet, check_point, reduced_mode, True)
    #         # evaluate(reduced_mode=reduced_mode)
    #         logging.info('Post-Quantization Validation...')
    #         val_results = validate(args.epochs + 1, anchornet_copy, localnet_copy, val_data, anchors, args, quantized_mode=True)
    #         log_match_result(val_results, dis_criterion, rot_criterion)
    #         log_and_save(args,
    #                         tb,
    #                         val_results,
    #                         args.epochs + 1,
    #                         anchornet_copy,
    #                         localnet_copy,
    #                         optimizer,
    #                         anchors,
    #                         save_folder,
    #                         mode='graspnet')


def load_models(check_point, args):
    # Init the model
    anchornet = AnchorGraspNet(in_dim=4,
                               ratio=args.ratio,
                               anchor_k=args.anchor_k)
    localnet = PointMultiGraspNet(info_size=3, k_cls=args.anchor_num**2)

    # multi gpu
    # anchornet = anchornet.cuda()
    # localnet = localnet.cuda()
    
    if args.q_anchornet_type == "8-bit" or args.q_anchornet_type == "4-bit" or args.q_anchornet_type == "QAT":
        anchornet = prepare_model(anchornet, args.q_anchornet_type, args.q_anchornet_scales)
        anchornet = convert_model(anchornet)
    elif args.q_anchornet_type != "None":
        raise ValueError('Error! Incorrect anchornet quantization type argument!')

    check_point['anchor'] = remap_anchornet(check_point['anchor'])
    anchornet.load_state_dict(check_point['anchor'])

    if args.q_localnet_type == "Normal" or args.q_localnet_type == "Optimized" or args.q_localnet_type == "QAT":
        localnet = prepare_model(localnet, args.q_localnet_type, None)
        localnet = convert_model(localnet)
    elif args.q_localnet_type != "None":
        raise ValueError('Error! Incorrect localnet quantization type argument!')

    check_point['local'] = remap_checkpoint(check_point['local'])
    localnet.load_state_dict(check_point['local'])

    # network eval mode
    anchornet.eval()
    localnet.eval()

    return anchornet, localnet

def load_callibration_data(args):
    val_list = list(range(0, 100))
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
    val_data = torch.utils.data.DataLoader(Val_Dataset,
                                           batch_size=1,
                                           pin_memory=True,
                                           shuffle=True)
    return val_data

# Adapted from (https://pytorch.org/blog/quantization-in-practice/)
# Eager mode is used - a more manual but more customizable approach
# Static quantization is used - it's done once and baked in, as opposed to doing it at runtime
def Q_callibration_and_training(anchornet :nn.Module, localnet :nn.Module, val_data, anchors, args):

    if args.q_anchornet_type == 'None':
        anchornet_quant = anchornet
    else:
        anchornet_quant = prepare_model(anchornet, args.q_anchornet_type, args.q_anchornet_scales)    

    if args.q_localnet_type == 'None':
        localnet_quant = localnet
    else:
        localnet_quant = prepare_model(localnet, args.q_localnet_type, None)    

    # Callibrate the quantization by feeding it a bunch of real-world data
    if (args.q_anchornet_type == "8-bit" or 
        args.q_anchornet_type == "4-bit" or 
        args.q_localnet_type == "Normal" or 
        args.q_localnet_type == "Optimized"):
        print("Calibrating...")
        validate(args.epochs, anchornet_quant, localnet_quant, load_callibration_data(args), anchors, args, end_early=args.callibration_samples)
    elif (args.q_anchornet_type == "QAT" or 
        args.q_localnet_type == "QAT"):
        print("QAT training...")
        args.lr /= 100
        training_loop(args, anchornet_quant, localnet_quant, args.epochs, args.epochs + args.qat_epochs, anchors)
        
    anchornet_quant.eval()
    localnet_quant.eval()

    if args.q_anchornet_type != 'None':
        anchornet_quant = convert_model(anchornet_quant)
    if args.q_localnet_type != 'None':
        localnet_quant = convert_model(localnet_quant)

    return (anchornet_quant, localnet_quant)    

def convert_model(model :nn.Module):
    quant_model = model.cpu()
    torch.quantization.convert(quant_model, inplace=True)
    quant_model.eval()
    return model

# Get the parent and the name
def get_parent_child(module, module_to_find, prefix = ""):
    for name, child in module.named_children():
        name_here = ""
        if prefix == "":
            name_here = name
        else:
            name_here = prefix + "." + name

        if name_here == module_to_find:
            #print("return", name_here)
            return (module, child, name)
        
        val = get_parent_child(child, module_to_find, name_here)
        if val != None:
            #print(val)
            return val

def prepare_model(model :nn.Module, q_type, q_scales):
    quant_model = copy.deepcopy(model)
    quant_model.eval()

    # If the model is wrapped in DataParallel, unwrap it
    if isinstance(quant_model, torch.nn.DataParallel):
        quant_model = quant_model.module

    # Fusing Conv-ReLU pairs (and similar), to improve performance
    fuse_all(quant_model)

    qconfig = load_qconfig(quant_model, q_type, q_scales)

    # Insert stubs
    if isinstance(quant_model, AnchorGraspNet):
        quant_model = QuantStubbed(quant_model, 1, 6)
    elif isinstance(quant_model, PointMultiGraspNet):
        if q_type == "Optimized" or q_type == "QAT":
            quant_model.add_quant_stubs()
        quant_model = QuantStubbed(quant_model, 2, 3)
    else:
        raise Exception("Unexpeccted model passed for quantization!")
    
    quant_model.qconfig = qconfig
    # quant_model = quant_model.cuda()

    if q_type == "QAT":
        quant_model.train()
        torch.quantization.prepare_qat(quant_model, inplace=True)
    else:
        torch.quantization.prepare(quant_model, inplace=True)

    return quant_model