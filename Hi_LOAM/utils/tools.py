import copy
from typing import List
import sys
import os
import multiprocessing
import getpass
import time
from pathlib import Path
from datetime import datetime
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.autograd import grad
import torch
import roma
import torch.nn as nn
import numpy as np
import wandb
import json
import open3d as o3d
import math
from collections import defaultdict

from utils.config import SHINEConfig


# setup this run
def setup_experiment(config: SHINEConfig): 

    os.environ["NUMEXPR_MAX_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp
    run_name = config.name + "_" + ts  # modified to a name that is easier to index
        
    run_path = os.path.join(config.output_root, run_name)
    access = 0o755
    os.makedirs(run_path, access, exist_ok=True)
    assert os.access(run_path, os.W_OK)
    print(f"Start {run_path}")

    mesh_path = os.path.join(run_path, "mesh")
    map_path = os.path.join(run_path, "map")
    model_path = os.path.join(run_path, "model")
    os.makedirs(mesh_path, access, exist_ok=True)
    os.makedirs(map_path, access, exist_ok=True)
    os.makedirs(model_path, access, exist_ok=True)
    
    if config.wandb_vis_on:
        # set up wandb
        setup_wandb()
        wandb.init(project="SHINEMapping", config=vars(config), dir=run_path) # your own worksapce
        wandb.run.name = run_name         
    
    # set the random seed (for deterministic experiment results)
    o3d.utility.random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed) 

    return run_path


def setup_optimizer(config: SHINEConfig, octree_feat, mlp_geo_param, mlp_sem_param, sigma_size) -> Optimizer:
    lr_cur = config.lr
    opt_setting = []
    # weight_decay is for L2 regularization, only applied to MLP
    if mlp_geo_param is not None:
        mlp_geo_param_opt_dict = {'params': mlp_geo_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_geo_param_opt_dict)
    if config.semantic_on and mlp_sem_param is not None:
        mlp_sem_param_opt_dict = {'params': mlp_sem_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_sem_param_opt_dict)
    # feature octree
    for i in range(config.tree_level_feat):
        # try to also add L2 regularization on the feature octree (results not quite good)
        feat_opt_dict = {'params': octree_feat[config.tree_level_feat-i-1], 'lr': lr_cur} 
        lr_cur *= config.lr_level_reduce_ratio
        opt_setting.append(feat_opt_dict)
    # make sigma also learnable for differentiable rendering (but not for our method)
    if config.ray_loss:
        sigma_opt_dict = {'params': sigma_size, 'lr': config.lr}
        opt_setting.append(sigma_opt_dict)
    
    if config.opt_adam:
        opt = optim.Adam(opt_setting, betas=(0.9,0.99), eps = config.adam_eps) 
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)
    
    return opt


def setup_optimizer_traj(config: SHINEConfig, octree_feat, mlp_geo_param, mlp_sem_param, mlp_traj_param, sigma_size) -> Optimizer:
    lr_cur = config.lr
    opt_setting = []
    # weight_decay is for L2 regularization, only applied to MLP
    if mlp_geo_param is not None:
        mlp_geo_param_opt_dict = {'params': mlp_geo_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_geo_param_opt_dict)
    if config.semantic_on and mlp_sem_param is not None:
        mlp_sem_param_opt_dict = {'params': mlp_sem_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_sem_param_opt_dict)
    if mlp_traj_param is not None:
        mlp_traj_param_opt_dict = {'params': mlp_traj_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_traj_param_opt_dict)

    # feature octree
    for i in range(config.tree_level_feat):
        # try to also add L2 regularization on the feature octree (results not quite good)
        feat_opt_dict = {'params': octree_feat[config.tree_level_feat-i-1], 'lr': lr_cur} 
        lr_cur *= config.lr_level_reduce_ratio
        opt_setting.append(feat_opt_dict)
    # make sigma also learnable for differentiable rendering (but not for our method)
    if config.ray_loss:
        sigma_opt_dict = {'params': sigma_size, 'lr': config.lr}
        opt_setting.append(sigma_opt_dict)
    
    if config.opt_adam:
        opt = optim.Adam(opt_setting, betas=(0.9,0.99), eps = config.adam_eps) 
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)
    
    return opt
    

# set up weight and bias
def setup_wandb():
    print("Weight & Bias logging option is on. Disable it by setting  wandb_vis_on: False  in the config file.")
    username = getpass.getuser()
    # print(username)
    wandb_key_path = username + "_wandb.key"
    if not os.path.exists(wandb_key_path):
        wandb_key = input(
            "[You need to firstly setup and login wandb] Please enter your wandb key (https://wandb.ai/authorize):"
        )
        with open(wandb_key_path, "w") as fh:
            fh.write(wandb_key)
    else:
        print("wandb key already set")
    os.system('export WANDB_API_KEY=$(cat "' + wandb_key_path + '")')

def step_lr_decay(
    optimizer: Optimizer,
    learning_rate: float,
    iteration_number: int,
    steps: List,
    reduce: float = 1.0):

    if reduce > 1.0 or reduce <= 0.0:
        sys.exit(
            "The decay reta should be between 0 and 1."
        )

    if iteration_number in steps:
        steps.remove(iteration_number)
        learning_rate *= reduce
        print("Reduce base learning rate to {}".format(learning_rate))

        for param in optimizer.param_groups:
            param["lr"] *= reduce

    return learning_rate


def num_model_weights(model: nn.Module) -> int:
    num_weights = int(
        sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, model.parameters())
            ]
        )
    )
    return num_weights


def print_model_summary(model: nn.Module):
    for child in model.children():
        print(child)


def get_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return points_grad


def freeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


def unfreeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True


def save_checkpoint(
    feature_octree, geo_decoder, sem_decoder, optimizer, run_path, checkpoint_name, iters
):
    torch.save(
        {
            "iters": iters,
            "feature_octree": feature_octree, # save the whole NN module (the hierachical features and the indexing structure)
            "geo_decoder": geo_decoder.state_dict(),
            "sem_decoder": sem_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(run_path, f"{checkpoint_name}.pth"),
    )
    print(f"save the model to {run_path}/{checkpoint_name}.pth")


def save_decoder(geo_decoder, sem_decoder, run_path, checkpoint_name):
    torch.save({"geo_decoder": geo_decoder.state_dict(), 
                "sem_decoder": sem_decoder.state_dict()},
        os.path.join(run_path, f"{checkpoint_name}_decoders.pth"),
    )

def save_geo_decoder(geo_decoder, run_path, checkpoint_name):
    torch.save({"geo_decoder": geo_decoder.state_dict()},
        os.path.join(run_path, f"{checkpoint_name}_geo_decoder.pth"),
    )

def save_sem_decoder(sem_decoder, run_path, checkpoint_name):
    torch.save({"sem_decoder": sem_decoder.state_dict()},
        os.path.join(run_path, f"{checkpoint_name}_sem_decoder.pth"),
    )

def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.
    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.
    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)
def intrinsic_correct(points, correct_deg=0.195):

    # # This function only applies for the KITTI dataset, and should NOT be used by any other dataset,
    # # the original idea and part of the implementation is taking from CT-ICP(Although IMLS-SLAM
    # # Originally introduced the calibration factor)
    if correct_deg == 0.:
        return points

    dist = np.linalg.norm(points[:,:3], axis=1)
    kitti_var_vertical_ang = correct_deg / 180. * math.pi
    v_ang = np.arcsin(points[:, 2] / dist)
    v_ang_c = v_ang + kitti_var_vertical_ang
    hor_scale = np.cos(v_ang_c) / np.cos(v_ang)
    #a= (dist * torch.cos(v_ang_c))**2
    points[:, 0] *= hor_scale
    points[:, 1] *= hor_scale
    points[:, 2] = dist * np.sin(v_ang_c)
    #b=points[0,0]**2+points[0,1]**2

    return points
def deskewing(points: torch.tensor, ts: torch.tensor, pose: torch.tensor, ts_mid_pose=0.5):
    if ts is None:
        return points  # no deskewing

    # pose as T_last<-cur
    # ts is from 0 to 1 as the ratio
    ts = ts.squeeze(-1)

    # Normalize the tensor to the range [0, 1]
    # NOTE: you need to figure out the begin and end of a frame because
    # sometimes there's only partial measurements, some part are blocked by some occlussions
    min_ts = torch.min(ts)
    max_ts = torch.max(ts)
    ts = (ts - min_ts) / (max_ts - min_ts)

    ts -= ts_mid_pose

    # print(ts)

    rotmat_slerp = roma.rotmat_slerp(torch.eye(3).to(points), pose[:3, :3].to(points), ts)

    tran_lerp = ts[:, None] * pose[:3, 3].to(points)

    points_deskewd = points
    points_deskewd[:, :3] = (rotmat_slerp @ points[:, :3].unsqueeze(-1)).squeeze(-1) + tran_lerp

    return points_deskewd


def voxel_down_sample_torch(frame_pc, voxel_size):
    """
        voxel based downsampling. Returns the indices of the points which are closest to the voxel centers.
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`

    Reference: Louis Wiesmann
    """
    points = np.asarray(frame_pc.points, dtype=np.float32)
    normal=np.asarray(frame_pc.normals)


    #points=torch.from_numpy(points_original)

    _quantization = 1000  # if change to 1, then it would be random sample

    offset = np.floor(points.min(axis=0) / voxel_size).astype(np.int64)
    grid = np.floor(points / voxel_size)
    center = (grid + 0.5) * voxel_size
    dist = np.sqrt(np.sum((points - center) ** 2,axis=1))
    dist = dist / dist.max() * (_quantization - 1)  # for speed up

    grid = grid.astype(np.int64) - offset
    v_size = np.ceil(grid.max())
    #v_size=np
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = np.unique(grid_idx, return_inverse=True)
    # a=unique[inverse[0]]
    idx_d = np.arange(inverse.size, dtype=inverse.dtype)
    #a=idx_d.max()
    offset = 10 ** len(str(idx_d.max()))

    idx_d = idx_d + dist.astype(np.int64) * offset
    idx = np.arange(unique.shape[0],dtype=inverse.dtype)
    index=inverse
    src=idx_d
    dict_data=defaultdict(lambda:[float('inf')])
    for idx_h,src_val in zip(index,src):
        if src_val<dict_data[idx_h][0]:
            dict_data[idx_h]=[src_val]

    for key,value in dict_data.items():
        idx[key]=value[0]
    #a=np.min(idx)
    idx = idx % offset
    #a = np.min(idx)
    points_keep=points[idx]
    normal_keep=normal[idx]
    frame_pc.points=o3d.utility.Vector3dVector(points_keep)
    frame_pc.normals=o3d.utility.Vector3dVector(normal_keep)
    #o3d.visualization.draw_geometries([frame_pc],window_name="aa",point_show_normal=True)




    return frame_pc,idx