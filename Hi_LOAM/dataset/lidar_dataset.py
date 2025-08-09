import os
import sys
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import copy
import torch
from torch.utils.data import Dataset
import open3d as o3d
from natsort import natsorted
from utils.tools import *

from utils.config import SHINEConfig
from utils.tools import get_time
from utils.pose import *
from utils.data_sampler import dataSampler
from utils.semantic_kitti_utils import *
from model.feature_octree import FeatureOctree
from utils.tools import voxel_down_sample_torch


# better to write a new dataloader for RGB-D inputs, not always converting them to KITTI Lidar format

class LiDARDataset(Dataset):
    def __init__(self, config: SHINEConfig, octree: FeatureOctree = None) -> None:

        super().__init__()

        self.config = config
        self.dtype = config.dtype
        torch.set_default_dtype(self.dtype)
        self.device = config.device

        self.calib = {}
        if config.calib_path != '':
            self.calib = read_calib_file(config.calib_path)
        else:
            self.calib['Tr'] = np.eye(4)
        if config.pose_path.endswith('txt'):
            self.poses_w = read_poses_file(config.pose_path, self.calib)
            self.ini_pose = self.poses_w[0]
        elif config.pose_path.endswith('csv'):
            self.poses_w = csv_odom_to_transforms(config.pose_path)
            self.ini_pose = self.poses_w[0]
        else:
            self.ini_pose = np.eye(4)
            #sys.exit(
            #    "Wrong pose file format. Please use either *.txt (KITTI format) or *.csv (xyz+quat format)"
            #)
        #self.ini_pose=self.poses_w[0]



        # point cloud files
        self.pc_filenames = natsorted(os.listdir(config.pc_path)) # sort files as 1, 2,… 9, 10 not 1, 10, 100 with natsort
        self.total_pc_count = len(self.pc_filenames)


        # feature octree
        self.octree = octree

        self.last_relative_tran = np.eye(4)

        # initialize the data sampler
        self.sampler = dataSampler(config)
        self.ray_sample_count = config.surface_sample_n + config.free_sample_n

        # merged downsampled point cloud
        self.map_down_pc = o3d.geometry.PointCloud()
        # map bounding box in the world coordinate system
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()
        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()

        # get the pose in the reference frame
        self.used_pc_count = 0
        begin_flag = False
        self.begin_pose_inv = np.eye(4)
        for frame_id in range(self.total_pc_count):
            if (
                frame_id < config.begin_frame
                or frame_id > config.end_frame
                or frame_id % config.every_frame != 0
            ):
                continue

            self.used_pc_count += 1
        # or we directly use the world frame as reference

        # to cope with the gpu memory issue (use cpu memory for the data pool, a bit slower for moving between cpu and gpu)
        if self.used_pc_count > config.pc_count_gpu_limit and not config.continual_learning_reg and not config.window_replay_on:
            self.pool_device = "cpu"
            self.to_cpu = True
            self.sampler.dev = "cpu"
            print("too many scans, use cpu memory")
        else:
            self.pool_device = config.device
            self.to_cpu = False

        # data pool
        self.coord_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.sdf_label_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.normal_label_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.color_label_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.sem_label_pool = torch.empty((0), device=self.pool_device, dtype=torch.long)
        self.weight_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.sample_depth_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.ray_depth_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.origin_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.time_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)

    def process_frame(self,frame_id,frame_pc,pose,cur_point_ts,last_tran, incremental_on=False):

        pc_radius = self.config.pc_radius
        min_z = self.config.min_z
        max_z = self.config.max_z
        normal_radius_m = self.config.normal_radius_m
        normal_max_nn = self.config.normal_max_nn
        rand_down_r = self.config.rand_down_r
        vox_down_m = self.config.vox_down_m
        sor_nn = self.config.sor_nn
        sor_std = self.config.sor_std

        self.cur_pose_ref = pose
        voxel_size = self.config.mapping_vox_down


        frame_pc,idx = voxel_down_sample_torch(frame_pc, voxel_size)

        frame_pc_tensor = torch.tensor(np.asarray(frame_pc.points, dtype=np.float32), device=self.device)
        # frame_pc_tensor = torch.tensor(np.asarray(frame_pc.points, dtype=np.float32), device=self.device)
        frame_normal = torch.from_numpy(np.asarray(frame_pc.normals, dtype=np.float32)).to(self.device)

        if cur_point_ts is not None:
            cur_ts = cur_point_ts.clone()
            cur_source_ts = cur_ts[idx]
        else:
            cur_source_ts = None


        if self.config.deskew :

            frame_pc_tensor = deskewing(frame_pc_tensor, cur_source_ts,
                                            torch.tensor(last_tran, device=self.pool_device,
                                                         dtype=self.dtype))
        frame_pc.points = o3d.utility.Vector3dVector(frame_pc_tensor.clone().detach().cpu().numpy())
        frame_pc.normals = o3d.utility.Vector3dVector(frame_normal.clone().detach().cpu().numpy())

        # voxel downsampling
        # frame_pc = frame_pc.voxel_down_sample(voxel_size=vox_down_m)

        # apply filter (optional)
        if self.config.filter_noise:
            frame_pc = frame_pc.remove_statistical_outlier(
                sor_nn, sor_std, print_progress=False
            )[0]

        # load the label from the color channel of frame_pc
        if self.config.semantic_on:
            frame_sem_label = np.asarray(frame_pc.colors)[:,0]*255.0 # from [0-1] tp [0-255]
            frame_sem_label = np.round(frame_sem_label, 0) # to integer value
            sem_label_list = list(frame_sem_label)
            frame_sem_rgb = [sem_kitti_color_map[sem_label] for sem_label in sem_label_list]
            frame_sem_rgb = np.asarray(frame_sem_rgb, dtype=np.float64)/255.0
            frame_pc.colors = o3d.utility.Vector3dVector(frame_sem_rgb)
        
        frame_origin = self.cur_pose_ref[:3, 3] * self.config.scale  # translation part
        frame_origin_torch = torch.tensor(frame_origin, dtype=self.dtype, device=self.pool_device)

        # transform to reference frame 
        frame_pc = frame_pc.transform(self.cur_pose_ref)
        # make a backup for merging into the map point cloud
        frame_pc_clone = copy.deepcopy(frame_pc)
        frame_pc_clone = frame_pc_clone.voxel_down_sample(voxel_size=self.config.map_vox_down_m) # for smaller memory cost
        self.map_down_pc += frame_pc_clone  #for solve kill9,build mesh need to cancel
        self.cur_frame_pc = frame_pc_clone

        self.map_bbx = self.map_down_pc.get_axis_aligned_bounding_box()   #for solve kill9,,build mesh need to cancel
        self.cur_bbx = self.cur_frame_pc.get_axis_aligned_bounding_box()
        # and scale to [-1,1] coordinate system
        frame_pc_s = frame_pc.scale(self.config.scale, center=(0,0,0))
        #a=self.config.scale*411

        frame_pc_s_torch = torch.tensor(np.asarray(frame_pc_s.points), dtype=self.dtype, device=self.pool_device)

        frame_normal_torch = None
        if self.config.estimate_normal:
            frame_normal_torch = torch.tensor(np.asarray(frame_pc_s.normals), dtype=self.dtype, device=self.pool_device)

        frame_label_torch = None
        if self.config.semantic_on:
            frame_label_torch = torch.tensor(frame_sem_label, dtype=self.dtype, device=self.pool_device)

        # print("Frame point cloud count:", frame_pc_s_torch.shape[0])

        # sampling the points
        (coord, sdf_label, normal_label, sem_label, weight, sample_depth, ray_depth) = \
            self.sampler.sample(frame_pc_s_torch, frame_origin_torch, \
            frame_normal_torch, frame_label_torch)

        origin_repeat = frame_origin_torch.repeat(coord.shape[0], 1)
        time_repeat = torch.tensor(frame_id, dtype=self.dtype, device=self.pool_device).repeat(coord.shape[0])
        self.cur_sample_count = sdf_label.shape[0]  # before filtering
        self.pool_sample_count = self.sdf_label_pool.shape[0]

        # update feature octree
        if frame_id % self.config.clear_table==0: #clear all feature to avoid other features interruption
            self.octree.clear_table()
            self.octree.clear_temp()
        if self.octree is not None:
            if self.config.octree_from_surface_samples:
                # update with the sampled surface points
                self.octree.update(coord[weight > 0, :].to(self.device), incremental_on)
            else:
                # update with the original points
                self.octree.update(frame_pc_s_torch.to(self.device), incremental_on)

                # get the data pool ready for training

        # ray-wise samples order
        if not incremental_on:  # for the incremental mapping with feature update regularization
            self.coord_pool = coord
            self.sdf_label_pool = sdf_label
            self.normal_label_pool = normal_label
            self.sem_label_pool = sem_label
            # self.color_label_pool = color_label
            self.weight_pool = weight
            self.sample_depth_pool = sample_depth
            self.ray_depth_pool = ray_depth
            self.origin_pool = origin_repeat
            self.time_pool = time_repeat

        else:  # batch processing

            # concat with current observations
            self.coord_pool = torch.cat((self.coord_pool, coord.to(self.pool_device)), 0)
            self.weight_pool = torch.cat((self.weight_pool, weight.to(self.pool_device)), 0)
            if self.config.ray_loss:
                self.sample_depth_pool = torch.cat((self.sample_depth_pool, sample_depth.to(self.pool_device)), 0)
                self.ray_depth_pool = torch.cat((self.ray_depth_pool, ray_depth.to(self.pool_device)), 0)
            else:
                self.sdf_label_pool = torch.cat((self.sdf_label_pool, sdf_label.to(self.pool_device)), 0)
                self.origin_pool = torch.cat((self.origin_pool, origin_repeat.to(self.pool_device)), 0)
                self.time_pool = torch.cat((self.time_pool, time_repeat.to(self.pool_device)), 0)

            if normal_label is not None:
                self.normal_label_pool = torch.cat((self.normal_label_pool, normal_label.to(self.pool_device)), 0)
            else:
                self.normal_label_pool = None

            if sem_label is not None:
                self.sem_label_pool = torch.cat((self.sem_label_pool, sem_label.to(self.pool_device)), 0)
            else:
                self.sem_label_pool = None
            # filter data

            pool_relative_dist = (self.coord_pool - frame_origin_torch).norm(2, dim=-1)
            dist_mask = pool_relative_dist < self.config.window_radius * self.config.scale
            filter_mask = dist_mask

            true_indices = torch.nonzero(filter_mask).squeeze()

            pool_sample_count = true_indices.shape[0]
            if pool_sample_count > self.config.pool_capacity:
                discard_count = pool_sample_count - self.config.pool_capacity
                # randomly discard some of the data samples if it already exceed the maximum number allowed in the data pool
                discarded_index = torch.randint(0, pool_sample_count, (discard_count,), device=self.pool_device)
                filter_mask[true_indices[
                    discarded_index]] = False  # Set the elements corresponding to the discard indices to False

            # and also have two filter mask options (delta frame, distance)
            # print(filter_mask)

            self.coord_pool = self.coord_pool[filter_mask]
            self.weight_pool = self.weight_pool[filter_mask]

            # FIX ME for ray-wise sampling
            # self.sample_depth_pool = self.sample_depth_pool[filter_mask]
            # self.ray_depth_pool = self.ray_depth_pool[filter_mask]

            self.sdf_label_pool = self.sdf_label_pool[filter_mask]
            self.origin_pool = self.origin_pool[filter_mask]
            self.time_pool = self.time_pool[filter_mask]

            if normal_label is not None:
                self.normal_label_pool = self.normal_label_pool[filter_mask]
            if sem_label is not None:
                self.sem_label_pool = self.sem_label_pool[filter_mask]
            if self.config.ray_loss:
                self.sample_depth_pool = self.sample_depth_pool[filter_mask]
                self.ray_depth_pool = self.ray_depth_pool[filter_mask]
            cur_sample_filter_mask = filter_mask[-self.cur_sample_count:]  # typically all true
            self.cur_sample_count = cur_sample_filter_mask.sum().item()  # number of current samples
            self.pool_sample_count = filter_mask.sum().item()
            cur_label_filtered = self.sdf_label_pool[-self.cur_sample_count:]

            self.new_idx = torch.where(
                (torch.abs(cur_label_filtered) < (self.config.surface_sample_range_m * 3. * self.config.scale)))[0]

            self.new_idx += (self.pool_sample_count - self.cur_sample_count)  # new idx in the data pool

            new_sample_count = self.new_idx.shape[0]

    def read_point_cloud(self, filename: str):
        # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
        if ".bin" in filename:
            points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
        elif ".ply" in filename or ".pcd" in filename:
            pc_load = o3d.io.read_point_cloud(filename)
            points = np.asarray(pc_load.points, dtype=np.float64)
        else:
            sys.exit(
                "The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)"
            )
        preprocessed_points = self.preprocess_kitti(
            points, self.config.min_z, self.config.min_range
        )
        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(preprocessed_points) # Vector3dVector is faster for np.float64 
        return pc_out

    def read_semantic_point_label(self, bin_filename: str, label_filename: str):

        # read point cloud (kitti *.bin format)
        if ".bin" in bin_filename:
            points = np.fromfile(bin_filename, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
        else:
            sys.exit(
                "The format of the imported point cloud is wrong (support only *bin)"
            )

        # read point cloud labels (*.label format)
        if ".label" in label_filename:
            labels = np.fromfile(label_filename, dtype=np.uint32).reshape((-1))
        else:
            sys.exit(
                "The format of the imported point labels is wrong (support only *label)"
            )

        points, sem_labels = self.preprocess_sem_kitti(
            points, labels, self.config.min_z, self.config.min_range, filter_moving=self.config.filter_moving_object
        )

        sem_labels = (np.asarray(sem_labels, dtype=np.float64)/255.0).reshape((-1, 1)).repeat(3, axis=1) # label 

        # TODO: better to use o3d.t.geometry.PointCloud(device)
        # a bit too cubersome
        # then you can use sdf_map_pc.point['positions'], sdf_map_pc.point['intensities'], sdf_map_pc.point['labels']
        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(points) # Vector3dVector is faster for np.float64 
        pc_out.colors = o3d.utility.Vector3dVector(sem_labels)

        return pc_out

    def preprocess_kitti(self, points, z_th=-3.0, min_range=2.5):
        # filter the outliers
        z = points[:, 2]
        points = points[z > z_th]
        points = points[np.linalg.norm(points, axis=1) >= min_range]
        return points

    def preprocess_sem_kitti(self, points, labels, min_range=2.75, filter_outlier = True, filter_moving = True):
        # TODO: speed up
        sem_labels = np.array(labels & 0xFFFF)

        range_filtered_idx = np.linalg.norm(points, axis=1) >= min_range
        points = points[range_filtered_idx]
        sem_labels = sem_labels[range_filtered_idx]

        # filter the outliers according to semantic labels
        if filter_moving:
            filtered_idx = sem_labels < 100
            points = points[filtered_idx]
            sem_labels = sem_labels[filtered_idx]

        if filter_outlier:
            filtered_idx = (sem_labels != 1) # not outlier
            points = points[filtered_idx]
            sem_labels = sem_labels[filtered_idx]
        
        sem_labels_main_class = np.array([sem_kitti_learning_map[sem_label] for sem_label in sem_labels]) # get the reduced label [0-20]

        return points, sem_labels_main_class
    
    def write_merged_pc(self, out_path):
        map_down_pc_out = copy.deepcopy(self.map_down_pc)
        map_down_pc_out.transform(inv(self.begin_pose_inv)) # back to world coordinate (if taking the first frame as reference)
        o3d.io.write_point_cloud(out_path, map_down_pc_out) 
        print("save the merged point cloud map to %s\n" % (out_path))    

    def __len__(self) -> int:
        if self.config.ray_loss:
            return self.ray_depth_pool.shape[0]  # ray count
        else:
            return self.sdf_label_pool.shape[0]  # point sample count

    # deprecated
    def __getitem__(self, index: int):
        # use ray sample (each sample containing all the sample points on the ray)
        if self.config.ray_loss:
            sample_index = torch.range(0, self.ray_sample_count - 1, dtype=int)
            sample_index += index * self.ray_sample_count

            coord = self.coord_pool[sample_index, :]
            # sdf_label = self.sdf_label_pool[sample_index]
            # normal_label = self.normal_label_pool[sample_index]
            # sem_label = self.sem_label_pool[sample_index]
            sample_depth = self.sample_depth_pool[sample_index]
            ray_depth = self.ray_depth_pool[index]

            return coord, sample_depth, ray_depth

        else:  # use point sample
            coord = self.coord_pool[index, :]
            sdf_label = self.sdf_label_pool[index]
            # normal_label = self.normal_label_pool[index]
            # sem_label = self.sem_label_pool[index]
            weight = self.weight_pool[index]

            return coord, sdf_label, weight
    def get_batch(self):

        if self.config.bs_new_sample > 0 and self.new_idx is not None :
            # half, half for the history and current samples
            new_idx_count = self.new_idx.shape[0]
            if new_idx_count > 0:
                bs_new = min(new_idx_count, self.config.bs_new_sample)
                bs_history =  self.config.bs - bs_new
                index_history = torch.randint(0, self.pool_sample_count, (bs_history,), device=self.device)
                index_new_batch = torch.randint(0, new_idx_count, (bs_new,), device=self.device)
                index_new = self.new_idx[index_new_batch]
                index = torch.cat((index_history, index_new), dim=0)
            else: # uniformly sample the pool
                index = torch.randint(0, self.pool_sample_count, (self.config.bs,), device=self.device)
        else: # uniformly sample the pool
            index = torch.randint(0, self.pool_sample_count, (self.config.bs,), device=self.device)


        coord = self.coord_pool[index, :]
        sdf_label = self.sdf_label_pool[index]
        ts = self.time_pool[index] # frame number as the timestamp
        weight = self.weight_pool[index]

        if self.sem_label_pool is not None:
            sem_label = self.sem_label_pool[index]
        else:
            sem_label = None
        origin = self.origin_pool[index].to(self.device)
        if self.normal_label_pool is not None:
            normal_label = self.normal_label_pool[index, :]
        else:
            normal_label = None

        return coord, sdf_label, origin,ts, normal_label, sem_label, weight
    '''
    def get_batch(self):
        # use ray sample (each sample containing all the sample points on the ray)
        if self.config.ray_loss:
            train_ray_count = self.ray_depth_pool.shape[0]
            ray_index = torch.randint(0, train_ray_count, (self.config.bs,), device=self.pool_device)

            ray_index_repeat = (ray_index * self.ray_sample_count).repeat(self.ray_sample_count, 1)
            sample_index = ray_index_repeat + torch.arange(0, self.ray_sample_count,\
                 dtype=int, device=self.device).reshape(-1, 1)
            index = sample_index.transpose(0,1).reshape(-1)

            coord = self.coord_pool[index, :].to(self.device)
            weight = self.weight_pool[index].to(self.device)
            sample_depth = self.sample_depth_pool[index].to(self.device)

            if self.normal_label_pool is not None:
                normal_label = self.normal_label_pool[index, :].to(self.device)
            else: 
                normal_label = None

            if self.sem_label_pool is not None:
                sem_label = self.sem_label_pool[ray_index * self.ray_sample_count].to(self.device) # one semantic label for one ray
            else: 
                sem_label = None

            ray_depth = self.ray_depth_pool[ray_index].to(self.device)

            return coord, sample_depth, ray_depth, normal_label, sem_label, weight
        
        else: # use point sample
            train_sample_count = self.sdf_label_pool.shape[0]
            index = torch.randint(0, train_sample_count, (self.config.bs,), device=self.pool_device)
            coord = self.coord_pool[index, :].to(self.device)
            sdf_label = self.sdf_label_pool[index].to(self.device)
            origin = self.origin_pool[index].to(self.device)
            ts = self.time_pool[index].to(self.device) # frame number or the timestamp

            if self.normal_label_pool is not None:
                normal_label = self.normal_label_pool[index, :].to(self.device)
            else: 
                normal_label = None
            
            if self.sem_label_pool is not None:
                sem_label = self.sem_label_pool[index].to(self.device)
            else: 
                sem_label = None

            weight = self.weight_pool[index].to(self.device)

            return coord, sdf_label, origin, ts, normal_label, sem_label, weight
    '''

