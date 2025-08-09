import os
import sys
import torch
import numpy as np
from utils.config import SHINEConfig
from model.feature_octree import FeatureOctree
from dataset.lidar_dataset import LiDARDataset
from model.decoder import Decoder
from copy import deepcopy
from utils.mesher import Mesher
from utils.loss import *
from utils.tools import *
from utils.data_sampler import dataSampler
from natsort import natsorted
import open3d as o3d
import copy
from utils.semantic_kitti_utils import *
from utils.optimize_pose import OptimizablePose
from tqdm import tqdm
import matplotlib.pyplot as plt


class Tracking():
    def __init__(self,config: SHINEConfig,dataset:LiDARDataset, octree:FeatureOctree,geo_decoder:Decoder,mesher:Mesher):
        #super().__init__()

        self.config=config
        self.mesher=mesher
        self.octree=octree
        self.geo_decoder=geo_decoder
        self.dataset=dataset
        self.final_pose=[]#store final pose
        #self.init_transform = torch.nn.Parameter(torch.eye(4))
        #self.init_transform =np.eye(4,dtype=np.float32)

        #self.pose_i=np.eye(4,dtype=np.float32)
        self.cur_pose_ref = self.dataset.ini_pose

        #pose_i[:3,3]=pose_i[:3,3]+0*self.config.scale
        #pose_i=torch.tensor(pose_i,requires_grad=True, dtype=torch.float32)
        #self.init_transform=OptimizablePose.from_init_matrix(pose_i)
        #self.init_transform= torch.nn.Parameter(pose_)
        self.pc_filenames = natsorted(os.listdir(config.pc_path))
        self.map_down_pc = o3d.geometry.PointCloud()
        # map bounding box in the world coordinate system
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()
        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()
        self.dtype = config.dtype
        self.pool_device = self.dataset.pool_device
        self.device = self.dataset.device
        self.sampler = dataSampler(config)
        self.last_pose_ref =  np.eye(4, dtype=np.float64)
        self.last_odom_tran =  np.eye(4, dtype=np.float64)
        if self.config.intrinsic_correct:
            self.last_odom_tran[0,3] = self.config.pc_radius*1e-2 # inital guess for booting on x aixs
        else:
            self.last_odom_tran[1, 3]=self.config.init_trans
        self.cur_point_ts_torch = None
        #self.last_odom_tran[0, 3] = 0.8
        #self.cur_pose_ref = np.eye(4, dtype=np.float64)


        #1.是否需要把采样点放到优化这一部分来？---->已经放进来了
        #2.mlp,以及特征向量参数需要冻结-------->already freeze
        #3.need to initialize the first frame---->yes
        #4.if do not find out feature vectors------->guess mc_mask decide whether if the point is valid
        #5. literratio number ? translation need to add 2000?,freezesigma ? add moreloss? getbatch? pose question or mapping question?

    def op_pose(self,sigma_sigmoid,frame_id,frame_pc):
        #coord=sdf_label=normal_label=sem_label=weight=sample_depth=ray_depth=frame_origin_torch=frame_pc_clone=frame_pc_s_torch=torch.Tensor()
        if frame_id== self.config.begin_frame :#第一帧使用单位矩阵, no need to query feature vectors
            #poses=self.init_transform
            #poses = self.pose_i
            poses = self.cur_pose_ref
            #self.cur_pose_ref=self.pose_i
            #self.cur_pose_ref =poses.matrix().detach()

            #self.last_pose_ref = self.cur_pose_ref.numpy()

            #poses=poses.to(self.dataset.pool_device)
            #poses=poses.detach()
            #pose=poses.matrix().detach().numpy()
            #coord, sdf_label, normal_label, sem_label, weight, sample_depth, ray_depth,frame_origin_torch,frame_pc_clone,frame_pc_s_torch = self.process_frame(frame_id,poses,frame_pc)
            #self.final_pose.append(poses.matrix().detach().cpu().numpy())
            self.final_pose.append(poses)
            self.last_pose_ref = self.cur_pose_ref
            return poses

        voxel_size = self.config.tracking_vox_down

        sor_nn = self.config.sor_nn
        sor_std = self.config.sor_std


        frame_pc,idx = voxel_down_sample_torch(frame_pc, voxel_size)
            # voxel downsampling
            # frame_pc = frame_pc.voxel_down_sample(voxel_size=vox_down_m)

            # apply filter (optional)
        if self.config.filter_noise: #default false  ---not change true ,if true need to change timestamp
            frame_pc = frame_pc.remove_statistical_outlier(
                sor_nn, sor_std, print_progress=False
            )[0]

        cur_pose_init_guess = self.last_pose_ref @ self.last_odom_tran # naddary


        if self.config.normal_loss_on or self.config.ekional_loss_on or self.config.proj_correction_on or self.config.consistency_loss_on:
            require_gradient = True
        else:
            require_gradient = False


        frame_pc_tensor = torch.tensor(np.asarray(frame_pc.points, dtype=np.float32),device=self.device , requires_grad=True)
        #frame_pc_tensor = torch.tensor(np.asarray(frame_pc.points, dtype=np.float32), device=self.device)
        frame_normal=torch.from_numpy(np.asarray(frame_pc.normals, dtype=np.float32)).to(self.device)
        #frame_pc_tensor.requires_grad_(True)
        loss=[]
        last_sdf_residual_cm = 1e5
        max_increment_sdf_residual_ratio = 1.1
        converged = False
        valid_flag = True
        min_valid_points = 30
        max_rot_deg = 10
        min_valid_ratio = 0.2
        max_valid_final_sdf_residual_cm = self.config.surface_sample_range_m * 0.6 * 100.0
        #max_valid_final_sdf_residual_cm = 2 * 0.5 * 100.0
        term_thre_deg = self.config.reg_term_thre_deg
        term_thre_m = self.config.reg_term_thre_m
        #init_pose_o=torch.tensor(cur_pose_init_guess,device=self.device)
        lamb=self.config.lm_lambda
        init_pose_T = torch.tensor(cur_pose_init_guess, dtype=torch.float64, device=self.device)
        last_iter_sdf_residual=0

        if self.cur_point_ts_torch is not None:
            cur_ts = self.cur_point_ts_torch.clone()
            cur_source_ts = cur_ts[idx]
        else:
            cur_source_ts = None


        if self.config.deskew:
            with torch.no_grad():


                frame_pc_tensor = deskewing(frame_pc_tensor, cur_source_ts,
                                            torch.tensor(self.last_odom_tran, device=self.pool_device,
                                                         dtype=self.dtype))  # T_last<-cur

        #frame_pc_tensor.requires_grad_(True)


        for iter in tqdm(range(self.config.num_iteration)):




            cur_point=transform_torch(frame_pc_tensor,init_pose_T)
            cur_point=cur_point * self.config.scale
            #cur_normal=rotation_torch(frame_normal,init_pose_T)
            '''
            cur_point=cur_point * self.config.scale
            source_point_count=cur_point.shape[0]
            cur_point.requires_grad_(True)
            feature = self.octree.query_feature(cur_point)
            pred_sdf = self.geo_decoder.sdf(feature)
            if require_gradient:
                g = get_gradient(cur_point, pred_sdf)
            '''
            source_point_count = cur_point.shape[0]
            g = torch.zeros((source_point_count, 3), device=self.device)
            pred_sdf = torch.zeros(source_point_count, device=self.device)

            iter_n = math.ceil(source_point_count / self.config.tracking_bs)
            for n in range(iter_n):
                head = n * self.config.tracking_bs
                tail = min((n + 1) * self.config.tracking_bs, source_point_count)
                batch_coords = cur_point[head:tail, :]

                batch_coords.requires_grad_(True)

                # cur_point.requires_grad_(True)
                batch_feature = self.octree.query_feature(batch_coords)
                batch_sdf = self.geo_decoder.sdf(batch_feature)

                # count = pred_sdf.shape[0]

                batch_g = get_gradient(batch_coords, batch_sdf)
                pred_sdf[head:tail] = batch_sdf.detach()
                g[head:tail, :] = batch_g.detach()
            #pred_sdf,g=self.query_sdf_and_grad(source_point_count,cur_point)

            #nonzero_indice=torch.nonzero(torch.any(feature !=0,dim=1)).squeeze()
            #cur_point=cur_point[nonzero_indice]
            #g=g[nonzero_indice]
            #pred_sdf = pred_sdf[nonzero_indice]

            count=pred_sdf.shape[0]
            sdf_label = torch.zeros(count, device=self.dataset.pool_device)
            #noise=(torch.rand(count, device=self.dataset.pool_device)-0.5)*2*0.005*self.config.scale
            #sdf_label=sdf_label+noise

            sdf_residual = pred_sdf - sdf_label


            sdf_residual_cm=torch.mean(torch.abs(sdf_residual)).item() * 100.0

            empty=sdf_residual.numel()
            if not empty:
                break

            delta_T, jicob, delta_x = self.implicit_reg(cur_point/self.config.scale, g*self.config.scale, sdf_residual,lm_lambda=lamb)
            #delta_T, _, _ = self.implicit_reg(cur_point , g , sdf_residual, lm_lambda=self.config.lm_lambda)
            #here to adjust lm_lambda according to ruo
            if iter !=0:
                differ_sdf_residual = torch.mean(torch.abs(sdf_residual)) / torch.mean(torch.abs( last_iter_sdf_residual)) # n
                #if differ_sdf_residual <1:
                    #lamb=2*lamb
                #if differ_sdf_residual==1:
                    #lamb=lamb
                #else:
                    #lamb=lamb/2

                #fenmu= jicob @ delta_x # n

                #ruo=differ_sdf_residual/fenmu
                #non_mask=~(torch.isnan(ruo)|torch.isinf(ruo))
                #ruo=ruo[non_mask]
                #ruo=torch.mean(ruo)
                #if ruo >0.75:
                    #lamb=2*lamb
                #if ruo<0.25:
                   # lamb=lamb/2
                #else:
                    #lamb=lamb

            last_iter_sdf_residual = sdf_residual



            init_pose_T = delta_T @ init_pose_T
            if (sdf_residual_cm - last_sdf_residual_cm) / last_sdf_residual_cm > max_increment_sdf_residual_ratio:
                print("[bold yellow](Warning) registration failed: wrong optimization[/bold yellow]")
                valid_flag = False
            else:
                last_sdf_residual_cm = sdf_residual_cm


            if (count < min_valid_points) or (
                    1.0 * count / source_point_count < min_valid_ratio):
                print("[bold yellow](Warning) registration failed: not enough valid points[/bold yellow]")
                valid_flag = False

            #if not valid_flag or converged:
                #break

            rot_angle_deg = rotation_matrix_to_axis_angle(delta_T[:3, :3]) * 180. / np.pi
            tran_m = delta_T[:3, 3].norm()


            if abs(rot_angle_deg) < term_thre_deg and tran_m < term_thre_m or iter == self.config.num_iteration - 2:
                converged = True  # for the visualization (save the computation)
            if sdf_residual_cm > max_valid_final_sdf_residual_cm:
                print("[bold yellow](Warning) registration failed: too large final residual[/bold yellow]")
                valid_flag = False
            #if frame_id !=1:
                #if abs(rot_angle_deg) > max_rot_deg:
                   # print("[bold yellow](Warning) registration failed: too large final rotation[/bold yellow]")
                   # print(abs(rot_angle_deg))
                   # valid_flag = False
            #if abs(rot_angle_deg) > max_rot_deg:
                #print("[bold yellow](Warning) registration failed: too large final rotation[/bold yellow]")
                #valid_flag = False

            if not valid_flag: # NOTE: if not valid, just take the initial guess
                init_pose_T = torch.tensor(cur_pose_init_guess, dtype=torch.float64, device=self.device)

            if not valid_flag or converged:

                break


            #aaa=init_pose_T.detach().cpu().numpy()
            #loss.append(aaa[0,3])


            #init_pose = torch.tensor(init_pose_T.detach().cpu().numpy(), requires_grad=True, dtype=torch.float32)

            #init_pose = OptimizablePose.from_matrix(init_pose)

        #print(init_pose_T)
        #print(init_pose.matrix())
        self.update_odom_pose(init_pose_T)
        self.final_pose.append(init_pose_T.clone().detach().cpu().numpy())
        #print(loss[-1])
        #print(init_pose_o)
        #print(init_pose_T)
        # loss=np.array(loss)
        # i=range(100)
        # plt.plot(loss)

        '''
        plt.plot(loss)
        plt.xlabel("num_iteration")
        #plt.ylabel("pose")
        plt.ylabel("pose")
        plt.legend()
        plt.show()
        plt.pause(1)
        '''



        #init_pose.requires_grad_(False)
        #coord.requires_grad_(False)
        return init_pose_T.clone().detach().cpu().numpy()
        #return coord, sdf_label, normal_label, sem_label, weight, sample_depth, ray_depth,frame_origin_torch,frame_pc_clone,frame_pc_s_torch

    def process_frame(self, frame_id,cur_pose,frame_pc):



        self.cur_pose_p= cur_pose

        # load point cloud (support *pcd, *ply and kitti *bin format)

        # load the label from the color channel of frame_pc
        if self.config.semantic_on:


            frame_sem_label = np.asarray(frame_pc.colors)[:,0]*255.0 # from [0-1] tp [0-255]
            frame_sem_label = np.round(frame_sem_label, 0) # to integer value
            sem_label_list = list(frame_sem_label)
            frame_sem_rgb = [sem_kitti_color_map[sem_label] for sem_label in sem_label_list]
            frame_sem_rgb = np.asarray(frame_sem_rgb, dtype=np.float64)/255.0
            frame_pc.colors = o3d.utility.Vector3dVector(frame_sem_rgb)

        frame_origin_torch = self.cur_pose_p.translation() * self.config.scale  # translation part
        #frame_origin_torch = self.cur_pose_ref[:3,3]* self.config.scale
        frame_origin_torch=frame_origin_torch.to(self.dataset.pool_device)

        #frame_origin_torch = torch.tensor(frame_origin, dtype=self.dtype, device=self.dataset.pool_device)

        # transform to reference frame
        #frame_pc=frame_pc.transform(self.cur_pose_ref)
        frame_pc_tensor=torch.from_numpy(np.asarray(frame_pc.points,dtype=np.float32))
        frame_pc_tensor=frame_pc_tensor.to(self.dataset.pool_device)
        R=self.cur_pose_p.rotation().transpose(-1,-2)
        #R=self.cur_pose_ref[:3,:3].transpose(-1,-2)
        R=R.to(self.dataset.pool_device)
        t=self.cur_pose_p.translation().reshape(-1,3)
        t=t.to(self.dataset.pool_device)
        frame_pc_tensor = frame_pc_tensor @ R + t
        #frame_pc_tensor=torch.mm(frame_pc_tensor , R)+t
        #frame_pc_t=frame_pc_tensor.cpu().detach().numpy()
        #frame_pc.points=o3d.utility.Vector3dVector(frame_pc_t)

        # make a backup for merging into the map point cloud
        frame_pc_clone = copy.deepcopy(frame_pc)
        frame_pc_clone = frame_pc_clone.voxel_down_sample(
            voxel_size=self.config.map_vox_down_m)  # for smaller memory cost
        #self.map_down_pc += frame_pc_clone
        #aaa=np.asarray(frame_pc.points, dtype=np.float32)

        self.cur_frame_pc = frame_pc_clone

        self.map_bbx = self.map_down_pc.get_axis_aligned_bounding_box()
        self.cur_bbx = self.cur_frame_pc.get_axis_aligned_bounding_box()
        # and scale to [-1,1] coordinate system
        #frame_pc_s = frame_pc.scale( self.config.scale, center=(0, 0, 0))
        frame_pc_s_torch=frame_pc_tensor*self.config.scale
        frame_pc_s_torch=frame_pc_s_torch.to(self.dataset.pool_device)

        #frame_pc_s_torch = torch.tensor(np.asarray(frame_pc_s.points), dtype=self.dtype, device=self.dataset.pool_device)
        #frame_pc_s_torch = torch.tensor(frame_pc_s, dtype=self.dtype, device=self.dataset.pool_device)

        frame_normal_torch = None
        if self.config.estimate_normal:
            frame_normal_torch = torch.tensor(np.asarray(frame_pc.normals), dtype=self.dtype, device=self.dataset.pool_device)
            frame_normal_torch=frame_normal_torch @ R

        frame_label_torch = None
        if self.config.semantic_on:
            #frame_label_torch=frame_label_torch
            frame_label_torch = torch.tensor(frame_sem_label, dtype=self.dtype, device=self.dataset.pool_device)

        # print("Frame point cloud count:", frame_pc_s_torch.shape[0])

        # sampling the points
        (coord, sdf_label, normal_label, sem_label, weight, sample_depth, ray_depth) = \
            self.sampler.sample(frame_pc_s_torch, frame_origin_torch, \
                                frame_normal_torch, frame_label_torch)
        origin_repeat = frame_origin_torch.repeat(coord.shape[0], 1)
        time_repeat = torch.tensor(frame_id, dtype=self.dtype, device=self.pool_device).repeat(coord.shape[0])
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

        #self.real_sample_depth_pool = real_sample_depth

    def get_tracking_batch(self):
        # use ray sample (each sample containing all the sample points on the ray)
        if self.config.ray_loss:
            train_ray_count = self.ray_depth_pool.shape[0]
            ray_index = torch.randint(0, train_ray_count, (self.config.tracking_bs,), device=self.pool_device)

            ray_index_repeat = (ray_index * self.ray_sample_count).repeat(self.ray_sample_count, 1)
            sample_index = ray_index_repeat + torch.arange(0, self.ray_sample_count, dtype=int,
                                                           device=self.device).reshape(-1, 1)
            index = sample_index.transpose(0, 1).reshape(-1)

            coord = self.coord_pool[index, :].to(self.device)
            weight = self.weight_pool[index].to(self.device)
            sample_depth = self.sample_depth_pool[index].to(self.device)

            if self.normal_label_pool is not None:
                normal_label = self.normal_label_pool[index, :].to(self.device)
            else:
                normal_label = None

            if self.sem_label_pool is not None:
                sem_label = self.sem_label_pool[ray_index * self.ray_sample_count].to(
                    self.device)  # one semantic label for one ray
            else:
                sem_label = None

            ray_depth = self.ray_depth_pool[ray_index].to(self.device)

            return coord, sample_depth, ray_depth, normal_label, sem_label, weight

        else:  # use point sample
            train_sample_count = self.sdf_label_pool.shape[0]
            index = torch.randint(0, train_sample_count, (self.config.tracking_bs,), device=self.pool_device)
            coord = self.coord_pool[index, :].to(self.device)
            sdf_label = self.sdf_label_pool[index].to(self.device)
            origin = self.origin_pool[index].to(self.device)
            ts = self.time_pool[index].to(self.device)  # frame number or the timestamp


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



    def read_point_cloud(self, filename: str):
        # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
        point_ts=None
        if ".bin" in filename:
            #points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
            points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3]
        elif ".ply" in filename or ".pcd" in filename:
            pc_load = o3d.io.read_point_cloud(filename)
            points = np.asarray(pc_load.points, dtype=np.float64)
        else:
            sys.exit(
                "The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)"
            )
        if self.config.deskew:
            self.get_point_ts(points, point_ts)
        preprocessed_points = self.preprocess_kitti(
            points, self.config.pc_radius,self.config.max_z,self.config.min_z, self.config.min_range
        )
        if self.config.intrinsic_correct:
            preprocessed_points = intrinsic_correct(preprocessed_points, 0.195)  # deskew vertical distortion


        #a=torch.from_numpy(preprocessed_points)
        #b=a.numpy()
        #c=b-preprocessed_points


        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(preprocessed_points)  # Vector3dVector is faster for np.float64
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

        sem_labels = (np.asarray(sem_labels, dtype=np.float64) / 255.0).reshape((-1, 1)).repeat(3, axis=1)  # label

        # TODO: better to use o3d.t.geometry.PointCloud(device)
        # a bit too cubersome
        # then you can use sdf_map_pc.point['positions'], sdf_map_pc.point['intensities'], sdf_map_pc.point['labels']
        pc_out = o3d.geometry.PointCloud()
        if self.config.intrinsic_correct:
            points = intrinsic_correct(points, 0.195)  # deskew vertical distortion

        pc_out.points = o3d.utility.Vector3dVector(points)  # Vector3dVector is faster for np.float64
        pc_out.colors = o3d.utility.Vector3dVector(sem_labels)

        return pc_out

    def preprocess_kitti(self, points, max_range,max_z,z_th=-3.0, min_range=2.5):
        # filter the outliers
        #z = points[:, 2]
        #points = points[z > z_th]
        #points = points[np.linalg.norm(points, axis=1) >= min_range]
        dist=np.linalg.norm(points, axis=1)
        filtered_idx = (dist > min_range) & (dist < max_range) & (points[:, 2] > z_th) & ( points[:, 2] < max_z)
        points=points[filtered_idx]
        if self.cur_point_ts_torch is not None:
            self.cur_point_ts_torch = self.cur_point_ts_torch[filtered_idx]
        return points

    def preprocess_sem_kitti(self, points, labels, min_range=2.75, filter_outlier=True, filter_moving=True):
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
            filtered_idx = (sem_labels != 1)  # not outlier
            points = points[filtered_idx]
            sem_labels = sem_labels[filtered_idx]

        sem_labels_main_class = np.array(
            [sem_kitti_learning_map[sem_label] for sem_label in sem_labels])  # get the reduced label [0-20]

        return points, sem_labels_main_class

    # original code from PIN_slam

    def update_odom_pose(self, cur_pose_torch: torch.tensor):
        # needed to be at least the second frame

        self.cur_pose_torch = cur_pose_torch.detach()  # need to be out of the computation graph, used for mapping

        self.cur_pose_ref = self.cur_pose_torch.cpu().numpy()

        self.last_odom_tran = np.linalg.inv(self.last_pose_ref) @ self.cur_pose_ref  # T_last<-cur# consecutive scan's transformation



        self.last_pose_ref = self.cur_pose_ref  # update for the next frame

        # deskewing (motion undistortion using the estimated transformation) for the sampled points for mapping

    def implicit_reg(self,points, sdf_grad, sdf_residual,  lm_lambda=0., require_cov=False, require_eigen=False):


        cross = torch.cross(points, sdf_grad)  # N,3 x N,3
        J = torch.cat([cross, sdf_grad], -1)  # first rotation, then translation # N, 6

        N = J.T @  J  # approximate Hessian matrix # first rot, then tran # 6, 6

        # you may try to do the SVD of N to analyzie its eigen value (contribution on each degree of freedom)
        if require_cov or require_eigen:
            N_old = N.clone()

        # use LM optimization
        #N += lm_lambda  * torch.eye(6, device=points.device)
        #b=torch.diag(N)

        # print(torch.diag(torch.diag(N)))
        N += lm_lambda * torch.diag(torch.diag(N))
        #b1= J.T
        #b2=b1[:3,:]
        #b3=b1[3:,:]

        # about lambda
        # If the lambda parameter is large, it implies that the algorithm is relying more on the gradient descent component of the optimization. This can lead to slower convergence as the steps are smaller, but it may improve stability and robustness, especially in the presence of noisy or ill-conditioned data.
        # If the lambda parameter is small, it implies that the algorithm is relying more on the Gauss-Newton component, which can make convergence faster. However, if the problem is ill-conditioned, setting lambda too small might result in numerical instability or divergence.

        g = -J.T @ sdf_residual
        #g1=-b2 @ sdf_residual
        #g2=-b3 @ sd
        #g=torch.cat([g1,g2],-1)

        t = torch.linalg.inv(N.to(dtype=torch.float64)) @ g.to(dtype=torch.float64)  # 6dof

        T = torch.eye(4, device=points.device, dtype=torch.float64)
        T[:3, :3] = expmap(t[:3])
        #T[:3, 3] = t[3:]/(self.config.scale)  # translation part
        T[:3, 3] = t[3:]

        eigenvalues = None  # the weight are also included, we need to normalize the weight part
        if require_eigen:
            N_old_tran_part = N_old[3:, 3:]
            eigenvalues = torch.linalg.eigvals(N_old_tran_part).real
            # # we need to set a threshold for the minimum eigenvalue for degerancy determination
            # smallest_value = min(eigenvalues)
            # print(smallest_value)

        cov_mat = None
        if require_cov:
            # Compute the covariance matrix (using a scaling factor)
            mse = torch.mean(weight.squeeze(1) * sdf_residual ** 2)
            cov_mat = torch.linalg.inv(N_old) * mse  # rotation , translation

            # Extract standard errors (square root of diagonal elements)
            # std = torch.sqrt(torch.diag(cov_mat)) # rotation , translation

            # print("Covariance Matrix:")
            # print(cov_mat)
            # print("Standard Errors:")
            # print(std)

        return T, J, t
    def implicit_normal_reg(self,points, sdf_grad, sdf_residual,  lm_lambda=0., require_cov=False, require_eigen=False):



        J = sdf_grad # first rotation, then translation # N, 3

        N = J.T @  J  # approximate Hessian matrix # first rot, then tran # 3, 3

        # you may try to do the SVD of N to analyzie its eigen value (contribution on each degree of freedom)
        if require_cov or require_eigen:
            N_old = N.clone()

        # use LM optimization
        # N += lm_lambda * 1e3 * torch.eye(6, device=points.device)

        # print(torch.diag(torch.diag(N)))
        N += lm_lambda * torch.diag(torch.diag(N))

        # about lambda
        # If the lambda parameter is large, it implies that the algorithm is relying more on the gradient descent component of the optimization. This can lead to slower convergence as the steps are smaller, but it may improve stability and robustness, especially in the presence of noisy or ill-conditioned data.
        # If the lambda parameter is small, it implies that the algorithm is relying more on the Gauss-Newton component, which can make convergence faster. However, if the problem is ill-conditioned, setting lambda too small might result in numerical instability or divergence.

        g = -J.T @ sdf_residual

        t = torch.linalg.inv(N.to(dtype=torch.float32)) @ g.to(dtype=torch.float32)  # 6dof

        T = torch.eye(4, device=points.device, dtype=torch.float32)
        T[:3, :3] = expmap(t[:3])
        T[:3, 3] = t[3:]/(self.config.scale)  # translation part

        eigenvalues = None  # the weight are also included, we need to normalize the weight part
        if require_eigen:
            N_old_tran_part = N_old[3:, 3:]
            eigenvalues = torch.linalg.eigvals(N_old_tran_part).real
            # # we need to set a threshold for the minimum eigenvalue for degerancy determination
            # smallest_value = min(eigenvalues)
            # print(smallest_value)

        cov_mat = None
        if require_cov:
            # Compute the covariance matrix (using a scaling factor)
            mse = torch.mean(weight.squeeze(1) * sdf_residual ** 2)
            cov_mat = torch.linalg.inv(N_old) * mse  # rotation , translation

            # Extract standard errors (square root of diagonal elements)
            # std = torch.sqrt(torch.diag(cov_mat)) # rotation , translation

            # print("Covariance Matrix:")
            # print(cov_mat)
            # print("Standard Errors:")
            # print(std)

        return T, cov_mat, eigenvalues
    def query_sdf_and_grad(self,source_point_count,cur_point):
        g = torch.zeros((source_point_count, 3), device=self.device)
        pred_sdf = torch.zeros(source_point_count, device=self.device)

        iter_n = math.ceil(source_point_count / self.config.tracking_bs)
        for n in range(iter_n):
            head = n * self.config.tracking_bs
            tail = min((n + 1) * self.config.tracking_bs, source_point_count)
            batch_coord = cur_point[head:tail, :]

            batch_coords = batch_coord * self.config.scale

            batch_coords.requires_grad_(True)

            # cur_point.requires_grad_(True)
            batch_feature = self.octree.query_feature(batch_coords)
            batch_sdf = self.geo_decoder.sdf(batch_feature)

            # count = pred_sdf.shape[0]


            batch_g = get_gradient(batch_coords, batch_sdf)
            pred_sdf[head:tail] = batch_sdf.detach()
            g[head:tail, :] = batch_g.detach()
            return pred_sdf,g
    def get_point_ts(self, points,point_ts = None):
        points= torch.from_numpy(points).clone().to(self.pool_device)
        if self.config.deskew:
            if point_ts is not None:
                self.cur_point_ts_torch = torch.tensor(point_ts, device=self.device, dtype=self.dtype)
            else:
                H = 64
                W = 1024
                #a=points.shape[0]
                #points.shape[0] == H * W:  # for Ouster 64-beam LiDAR


                if points.shape[0] == H*W:  # for Ouster 64-beam LiDAR
                    #if not self.silence:
                    print("Ouster-64 point cloud deskewed")
                    self.cur_point_ts_torch = (torch.floor(torch.arange(H * W) / H) / W).reshape(-1, 1).to(self.pool_device)
                

                elif(points.shape[0] == 128*1024):
                    print("Ouster-128 point cloud deskewed")
                    self.cur_point_ts_torch = (
                        (torch.floor(torch.arange(128 * 1024) / 128) / 1024)
                        .reshape(-1, 1)
                        .to(self.pool_device))
                

                else:
                    yaw = -torch.atan2(points[:,1], points[:,0])  # y, x -> rad (clockwise)
                    #yaw = -np.arctan2(points[:, 1], points[:, 0])  # y, x -> rad (clockwise)
                    if self.config.lidar_type_guess == "velodyne":
                        # for velodyne LiDAR (from -x axis, clockwise)
                        self.cur_point_ts_torch = 0.5 * (yaw / math.pi + 1.0) # [0,1]

                        #if not self.silence:
                        print("Velodyne point cloud deskewed")
                    else:
                        # for Hesai LiDAR (from +y axis, clockwise)
                        self.cur_point_ts_torch = 0.5 * (yaw / math.pi + 0.5) # [-0.25,0.75]
                        self.cur_point_ts_torch[self.cur_point_ts_torch < 0] += 1.0 # [0,1]

                        #if not self.silence:
                        print("HESAI point cloud deskewed")



def expmap(axis_angle: torch.Tensor):

    angle = axis_angle.norm()
    axis = axis_angle / angle
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    S = skew(axis)
    R = eye + S * torch.sin(angle) + (S @ S) * (1. - torch.cos(angle))
    return R



def skew(v):

    S = torch.zeros(3, 3, device=v.device, dtype=v.dtype)
    S[0, 1] = -v[2]
    S[0, 2] = v[1]
    S[1, 2] = -v[0]
    return S - S.T


def transform_torch(points: torch.tensor, transformation: torch.tensor):
    # points [N, 3]
    # transformation [4, 4]

    # Add a homogeneous coordinate to each point in the point cloud
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1).to(points)], dim=1)

    # Apply the transformation by matrix multiplication
    transformed_points_homo = torch.matmul(points_homo, transformation.to(points).T)

    # Remove the homogeneous coordinate from each transformed point
    transformed_points = transformed_points_homo[:, :3]

    return transformed_points
def rotation_torch(points: torch.tensor, transformation: torch.tensor):
    # points [N, 3]
    # transformation [4, 4]

    # Add a homogeneous coordinate to each point in the point cloud
    transformation=transformation[0:3,0:3]

    # Apply the transformation by matrix multiplication
    transformed_points_homo = torch.matmul(points, transformation.to(points).T)

    # Remove the homogeneous coordinate from each transformed point
    transformed_points = transformed_points_homo

    return transformed_points
def rotation_matrix_to_axis_angle(R):
    # epsilon = 1e-8  # A small value to handle numerical precision issues
    # Ensure the input matrix is a valid rotation matrix
    assert torch.is_tensor(R) and R.shape == (3, 3), "Invalid rotation matrix"
    # Compute the trace of the rotation matrix
    trace = torch.trace(R)
    # Compute the angle of rotation
    angle = torch.acos((trace - 1) / 2)

    return angle # rad






