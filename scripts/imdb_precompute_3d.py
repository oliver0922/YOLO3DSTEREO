import numpy as np
import os
import pickle
import time
import cv2
from copy import deepcopy
import skimage.measure
import torch

from _path_init import *
from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.networks.utils.utils import calc_iou, BBox3dProjector
from visualDet3D.data.pipeline import build_augmentator
from visualDet3D.data.kitti.kittidata import KittiData
from visualDet3D.utils.timer import Timer
from visualDet3D.utils.utils import cfg_from_file

def process_train_val_file(cfg):
    train_file = cfg.data.train_split_file #train_file = '/path/to/visualDet3D/visualDet3D/data/kitti/test_split/train.txt'
    val_file   = cfg.data.val_split_file #val_file = '/path/to/visualDet3D/visualDet3D/data/kitti/test_split/val.txt'

    with open(train_file) as f:
        train_lines = f.readlines() #train_lines=["000000\n","000001\n",...]
        for i  in range(len(train_lines)):
            train_lines[i] = train_lines[i].strip() #train_lines=["000000","000001",...]

    with open(val_file) as f: 
        val_lines = f.readlines()
        for i  in range(len(val_lines)):
            val_lines[i] = val_lines[i].strip()

    return train_lines, val_lines

def read_one_split(cfg, index_names, data_root_dir, output_dict, data_split = 'training', time_display_inter=100):
    #save_dir=/content/gdrive/MyDrive/visualDet3D/workdirs/Stereo3D/output/training
    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if data_split == 'training':
        disp_dir = os.path.join(save_dir, 'disp') #save_dir=/content/gdrive/MyDrive/visualDet3D/workdirs/Stereo3D/output/training/disp
        if not os.path.isdir(disp_dir):
            os.mkdir(disp_dir)

    N = len(index_names) #training, validation dataset 개수
    frames = [None] * N 
    print("start reading {} data".format(data_split))
    timer = Timer()

    anchor_prior = getattr(cfg, 'anchor_prior', True)

    total_objects = [0 for _ in range(len(cfg.obj_types))] #total_objects=[0,0]
    total_usable_objects = [0 for _ in range(len(cfg.obj_types))] #total_usable_objects=[0,0]
    if anchor_prior: #무조건 실행
        #anchor_manager=Anchors("/content/gdrive/MyDrive/visualDet3D/workdirs/Stereo3D/output")
        anchor_manager = Anchors(cfg.path.preprocessed_path, readConfigFile=False, **cfg.detector.head.anchors_cfg)
        preprocess = build_augmentator(cfg.data.test_augmentation)
        total_objects = [0 for _ in range(len(cfg.obj_types))] #total_objects=[0,0]
        total_usable_objects = [0 for _ in range(len(cfg.obj_types))] #total_usable_objects=[0,0]
        
        len_scale = len(anchor_manager.scales) #len_scale=16 (anchor_manager.scales=[2**0,2**1/4,2**2/4,...,2**15/4])
        len_ratios = len(anchor_manager.ratios)#len_ratios=3 (anchor_magager.ratios=np.array([0.5, 1, 2.0])
        len_level = len(anchor_manager.pyramid_levels) #len_level=1 (anchor_manager.pyramid_levels=[4])

        examine = np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios]) #examine=np.zeros([2,16,3])
        sums    = np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios, 3]) #sums=np.sums([2,16,3,3])
        squared = np.zeros([len(cfg.obj_types), len_level * len_scale, len_ratios, 3], dtype=np.float64) #squared=np.sums([2,16,3,3],dtype=np.float64)

        uniform_sum_each_type = np.zeros((len(cfg.obj_types), 6), dtype=np.float64)  #[z, sin2a, cos2a, w, h, l]
        uniform_square_each_type = np.zeros((len(cfg.obj_types), 6), dtype=np.float64)

    for i, index_name in enumerate(index_names):

        # read data with dataloader api
        data_frame = KittiData(data_root_dir, index_name, output_dict)
        calib, image, label, velo = data_frame.read_data()

        # store the list of kittiObjet and kittiCalib
        max_occlusion = getattr(cfg.data, 'max_occlusion', 2)
        min_z        = getattr(cfg.data, 'min_z', 3)
        if data_split == 'training':
            data_frame.label = [obj for obj in label.data if obj.type in cfg.obj_types and obj.occluded < max_occlusion and obj.z > min_z] #거르는 부분
            #data_fram.label=사진에 들어있고 조건을 만족하는 객체들로 이루어진 리스트[Kittiobj(pedestrain 0,3 ,...),Kittiobj(car 0,6,53...)] 
            if anchor_prior:
                for j in range(len(cfg.obj_types)):
                    total_objects[j] += len([obj for obj in data_frame.label if obj.type==cfg.obj_types[j]])
                    # total_objects=[image에 있는 'Car'의 개수(occlusion이 2보다 작아야하고, z값이 3보다 커야함),
                    # image에 있는 'Pedestrian'의 개수(occlusion이 2보다 작아야하고 , z값이 3보다 커야함)]
                    data = np.array(
                        [
                            [obj.z, np.sin(2 * obj.alpha), np.cos(2 * obj.alpha), obj.w, obj.h, obj.l] 
                                for obj in data_frame.label if obj.type==cfg.obj_types[j]
                        ]
                    ) #[N, 6]
                    if data.any():
                        uniform_sum_each_type[j, :] += np.sum(data, axis=0) #uniform_sum_each_type=[Car의 z 합,Pedestrian z합]
                        uniform_square_each_type[j, :] += np.sum(data ** 2, axis=0) #uniform_square_each_type=[Car의 z 합, Pedestrian z합]
        else:
            data_frame.label = [obj for obj in label.data if obj.type in cfg.obj_types]
            #data_fram.label=사진에 들어있는 객체들로 이루어진 리스트[Kittiobj(pedestrain 0,3 ,...),Kittiobj(car 0,6,53...)] ->조건 만족 필요 x 
        data_frame.calib = calib
        


        if data_split == 'training' and anchor_prior:
            original_image = image.copy()
            baseline = (calib.P2[0, 3] - calib.P3[0, 3]) / calib.P2[0, 0] #baseline=(-fu(left)*bx(left)-fu(right)*bx(right))/fu(left)
            image, P2, label = preprocess(original_image, p2=deepcopy(calib.P2), labels=deepcopy(data_frame.label))
            _,  P3 = preprocess(original_image, p2=deepcopy(calib.P3))

            ## Computing statistic for positive anchors
            if len(data_frame.label) > 0:
                anchors, _ = anchor_manager(image[np.newaxis].transpose([0,3,1,2]), torch.tensor(P2).reshape([-1, 3, 4]))

                for j in range(len(cfg.obj_types)):
                    bbox2d = torch.tensor([[obj.bbox_l, obj.bbox_t, obj.bbox_r, obj.bbox_b] for obj in label if obj.type == cfg.obj_types[j]]).cuda()
                    #bbox2d=[class가 Car 또는 Pedestrian 에 해당하는 2d bounding box 좌표들]
                    if len(bbox2d) < 1:
                        continue
                    bbox3d = torch.tensor([[obj.x, obj.y, obj.z, np.sin(2 * obj.alpha), np.cos(2 * obj.alpha)] for obj in label if obj.type == cfg.obj_types[j]]).cuda()
                    #bbox3d=[class가 Car 또는 Pedestrian 에 해당하는 3d bounding box 좌표들]
                    
                    usable_anchors = anchors[0]

                    IoUs = calc_iou(usable_anchors, bbox2d) #[N, K]
                    IoU_max, IoU_argmax = torch.max(IoUs, dim=0)
                    IoU_max_anchor, IoU_argmax_anchor = torch.max(IoUs, dim=1)

                    num_usable_object = torch.sum(IoU_max > cfg.detector.head.loss_cfg.fg_iou_threshold).item()
                    total_usable_objects[j] += num_usable_object

                    positive_anchors_mask = IoU_max_anchor > cfg.detector.head.loss_cfg.fg_iou_threshold
                    positive_ground_truth_3d = bbox3d[IoU_argmax_anchor[positive_anchors_mask]].cpu().numpy()

                    used_anchors = usable_anchors[positive_anchors_mask].cpu().numpy() #[x1, y1, x2, y2]

                    sizes_int, ratio_int = anchor_manager.anchors2indexes(used_anchors)
                    for k in range(len(sizes_int)):
                        examine[j, sizes_int[k], ratio_int[k]] += 1
                        sums[j, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5]
                        squared[j, sizes_int[k], ratio_int[k]] += positive_ground_truth_3d[k, 2:5] ** 2

        frames[i] = data_frame

        if (i+1) % time_display_inter == 0:
            avg_time = timer.compute_avg_time(i+1)
            eta = timer.compute_eta(i+1, N)
            print("{} iter:{}/{}, avg-time:{}, eta:{}, total_objs:{}, usable_objs:{}".format(
                data_split, i+1, N, avg_time, eta, total_objects, total_usable_objects), end='\r')

    save_dir = os.path.join(cfg.path.preprocessed_path, data_split)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if data_split == 'training' and anchor_prior:
        
        for j in range(len(cfg.obj_types)):
            global_mean = uniform_sum_each_type[j] / total_objects[j]
            global_var  = np.sqrt(uniform_square_each_type[j] / total_objects[j] - global_mean ** 2)

            avg = sums[j] / (examine[j][:, :, np.newaxis] + 1e-8)
            EX_2 = squared[j] / (examine[j][:, :, np.newaxis] + 1e-8)
            std = np.sqrt(EX_2 - avg ** 2)

            avg[examine[j] < 10, :] = -100  # with such negative mean Z, anchors/losses will filter them out
            std[examine[j] < 10, :] = 1e10
            avg[np.isnan(std)]      = -100
            std[np.isnan(std)]      = 1e10
            avg[std < 1e-3]         = -100
            std[std < 1e-3]         = 1e10

            whl_avg = np.ones([avg.shape[0], avg.shape[1], 3]) * global_mean[3:6]
            whl_std = np.ones([avg.shape[0], avg.shape[1], 3]) * global_var[3:6]

            avg = np.concatenate([avg, whl_avg], axis=2)
            std = np.concatenate([std, whl_std], axis=2)

            npy_file = os.path.join(save_dir,'anchor_mean_{}.npy'.format(cfg.obj_types[j]))
            np.save(npy_file, avg)
            std_file = os.path.join(save_dir,'anchor_std_{}.npy'.format(cfg.obj_types[j]))
            np.save(std_file, std)
    pkl_file = os.path.join(save_dir,'imdb.pkl')
    pickle.dump(frames, open(pkl_file, 'wb'))
    print("{} split finished precomputing".format(data_split))




def main(config:str="config/config.py"):
    cfg = cfg_from_file(config)
    torch.cuda.set_device(cfg.trainer.gpu) #cuda 위에 trainer 올린다.
    
    time_display_inter = 100 # define the inverval displaying time consumed in loop
    data_root_dir = cfg.path.data_path # the base directory of training dataset
    calib_path = os.path.join(data_root_dir, 'calib') 
    list_calib = os.listdir(calib_path) #list_calib=calib 디렉토리 내부의 파일들을 list화
    N = len(list_calib) #training data개수 =7481개
    # no need for image, could be modified for extended use
    output_dict = {
                "calib": True,
                "image": True,
                "label": True,
                "velodyne": False,
            }

    train_names, val_names = process_train_val_file(cfg) #train_names=["000000","000001",...], val_names=["000058","000090",...]
    #read_one_split(cfg, ["000000","000001",....],"/content/gdrive/MyDrive/KITTI_YOLO3D/training",output_dict,"training",100)
    read_one_split(cfg, train_names, data_root_dir, output_dict, 'training', time_display_inter) 
    output_dict = {
                "calib": True,
                "image": False,
                "label": True,
                "velodyne": False,
            }
    read_one_split(cfg, val_names, data_root_dir, output_dict, 'validation', time_display_inter)

    print("Preprocessing finished")

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
