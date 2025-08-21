import os
import argparse
import cv2
import dnnlib
import math
import pickle
import mrcfile
import pyshtools
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
from torch_utils import misc

import copy
import imageio
from torchvision import transforms
from torchvision.utils import make_grid

from matplotlib import pyplot as plt
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import persistence
from torchvision import transforms
from torchvision.utils import make_grid


import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init

imageio.plugins.freeimage.download()

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random

parser = argparse.ArgumentParser(description='parameter for evaluation')
parser.add_argument('--network', help='network for evaluate')
parser.add_argument('--seed',default=0, type=int,help='set a seed')
args = parser.parse_args()
device = "cuda"

if __name__ == "__main__":

    G = torch.load(f"./{args.network}").to(device).eval().requires_grad_(False) 


    G.rendering_kwargs['depth_resolution'] = 96
    G.rendering_kwargs['depth_resolution_importance'] = 96


    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)

    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    def mapping(G, z: torch.Tensor, conditioning_params: torch.Tensor, truncation_psi=1., truncation_cutoff=14, update_emas=True):
        return G.backbone.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)


    criterion = nn.CrossEntropyLoss()

    def compute_miou(preds, targets, num_classes):
        batch_miou = 0.0
        for i in range(preds.size(0)): 
            intersection = torch.zeros(num_classes).to(preds.device)  
            union = torch.zeros(num_classes).to(preds.device)  

            for cls in range(num_classes):  
                pred_mask = preds[i] == cls
                target_mask = targets[i] == cls

                intersection[cls] = (pred_mask & target_mask).sum().item()  
                union[cls] = (pred_mask | target_mask).sum().item() 

            valid_classes = union != 0
            class_iou = torch.where(valid_classes, intersection / union, torch.zeros_like(union))
            class_miou = class_iou[valid_classes].mean()

            batch_miou += class_miou

        return batch_miou / preds.size(0)  

    
    eval_list = [ 10, 13, 14, 15, 16, 26, 29, 33, 35, 45, 50, 59, 60, 61, 62, 63, 64, 65, 68, 69, 70, 71, 72, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 92, 93, 97, 98, 99]

    loss_mean_list = []
    miou_mean_list = []
    acc_mean_list = []


    def average_of_list(numbers):
        total = sum(numbers)
        count = len(numbers)
        average = total / count
        return average

    for img_num in eval_list:
        with torch.no_grad():
            input1_dir = f'./data/ws/ws{img_num:04d}.pt'
            input2_dir = f'./data/camera_params/c{img_num:04d}.pt'
            label_dir = f'./data/labels_base/label{img_num:04d}.pt'
            ws = torch.load(input1_dir).to(device)
            camera_params = torch.load(input2_dir).to(device)

            label = torch.load(label_dir)

            outputs = G.synthesis(ws, camera_params)["image_seg"]

            output = torch.softmax(outputs, dim=1).to(device)
            miou = compute_miou(output.argmax(dim=1), label.to(device).unsqueeze(0), 17)

            miou_mean_list.append(miou.item())

    avg_miou = average_of_list(miou_mean_list)

    print(f'average miou: ', avg_miou)










