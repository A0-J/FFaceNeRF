import os
import cv2
import dnnlib
import torch
import math
import pickle
import mrcfile
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
from torch_utils import misc
from torch.utils.tensorboard import SummaryWriter
import random
import copy
import imageio
from torchvision import transforms
from torchvision.utils import make_grid

from matplotlib import pyplot as plt
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import persistence
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
from torch.nn import functional as F


from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from training.triplane_mod import TriPlaneGenerator_base, TriPlaneGenerator_simple, TriPlaneGenerator_eyes, TriPlaneGenerator_nose, TriPlaneGenerator_chin

imageio.plugins.freeimage.download()

import argparse

def get_argparse():
    parser = argparse.ArgumentParser(description='parameter for training')
    parser.add_argument('--lr',default=0.01, type=float, help='max learning rate')
    parser.add_argument('--mode', default = 'base',help='simple,base,eyes,nose,chin')
    parser.add_argument('--aug',default='True', help='ws augment or not, True or False')
    parser.add_argument('--d_lambda',default=0.1,type=float, help='lambda for overlap loss')
    parser.add_argument('--alpha',default=0.5, type=float, help='parameter for w augmentation')
    parser.add_argument('--layers',default='9,10,11,12,13', help='layers for w augmentation')
    parser.add_argument('--iterations',default=5000,type=int, help='layers for w augmentation')
    parser.add_argument('--stage_iter',default=1000, help='layers for w augmentation')
    parser.add_argument('--seed',default=0, help='layers for w augmentation')
    parser.add_argument('--numdata',default=10, type=int, help='parameter for w augmentation')


    args = parser.parse_args()
    return args

device = "cuda"


def mapping(G, z: torch.Tensor, conditioning_params: torch.Tensor, truncation_psi=1., truncation_cutoff=14, update_emas=True):
    return G.backbone.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    

def calc_ovlp_loss(pred, target, smooth=1):
    # Convert target to one-hot encoding
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
    pred = torch.nn.functional.softmax(pred, dim=1)
    
    pred_flat = pred.contiguous().reshape(pred.size(0), pred.size(1), -1)
    target_flat = target_one_hot.contiguous().reshape(target_one_hot.size(0), target_one_hot.size(1), -1)
    
    intersection = (pred_flat * target_flat).sum(2)
    denominator = pred_flat.sum(2) + target_flat.sum(2)
    
    dice_ovlp_score = (2. * intersection + smooth) / (denominator + smooth)
    ovlp_loss = 1 - dice_ovlp_score.mean(1)
    return ovlp_loss.mean()


def prepare_data():
    
    print("data processing ...")
 
    latent_dir = './data/ws'
    latent_all=[]
    
    if args.mode == 'base':
        train_data = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 30, 31, 32, 34, 36, 37, 38, 39, 41, 42, 44, 46, 47, 48, 49, 52, 54, 55, 57, 58]
        data_list = random.sample(train_data, args.numdata)
        print(f"selected data : {data_list}")
    else:
        data_list = [5,22,27,30,32,34,41,42,44,52] # max 10 data for other models
        print(f"selected data : {data_list}")
     
            
    for i in data_list:
        single_latent = os.path.join(latent_dir ,f"ws{i:04d}.pt") 
        if os.path.isfile(single_latent):
            latent_all.append(single_latent)
        
    
    cam_dir = './data/camera_params'
    cam_all=[]

    for i in data_list:
        single_cam = os.path.join(cam_dir ,f"c{i:04d}.pt") 
        if os.path.isfile(single_cam):
            cam_all.append(single_cam)
                        
            
            
    if ('simple' in args.mode) or ('base' in args.mode) : 
        label_dir = './data/labels_base'
    elif args.mode == 'eyes':
        label_dir = './data/labels_eyes'
    elif args.mode == 'nose':
        label_dir = './data/labels_nose'    
    elif args.mode == 'chin':
        label_dir = './data/labels_chin'        
    label_all=[]
 
            
    for i in data_list:
        single_label = os.path.join(label_dir ,f"label{i:04d}.pt") # pt files
        if os.path.isfile(single_label):
            label_all.append(single_label)
        else:
            if os.path.isfile(f"./data/labels_62/label{i:04d}.pt"):
                label_all.append(f"./data/labels_62/label{i:04d}.pt")

    assert len(data_list) == args.numdata, 'latent missing'
    assert len(cam_all) == args.numdata, 'cam missing'
    assert len(label_all) == args.numdata, 'label missing'
    
    return latent_all, cam_all, label_all, args.numdata # ws, camera parameter, label, number of data





class CustomDataset(Dataset):
    def __init__(self, x1_list, x2_list, y_list):
        self.x1 = x1_list
        self.x2 = x2_list
        self.y = y_list

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        x1_ws = torch.load(self.x1[idx]).to('cuda')
        x2_cam = torch.load(self.x2[idx]).to('cuda')
        y_label = torch.load(self.y[idx]).to('cuda')
        return x1_ws, x2_cam, y_label

def train(args):
    
    if args.numdata<4:
        batch_size =args.numdata
    else:
        batch_size = 4

    iter_per_epoch = math.ceil(args.numdata/batch_size)
    t_epochs = math.ceil(args.iterations/iter_per_epoch)




    print(f'learning rate : {args.lr}')
    writer = SummaryWriter(log_dir='runs/training_logs')


    with open("./networks/NeRFFaceEditing-ffhq-64.pkl", "rb") as f:
        G = pickle.load(f)['G_ema'].to(device).eval().requires_grad_(False)
    


    G.rendering_kwargs['depth_resolution'] = 96
    G.rendering_kwargs['depth_resolution_importance'] = 96


    if args.mode == 'simple': 
        G_new = TriPlaneGenerator_simple(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        print('training without injection!')
    elif args.mode == 'base':
        G_new = TriPlaneGenerator_base(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        print('training with base')
    elif args.mode == 'eyes' :
        G_new = TriPlaneGenerator_eyes(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        print('training with injection and eyes addition!')
    elif args.mode == 'nose':
        G_new = TriPlaneGenerator_eyes(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        print('training with injection and nose addition!')     
    elif args.mode == 'chin':
        G_new = TriPlaneGenerator_chin(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        print('training with injection and chin addition!')     
    else:
        print('wrong model name!')
        return

    misc.copy_params_and_buffers(G, G_new, require_all=False)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

    # check current network's state
    print(type(G.state_dict()))


    # we have to train just some parts of decoders, let's freeze the rest of the parts! 
    for name, param in G.decoder.named_parameters():
        if '_appdx' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


    # use parameter from EG3D and NeRFFaceEditing
    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)

    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    optimizer = torch.optim.Adam(G.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=iter_per_epoch, 
                                                    epochs=t_epochs+1, anneal_strategy='cos')

    criterion = nn.CrossEntropyLoss()

    
    input_list, c_list, label_list, num_data = prepare_data()
    custom_dataset = CustomDataset(input_list, c_list, label_list)
    train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    # Training
    for epoch in tqdm(range(t_epochs)):
        running_loss = 0.0
        G.train()

        for i, (batch_x1, batch_x2, batch_y) in enumerate(train_loader):            

            optimizer.zero_grad()

            ws = batch_x1.squeeze(1).to(device)

            if args.aug == 'True':
                # LMTA augmentation ratio
                choices = ['no_aug','aug']
                probabilities = [0.5,0.5] 

                selected_choice = random.choices(choices, probabilities)[0]

                if selected_choice == 'aug':
                    alpha = 1-args.alpha
                    z2 = torch.randn([1, 512]).to(device)
                    mean_w = mapping(G, z2, conditioning_params.expand(z2.size(0), -1), truncation_psi=.7)
                    in_latent = ws.clone()

                    #split layers with comma fpr LMTA
                    if ',' in args.layers:
                        layer_list= args.layers.split(',')

                    for layer2mix in layer_list:
                        layer2mix = int(layer2mix)
                        in_latent[:, layer2mix, :] = alpha*ws[:, layer2mix, :] + (1-alpha)*mean_w[:, layer2mix, :]
                    ws = in_latent
            else:
                layer_list = []

            camera_params = batch_x2.to(device)
            labels = batch_y.to(device)
            labels = labels.to(torch.long).to(device)
            outputs = G.synthesis(ws, camera_params.squeeze(1))["image_seg"]

            # Loss 
            cross_entropy_loss = criterion(outputs, labels)
            loss= cross_entropy_loss
            if args.d_lambda>0 and (epoch*iter_per_epoch + i > args.stage_iter):
                ovlp_loss = args.d_lambda*calc_ovlp_loss(outputs, labels)
                loss+=ovlp_loss
            else:
                ovlp_loss=0
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss

        if epoch%10==0:
            writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, c_entropy_loss: {cross_entropy_loss},ovlp_loss: {args.d_lambda*ovlp_loss}")
    model_save_path = f"./networks/ckpt_{args.mode}_{args.numdata}.pth"
    torch.save(G, model_save_path)                 


    writer.close()
    print('train complete')    


if __name__== '__main__':
    args = get_argparse()
    train(args)