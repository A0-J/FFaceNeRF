# ===== train_ffacenerf.py (DDP + AMP 적용, 디바이스/샘플러/저장 정리) =====

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

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from training.triplane_mod import (
    TriPlaneGenerator_base,
    TriPlaneGenerator_simple,
    TriPlaneGenerator_eyes,
    TriPlaneGenerator_nose,
    TriPlaneGenerator_chin,
)

imageio.plugins.freeimage.download()

import argparse

# -----------------------------
# DDP 초기화 유틸
# -----------------------------
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, device

# -----------------------------
# Args
# -----------------------------
def get_argparse():
    parser = argparse.ArgumentParser(description='parameter for training')
    parser.add_argument('--lr', default=0.01, type=float, help='max learning rate')
    parser.add_argument('--mode', default='base', help='simple,base,eyes,nose,chin')
    parser.add_argument('--aug', default='True', help='ws augment or not, True or False')
    parser.add_argument('--d_lambda', default=0.1, type=float, help='lambda for overlap loss')
    parser.add_argument('--alpha', default=0.5, type=float, help='parameter for w augmentation')
    parser.add_argument('--layers', default='9,10,11,12,13', help='layers for w augmentation')
    parser.add_argument('--iterations', default=5000, type=int, help='total iterations')
    parser.add_argument('--stage_iter', default=1000, type=int, help='ovlp stage start iteration')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--numdata', default=10, type=int, help='number of training samples')
    args = parser.parse_args()
    return args

# -----------------------------
# Helper
# -----------------------------
def mapping(G_like, z: torch.Tensor, conditioning_params: torch.Tensor,
            truncation_psi=1., truncation_cutoff=14, update_emas=True):
    # G_like: G.module 또는 G (둘 다 지원)
    backbone = G_like.backbone if hasattr(G_like, "backbone") else G_like.module.backbone
    return backbone.mapping(
        z, conditioning_params,
        truncation_psi=truncation_psi,
        truncation_cutoff=truncation_cutoff,
        update_emas=update_emas
    )

def calc_ovlp_loss(pred, target, smooth=1):
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
    pred = torch.nn.functional.softmax(pred, dim=1)
    pred_flat = pred.contiguous().reshape(pred.size(0), pred.size(1), -1)
    target_flat = target_one_hot.contiguous().reshape(target_one_hot.size(0), target_one_hot.size(1), -1)
    intersection = (pred_flat * target_flat).sum(2)
    denominator = pred_flat.sum(2) + target_flat.sum(2)
    dice_ovlp_score = (2. * intersection + smooth) / (denominator + smooth)
    ovlp_loss = 1 - dice_ovlp_score.mean(1)
    return ovlp_loss.mean()

# -----------------------------
# Data
# -----------------------------
def prepare_data(args):
    print("data processing ...")
    latent_dir = './data/ws'
    latent_all = []

    if args.mode == 'base':
        train_data = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 30, 31, 32, 34, 36, 37, 38, 39, 41, 42, 44, 46, 47, 48, 49, 52, 54, 55, 57, 58]
        data_list = random.sample(train_data, args.numdata)
        print(f"selected data : {data_list}")
    else:
        data_list = [5, 22, 27, 30, 32, 34, 41, 42, 44, 52]  # max 10 data for other models
        print(f"selected data : {data_list}")

    for i in data_list:
        p = os.path.join(latent_dir, f"ws{i:04d}.pt")
        if os.path.isfile(p):
            latent_all.append(p)

    cam_dir = './data/camera_params'
    cam_all = []
    for i in data_list:
        p = os.path.join(cam_dir, f"c{i:04d}.pt")
        if os.path.isfile(p):
            cam_all.append(p)

    if ('simple' in args.mode) or ('base' in args.mode):
        label_dir = './data/labels_base'
    elif args.mode == 'eyes':
        label_dir = './data/labels_eyes'
    elif args.mode == 'nose':
        label_dir = './data/labels_nose'
    elif args.mode == 'chin':
        label_dir = './data/labels_chin'
    label_all = []
    for i in data_list:
        p = os.path.join(label_dir, f"label{i:04d}.pt")
        if os.path.isfile(p):
            label_all.append(p)
        else:
            fallback = f"./data/labels_62/label{i:04d}.pt"
            if os.path.isfile(fallback):
                label_all.append(fallback)

    assert len(latent_all) == args.numdata, 'latent missing'
    assert len(cam_all) == args.numdata, 'cam missing'
    assert len(label_all) == args.numdata, 'label missing'
    return latent_all, cam_all, label_all, args.numdata

class CustomDataset(Dataset):
    def __init__(self, x1_list, x2_list, y_list):
        self.x1 = x1_list
        self.x2 = x2_list
        self.y = y_list

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        # DDP에서는 여기서 .to('cuda') 금지! CPU로 반환
        x1_ws = torch.load(self.x1[idx])   # CPU
        x2_cam = torch.load(self.x2[idx])  # CPU
        y_label = torch.load(self.y[idx])  # CPU
        return x1_ws, x2_cam, y_label

# -----------------------------
# Train
# -----------------------------
def train(args):
    torch.backends.cuda.matmul.allow_tf32 = True  # Ampere에서 성능 ↑

    # DDP 세팅
    local_rank, device = setup_ddp()

    if args.numdata < 4:
        batch_size = args.numdata
    else:
        batch_size = 4

    iter_per_epoch = math.ceil(args.numdata / batch_size)
    t_epochs = math.ceil(args.iterations / iter_per_epoch)

    # SummaryWriter는 rank 0에서만
    is_main = (dist.get_rank() == 0)
    writer = SummaryWriter(log_dir='runs/training_logs') if is_main else None

    print(f'learning rate : {args.lr}')

    # G_ema 로드(우선 CPU)
    with open("./networks/NeRFFaceEditing-ffhq-64.pkl", "rb") as f:
        G_ema = pickle.load(f)['G_ema']  # CPU

    # 렌더링 관련 기본 스텝(필요 시 조정)
    G_ema.rendering_kwargs['depth_resolution'] = 96
    G_ema.rendering_kwargs['depth_resolution_importance'] = 96

    # 모드에 따라 새 네트 만들고 파라미터 복사
    if args.mode == 'simple':
        G_new = TriPlaneGenerator_simple(*G_ema.init_args, **G_ema.init_kwargs).eval().requires_grad_(False)
        if is_main: print('training without injection!')
    elif args.mode == 'base':
        G_new = TriPlaneGenerator_base(*G_ema.init_args, **G_ema.init_kwargs).eval().requires_grad_(False)
        if is_main: print('training with base')
    elif args.mode == 'eyes':
        G_new = TriPlaneGenerator_eyes(*G_ema.init_args, **G_ema.init_kwargs).eval().requires_grad_(False)
        if is_main: print('training with injection and eyes addition!')
    elif args.mode == 'nose':
        G_new = TriPlaneGenerator_eyes(*G_ema.init_args, **G_ema.init_kwargs).eval().requires_grad_(False)
        if is_main: print('training with injection and nose addition!')
    elif args.mode == 'chin':
        G_new = TriPlaneGenerator_chin(*G_ema.init_args, **G_ema.init_kwargs).eval().requires_grad_(False)
        if is_main: print('training with injection and chin addition!')
    else:
        if is_main: print('wrong model name!')
        dist.destroy_process_group()
        return

    misc.copy_params_and_buffers(G_ema, G_new, require_all=False)
    G_new.neural_rendering_resolution = G_ema.neural_rendering_resolution
    G_new.rendering_kwargs = G_ema.rendering_kwargs

    # 디바이스 올리고 DDP 래핑
    G = G_new.to(device)
    G = DDP(G, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 디코더 학습 범위 설정 (DDP 후엔 G.module로 접근)
    for name, param in G.module.decoder.named_parameters():
        param.requires_grad = ('_appdx' in name)

    # 카메라/조건 파라미터(디바이스 고정)
    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.module.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.module.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)  # (1,25)

    # Optim & Sched & Loss
    optimizer = torch.optim.Adam(G.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=iter_per_epoch,
        epochs=t_epochs+1, anneal_strategy='cos'
    )
    criterion = nn.CrossEntropyLoss()

    # Data
    input_list, c_list, label_list, num_data = prepare_data(args)
    custom_dataset = CustomDataset(input_list, c_list, label_list)

    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(custom_dataset, shuffle=True)
    train_loader = DataLoader(
        custom_dataset,
        batch_size=batch_size,        # GPU당 배치
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # AMP 준비
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

    # ------------------ Training ------------------
    for epoch in tqdm(range(t_epochs)):
        running_loss = 0.0
        G.train()

        train_sampler.set_epoch(epoch)

        for i, (batch_x1, batch_x2, batch_y) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            # 반드시 여기서 디바이스 이동
            ws = batch_x1.squeeze(1).to(device, non_blocking=True)
            camera_params = batch_x2.to(device, non_blocking=True)
            labels = batch_y.to(device, non_blocking=True).to(torch.long)

            # 선택적 augmentation
            if args.aug == 'True':
                choices = ['no_aug', 'aug']
                probabilities = [0.5, 0.5]
                selected_choice = random.choices(choices, probabilities)[0]
                if selected_choice == 'aug':
                    alpha = 1 - args.alpha
                    z2 = torch.randn([1, 512], device=device)
                    mean_w = mapping(G.module, z2, conditioning_params.expand(z2.size(0), -1), truncation_psi=.7)
                    in_latent = ws.clone()
                    layer_list = args.layers.split(',') if ',' in args.layers else []
                    for layer2mix in layer_list:
                        l = int(layer2mix)
                        in_latent[:, l, :] = alpha * ws[:, l, :] + (1 - alpha) * mean_w[:, l, :]
                    ws = in_latent

            with autocast():  # AMP
                outputs = G.module.synthesis(ws, camera_params.squeeze(1))["image_seg"]
                cross_entropy_loss = criterion(outputs, labels)
                loss = cross_entropy_loss
                global_iter = epoch * iter_per_epoch + i
                if args.d_lambda > 0 and (global_iter > int(args.stage_iter)):
                    ovlp_loss = args.d_lambda * calc_ovlp_loss(outputs, labels)
                    loss += ovlp_loss
                else:
                    ovlp_loss = 0

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())

        if is_main and (epoch % 10 == 0):
            writer.add_scalar('Loss/train', running_loss, epoch)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, c_entropy_loss: {cross_entropy_loss}, ovlp_loss: {args.d_lambda * ovlp_loss}")

    # 저장은 rank 0만
    if is_main:
        os.makedirs("./networks", exist_ok=True)
        model_save_path = f"./networks/ckpt_{args.mode}_{args.numdata}.pth"
        torch.save(G.module.state_dict(), model_save_path)
        print(f"saved: {model_save_path}")

    if writer is not None:
        writer.close()
    print('train complete')

    # DDP 정리
    dist.destroy_process_group()

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    args = get_argparse()
    # 재현성(선택)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train(args)
