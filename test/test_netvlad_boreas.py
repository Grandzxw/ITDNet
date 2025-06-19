import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter
from sklearn import metrics
import numpy as np
import yaml
import time
import random
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from tools.database_boreas import boreas_dataset, PromptTrainDataset, DeweatherDataset
from tools.schedulers import LinearWarmupCosineAnnealingLR
from model.loss import triplet_margin_loss
from model.ITDNet.ITD_res import ITDNet_D 
from model.ITDNet.ITD_lpr import ITDNet_P
import matplotlib.pyplot as plt
from tools.val_utils import AverageMeter, compute_psnr_ssim
from tools.image_io import save_image_tensor, save_npy, torch_to_np, process_channels, torch_to_np_batch
import datetime
from collections import OrderedDict
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_auc_score


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
def test_Deweather(net, vlad, valloader, datalen):
    # subprocess.check_output(['mkdir', '-p', testopt.output_path])
    psnr = AverageMeter()
    ssim = AverageMeter()
    vlad_arr = np.zeros((datalen, 1024), dtype=np.float32)
    with torch.no_grad():
        for ([degraded_name], degrad_patch, idx) in tqdm(valloader):
            degrad_patch = degrad_patch.to(device=device)
            restored = net(degrad_patch)
            vlad_out = vlad(restored)

            batsize = degrad_patch.shape[0]
            for batch in range(batsize):
                id = idx[batch].detach().cpu().numpy()
                vlad_arr[id] = vlad_out[batch].detach().cpu().numpy()
    return vlad_arr


def test_Deweather_database(vlad, valloader, datalen):
    # subprocess.check_output(['mkdir', '-p', testopt.output_path])
    psnr = AverageMeter()
    ssim = AverageMeter()
    vlad_arr = np.zeros((datalen, 1024), dtype=np.float32)
    with torch.no_grad():
        for ([degraded_name], degrad_patch, idx) in tqdm(valloader):
            degrad_patch = degrad_patch.to(device=device)
            vlad_out = vlad(degrad_patch)
            batsize = degrad_patch.shape[0]
            for batch in range(batsize):
                id = idx[batch].detach().cpu().numpy()
                vlad_arr[id] = vlad_out[batch].detach().cpu().numpy()
    return vlad_arr



def test(config):

    test_query = config["test_config"]["datapath_query"]
    test_database = config["test_config"]["datapath_database"]
    pretrained_vlad_model = config["test_config"]["test_vlad_model"]
    pretrained_resnet_model = config["test_config"]["test_res_model"]
    th_min = config["test_config"]["th_min"]
    th_max = config["test_config"]["th_max"]
    th_max_pre = config["test_config"]["th_max_pre"]
    skip = config["test_config"]["skip"]

    test_dataset_query = DeweatherDataset(target_path=test_query)
    val_loader_query = DataLoader(
        dataset=test_dataset_query, batch_size=20,
        shuffle=True, num_workers=8)
    print("val_dataset query len is: ", len(test_dataset_query))
    
    test_dataset_database = DeweatherDataset(target_path=test_database)
    val_loader_database = DataLoader(
        dataset=test_dataset_database, batch_size=20,
        shuffle=True, num_workers=8)
    print("val_dataset database len is: ", len(test_dataset_database))

    resnet = ITDNet_D().to(device)
    vlad = ITDNet_P().to(device)

    checkpoint = torch.load(config["test_config"]["test_res_model"], map_location=device)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # 去掉 "module."
        new_state_dict[new_key] = v
    resnet.load_state_dict(new_state_dict, strict=False)  # strict=False 避免额外 key 报错

    checkpoint_vlad = torch.load(config["test_config"]["test_vlad_model"], map_location=device)
    state_dict_vlad = checkpoint_vlad["state_dict"]
    new_state_dict_vlad = OrderedDict()
    for k, v in state_dict_vlad.items():
        new_key = k.replace("module.", "")  # 去掉 "module."
        new_state_dict_vlad[new_key] = v
    vlad.load_state_dict(new_state_dict_vlad, strict=False)  # strict=False 避免额外 key 报错

    resnet.eval()
    vlad.eval()

    datalen = len(test_dataset_query)
    datalen_dabase = len(test_dataset_database)

    query_vlad = test_Deweather(resnet,vlad,val_loader_query,datalen)
    database_vlad = test_Deweather_database(vlad,val_loader_database,datalen_dabase)

    th_min = 0
    th_max = 25
    th_max_pre = 25
    skip = 50
    topk = 50
    
    query_pose = pd.read_csv("/data1/Boreas/transform/range_image_vlad/heavy/lidar_poses.csv").iloc[:8900]

    data_pose = pd.read_csv("/data1/Boreas/transform/range_image_vlad/test1/lidar_poses.csv").iloc[:5700]



    correct_at_k = np.zeros(topk)
    whole_test_size = 0
    length = len(query_pose)

    for i in tqdm(range(length), desc="evaluating", total=length):
        # index = query_pose['GPSTime'][i]
        pose_i = query_pose.iloc[i][['easting', 'northing']].values.reshape(1, -1)
        data_poses = data_pose[['easting', 'northing']].values
        # data_poses = query_pose[['easting', 'northing']].values
        distances = np.sqrt(np.sum((data_poses - pose_i)**2, axis=1))
        mask = (distances < th_min)
        distances[mask] = np.inf
        mindis_gt = np.min(distances)
        if mindis_gt < th_max:
            whole_test_size += 1
            vlad_dis = np.linalg.norm(database_vlad - query_vlad[i], axis=1)
            # vlad_dis[max(i - skip, 0):] = np.inf
            vlad_dis[mask] = np.inf
            vlad_topks = np.argsort(vlad_dis)[:topk]
            for k, k_idx in enumerate(vlad_topks):
                dis_gt = distances[k_idx]
                if dis_gt < th_max_pre:
                    correct_at_k[k:] += 1
                    break  
    recalls = correct_at_k / whole_test_size
    print("correct_at_k is: ", correct_at_k)
    print("whole_test_size is: ", whole_test_size)
    print("recall@N is: ", recalls)



if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    config_path = "/data1/Code/ITDNet/config/config_boreas.yaml"
    config = yaml.safe_load(open(config_path))
    test(config)  # 直接运行 train()