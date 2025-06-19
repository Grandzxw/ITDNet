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
from tools.database_kitti import kitti_dataset, PromptTrainDataset, DeweatherDataset
from tools.schedulers import LinearWarmupCosineAnnealingLR
from model.loss import triplet_margin_loss
from model.ITDNet.ITD_res import ITDNet_D 
from model.ITDNet.ITD_lpr import ITDNet_P
import matplotlib.pyplot as plt
from tools.val_utils import AverageMeter, compute_psnr_ssim
from tools.image_io import save_image_tensor, save_npy, torch_to_np, process_channels, torch_to_np_batch
import datetime
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score



device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
def test_Deweather(net, vlad, valloader,datalen):
    # subprocess.check_output(['mkdir', '-p', testopt.output_path])
    psnr = AverageMeter()
    ssim = AverageMeter()
    vlad_arr = np.zeros((datalen, 1024), dtype=np.float32)
    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch, idx) in tqdm(valloader):
            degrad_patch, clean_patch = degrad_patch.to(device=device), clean_patch.to(device=device)
            
            restored = net(degrad_patch)
            vlad_out = vlad(restored)
            
            batsize = degrad_patch.shape[0]
            for batch in range(batsize):
                
                id = idx[batch].detach().cpu().numpy()
                vlad_arr[id] = vlad_out[batch].detach().cpu().numpy()
                
                restored_npy = torch_to_np_batch(restored[batch])
                restored_npy = restored_npy.transpose(1, 2, 0)
                restored_npy_handled = process_channels(restored_npy)


                clean_npy = torch_to_np_batch(clean_patch[batch])
                clean_npy = clean_npy.transpose(1, 2, 0)
                
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored_npy_handled, clean_npy)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)
        
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

    return vlad_arr, psnr.avg, ssim.avg



def test_Deweather_database(vlad, valloader,datalen):
    psnr = AverageMeter()
    ssim = AverageMeter()
    vlad_arr = np.zeros((datalen, 1024), dtype=np.float32)
    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch, idx) in tqdm(valloader):
            degrad_patch, clean_patch = degrad_patch.to(device=device), clean_patch.to(device=device)
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
        dataset=test_dataset_query, batch_size=24,
        shuffle=True, num_workers=8)
    print("val_dataset query len is: ", len(test_dataset_query))
    
    test_dataset_database = DeweatherDataset(target_path=test_database)
    val_loader_database = DataLoader(
        dataset=test_dataset_database, batch_size=24,
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
    query_vlad = np.zeros((datalen, 1024), dtype=np.float32)
    database_vlad = np.zeros((datalen, 1024), dtype=np.float32)

    query_vlad, psnr_query, ssim_query = test_Deweather(resnet,vlad,val_loader_query,datalen)
    database_vlad = test_Deweather_database(vlad,val_loader_database,datalen)

    th_min = 0
    th_max = 5
    th_max_pre = 5
    skip = 50
    topk = 50
    
    y_true = []  # 1: 正确匹配, 0: 错误匹配
    y_pred = []  # 1: 判定为匹配, 0: 判定为不匹配

    pose = np.genfromtxt('/data1/KITTI/sequences/07/07.txt')[:, [3, 11]]
    length = len(pose)
    correct_at_k = np.zeros(topk)
    whole_test_size = 0
    for i in tqdm(range(length), desc="evaluating", total=length):
        pos_dis = np.linalg.norm(pose - pose[i], axis=1)
        # 50帧以内不进行比较
        pos_dis[max(i - skip, 0):] = np.inf
        mask = (pos_dis < th_min)
        pos_dis[mask] = np.inf
        mindis_gt = np.min(pos_dis)
        #如果小于最小样本则是有效的
        if mindis_gt < th_max:
            whole_test_size += 1
            vlad_dis = np.linalg.norm(database_vlad - query_vlad[i], axis=1)
            vlad_dis[max(i - skip, 0):] = np.inf
            vlad_dis[mask] = np.inf
            vlad_topks = np.argsort(vlad_dis)[:topk]

            pred_idx = np.argmin(vlad_dis)  # 最相似的帧
            y_true.append(1 if pos_dis[pred_idx] < th_max_pre else 0)
            y_pred.append(1)  # 因为你总是预测一个最相近的帧
            ## 为什么top3正确那么top3到topk一定正确
            for k, k_idx in enumerate(vlad_topks):
                dis_gt = pos_dis[k_idx]
                if dis_gt < th_max_pre:
                    correct_at_k[k:] += 1
                    break  
    recalls = correct_at_k / whole_test_size
    print("recalls is: ", recalls)
    print("correct_at_k is: ", correct_at_k)
    print("whole_test_size is: ", whole_test_size)

    f1 = f1_score(y_true, y_pred)
    print("F1 Score:", f1)



if __name__ == "__main__":
    # world_size = torch.cuda.device_count()
    config_path = "/data1/Code/ITDNet/config/config_kitti.yaml"
    config = yaml.safe_load(open(config_path))
    test(config)  # 直接运行 train()