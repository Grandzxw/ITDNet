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
from model.ITRNet.ITR_res import ITRNet_D 
from model.ITRNet.ITR_lpr import ITRNet_P
import matplotlib.pyplot as plt
import datetime
import pandas as pd

torch.autograd.set_detect_anomaly(True)


def setup():
    """初始化分布式训练环境"""
    timeout = datetime.timedelta(seconds=300)  # 🔥 设置超时时间 300s
    dist.init_process_group(backend="nccl", timeout=timeout)
    local_rank = int(os.environ["LOCAL_RANK"])  # torchrun 会自动传递
    torch.cuda.set_device(local_rank)  # 绑定到当前 GPU
    return local_rank


def cleanup():
    """清理分布式进程组"""
    dist.destroy_process_group()


def val_Deweather_query(net, vlad, valloader, device):
    """分布式计算 VLAD 特征，并同步 PSNR 和 SSIM 计算"""
    local_vlad_list = []  # 每个 rank 计算的 VLAD 结果
    local_indices_list = []  # 每个 rank 计算的索引
    with torch.no_grad():
        for ([degraded_name], degrad_patch, idx) in tqdm(valloader):
            degrad_patch = degrad_patch.to(device=device)
            restored = net(degrad_patch)
            vlad_out = vlad(restored)
            for batch in range(degrad_patch.shape[0]):
                local_vlad_list.append(vlad_out[batch].cpu().numpy())  # 只存储计算出的 VLAD
                local_indices_list.append(idx[batch].item())  # 记录对应索引

    local_vlad_arr = np.array(local_vlad_list, dtype=np.float32)  # [batch_size, 256]
    local_indices_arr = np.array(local_indices_list, dtype=np.int32)  # [batch_size]

    gathered_vlad = [torch.zeros_like(torch.tensor(local_vlad_arr, dtype=torch.float32, device=device)) for _ in range(dist.get_world_size())]
    gathered_indices = [torch.zeros_like(torch.tensor(local_indices_arr, dtype=torch.int32, device=device)) for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_vlad, torch.tensor(local_vlad_arr, dtype=torch.float32, device=device))
    dist.all_gather(gathered_indices, torch.tensor(local_indices_arr, dtype=torch.int32, device=device))

    if dist.get_rank() == 0:
        all_vlad = []
        all_indices = []
        for rank in range(dist.get_world_size()):
            all_vlad.append(gathered_vlad[rank].cpu().numpy())
            all_indices.extend(gathered_indices[rank].cpu().numpy())

        all_vlad = np.concatenate(all_vlad, axis=0)
        sorted_indices = np.argsort(all_indices)  # 重新排序
        vlad_arr = all_vlad[sorted_indices]  # 按照原数据索引排序
    else:
        vlad_arr = None  # 其他 rank 只返回 None

    return vlad_arr




def val_Deweather_database(vlad, valloader, device):
    local_vlad_list = []  # 用于存储 rank 计算的 vlad 结果
    local_indices_list = []  # 用于存储 rank 计算的索引

    with torch.no_grad():
        for ([degraded_name], degrad_patch, idx) in tqdm(valloader):
            degrad_patch = degrad_patch.to(device=device)
            vlad_out = vlad(degrad_patch)

            for batch in range(degrad_patch.shape[0]):
                local_vlad_list.append(vlad_out[batch].cpu().numpy())  # 只存储计算出的 VLAD
                local_indices_list.append(idx[batch].item())  # 记录对应索引

    local_vlad_arr = np.array(local_vlad_list, dtype=np.float32)  # [batch_size, 256]
    local_indices_arr = np.array(local_indices_list, dtype=np.int32)  # [batch_size]

    gathered_vlad = [torch.zeros_like(torch.tensor(local_vlad_arr, dtype=torch.float32, device=device)) for _ in range(dist.get_world_size())]
    gathered_indices = [torch.zeros_like(torch.tensor(local_indices_arr, dtype=torch.int32, device=device)) for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_vlad, torch.tensor(local_vlad_arr, dtype=torch.float32, device=device))
    dist.all_gather(gathered_indices, torch.tensor(local_indices_arr, dtype=torch.int32, device=device))

    if dist.get_rank() == 0:
        all_vlad = []
        all_indices = []
        for rank in range(dist.get_world_size()):
            all_vlad.append(gathered_vlad[rank].cpu().numpy())
            all_indices.extend(gathered_indices[rank].cpu().numpy())

        all_vlad = np.concatenate(all_vlad, axis=0)
        sorted_indices = np.argsort(all_indices)  # 重新排序
        vlad_arr = all_vlad[sorted_indices]  # 按照原数据索引排序
    else:
        vlad_arr = None  # 其他 rank 只返回 None

    return vlad_arr




def train(config):
    local_rank = setup()  # 自动获取 local_rank
    device = torch.device(f"cuda:{local_rank}")

    vlad_root = config["data_root"]["vlad_data_root_folder"]
    res_root = config["data_root"]["res_data_root_folder"]
    valpath_query = config["val_config"]["datapath_query"]
    valpath_database = config["val_config"]["datapath_database"]
    
    log_folder = config["training_config"]["log_folder"]
    restore_training_seqs = config["training_config"]["restore_training_seqs"]
    vlad_training_seqs = config["training_config"]["vlad_training_seqs"]

    patch_height = config["training_config"]["patch_height"]
    patch_width = config["training_config"]["patch_width"]
    
    pretrained_vlad_model = config["training_config"]["pretrained_vlad_model"]
    pretrained_resnet_model = config["training_config"]["pretrained_resnet_model"]
    
    pos_threshold = config["training_config"]["pos_threshold"]
    neg_threshold = config["training_config"]["neg_threshold"]
    res_batch_size = config["training_config"]["res_batch_size"]
    vlad_batch_size = config["training_config"]["vlad_batch_size"]
    all_epochs = config["training_config"]["epoch"]
    save_res_path = config["training_config"]["save_res_path"]
    save_vlad_path = config["training_config"]["save_vlad_path"]

    
    writer = SummaryWriter()
    train_dataset_restore = PromptTrainDataset(root=res_root, 
                                               seqs=restore_training_seqs,
                                               patch_height=patch_height, 
                                               patch_width=patch_width)
    train_sampler_restore = DistributedSampler(train_dataset_restore, shuffle=True)
    train_loader_restore = DataLoader(
        dataset=train_dataset_restore, batch_size=res_batch_size, pin_memory=True,
        drop_last=True, num_workers=6, sampler=train_sampler_restore)
    print("train_dataset_restore len is: ", len(train_dataset_restore))


    train_dataset_vlad = boreas_dataset(
        root=vlad_root,
        seqs=vlad_training_seqs,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold)
    train_sampler_vlad = DistributedSampler(train_dataset_vlad, shuffle=True)
    train_loader_vlad = DataLoader(
        dataset=train_dataset_vlad, batch_size=vlad_batch_size,
        num_workers=6, sampler=train_sampler_vlad)
    print("train_dataset_vlad len is: ", len(train_dataset_vlad))


    val_dataset_query = DeweatherDataset(target_path=valpath_query)
    val_sampler_query = DistributedSampler(val_dataset_query, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False)
    val_loader_query = DataLoader(dataset=val_dataset_query, batch_size=vlad_batch_size, shuffle=False, num_workers=8, sampler=val_sampler_query)
    print("val_dataset query len is: ", len(val_dataset_query))
    
    val_dataset_database = DeweatherDataset(target_path=valpath_database)
    val_sampler_database = DistributedSampler(val_dataset_database, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False)
    val_loader_database = DataLoader(dataset=val_dataset_database, batch_size=vlad_batch_size, shuffle=False, num_workers=8, sampler=val_sampler_database)
    print("val_dataset database len is: ", len(val_dataset_database))


    resnet = ITRNet_D().to(device)
    vlad = ITRNet_P().to(device)
    

    if pretrained_vlad_model:
        checkpoint = torch.load(pretrained_vlad_model, map_location=device)
        vlad.load_state_dict(checkpoint['state_dict'])

    if pretrained_resnet_model:
        checkpoint = torch.load(pretrained_resnet_model, map_location=device)
        resnet.load_state_dict(checkpoint['state_dict'])


    resnet = DDP(resnet, device_ids=[local_rank],find_unused_parameters=True)
    vlad = DDP(vlad, device_ids=[local_rank], find_unused_parameters=True)


    criterion_res = nn.L1Loss().to(device)
    optimizer_restore = optim.AdamW(resnet.parameters(), lr=2e-4)
    optimizer_vlad = optim.AdamW(vlad.parameters(), lr=1e-5)
    scheduler_restore = optim.lr_scheduler.CosineAnnealingLR(optimizer_restore, T_max=100, eta_min=1e-6)
    scheduler_vlad = optim.lr_scheduler.CosineAnnealingLR(optimizer_vlad, T_max=100, eta_min=1e-6)

    vlad_loss_all = []
    res_loss_all = []
    vlad_loss_all_eval = []
    
    psnr_all =[]
    best_psnr = 0
    
    ssim_all = []
    best_ssim = 0
    for epoch in range(all_epochs):
        vlad.train()
        resnet.train()
        
        # train resnet
        if(epoch%2==0):

            # print("enter res epoch!")
            for param in resnet.parameters():
                param.requires_grad = True
            for param in vlad.parameters():
                param.requires_grad = False
            
            vlad.eval()
            resnet.train()
            
            batch_count = 0
            res_loss_epoch = 0
            vlad_loss_epoch = 0
            for i_batch, sample_batch in tqdm(enumerate(train_loader_restore), total=len(
                    train_loader_restore), desc='Train restore epoch ' + str(epoch), leave=False):
                optimizer_restore.zero_grad()
                degrad_patch = sample_batch['degrad_patch'].to(device)
                clean_patch = sample_batch['clean_patch'].to(device)
                
                out_res  = resnet(degrad_patch)
                res_loss = criterion_res(out_res,clean_patch)
                
                degrad_dec =  vlad(out_res)
                clean_dec = vlad(clean_patch)
                

                # p = F.softmax(degrad_dec, dim=1)
                # q = F.softmax(clean_dec, dim=1)

                # # 计算 KL 散度损失
                vlad_loss = torch.norm(degrad_dec - clean_dec, dim=1, p=2)**2
                vlad_loss = vlad_loss.mean()
                

                total_loss = res_loss + vlad_loss * 10
                
                res_loss_epoch += res_loss.item()
                vlad_loss_epoch += vlad_loss.item()
                batch_count += 1
                
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(resnet.parameters(), max_norm=1.0)
                optimizer_restore.step()
                
            res_loss_all.append(res_loss_epoch/batch_count)
            vlad_loss_all_eval.append(vlad_loss_epoch/batch_count)
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(res_loss_all) + 1), res_loss_all, label='Res Loss', color='blue')
            plt.xlabel('Epochs')
            plt.ylabel('Res Loss')
            plt.title('Res Loss vs Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'/data1/Code/ITRNet/logs/boreas_02-04_clean/fig/res_loss_vs_epochs.png')
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(vlad_loss_all_eval) + 1), vlad_loss_all_eval, label='Vlad eval Loss', color='blue')
            plt.xlabel('Epochs')
            plt.ylabel('Vlad eval Loss')
            plt.title('Vlad eval Loss vs Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'/data1/Code/ITRNet/logs/boreas_02-04_clean/fig/vlad_eval_loss_vs_epochs.png')
            plt.close()
            
            scheduler_restore.step()

        else:
            for param in vlad.parameters():
                param.requires_grad = True
            for param in resnet.parameters():
                param.requires_grad = False
            vlad.train()
            resnet.eval()

            batch_count = 0
            vlad_loss_epoch = 0
            for i_batch, sample_batch in tqdm(enumerate(train_loader_vlad), total=len(
                train_loader_vlad), desc='Train vlad epoch ' + str(epoch), leave=False):
                optimizer_vlad.zero_grad()
                
                degrad_patch = sample_batch['query_desc'].to(device)
                pos_patch = sample_batch['pos_desc'].to(device)
                neg_patch = sample_batch['neg_desc'].to(device)
                
                                
                input = torch.cat([degrad_patch,
                                pos_patch,
                                neg_patch], dim=0)
                
                out = vlad(input)
                split_size = input.shape[0] // 3
                query_fea, pos_fea, neg_fea = torch.split(
                    out, split_size, dim=0)
                
                train_dataset_vlad.update_latent_vectors(
                    query_fea.detach().cpu().numpy(),
                    sample_batch['query_id'].detach().cpu().numpy())
                
                train_dataset_vlad.update_latent_vectors_data(
                    pos_fea.detach().cpu().numpy(),
                    sample_batch['pos_id'].detach().cpu().numpy())
                
                train_dataset_vlad.update_latent_vectors_data(
                    neg_fea.detach().cpu().numpy(),
                    sample_batch['neg_id'].detach().cpu().numpy())

                vlad_loss = triplet_margin_loss(query_fea, pos_fea, neg_fea, margin=0.1)
                vlad_loss_epoch += vlad_loss.item()
                batch_count += 1
                
                vlad_loss.backward()
                optimizer_vlad.step()
        
        
            vlad_loss_all.append(vlad_loss_epoch/batch_count)
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(vlad_loss_all) + 1), vlad_loss_all, label='Vlad Loss', color='blue')
            plt.xlabel('Epochs')
            plt.ylabel('Vlad Loss')
            plt.title('Vlad Loss vs Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'/data1/Code/ITRNet/logs/boreas_02-04_clean/fig/Vlad_loss_vs_epochs.png')
            plt.close()

            scheduler_vlad.step()


        if (epoch > 60 and epoch%5==0):
            print(f"[Rank {dist.get_rank()}] start val train")

            with torch.no_grad():
                resnet.eval()
                vlad.eval()

            datalen_query = len(val_dataset_query)
            datalen_database = len(val_dataset_database)

            query_vlad = val_Deweather_query(resnet,vlad,val_loader_query, device)
            database_vlad = val_Deweather_database(vlad,val_loader_database,device)

            if dist.get_rank() == 0:
                print("database_vlad_arr is: ", database_vlad.shape)
                print("vlad_arr is: ", query_vlad.shape)
                th_min = 0
                th_max = 30
                th_max_pre = 30
                topk = 1

                # query_pose = pd.read_csv("/data1/Boreas/transform/range_image_vlad/test2/lidar_poses.csv").iloc[:5300]
                
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
                    # distances[max(i - skip, 0):] = np.inf
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

                print("whole_test_size is: ", whole_test_size)
                print("correct_at_k is: ", correct_at_k)
                print("recall@N is: ", recalls)


                res_folder = os.path.join(save_res_path, "Restore_model_lakder")
                if (not os.path.exists(res_folder)):
                    os.makedirs(res_folder)
                
                vlad_folder = os.path.join(save_vlad_path, "vlad_model_lakder")
                if (not os.path.exists(vlad_folder)):
                    os.makedirs(vlad_folder)

                torch.save({'epoch': epoch,
                    'state_dict': resnet.state_dict(),
                    'optimizer': optimizer_restore.state_dict(),
                    'scheduler': scheduler_restore.state_dict() if scheduler_restore else None},
                    os.path.join(res_folder,"Epoch_{}.ckpt".format(epoch)))

                torch.save({'epoch': epoch,
                    'state_dict': vlad.state_dict(),
                    'optimizer': optimizer_vlad.state_dict(),
                    'scheduler': scheduler_vlad.state_dict() if scheduler_vlad else None},
                    os.path.join(vlad_folder,"Epoch_{}_recall_{}.ckpt".format(epoch, recalls)))

            
    cleanup() 





if __name__ == '__main__':
    config_file = os.path.join(p, 'config/config_boreas.yaml')
    config = yaml.safe_load(open(config_file))
    train(config)  # 直接运行 train()