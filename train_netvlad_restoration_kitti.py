import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
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

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def val_Deweather(net, vlad, valloader,datalen):
    # subprocess.check_output(['mkdir', '-p', testopt.output_path])
    psnr = AverageMeter()
    ssim = AverageMeter()
    vlad_arr = np.zeros((datalen, 256), dtype=np.float32)
    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch, idx) in tqdm(valloader):
            degrad_patch, clean_patch = degrad_patch.to(device=device), clean_patch.to(device=device)
            # combined_patch = torch.cat((degrad_patch, clean_patch), dim=0)
            # start_time = time.time()
            
            restored = net(degrad_patch)
            vlad_out = vlad(restored)
            # print("restored is: ", restored.shape)
            # print("vlad_out is: ", vlad_out.shape)
            
            batsize = degrad_patch.shape[0]
            # print("restored all is: ", restored.shape)
            for batch in range(batsize):
                
                id = idx[batch].detach().cpu().numpy()
                vlad_arr[id] = vlad_out[batch].detach().cpu().numpy()
                
                # print("restored[batch] is :", restored[batch].shape)
                restored_npy = torch_to_np_batch(restored[batch])
                # restored_npy = image_np.detach().cpu().numpy()
                restored_npy = restored_npy.transpose(1, 2, 0)
                restored_npy_handled = process_channels(restored_npy)


                clean_npy = torch_to_np_batch(clean_patch[batch])
                # restored_npy = image_np.detach().cpu().numpy()
                clean_npy = clean_npy.transpose(1, 2, 0)
                
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored_npy_handled, clean_npy)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)
        
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

    return vlad_arr, psnr.avg, ssim.avg





def train(config):
    root = config["data_root"]["data_root_folder"]
    valpath_query = config["val_config"]["datapath_query"]
    valpath_database = config["val_config"]["datapath_database"]
    
    log_folder = config["training_config"]["log_folder"]
    training_seqs = config["training_config"]["training_seqs"]
    patch_height = config["training_config"]["patch_height"]
    patch_width = config["training_config"]["patch_width"]
    
    pretrained_vlad_model = config["training_config"]["pretrained_vlad_model"]
    pretrained_resnet_model = config["training_config"]["pretrained_resnet_model"]
    
    pos_threshold = config["training_config"]["pos_threshold"]
    neg_threshold = config["training_config"]["neg_threshold"]
    batch_size = config["training_config"]["batch_size"]
    all_epochs = config["training_config"]["epoch"]
    save_res_path = config["training_config"]["save_res_path"]
    save_vlad_path = config["training_config"]["save_vlad_path"]

    
    writer = SummaryWriter()
    train_dataset_restore = PromptTrainDataset(root=root, 
                                               seqs=training_seqs, 
                                               patch_height=patch_height, 
                                               patch_width=patch_width)
    trainloader_restore = DataLoader(dataset=train_dataset_restore, batch_size=batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=8)
    print("train_dataset_restore len is: ", len(train_dataset_restore))


    train_dataset_vlad = kitti_dataset(
        root=root,
        seqs=training_seqs,
        pos_threshold=pos_threshold,
        neg_threshold=neg_threshold)
    train_loader_vlad = DataLoader(
        dataset=train_dataset_vlad, batch_size=batch_size,
        shuffle=True, num_workers=8)
    print("train_dataset_vlad len is: ", len(train_dataset_vlad))
    
    

    val_dataset_query = DeweatherDataset(target_path=valpath_query)
    val_loader_query = DataLoader(
        dataset=val_dataset_query, batch_size=6,
        shuffle=True, num_workers=8)
    print("val_dataset query len is: ", len(val_dataset_query))
    
    val_dataset_database = DeweatherDataset(target_path=valpath_database)
    val_loader_database = DataLoader(
        dataset=val_dataset_database, batch_size=6,
        shuffle=True, num_workers=8)
    print("val_dataset database len is: ", len(val_dataset_database))

    
    resnet = ITDNet_D().to(device=device)
    vlad = ITDNet_P().to(device=device)
    

    if not pretrained_vlad_model == "":
        checkpoint = torch.load(pretrained_vlad_model)
        vlad.load_state_dict(checkpoint['state_dict'])

    if not pretrained_resnet_model == "":
        checkpoint = torch.load(pretrained_resnet_model)
        resnet.load_state_dict(checkpoint['state_dict'])
    
    
    criterion_res = nn.L1Loss().to(device=device)
    
    optimizer_restore = torch.optim.AdamW(resnet.parameters(), lr=2e-4)
    scheduler_restore = LinearWarmupCosineAnnealingLR(optimizer=optimizer_restore, warmup_epochs=15, max_epochs=150)

    optimizer_vlad = torch.optim.AdamW(vlad.parameters(), lr=1e-5)
    scheduler_vlad = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vlad, T_max=100,
                                                           eta_min=1e-6)
 

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
        #if(0):
            for param in resnet.parameters():
                param.requires_grad = True
            for param in vlad.parameters():
                param.requires_grad = False
            
            vlad.eval()
            resnet.train()
            
            batch_count = 0
            res_loss_epoch = 0
            vlad_loss_epoch = 0
            for i_batch, sample_batch in tqdm(enumerate(trainloader_restore), total=len(
                    trainloader_restore), desc='Train restore epoch ' + str(epoch), leave=False):
                optimizer_restore.zero_grad()
                
                degrad_patch = sample_batch['degrad_patch'].to(device)
                clean_patch = sample_batch['clean_patch'].to(device)
                
                out_res  = resnet(degrad_patch)
                res_loss = criterion_res(out_res,clean_patch)
                
                degrad_dec =  vlad(out_res)
                clean_dec = vlad(clean_patch)
                

                # # 计算 KL 散度损失
                vlad_loss = torch.norm(degrad_dec - clean_dec, dim=1, p=2)**2
                vlad_loss = vlad_loss.mean()
                
                # print("vlad_loss is: ", vlad_loss.item())

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
            plt.savefig(f'/data1/Code/ITDNet/logs/kitti_03-10_clean/fig/res_loss_vs_epochs.png')
            plt.close()
            
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(vlad_loss_all_eval) + 1), vlad_loss_all_eval, label='Vlad eval Loss', color='blue')
            plt.xlabel('Epochs')
            plt.ylabel('Vlad eval Loss')
            plt.title('Vlad eval Loss vs Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'/data1/Code/ITDNet/logs/kitti_03-10_clean/fig/vlad_eval_loss_vs_epochs.png')
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
                   
                # with torch.no_grad():
                #     degrad_patch = resnet(degrad_patch)
                #     pos_patch = resnet(pos_patch)
                #     neg_patch = resnet(neg_patch)
                                
                input = torch.cat([degrad_patch,
                                pos_patch,
                                neg_patch], dim=0)
                
                out = vlad(input)
                split_size = input.shape[0] // 3
                query_fea, pos_fea, neg_fea = torch.split(
                    out, split_size, dim=0)
                

                train_dataset_vlad.update_latent_vectors(
                    query_fea.detach().cpu().numpy(),
                    sample_batch['id'].detach().cpu().numpy())

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
            plt.savefig(f'/data1/Code/ITDNet/logs/kitti_03-10_clean/fig/Vlad_loss_vs_epochs.png')
            plt.close()

            scheduler_vlad.step()

        
        if (epoch > 50 and epoch%5==0):
            resnet.eval()
            vlad.eval()

            datalen = len(val_dataset_query)
            query_vlad = np.zeros((datalen, 256), dtype=np.float32)
            database_vlad = np.zeros((datalen, 256), dtype=np.float32)

            query_vlad, psnr_query, ssim_query = val_Deweather(resnet,vlad,val_loader_query,datalen)
            database_vlad, _, _ = val_Deweather(resnet,vlad,val_loader_database,datalen)

            if(psnr_query > best_psnr):
                th_min = 0
                th_max = 5
                th_max_pre = 5
                skip = 50
                topk = 1
                
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
                

                res_folder = os.path.join(save_res_path, "Restore_model_lakder")
                if (not os.path.exists(res_folder)):
                    os.makedirs(res_folder)
                
                vlad_folder = os.path.join(save_vlad_path, "Restore_model_lakder")
                if (not os.path.exists(vlad_folder)):
                    os.makedirs(vlad_folder)


                torch.save({'epoch': epoch,
                    'state_dict': resnet.state_dict(),
                    'optimizer': optimizer_restore.state_dict(),
                    'scheduler': scheduler_restore.state_dict() if scheduler_restore else None},
                    os.path.join(res_folder,"Epoch_{}_psnr_{}_ssim_{}.ckpt".format(epoch,
                                                                        psnr_query,
                                                                        ssim_query)))

                torch.save({'epoch': epoch,
                    'state_dict': vlad.state_dict(),
                    'optimizer': optimizer_vlad.state_dict(),
                    'scheduler': scheduler_vlad.state_dict() if scheduler_vlad else None},
                    os.path.join(vlad_folder,"Epoch_{}_recall_{}.ckpt".format(epoch, recalls)))


if __name__ == '__main__':
    config_file = os.path.join(p, 'config/config_kitti.yaml')
    config = yaml.safe_load(open(config_file))
    train(config)
