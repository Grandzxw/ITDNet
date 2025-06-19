from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from scipy.linalg import norm
from tqdm import tqdm
import threading
from glob import glob
from torchvision.transforms import ToTensor
from tools.image_utils import crop_img, random_augmentation
import torch.distributed as dist


# /home/hit/sda/Dataset/Ori_KITTI/sequences/00/00.txt
# /home/hit/sda/place_recognition/dataset_overlapnetvlad/
class kitti_dataset(Dataset):
    def __init__(self, root, seqs, pos_threshold, neg_threshold) -> None:
        super().__init__()
        self.root = root
        self.seqs = seqs
        self.poses = []
        for seq in seqs:
            pose = np.genfromtxt(os.path.join("/data1/KITTI/sequences", seq ,seq + '.txt'))[:, [3, 11]]
            self.poses.append(pose)
        self.pairs = {}
        key = 0
        acc_num = 0
        for i in range(len(self.poses)):
            pose = self.poses[i]
            inner = 2 * np.matmul(pose, pose.T)
            xx = np.sum(pose**2, 1, keepdims=True)
            dis = xx - inner + xx.T
            dis = np.sqrt(np.abs(dis))
            id_pos = np.argwhere((dis < pos_threshold) & (dis > 0))
            id_neg = np.argwhere(dis < neg_threshold)
            for j in range(len(pose)):
                positives = id_pos[:, 1][id_pos[:, 0] == j] + acc_num
                negatives = id_neg[:, 1][id_neg[:, 0] == j] + acc_num
                self.pairs[key] = {
                    "query_seq": i,
                    "query_id": j,
                    "positives": positives.tolist(),
                    "negatives": set(
                        negatives.tolist())}
                key += 1
            acc_num += len(pose)
        self.all_ids = set(range(len(self.pairs)))
        self.traing_latent_vectors = [None] * len(self.pairs)

    def get_random_positive(self, idx):
        positives = self.pairs[idx]["positives"]
        randid = random.randint(0, len(positives) - 1)

        return positives[randid]


    def get_random_negative(self, idx):
        negatives = list(self.all_ids - self.pairs[idx]["negatives"])
        randid = random.randint(0, len(negatives) - 1)

        return negatives[randid]


    def get_random_hard_positive(self, idx):
        random_pos = self.pairs[idx]["positives"]
        qurey_vec = self.traing_latent_vectors[idx]
        if qurey_vec is None:
            randid = random.randint(0, len(random_pos) - 1)
            return random_pos[randid]

        latent_vecs = []
        for j in range(len(random_pos)):
            latent_vecs.append(self.traing_latent_vectors[random_pos[j]])
 
        latent_vecs = np.array(latent_vecs)
        query_vec = self.traing_latent_vectors[idx]
        query_vec = query_vec.reshape(1, -1)
        query_vec = np.repeat(query_vec, latent_vecs.shape[0], axis=0)
        diff = query_vec - latent_vecs
        diff = np.linalg.norm(diff, axis=1)
        maxid = np.argmax(diff)

        return random_pos[maxid]



    def get_random_hard_negative(self, idx):
        random_neg = list(self.all_ids - self.pairs[idx]["negatives"])
        qurey_vec = self.traing_latent_vectors[idx]
        if qurey_vec is None:
            randid = random.randint(0, len(random_neg) - 1)
            return random_neg[randid]

        latent_vecs = []

        # print(f"random_neg length: {len(random_neg)}")

        for j in range(len(random_neg)):
            latent_vecs.append(self.traing_latent_vectors[random_neg[j]])

        # print(f"latent_vecs type: {type(latent_vecs)}")
        # print(f"latent_vecs length: {len(latent_vecs)}")

        # if len(latent_vecs) == 0:
        #     print(f"[RANK {dist.get_rank()}] Empty latent_vecs for query {idx}")

        # print(f"[RANK {dist.get_rank()}] latent_vecs shape: {[np.array(v).shape for v in latent_vecs]}")

        # print(f"latent_vecs shape: {[np.array(v).shape for v in latent_vecs]}")


        latent_vecs = np.array(latent_vecs)
        query_vec = self.traing_latent_vectors[idx]
        query_vec = query_vec.reshape(1, -1)
        query_vec = np.repeat(query_vec, latent_vecs.shape[0], axis=0)
        diff = query_vec - latent_vecs
        diff = np.linalg.norm(diff, axis=1)
        minid = np.argmin(diff)

        return random_neg[minid]



    def get_other_neg(self, id_pos, id_neg):
        random_neg = list(
            self.all_ids -
            self.pairs[id_pos]["negatives"] -
            self.pairs[id_neg]["negatives"])
        randid = random.randint(0, len(random_neg) - 1)

        return random_neg[randid]




    def update_latent_vectors_dis(self, fea, idx):
        # Step 1: 转换 `idx` 和 `fea` 为 `tensor`
        idx_tensor = torch.tensor(idx, dtype=torch.int64, device="cuda")
        fea_tensor = torch.tensor(fea, dtype=torch.float32, device="cuda")

        # Step 2: 使用 `all_gather()` 同步所有 GPU 的 `idx` 和 `fea`
        world_size = dist.get_world_size()
        gathered_idx = [torch.zeros_like(idx_tensor) for _ in range(world_size)]
        gathered_fea = [torch.zeros_like(fea_tensor) for _ in range(world_size)]

        dist.all_gather(gathered_idx, idx_tensor)
        dist.all_gather(gathered_fea, fea_tensor)

        # Step 3: 在 rank 0 上整合所有 `rank` 采样到的 `latent_vectors`
        if dist.get_rank() == 0:
            for r in range(world_size):
                for i in range(len(gathered_idx[r])):
                    self.traing_latent_vectors[gathered_idx[r][i].item()] = gathered_fea[r][i].cpu().numpy()



    def update_latent_vectors(self, fea, idx):
        for i in range(len(idx)):
            self.traing_latent_vectors[idx[i]] = fea[i]


    def load_fea(self, idx, query_ot = True):
        query = self.pairs[idx]
        seq = self.seqs[query["query_seq"]]
        id = str(query["query_id"]).zfill(6)
        
        if query_ot:
            file = os.path.join(self.root,seq, id + '.npy')
        else:
            file = os.path.join(self.root.replace("Weather_KITTI_PR_RANGE", "ORI_KITTI_PR_RANGE"),seq, id + '.npy')

        # print(file)
        ri = np.load(file)
        
        # ri = np.transpose(ri, (2, 0, 1))
        # ri = ri.astype('float32')
        return ri


    def _crop_patch(self, img_1, pos_img , neg_img):
        H = img_1.shape[0]
        W = img_1.shape[1]
        
        patch_height = 48
        patch_width = 720
        
        # Ensure the patch size is within the image dimensions
        if H < patch_height or W < patch_width:
            raise ValueError("Patch size is larger than the image dimensions.")

        ind_H = random.randint(0, H - patch_height)
        ind_W = random.randint(0, W - patch_width)

        patch_1 = img_1[ind_H:ind_H + patch_height, ind_W:ind_W + patch_width]
        
        patch_3 = pos_img[ind_H:ind_H + patch_height, ind_W:ind_W + patch_width]
        patch_4 = neg_img[ind_H:ind_H + patch_height, ind_W:ind_W + patch_width]

        return patch_1, patch_3, patch_4



    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        queryid = idx % len(self.pairs)
        negid = self.get_random_hard_negative(queryid)
        posid = self.get_random_hard_positive(queryid)

        id = str(queryid).zfill(6)
        
        # file = os.path.join(self.root.replace("Weather_KITTI_PR_RANGE", "ORI_KITTI_PR_RANGE"),seq, id + '.npy')
        # true_fea = np.load(file)
        # true_fea = np.transpose(true_fea, (2, 0, 1))
        # true_fea = true_fea.astype('float32')

        query_fea = self.load_fea(queryid, query_ot = False)
        pos_fea = self.load_fea(posid, query_ot = False)
        neg_fea = self.load_fea(negid, query_ot = False)
        # other_fea = self.load_fea(otherid, query_ot = False)
        # print(query_fea.shape, pos_fea.shape, neg_fea.shape, other_fea.shape)

        degrad_patch, pos_patch, neg_patch = self._crop_patch(query_fea, pos_fea, neg_fea)

        degrad_patch = np.transpose(degrad_patch, (2, 0, 1))
        degrad_patch = degrad_patch.astype('float32')

        pos_patch = np.transpose(pos_patch, (2, 0, 1))
        pos_patch = pos_patch.astype('float32')

        neg_patch = np.transpose(neg_patch, (2, 0, 1))
        neg_patch = neg_patch.astype('float32')
        
        return {
            "id": queryid,
            "query_desc": degrad_patch,
            "pos_desc": pos_patch,
            "neg_desc": neg_patch}





class PromptTrainDataset(Dataset):
    def __init__(self, root, seqs, patch_height, patch_width):
        super(PromptTrainDataset, self).__init__()
        self.root = root
        self.degra_ids = []
        self.seqs = seqs
        self.patch_height = patch_height
        self.patch_width = patch_width
        for seq in self.seqs:
            bin_files = (
                glob(
                    os.path.join(
                        self.root,
                        str(seq).zfill(2),
                        "*.npy",
                    )
                )
            )
            bin_files_sorted = sorted(bin_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.degra_ids.extend(bin_files_sorted)

        self.toTensor = ToTensor()
    
    
    def __len__(self):
        return len(self.degra_ids)


    def load_and_crop_npy(self, file_path, base=16):
        """
        读取.npy文件并裁剪图像
        """
        image = np.load(file_path)  # 加载.npy文件
        # 调整图像维度从 H*W*2 变为 2*H*W
        # image = np.transpose(image, (2, 0, 1))
        cropped_image = crop_img(image, base)
        return cropped_image



    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        
        # Ensure the patch size is within the image dimensions
        if H < self.patch_height or W < self.patch_width:
            raise ValueError("Patch size is larger than the image dimensions.")

        ind_H = random.randint(0, H - self.patch_height)
        ind_W = random.randint(0, W - self.patch_width)

        patch_1 = img_1[ind_H:ind_H + self.patch_height, ind_W:ind_W + self.patch_width]
        patch_2 = img_2[ind_H:ind_H + self.patch_height, ind_W:ind_W + self.patch_width]
        

        return patch_1, patch_2


    # /home/hit/sda/Dataset/Weather_KITTI_PR_RANGE/00/000000.npy
    # /home/hit/sda/Dataset/ORI_KITTI_PR_RANGE/00/000000.npy

    def __getitem__(self, idx):
        
        degra_queryid = idx
        degrad_img = self.load_and_crop_npy(self.degra_ids[degra_queryid], base=16)
        clean_querypath = self.degra_ids[degra_queryid].replace("Weather_KITTI_PR_RANGE", "ORI_KITTI_PR_RANGE")
        # print(degra_queryid)
        # print("degra_queryid is: ", self.degra_ids[degra_queryid])
        # print("clean_querypath is: ", clean_querypath)
        
        clean_img = self.load_and_crop_npy(clean_querypath, base=16)
        
        # print("pos_path is: ", pos_path)
        # print("neg_path is: ", neg_path)
        
        ## 暂时先这样
        degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        # clean_patch = self.toTensor(clean_patch)
        # degrad_patch = self.toTensor(degrad_patch)

        # 将数据转换为张量
        clean_patch = self.toTensor(clean_patch).float()  # 转换为 FloatTensor 并放到 GPU
        degrad_patch = self.toTensor(degrad_patch).float()  # 转换为 FloatTensor 并放到 GPU

        return {
            "degra_queryid": degra_queryid,
            "degrad_patch": degrad_patch,
            "clean_patch": clean_patch}



class DeweatherDataset(Dataset):
    def __init__(self, target_path=None):
        super(DeweatherDataset, self).__init__()
        self.ids = []
        self.toTensor = ToTensor()
        self.target_path = target_path
        bin_files = (
            glob(
                os.path.join(
                    self.target_path,
                    "*.npy",
                )
            )
        )
        bin_files_sorted = sorted(bin_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.ids.extend(bin_files_sorted)


    def load_and_crop_npy(self, file_path, base=16):
        image = np.load(file_path)  # 加载.npy文件
        # 调整图像维度从 H*W*2 变为 2*H*W
        # image = np.transpose(image, (2, 0, 1))
        cropped_image = crop_img(image, base)
        return cropped_image

    def _get_gt_path(self, degraded_name):
        #print("degraded_name is: ", degraded_name)
        gt_name = degraded_name.replace("Weather_KITTI_PR_RANGE", "ORI_KITTI_PR_RANGE")
        #print("gt_name is: ", gt_name)
        return gt_name


    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        
        patch_height = 48
        patch_width = 720
        # Ensure the patch size is within the image dimensions
        if H < patch_height or W < patch_width:
            raise ValueError("Patch size is larger than the image dimensions.")
        ind_H = random.randint(0, H - patch_height)
        ind_W = random.randint(0, W - patch_width)
        patch_1 = img_1[ind_H:ind_H + patch_height, ind_W:ind_W + patch_width]
        patch_2 = img_2[ind_H:ind_H + patch_height, ind_W:ind_W + patch_width]
        
        return patch_1, patch_2


    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)
        degraded_img = self.load_and_crop_npy(degraded_path, base=16)
        clean_img = self.load_and_crop_npy(clean_path, base=16)
        # degraded_img, clean_img = self._crop_patch(degraded_img, clean_img)
        
        clean_img, degraded_img = self.toTensor(clean_img).float(), self.toTensor(degraded_img).float()
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img, idx


    def __len__(self):
        return len(self.ids)









class DeweatherDataset_res(Dataset):
    def __init__(self, target_path=None):
        super(DeweatherDataset_res, self).__init__()
        self.ids = []
        self.toTensor = ToTensor()
        self.target_path = target_path
        bin_files = (
            glob(
                os.path.join(
                    self.target_path,
                    "*.npy",
                )
            )
        )
        bin_files_sorted = sorted(bin_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.ids.extend(bin_files_sorted)


    def load_and_crop_npy(self, file_path, base=16):
        image = np.load(file_path)  # 加载.npy文件
        # 调整图像维度从 H*W*2 变为 2*H*W
        # image = np.transpose(image, (2, 0, 1))
        cropped_image = crop_img(image, base)
        return cropped_image

    def _get_gt_path(self, degraded_name):
        # print("degraded_name is: ", degraded_name)
        gt_name = degraded_name.replace("Weather_KITTI_PR_RANGE", "ORI_KITTI_PR_RANGE")
        # print("gt_name is: ", gt_name)
        return gt_name


    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        
        patch_height = 48
        patch_width = 720
        # Ensure the patch size is within the image dimensions
        if H < patch_height or W < patch_width:
            raise ValueError("Patch size is larger than the image dimensions.")
        ind_H = random.randint(0, H - patch_height)
        ind_W = random.randint(0, W - patch_width)
        patch_1 = img_1[ind_H:ind_H + patch_height, ind_W:ind_W + patch_width]
        patch_2 = img_2[ind_H:ind_H + patch_height, ind_W:ind_W + patch_width]
        
        return patch_1, patch_2


    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)
        degraded_img = self.load_and_crop_npy(degraded_path, base=16)
        clean_img = self.load_and_crop_npy(clean_path, base=16)
        # degraded_img, clean_img = self._crop_patch(degraded_img, clean_img)
        
        clean_img, degraded_img = self.toTensor(clean_img).float(), self.toTensor(degraded_img).float()
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img, idx


    def __len__(self):
        return len(self.ids)







if __name__ == "__main__":
    pass
