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
import pandas as pd


# /home/hit/sda/Dataset/Ori_KITTI/sequences/00/00.txt
# /home/hit/sda/place_recognition/dataset_overlapnetvlad/
class boreas_dataset(Dataset):
    def __init__(self, root, seqs, pos_threshold, neg_threshold) -> None:
        super().__init__()
        self.root = root
        self.pairs = {}
        self.query_pose = pd.read_csv("/data1/Boreas/transform/range_image_vlad/test2/lidar_poses.csv").iloc[:5300]
        self.data_pose = pd.read_csv("/data1/Boreas/transform/range_image_vlad/test1/lidar_poses.csv").iloc[:5700]
        key = 0
        for j in range(len(self.query_pose)):
            pose_i = self.query_pose.iloc[j][['easting', 'northing']].values.reshape(1, -1)
            data_poses = self.data_pose[['easting', 'northing']].values
            distances = np.sqrt(np.sum((data_poses - pose_i)**2, axis=1))
            id_pos = np.argwhere((distances < pos_threshold) & (distances > 0))
            id_neg = np.argwhere(distances < neg_threshold)
            self.pairs[key] = {
                "query_id": j,
                "positives": id_pos.flatten().tolist(),
                "negatives": set(id_neg.flatten().tolist())}
            key += 1
        self.all_ids = set(range(len(self.pairs)))
        self.data_ids = set(range(len(self.data_pose)))
        self.traing_latent_vectors = [None] * len(self.pairs)
        self.traing_latent_vectors_data = [None] * len(self.data_pose)


    def get_random_hard_positive(self, idx):
        random_pos = self.pairs[idx]["positives"]

        qurey_vec = self.traing_latent_vectors[idx]
        if qurey_vec is None:
            randid = random.randint(0, len(random_pos) - 1)
            return random_pos[randid]

        latent_vecs = []
        for j in range(len(random_pos)):
            vec = self.traing_latent_vectors_data[random_pos[j]]
            if vec is not None:
                latent_vecs.append(vec)
 
        latent_vecs = np.array(latent_vecs)
        query_vec = self.traing_latent_vectors[idx]
        query_vec = query_vec.reshape(1, -1)
        query_vec = np.repeat(query_vec, latent_vecs.shape[0], axis=0)
        diff = query_vec - latent_vecs
        diff = np.linalg.norm(diff, axis=1)
        maxid = np.argmax(diff)

        return random_pos[maxid]


    def get_random_hard_negative(self, idx):
        random_neg = list(self.data_ids - self.pairs[idx]["negatives"])
        qurey_vec = self.traing_latent_vectors[idx]
        if qurey_vec is None:
            randid = random.randint(0, len(random_neg) - 1)
            return random_neg[randid]

        latent_vecs = []
        for j in range(len(random_neg)):
            vec = self.traing_latent_vectors_data[random_neg[j]]
            if vec is not None:
                latent_vecs.append(vec)
            # latent_vecs.append(self.traing_latent_vectors_data[random_neg[j]])

        latent_vecs = np.array(latent_vecs)
        query_vec = self.traing_latent_vectors[idx]
        query_vec = query_vec.reshape(1, -1)
        query_vec = np.repeat(query_vec, latent_vecs.shape[0], axis=0)
        diff = query_vec - latent_vecs
        diff = np.linalg.norm(diff, axis=1)
        minid = np.argmin(diff)

        return random_neg[minid]


    def update_latent_vectors(self, fea, idx):
        for i in range(len(idx)):
            self.traing_latent_vectors[idx[i]] = fea[i]


    def update_latent_vectors_data(self, fea, idx):
        for i in range(len(idx)):
            self.traing_latent_vectors_data[idx[i]] = fea[i]


    def load_fea_query(self, idx):
        query_id = self.query_pose['GPSTime'][idx]
        file = os.path.join(self.root,str(query_id) + '.npy')
        # print("idx is:", idx)
        # print("query file is:", file)
        ri = np.load(file)
        return ri

    def load_fea_data(self, idx):
        data_id = self.data_pose['GPSTime'][idx]
        file = os.path.join(self.root.replace("Rangeimage_Clean_test2_5300","Rangeimage_Clean_test1_5700"),str(data_id) + '.npy')
        ri = np.load(file)
        return ri


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        queryid = idx % len(self.pairs)
        query_fea = self.load_fea_query(queryid)

        negid = self.get_random_hard_negative(queryid)
        posid = self.get_random_hard_positive(queryid)

        pos_fea = self.load_fea_data(posid)
        neg_fea = self.load_fea_data(negid)

        query_fea = np.transpose(query_fea, (2, 0, 1))
        query_fea = query_fea.astype('float32')

        pos_fea = np.transpose(pos_fea, (2, 0, 1))
        pos_fea = pos_fea.astype('float32')

        neg_fea = np.transpose(neg_fea, (2, 0, 1))
        neg_fea = neg_fea.astype('float32')
        
        return {
            "query_id": queryid,
            "query_desc": query_fea,
            "pos_id": posid,
            "pos_desc": pos_fea,
            "neg_id": negid,
            "neg_desc": neg_fea}






class PromptTrainDataset(Dataset):
    def __init__(self, root, seqs, patch_height, patch_width):
        super(PromptTrainDataset, self).__init__()
        self.root = root
        # self.args = args
        self.degra_ids = []
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.seqs = seqs
        self.bin_mapping = {
                        "02": "Weather_Range_heavy_snow",
                        "03": "Weather_Range_light_snow",
                        "04": "Weather_Range_rain_test1",
                        "05": "Weather_Range_rain_test2",
                    }
        self.match_mapping = {
                        "02": "matched_results_under_heavy_snow",
                        "03": "matched_results_under_light_snow",
                        "04": "matched_results_under_rain_test1",
                        "05": "matched_results_under_rain_test2",
                    }
        
        all_lidar_data = []
        for seq in self.seqs:
            seq_name = self.bin_mapping.get(seq, seq)
            seq_csv = self.match_mapping.get(seq, seq)
            bin_files = (
                glob(
                    os.path.join(
                        self.root,
                        seq_name,
                        "*.npy",
                    )
                )
            )
            bin_files_sorted = sorted(bin_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.degra_ids.extend(bin_files_sorted)
            df_lidar = pd.read_csv(os.path.join(self.root,seq_csv+".csv"))
            df_lidar["seq"] = seq
            all_lidar_data.append(df_lidar)
        
        self.df_lidar = pd.concat(all_lidar_data, ignore_index=True)
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


    def __getitem__(self, idx):

        degra_queryid = idx
        degrad_img = self.load_and_crop_npy(self.degra_ids[degra_queryid], base=16)

        degra_timestamp = os.path.basename(self.degra_ids[degra_queryid]).split('.')[0]

        matched_row = self.df_lidar[self.df_lidar['GPSTime'].astype(str) == degra_timestamp]
        clean_timestamp = matched_row['matched_GPSTime'].iloc[0]

        clean_dir = self.root.replace("range_image_restore", "range_image_vlad")

        clean_path = os.path.join(clean_dir, "Rangeimage_Clean_test1_5700", str(clean_timestamp)  + ".npy")
        # print("clean_path is: ", clean_path)
        clean_img = self.load_and_crop_npy(clean_path, base=16)

        degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch).float()  # 转换为 FloatTensor 并放到 GPU
        degrad_patch = self.toTensor(degrad_patch).float()  # 转换为 FloatTensor 并放到 GPU

        # return degrad_patch, clean_patch, degra_queryid
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


    def __getitem__(self, idx):

        degraded_path = self.ids[idx]
        degraded_img = self.load_and_crop_npy(degraded_path, base=16)        

        degraded_img = self.toTensor(degraded_img).float()
        
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, idx


    def __len__(self):
        return len(self.ids)



if __name__ == "__main__":
    pass
