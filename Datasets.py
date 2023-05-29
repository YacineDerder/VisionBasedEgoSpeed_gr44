from nuimages import NuImages
import os
import sys
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as T
import torch.nn.functional as F
import time

# 1. NuImage dataset
class NuImageDataset(Dataset):
    def __init__(self, dataset_loc, mode='train', transform=None, target_transform=None):
        
        version = 'v1.0-' + mode
        self.nuim = NuImages(dataroot=dataset_loc, version=version, verbose=True, lazy=True)
        self.dataset_loc = self.nuim.dataroot
        self.transform = transform
        self.target_transform = target_transform
        
        self.all_kfst = [] # Key Frame Samples Tokens
        for i in self.nuim.sample:
            i_kc = self.nuim.get('sample_data', i['key_camera_token'])
            i_sd = self.nuim.get_sample_content(i['token'])
            if i_kc["filename"][8:18] == "CAM_FRONT/" and len(i_sd) == 13:
                self.all_kfst.append(i['token'])

    def __len__(self):
        return len(self.all_kfst)

    def __getitem__(self, idx):
        
        images = []
        label = 0
        for idx, s_d_t in enumerate(self.nuim.get_sample_content(self.all_kfst[idx])):
            s_d = self.nuim.get('sample_data', s_d_t)
            #image = cv2.imread(os.path.join(self.dataset_loc, s_d['filename']), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(os.path.join(self.dataset_loc, s_d['filename']))
            #image = cv2.resize(image, (64,64))
            if self.transform:
                image = self.transform(image)
            images.append(image)
            if idx == 6:
                label = self.nuim.get('ego_pose', s_d['ego_pose_token'])['speed']

        images = np.array(images)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return images, label

# 2. YOLOP Network
def clip_to_laneline(laneline_NN, clip, imshow=False):
    """
    in :
    LaneLine_NN is the NN that performes the laneline determination on a given image batch
    Clip is a 13 frames (for NuImages) sequence of 1600x900x3 frames given as a tenor in a [D,H,W,C] format
    out :
    a clip of 13 binary frames (64x64) : the output of the LaneLine_NN
    """

    # t1 = time.time()
    clip = clip.permute(0,3,1,2)
    clip = clip.to(torch.float32)
    # t2 = time.time()
    # print(f"clip.to(torch.float32) time = {t2 - t1} [s]")
    clip = F.interpolate(clip, size=(64, 64), mode='bilinear', align_corners=False)
    # t3 = time.time()
    # print(f"F.interpolate(clip) time = {t3 - t2} [s]")
    _, _,ll_seg_out = laneline_NN(clip)
    # t4 = time.time()
    # print(f"laneline_NN(clip) time = {t4 - t3} [s]")
    clip_ll = ll_seg_out
    clip_ll = clip_ll.permute(0,2,3,1)
    
    clip_resize = F.interpolate(clip, size=(64, 64), mode='bilinear', align_corners=False)
    clip_resize = clip_resize.permute(0,2,3,1).numpy().astype(np.uint8)
    
    clip_gray = np.zeros([clip_resize.shape[0], clip_resize.shape[1], clip_resize.shape[2]])
    for i in range(clip_resize.shape[0]):
        clip_gray[i,:,:] = cv2.cvtColor(clip_resize[i], cv2.COLOR_BGR2GRAY)

    if imshow: # Show the middle image of the clip
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        img_idx = 6
        clip_ll_show = clip_ll[img_idx].detach().numpy()
        ax[0].imshow(clip_gray[img_idx], cmap='gray')
        ax[1].imshow(clip_ll_show[:,:,1], cmap="gray")
        plt.show()
        
    return clip_gray, clip_ll.detach().numpy()[:,:,:,1]

# 3. NuImage preprocessed dataset
class PreprocessedDataset(Dataset):
    def __init__(self, dataset_loc, dataset='NuImage', mode='train'):
        
        file_in_name = dataset + "_input_" + mode + ".npy"
        file_out_name = dataset + "_output_" + mode + ".npy"
        file_in_loc = os.path.join(dataset_loc, file_in_name)
        file_out_loc = os.path.join(dataset_loc, file_out_name)
        self.dataset_in = np.load(file_in_loc)
        self.dataset_out = np.load(file_out_loc)
        
    def __len__(self):
        return len(self.dataset_out)

    def __getitem__(self, idx):
        
        images = self.dataset_in[idx]
        labels = self.dataset_out[idx]
            
        return images, labels

# Run only if module is run as a standalone script
if __name__ == '__main__':
    pass