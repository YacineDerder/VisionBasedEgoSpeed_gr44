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
import pandas as pd
from sympy import factorint
import multiprocessing
from Datasets import NuImageDataset, clip_to_laneline

# Run only if module is run as a standalone script
if __name__ == '__main__':
    # NuImage Dataset
    mode = "test"
    dataset_loc = r'C:\Users\yacin\Documents\Datasets\NuImage'
    training_data = NuImageDataset(dataset_loc, mode=mode)

    # YOLOP model
    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

    # Test with imshow
    test = False
    if test:
        clip_gray, clip_ll = clip_to_laneline(model, training_data[0][0], imshow=True)
        print(clip_gray.shape)
        print(clip_ll.shape)

    # 3. Make the new dataset and save it to file

    # Define Dataloader
    load = False
    if load:
        features_final = np.load('NN_input.npy')
        labels_final = np.load('NN_output.npy')
    else:
        features_final = np.zeros([len(training_data), 13, 64, 64, 2])
        labels_final = np.zeros(len(training_data))
    
    batch_size = 16
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
    train_iterator = iter(train_dataloader)
    save = False
    
    print(len(training_data))

    # Fill features
    for i in range(len(training_data)//batch_size+1):
        print(f"Processing batch {i+1} / {len(training_data)//batch_size+1}")
        train_features, train_labels = next(train_iterator)
        
        for j in range(train_features.shape[0]):
            clip_gray, clip_ll = clip_to_laneline(model, train_features[j], imshow=False)
            
            features_final[batch_size*i+j, :, :, :, 0] = clip_gray
            features_final[batch_size*i+j, :, :, :, 1] = clip_ll
            labels_final[batch_size*i+j] = train_labels[j]
            
            del clip_gray, clip_ll

        # Save checkpoint dataset as a .npy
        if save:
            np.save("NN_input.npy", features_final)
            np.save("NN_output.npy", labels_final)
            
        del train_features, train_labels
    
    # Save new dataset as a .npy
    if save:
        np.save("NN_input.npy", features_final)
        np.save("NN_output.npy", labels_final)