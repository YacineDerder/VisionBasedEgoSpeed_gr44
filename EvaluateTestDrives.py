from nuimages import NuImages
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.image as mpimg
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import *
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from Datasets import NuImageDataset, clip_to_laneline, PreprocessedDataset
from conv_NN import CNN
from Evaluate_NN import evaluate
import time
import json

save = False
animate = False
print_detail = False

# Load models
LL_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
CNN_model = CNN()
results_loc = r"C:\Users\yacin\Documents\Datasets\DLAV\Results"
exp_name = "Exp75EpochNuImage"
model_name = "model.pth"
CNN_model.load_state_dict(torch.load(os.path.join(results_loc, exp_name, model_name)))

to_process = [2,3,4,5]
for test_nbr in to_process:
    # Load dataset
    dataset_loc = r"C:\Users\yacin\Documents\Datasets\DLAV\Test videos"
    test_name = "Test" + str(test_nbr)
    test_loc = os.path.join(dataset_loc, test_name)

    # Test reading correctly and get shape
    test_file = os.path.join(test_loc, "scene"+str(test_nbr)+"_00001.png")
    img = cv2.imread(test_file)
    img_shape = img.shape

    # Sub-Sampling
    model_freq = 2
    test_freq = 30
    sub_sampling = int(test_freq/model_freq) # 15
    n_model_frames = 13 # NuImage
    n_test_frames = len([entry for entry in os.listdir(test_loc) if os.path.isfile(os.path.join(test_loc, entry))])

    output = []
    count=0
    for count in range(n_test_frames-sub_sampling*(n_model_frames-1)):
        mid_frame_idx = int(np.floor(n_model_frames/2)*sub_sampling+count+1)
        frame_idx = []
        for j in range(n_model_frames):
            frame_idx.append(int(mid_frame_idx - np.floor(n_model_frames/2)*sub_sampling + j*sub_sampling))
        if max(frame_idx) > n_test_frames:
            break

        clip = np.zeros([n_model_frames, img_shape[0], img_shape[1], img_shape[2]], dtype=np.uint8)
        for i in range(len(frame_idx)):
            file = os.path.join(test_loc, "scene"+str(test_nbr)+"_"+str(frame_idx[i]).zfill(5)+".png")
            img = cv2.imread(file)
            clip[i,:,:,:] = img

        if animate:
            frames = [] # for storing the generated images
            fig = plt.figure()
            for j in range(clip.shape[0]):
                frames.append([plt.imshow(clip[j], cmap=cm.Greys_r,animated=True)])
            ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                            repeat_delay=200)
            plt.show()
        
        start_t = time.time()
        ego_speed = evaluate(torch.tensor(clip), CNN_model, LL_model, use_cuda=False).detach().numpy()[0]
        end_t = time.time()
        if print_detail:
            print(f"Estimated ego speed = {ego_speed} [m/s], elapsed time = {end_t - start_t}\n")

        output.append({
            "frame" : mid_frame_idx,
            "ego speed" : float(ego_speed)
        })
        if ((count+1)%10 == 0) or (count+1 == n_test_frames-sub_sampling*(n_model_frames-1)):
            print(f"Clip {count+1}/{n_test_frames-sub_sampling*(n_model_frames-1)}, number of test frames : {n_test_frames}, test dataset : {test_nbr}")

    # Save in .json file
    if save:
        data = {
            "project" : "20. Vision-based Vehicle Speed Estimation",
            "output" : output
        }
        json_path = os.path.join(dataset_loc, test_name+".json")
        with open(json_path, "w") as json_file:
            json.dump(data, json_file)
            json_file.close()

# Run only if module is run as a standalone script
if __name__ == '__main__':
    pass