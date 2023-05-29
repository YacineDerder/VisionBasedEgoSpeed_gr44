from nuimages import NuImages
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
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
import time

save = False
animate = True

# Dataset and network definition
dataset_loc = r"C:\Users\yacin\Documents\Datasets\DLAV"
batch_size = 10

def evaluate(clip, CNN_model, LL_model, use_cuda=False):
    # t1 = time.time()
    if use_cuda:
        clip = clip.cuda()
        LL_model = LL_model.cuda()
        CNN_model = CNN_model.cuda()
    clip_gray, clip_ll = clip_to_laneline(LL_model, clip, imshow=False)
    # t2 = time.time()
    # print(f"clip_to_laneline time = {t2 - t1} [s]")
    CNN_input = np.zeros([clip.shape[0], 64, 64, 2])
    CNN_input[:, :, :, 0] = clip_gray
    CNN_input[:, :, :, 1] = clip_ll
    CNN_input = torch.from_numpy(CNN_input)[None, :].float() # Cast to tensor and add a dimension
    # t3 = time.time()
    # print(f"torch.from_numpy time = {t3 - t2} [s]")

    output = CNN_model(CNN_input).flatten()
    # t4 = time.time()
    # print(f"CNN_model time = {t4 - t3} [s]")
    return output

def eval_test_dataset(results_loc, exp_name, model_name):
    # Import preprocessed test dataset
    data_test = PreprocessedDataset(dataset_loc, dataset='NuImage', mode="test")
    dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    # Load pretrained model (75 epoch, NuImage)
    model = CNN()
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(results_loc, exp_name, model_name)))

    # Define error metric
    running_loss = []

    # Evaluate whole (preprocessed) test dataset
    for i, (clip, label) in enumerate(dataloader_test):
        clip = clip.to(torch.float32).cuda()
        label = label.to(torch.float32).cuda()

        output = model(clip).flatten().cpu()
        label = label.cpu()
        loss = np.sqrt(np.sum((label.detach().numpy() - output.detach().numpy())**2)/batch_size)
        print(f"Batch = {i+1}, Loss = {loss}")
        running_loss.append(loss)

    print(f"Average loss over test dataset = {np.mean(running_loss)}")

# Run only if module is run as a standalone script
if __name__ == '__main__':
    # Evaluate whole test dataset
    results_loc = r"C:\Users\yacin\Documents\Datasets\DLAV\Results"
    exp_name = "Exp75EpochNuImage"
    model_name = "model.pth"
    # eval_test_dataset(results_loc, exp_name, model_name)

    # Load models
    LL_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
    CNN_model = CNN()
    CNN_model.load_state_dict(torch.load(os.path.join(results_loc, exp_name, model_name)))

    dataset_loc = r'C:\Users\yacin\Documents\Datasets\NuImage'
    n_tests = 5

    if not(animate):
        # Load train and test samples
        data_train = NuImageDataset(dataset_loc, mode="train")
        dataloader_train = DataLoader(data_train, batch_size=1, shuffle=True)
        data_test = NuImageDataset(dataset_loc, mode="test")
        dataloader_test = DataLoader(data_test, batch_size=1, shuffle=True  )

        for i in range(n_tests):
            # Extract new clips and labels from dataset
            clip, label = next(iter(dataloader_train))
            clip = clip.squeeze()
            label = label.squeeze()

            # Evaluate the chosen clip
            start_t = time.time()
            output = evaluate(clip, CNN_model, LL_model, use_cuda=False).detach().numpy()[0]
            end_t = time.time()
            print(f"################# TRAIN SAMPLE {i+1} #################")
            print(f"Total evaulate function time = {end_t - start_t} [s]")
            print(f"Estimated ego speed = {output} [m/s], actual speed = {label} [m/s], error = {np.absolute(output-label)} [m/s] \n")

        for i in range(n_tests):
            # Extract new clips and labels from dataset
            clip, label = next(iter(dataloader_test))
            clip = clip.squeeze()
            label = label.squeeze()

            # Evaluate the chosen clip
            start_t = time.time()
            output = evaluate(clip, CNN_model, LL_model, use_cuda=False).detach().numpy()[0]
            end_t = time.time()
            print(f"################# TEST SAMPLE {i+1} #################")
            print(f"Total evaulate function time = {end_t - start_t} [s]")
            print(f"Estimated ego speed = {output} [m/s], actual speed = {label} [m/s], error = {np.absolute(output-label)} [m/s] \n")

    if animate:
        # Load test samples
        data_test = NuImageDataset(dataset_loc, mode="test")
        dataloader_test = DataLoader(data_test, batch_size=1, shuffle=True)

        clip, label = next(iter(dataloader_test))
        clip = clip.squeeze()
        label = label.squeeze()

        frames = [] # for storing the generated images
        fig = plt.figure()
        for j in range(clip.shape[0]):
            frames.append([plt.imshow(clip[j], cmap=cm.Greys_r,animated=True)])
        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                        repeat_delay=200)
        if save:
            ani.save("DataInput.gif")

        plt.show()

        # Evaluate the chosen clip
        start_t = time.time()
        output = evaluate(clip, CNN_model, LL_model, use_cuda=False).detach().numpy()[0]
        end_t = time.time()
        print(f"################# TEST SAMPLE #################\n")
        print(f"Total evaulate function time = {end_t - start_t} [s]")
        print(f"Estimated ego speed = {output} [m/s], actual speed = {label} [m/s], error = {np.absolute(output-label)} [m/s] \n")

    
