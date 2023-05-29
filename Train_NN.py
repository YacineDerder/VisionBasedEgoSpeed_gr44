from nuimages import NuImages
import os
import sys
import matplotlib.pyplot as plt
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import *
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from Datasets import NuImageDataset, clip_to_laneline, PreprocessedDataset
from conv_NN import CNN
print("Imported without error")

# Dataset and network definition
dataset_loc = r"C:\Users\yacin\Documents\Datasets\DLAV"
batch_size = 10

data_train = PreprocessedDataset(dataset_loc, dataset='NuImage', mode="train")
dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)

data_test = PreprocessedDataset(dataset_loc, dataset='NuImage', mode="test")
dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)

train = bool(int(sys.argv[1]))
save = bool(int(sys.argv[2]))
print(f"train = {train}, save = {save}")

#Definition of hyperparameters
num_epochs = 75

# Create CNN
model = CNN()
model.cuda()
print(model)

# Cross Entropy Loss 
error = nn.MSELoss()

# SGD Optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# CNN model training
if train:
    count = 0
    train_loss_list = []
    test_loss_list = []
    iteration_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader_train):
            
            # images = images.to(torch.float32)
            # labels = labels.to(torch.float32)

            train = Variable(images.to(torch.float32)).cuda()
            labels = Variable(labels.to(torch.float32)).cuda()
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(train).flatten()
            # Calculate softmax and ross entropy loss
            train_loss = error(outputs, labels)
            # Calculating gradients
            train_loss.backward()
            # Update parameters
            optimizer.step()

            del train, labels, outputs
            
            count += 1
            if count % 50 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0

                # Iterate through test batch
                images, labels = next(iter(dataloader_test))
                test = Variable(images.to(torch.float32)).cuda()
                labels = Variable(labels.to(torch.float32)).cuda()

                # Forward propagation
                outputs = model(test).flatten()
                # 
                test_loss = error(outputs, labels)

                del test, labels, outputs
                
                # # store loss and iteration
                train_loss_list.append(train_loss.data.cpu())
                test_loss_list.append(test_loss.data.cpu())
                iteration_list.append(count)

                print(f"epoch = {epoch+1}/{num_epochs}, batch = {i+1}/{len(data_train)//batch_size+1}")
                
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  train_loss: {}  test_loss: {}'.format(count, train_loss.data, test_loss.data))
                if save:
                    np.save("iteration.npy", np.array(iteration_list))
                    np.save("train_loss.npy", np.array(train_loss_list))
                    np.save("test_loss.npy", np.array(test_loss_list))
                    torch.save(model.state_dict(), "model.pth")

    if save:
        np.save("iteration.npy", np.array(iteration_list))
        np.save("train_loss.npy", np.array(train_loss_list))
        np.save("test_loss.npy", np.array(test_loss_list))
        torch.save(model.state_dict(), "model.pth")
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(iteration_list, train_loss_list)
    ax[1].plot(iteration_list, test_loss_list)
    plt.show()

else: # Random tests when not training
    x = [10,10,9,9,8,8,6,6,6,5,5,4,3]
    y = [0,1,2,3,4,5,6,7,8,9,10,11,12]

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(x,y)
    ax[1].plot(y,x)
    plt.show()

# Run only if module is run as a standalone script
if __name__ == '__main__':
    pass
    