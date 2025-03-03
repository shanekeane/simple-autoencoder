import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import matplotlib.cm as cm
import numpy as np

def interpolate(model, share, sample1, sample2):
    """
    Uses autoencoder model to get output interpolated between two samples
    
    Input:   model   - autoencoder model
             share   - how much of sample1 to use (0-1)
             sample1 - first sample (1x28x28 torch tensor)
             sample2 - second sample (1x28x28 torch tensor)
                       
    Returns: the interpolated sample (28x28 torch tensor)
    """
    enc1 = model.encoder(sample1.unsqueeze(0))
    enc2 = model.encoder(sample2.unsqueeze(0))
    combined = enc1*share+enc2*(1.0-share)
    dec_combined = model.decoder(combined)
    return dec_combined.squeeze().detach().numpy()

def plot_interpolate(model, share, sample1, sample2):
    """
    Plots: 1. sample1 (1x28x28),
           2. sample2 (1x28x28), 
           3. simple interpolation of s1 and s2 via addition
           4. interpolation obtained via autoencoder
    
   Input:   model   - autoencoder model
            share   - how much of sample1 to use (0-1)
            sample1 - first sample (1x28x28 torch tensor)
            sample2 - second sample (1x28x28 torch tensor)
   
    """   
    fig, ax = plt.subplots(1,4)
    ax[0].imshow(sample1.squeeze(), cmap=plt.cm.binary)
    ax[1].imshow(sample2.squeeze(), cmap=plt.cm.binary)
    ax[2].imshow((share*sample1+(1.0-share)*sample2).squeeze(), cmap=plt.cm.binary)
    int_sample = interpolate(model, share, sample1, sample2)
    ax[3].imshow(int_sample, cmap=plt.cm.binary)
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    ax[3].axis("off")

def get_sample(data_ldr):
    """
    Return random sample from the dataloader set 
    
    Input:  data_ldr - data loader torch

    Output: random sample (1x28x28 tensor)
   
    """   
    sample = np.random.randint(0,len(data_ldr.dataset))
    return data_ldr.dataset[sample][0]

def load_data(data="mnist", batch_size=128, sample="all"):
    """
    Returns data loader with samples of MNIST or Fashion MNIST
    
    Input:  data       - "mnist" or "fmnist"
            batch_size - for dataloader
            sample     - 0-9, or "all"

    Output: data loader
   
    """   

    #Load training and testing data
    if data=="mnist":
        train_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
        test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    elif data=="fmnist":
        train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
        test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    if sample == "all":
        all_data = torch.utils.data.ConcatDataset([train_data, test_data])
 
    else:
        train_data.data = train_data.data[train_data.targets==sample]
        train_data.targets = train_data.targets[train_data.targets==sample]
        test_data.data = test_data.data[test_data.targets==sample]
        test_data.targets = test_data.targets[test_data.targets==sample]
        all_data = torch.utils.data.ConcatDataset([train_data, test_data])
 
    dataloader = DataLoader(all_data, batch_size=batch_size, shuffle=True)
    return dataloader
