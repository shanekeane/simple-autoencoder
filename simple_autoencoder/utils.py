import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import matplotlib.cm as cm
import numpy as np

def interpolate(model, share, sample1, sample2):
    enc1 = model.encoder(sample1.unsqueeze(0))
    enc2 = model.encoder(sample2.unsqueeze(0))
    combined = enc1*share+enc2*(1.0-share)
    dec_combined = model.decoder(combined)
    return dec_combined.squeeze().detach().numpy()

def plot_interpolate(model, share, sample1, sample2):
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
    sample = np.random.randint(0,len(data_ldr.dataset))
    return data_ldr.dataset[sample][0]

def load_data(data="mnist", batch_size=128, sample="all"):
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
