from .models import Autoencoder
import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, data_ldr, opt, loss_fn):
    """
    Train individual epoch for autoencoder.
    
    Input:   model    - model to be trained
             data_ldr - data loader
             opt      - optimizer
             loss_fn  - loss function
                       
    Returns: avg_loss - average loss for the epoch
    """
    for batch_num, input_data in enumerate(data_ldr):
        loss_epoch = []
        opt.zero_grad()
        X, y = input_data
        X = X.to(device).float()

        output = model(X)
        loss = loss_fn(output, X)
        loss.backward()
        loss_epoch.append(loss.item())

        opt.step()

    avg_loss = np.average(np.asarray(loss_epoch))

    return avg_loss

def train_autoencoder(data_ldr, epochs, enc_length):
    """
    Trains autoencoder
    
    Input:   data_ldr   - torch data loader
             epochs     - number of epochs
             enc_length - encoding length at centre of autoencoder
                       
    Returns: model      - trained model
             losses     - losses at each epoch
    """   
    #Setup
    model = Autoencoder(enc_length).to(device)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    #train
    model.train()
    losses=[]
    for epoch in range(epochs):
        epoch_loss = train_epoch(model, data_ldr, opt, loss_fn)
    
        losses.append(epoch_loss)
        print(epoch, losses[-1])

    return model, np.asarray(losses)
