from .models import Autoencoder
import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, data_ldr, opt, loss_fn):
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
