import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Autoencoder class to be used for generative task.
    
    Input:  enc_length - the length of the encoding of the image 
                         at the centre of the autoencoder. 
   
    """   
    def __init__(self, enc_length):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, 5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(24, 12, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(432, enc_length),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(enc_length, 432),
            nn.ReLU(),
            nn.Unflatten(1,(12,6,6)),
            nn.ConvTranspose2d(12, 24, 5,  stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 1, 5,  stride=2, padding=1, output_padding=1),
            nn.Sigmoid())
    
    def forward(self, x):
        #Encoder
        x = self.encoder(x)
        #Decoder
        x = self.decoder(x)
        
        return x
