# vanilla-GAN
Autoencoder used for basic image generation. Randomly chooses two samples from MNIST or Fashion MNIST and creates something "between" them. 

# Description
- May choose "mnist" or "fmnist" for the training set.
- Outputs a generator to generate samples similar to the training set.
- May choose how much of sample1 to include (e.g. 0.4 means 0.4 of sample1 and 0.6 of sample 2

# Packages
Works with:
- Python 3.10.12
- PyTorch 2.5.1+cu124
