import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2

import librosa
import librosa.display
import skimage.io

# number of workers
workers = 2

# batch size
batch_size = 64

# image size
image_size = 128

# number of channels 
nc = 1

# length of latent/random vector
nz = 100

# number of generator filters
ngf = 64 

# number of discriminator filters
ndf = 64

# momentum hyperparam for Adam optimizer
beta1 = 0.5

# num of gpus
ngpu = 1

# use cuda if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class EmotionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.imgs_files = [f for f in df.image_path.values]
        self.label = [f for f in df['emotion_id'].values]
    
    def __len__(self): 
        return len(self.label)
    
    def _getname__(self, idx): 
        return self.label[idx]
    
    def __getitem__(self, idx):
        i = idx%8
        idx = int(idx/8) 
        
        img_paths = self.imgs_files[8*idx:8*idx+8]
        label = self.label[idx]
        
        imgs = []
        # Read Image:
        for img_path in img_paths: 
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128,128))
            imgs.append(img)
                
        images = torch.from_numpy(np.array(imgs))
        
        # print(i)
        return images[i].unsqueeze(0), images
    
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
def plot_images(data_loader, real_batch, img_list, epoch=-1):
    real_batch = next(iter(data_loader))

    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[epoch],(1,2,0)))
    plt.show()
    
def plot_loss(train_losses, val_losses):
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    ax = ax.flatten()
    for i, k in enumerate(train_losses):
        ax[i].plot(train_losses[k], label='Train')
        ax[i].plot(val_losses[k], label='Val')
        ax[i].set_title(' '.join(k.split('_')))
        ax[i].set_xlabel('Epochs')
        ax[i].legend()
    # ax[-1].axis('off')
    plt.tight_layout()
    plt.show()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        f_size = 7
        padding = (f_size//2, f_size//2)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, f_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, f_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        # pooling
        self.pool = nn.MaxPool2d(2,2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ENCODER
        self.conv00 = ConvBlock(1, 32)
        self.conv10 = ConvBlock(32, 64)
        self.conv20 = ConvBlock(64, 128)
        self.conv30 = ConvBlock(128, 256)
        self.conv40 = ConvBlock(256, 512)
        
        # DECODER
        self.upconv31 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.conv31 = ConvBlock(2*256, 256)
        self.upconv22 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv22 = ConvBlock(2*128, 128)
        self.upconv13 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv13 = ConvBlock(2*64, 64)
        self.upconv04 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv04 = ConvBlock(2*32, 32)
        
        # final layers
        self.final04 = nn.Conv2d(32, 8, 1)
        
    def forward(self, x):
        
        # Encoder Path
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))
        
        
        # Up-sampling 
        x31 = self.upconv31(x40)
        x31 = self.conv31(torch.cat((x30,x31), dim=1))
        x22 = self.upconv22(x31)
        x22 = self.conv22(torch.cat((x20,x22),dim=1))
        x13 = self.upconv13(x22)
        x13 = self.conv13(torch.cat((x10,x13),dim=1))
        x04 = self.upconv04(x13)
        x04 = self.conv04(torch.cat((x00,x04),dim=1))
        
        # Outputs
        x04 = self.final04(x04)

        
        return x04

class NLayerDiscriminator(nn.Module):
    """
    This class was inspired from 
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
    Defines a PatchGAN discriminator
    """

    def __init__(self, input_nc=8, ndf=64, n_layers=4):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()

        f_size = 4
        padding = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=f_size,
                              stride=2, padding=padding),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = 2 ** n
            sequence += [nn.Conv2d(ndf * nf_mult_prev,
                                   ndf * nf_mult,
                                   kernel_size=f_size,
                                   stride=2,
                                   padding=padding),
                         nn.BatchNorm2d(ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = 2 ** n_layers
        sequence += [nn.Conv2d(ndf * nf_mult_prev,
                               ndf * nf_mult,
                               kernel_size=f_size,
                               stride=1, padding=padding),
                     nn.BatchNorm2d(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult,
                               1, kernel_size=f_size,
                               stride=1, padding=padding)]  # output 1 channel prediction map
        
        sequence += [nn.AvgPool2d(6)] 
        
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


def train(generator_model, discriminator_model, train_loader, val_loader,
          device=None, Lambda=0.01, num_epochs=1):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    
    generator_model = generator_model.to(device)
    discriminator_model = discriminator_model.to(device)

    gen_optimizer = optim.Adam(generator_model.parameters(), lr=2*0.0001)
    disc_optimizer = optim.Adam(discriminator_model.parameters(), lr=0.0001)

    BCE_loss = nn.BCEWithLogitsLoss()
    L1_loss = nn.L1Loss()

    loss_dictionary_train = {'Generator_BCE': [],
                             "L1": [],
                             'Discriminator_BCE_real': [],
                             'Discriminator_BCE_fake': [],
                             'Verdict_on_real': [],
                             'Verdict_on_fake': []}

    loss_dictionary_val = {'Generator_BCE': [],
                           "L1": [],
                           'Discriminator_BCE_real': [],
                           'Discriminator_BCE_fake': [],
                           'Verdict_on_real': [],
                           'Verdict_on_fake': []}

    epoch_metrics = [loss_dictionary_train, loss_dictionary_val]
    
    for epoch in range(num_epochs):
        print(f'\nepoch = {epoch:03d}')
        
        ld_train = {'Generator_BCE': [],
                    'L1': [],
                    'Discriminator_BCE_real':[],
                    'Discriminator_BCE_fake':[],
                    'Verdict_on_real': [],
                    'Verdict_on_fake': []}

        ld_val = {'Generator_BCE': [],
                  'L1': [],
                  'Discriminator_BCE_real':[],
                  'Discriminator_BCE_fake':[],
                  'Verdict_on_real': [],
                  'Verdict_on_fake': []}
                  
        batch_metrics = [ld_train, ld_val]

        for i, data_loader in enumerate([train_loader, val_loader]):
            for batch_idx, (input_img, target_images) in enumerate(data_loader):
                print(f'Batch = {batch_idx:04d}', end='\r')

                input_img = input_img.float().to(device)
                target_images = target_images.float().to(device)
                fake_images = generator_model(input_img)
                
                # Join fake images with its opiginal input & Pass through discriminator
                fake_paired = torch.cat((input_img, fake_images), 1).float()
                verdict_on_fake = discriminator_model(fake_paired.detach()).squeeze()
                batch_metrics[i]['Verdict_on_fake'].append(torch.sigmoid(verdict_on_fake).mean().item())

                # Join real images with its opiginal input & Pass through discriminator
                real_paired = torch.cat((input_img, target_images), 1).float()
                verdict_on_real = discriminator_model(real_paired).squeeze()
                batch_metrics[i]['Verdict_on_real'].append(torch.sigmoid(verdict_on_real).mean().item())

                # Labels for real (1s) and fake (0s) images
                ones = torch.ones_like(verdict_on_real).to(device)
                zeros = torch.zeros_like(verdict_on_fake).to(device)

                # Backprop on discriminator (for real data)
                disc_optimizer.zero_grad()
                loss = BCE_loss(verdict_on_real, ones)
                if not i:
                    loss.backward()
                batch_metrics[i]['Discriminator_BCE_real'].append(loss.item())
                

                # Backprop on discriminator (for fake data)
                loss = BCE_loss(verdict_on_fake, zeros)
                if not i:
                    loss.backward()
                    disc_optimizer.step()
                batch_metrics[i]['Discriminator_BCE_fake'].append(loss.item())
                
                # Backprop on generator
                verdict_on_fake = discriminator_model(fake_paired).squeeze()
                loss_bce = BCE_loss(verdict_on_fake, ones)
                loss_l1 = L1_loss(target_images, fake_images)
                loss_gen = loss_bce + Lambda * loss_l1
                if not i:
                    gen_optimizer.zero_grad()
                    loss_gen.backward()
                    gen_optimizer.step()
                batch_metrics[i]['Generator_BCE'].append(loss_bce.item())
                batch_metrics[i]['L1'].append(loss_l1.item())

            # Record average of losses from all batches
            for k, v in batch_metrics[i].items():
                epoch_metrics[i][k].append(sum(v)/len(v))
    
    return generator_model, discriminator_model, loss_dictionary_train, loss_dictionary_val

