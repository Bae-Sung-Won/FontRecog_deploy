#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torch import nn
import torch.nn.functional as F


# In[2]:


# autoencoder skip connection
# aesc_version_2
# Vram 22.8GB
# loss 0.0003110 0.0004677 (60 epoch)

class autoencoders(nn.Module):
    def __init__(self):
        super(autoencoders, self).__init__()

        def COV2d(input, output, kernel_size, stride, padding):
            layer = nn.Sequential(
                nn.Conv2d(input, output, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(output),
                nn.ReLU()
            )
            return layer


        # encoder 
        self.conv1 = nn.Sequential(
                COV2d(3, 32, 5, 1, 2),
                COV2d(32, 32, 5, 1, 2)
            )
        
        self.conv2 = nn.Sequential(
                COV2d(32, 64, 5, 1, 2),
                COV2d(64, 64, 5, 1, 2)
            )
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                       # down sampling
        
        self.conv3 = nn.Sequential(
                COV2d(64, 128, 5, 1, 2),
                COV2d(128, 128, 5, 1, 2)
            )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                       # down sampling

        # BottleNeck
        self.bottleneck = nn.Sequential(
                COV2d(128, 256, 5, 1, 2),
                COV2d(256, 256, 5, 1, 2)
            )
    

        # decoder
        self.upsampling1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # upsampling
        
        self.upconv1 = nn.Sequential(
                COV2d(128, 128, 5, 1, 2)
            )

        self.upsampling2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # upsampling
        
        self.upconv2 = nn.Sequential(
                COV2d(64, 64, 5, 1, 2)
            )

        self.upsampling3 = nn.ConvTranspose2d(64, 32, kernel_size=1, stride=1) 
        
        self.upconv3 = nn.Sequential(
                COV2d(32, 32, 5, 1, 2)
            )
        
        self.fc = nn.Sequential(
                nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2),     
                nn.Sigmoid()   
            )

    
    def forward(self, images):
        # encoder
        layer1 = self.conv1(images)      # 300x400x3 -> 300x400x32
        layer2 = self.conv2(layer1)      # 300x400x32 -> 300x400x64
        x = self.pool1(layer2)           # 300x400x64 -> 150x200x64
        layer3 = self.conv3(x)           # 150x200x64 -> 150x200x128
        x = self.pool2(layer3)           # 150x200x128 -> 75x100x128
        bottle = self.bottleneck(x)      # 75x100x128 -> 75x100x256
        
        # decoder
        upsampl1 = self.upsampling1(bottle)        # 75x100x256 -> 150x200x128
        add1 = torch.add(layer3, upsampl1)         # 150x200x128 -> 150x200x128    skip-connection add
        upconv1 = self.upconv1(add1)               # 150x200x128 -> 150x200x128
        upsampl2 = self.upsampling2(upconv1)       # 150x200x128 -> 300x400x64
        add2 = torch.add(layer2, upsampl2)         # 300x400x64 -> 300x400x64      skip-connection add
        upconv2 = self.upconv2(add2)               # 300x400x64 -> 300x400x64 
        upsampl3 = self.upsampling3(upconv2)       # 300x400x64 -> 300x400x32
        add3 = torch.add(layer1, upsampl3)         # 300x400x32 -> 300x400x32      skip-connection add
        upconv3 = self.upconv3(add3)               # 300x400x32 -> 300x400x32  
        x = self.fc(upconv3)                       # 300x400x32 -> 300x400x3

        return x

