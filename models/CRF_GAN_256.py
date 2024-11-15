import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.layers import SNConv3d, SNLinear


class Encoder(nn.Module):
    def __init__(self, channel=64):
        super(Encoder, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(1, channel//2, kernel_size=4, stride=2, padding=1) # in:[32,256,256], out:[16,128,128]
        self.bn1 = nn.GroupNorm(8, channel//2)
        self.conv2 = nn.Conv3d(channel//2, channel//2, kernel_size=3, stride=1, padding=1) # out:[16,128,128]
        self.bn2 = nn.GroupNorm(8, channel//2)
        self.conv3 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1) # out:[8,64,64]
        self.bn3 = nn.GroupNorm(8, channel)

    def forward(self, h):
        h = self.conv1(h)
        h = self.relu(self.bn1(h))

        h = self.conv2(h)
        h = self.relu(self.bn2(h))

        h = self.conv3(h)
        h = self.relu(self.bn3(h))
        return h

    """
    I used CRF implementation from https://github.com/baidu-research/NCRF 
    """
class CRF(nn.Module):
    def __init__(self, num_nodes, iteration=10):
        """Initialize the CRF module

        Args:
            num_nodes: int, number of nodes/patches within the fully CRF (it means the number of embeddings that I have not batches, for now I guess this)
            iteration: int, number of mean field iterations, e.g. 10
        """
        super(CRF, self).__init__()
        self.num_nodes = num_nodes
        self.iteration = iteration
        self.W = nn.Parameter(torch.zeros(1, num_nodes, num_nodes))

    def forward(self, a_inter, logits):
        """
        logits > 0 means tumor and logits < 0 means normal.
        if probs -> 1, then pairwise_potential promotes tumor probability;
        if probs -> 0, then -pairwise_potential promotes normal probability.

        Args:
            feats: 3D tensor with the shape of
            [batch_size, num_nodes, embedding_size], where num_nodes is the
            number of patches within a grid, e.g. 9 for a 3x3 grid;
            embedding_size is the size of extracted feature representation for
            each patch from ResNet, e.g. 512
            logits: 3D tensor with shape of [batch_size, num_nodes, 1], the
            logit of each patch within the grid being tumor before CRF

        Returns:
            logits: 3D tensor with shape of [batch_size, num_nodes, 1], the
            logit of each patch within the grid being tumor after CRF
        """
        batch_size, channels, _, height, width = a_inter.shape
        # Reshape tensor A to feats
        feats = a_inter[:, :, :self.num_nodes, :, :].reshape(batch_size, self.num_nodes, -1)
        logits = logits.unsqueeze(1).expand(-1, self.num_nodes, -1)
        feats_norm = torch.norm(feats, p=2, dim=2, keepdim=True)
        pairwise_norm = torch.bmm(feats_norm, torch.transpose(feats_norm, 1, 2))
        pairwise_dot = torch.bmm(feats, torch.transpose(feats, 1, 2))
        # cosine similarity between feats
        pairwise_sim = pairwise_dot / (pairwise_norm + 1e-6)
        # symmetric constraint for CRF weights
        W_sym = (self.W + torch.transpose(self.W, 1, 2)) / 2.
        pairwise_potential = pairwise_sim * W_sym
        unary_potential = logits.clone()
        for i in range(self.iteration):
            # current Q after normalizing the logits
            probs = torch.transpose(logits.sigmoid(), 1, 2)
            # taking expectation of pairwise_potential using current Q
            pairwise_potential_E = torch.sum(probs * pairwise_potential - (1 - probs) * pairwise_potential, dim=2, keepdim=True)
            logits = unary_potential + pairwise_potential_E
        return logits.mean(dim=1)


class Discriminator(nn.Module):
    def __init__(self, num_class=0, channel=512):
        super(Discriminator, self).__init__()        
        self.channel = channel
        self.num_class = num_class
        
        self.conv1 = SNConv3d(1, channel//32, kernel_size=4, stride=2, padding=1) # in:[32,256,256], out:[16,128,128]
        self.conv2 = SNConv3d(channel//32, channel//16, kernel_size=4, stride=2, padding=1) # out:[8,64,64,64]
        self.conv3 = SNConv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1) # out:[4,32,32,32]
        self.conv4 = SNConv3d(channel//8, channel//4, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)) # out:[2,16,16,16]
        self.conv5 = SNConv3d(channel//4, channel//2, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)) # out:[1,8,8,8]
        self.conv6 = SNConv3d(channel//2, channel, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)) # out:[1,4,4,4]
        self.conv7 = SNConv3d(channel, channel//4, kernel_size=(1,4,4), stride=1, padding=0) # out:[1,1,1,1]
        self.fc1 = SNLinear(channel//4+1, channel//8)
        self.fc2 = SNLinear(channel//8, 1)
        if num_class > 0:
            self.fc2_class = SNLinear(channel//8, num_class)


    def forward(self, h, crop_idx):
        h = F.leaky_relu(self.conv1(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv7(h), negative_slope=0.2).squeeze()
        h = torch.cat([h, (crop_idx / 224. * torch.ones((h.size(0), 1))).cuda()], 1) # 256*7/8
        h = F.leaky_relu(self.fc1(h), negative_slope=0.2)
        h_logit = self.fc2(h)
        if self.num_class > 0:
            h_class_logit = self.fc2_class(h)
            return h_logit, h_class_logit
        else:
            return h_logit


class Generator(nn.Module):
    def __init__(self, mode="train", latent_dim=1024, channel=32, num_class=0):
        super(Generator, self).__init__()
        _c = channel

        self.mode = mode
        self.relu = nn.ReLU()
        self.num_class = num_class

        self.fc1 = nn.Linear(latent_dim+num_class, 4*4*4*_c*16)

        self.tp_conv1 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(8, _c*16)

        self.tp_conv2 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(8, _c*16)

        self.tp_conv3 = nn.Conv3d(_c*16, _c*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.GroupNorm(8, _c*8)

        self.tp_conv4 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.GroupNorm(8, _c*4)

        self.tp_conv5 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.GroupNorm(8, _c*2)

        self.tp_conv6 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.GroupNorm(8, _c)

        self.tp_conv7 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, h, crop_idx=None, class_label=None, crf_need=False):
        # Generate from random noise
        if crop_idx != None or self.mode=='eval':
            if self.num_class > 0:
                h = torch.cat((h, class_label), dim=1)

            h = self.fc1(h)

            h = h.view(-1,512,4,4,4)
            h = self.tp_conv1(h)
            h = self.relu(self.bn1(h))

            h = F.interpolate(h,scale_factor = 2)
            h = self.tp_conv2(h)
            h = self.relu(self.bn2(h))

            h = F.interpolate(h,scale_factor = 2)
            h = self.tp_conv3(h)
            h = self.relu(self.bn3(h))

            h = F.interpolate(h,scale_factor = 2)
            h = self.tp_conv4(h)
            h = self.relu(self.bn4(h))

            h = F.interpolate(h,scale_factor = 2)
            h = self.tp_conv5(h)
            h_latent = self.relu(self.bn5(h)) # (64, 64, 64), channel:128

            if self.mode == "train":
                h = h_latent[:,:,crop_idx//4:crop_idx//4+8,:,:] # Crop, out: (8, 64, 64)
            else:
                h = h_latent

        # Generate from latent feature
        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv6(h)
        h = self.relu(self.bn6(h)) # (128, 128, 128)

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv7(h)
        h = torch.tanh(h) # (256, 256, 256)

        if crf_need:
            return h, h_latent
        return h
