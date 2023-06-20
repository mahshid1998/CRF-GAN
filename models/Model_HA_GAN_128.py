import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
from models.layers import SNConv3d, SNLinear

import sys
import gc


class Encoder(nn.Module):
    def __init__(self, channel=64):
        super(Encoder, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(1, channel // 2, kernel_size=4, stride=2, padding=1)  # in:[16,128,128], out:[8,64,64]
        self.bn1 = nn.GroupNorm(8, channel // 2)
        self.conv2 = nn.Conv3d(channel // 2, channel // 2, kernel_size=3, stride=1, padding=1)  # out:[8,64,64]
        self.bn2 = nn.GroupNorm(8, channel // 2)
        self.conv3 = nn.Conv3d(channel // 2, channel, kernel_size=4, stride=2, padding=1)  # out:[4,32,32]
        self.bn3 = nn.GroupNorm(8, channel)

    def forward(self, h):
        h = self.conv1(h)
        h = self.relu(self.bn1(h))

        h = self.conv2(h)
        h = self.relu(self.bn2(h))

        h = self.conv3(h)
        h = self.relu(self.bn3(h))
        return h


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

    def forward(self, feats, logits):
        '''
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
        '''
        # print(f"last embeddings {feats.shape}, logit size which should be [4, 112, 1]{logits.shape}")
        feats_norm = torch.norm(feats, p=2, dim=2, keepdim=True)
        pairwise_norm = torch.bmm(feats_norm, torch.transpose(feats_norm, 1, 2))
        pairwise_dot = torch.bmm(feats, torch.transpose(feats, 1, 2))
        # cosine similarity between feats
        pairwise_sim = pairwise_dot / pairwise_norm
        # symmetric constraint for CRF weights
        W_sym = (self.W + torch.transpose(self.W, 1, 2)) / 2
        pairwise_potential = pairwise_sim * W_sym
        unary_potential = logits.clone()
        for i in range(self.iteration):
            # current Q after normalizing the logits
            probs = torch.transpose(logits.sigmoid(), 1, 2)
            # taking expectation of pairwise_potential using current Q
            pairwise_potential_E = torch.sum(probs * pairwise_potential - (1 - probs) * pairwise_potential, dim=2, keepdim=True)
            logits = unary_potential + pairwise_potential_E
        return logits


class Discriminator(nn.Module):
    def __init__(self, num_class=0, channel=512):
        super(Discriminator, self).__init__()
        self.channel = channel
        self.num_class = num_class

        # D^H
        self.conv2 = SNConv3d(1, channel // 16, kernel_size=4, stride=2, padding=1)  # out:[8,64,64,64]
        self.conv3 = SNConv3d(channel // 16, channel // 8, kernel_size=4, stride=2, padding=1)  # out:[4,32,32,32]
        self.conv4 = SNConv3d(channel // 8, channel // 4, kernel_size=(2, 4, 4), stride=(2, 2, 2),
                              padding=(0, 1, 1))  # out:[2,16,16,16]
        self.conv5 = SNConv3d(channel // 4, channel // 2, kernel_size=(2, 4, 4), stride=(2, 2, 2),
                              padding=(0, 1, 1))  # out:[1,8,8,8]
        self.conv6 = SNConv3d(channel // 2, channel, kernel_size=(1, 4, 4), stride=(1, 2, 2),
                              padding=(0, 1, 1))  # out:[1,4,4,4]
        self.conv7 = SNConv3d(channel, channel // 4, kernel_size=(1, 4, 4), stride=1, padding=0)  # out:[1,1,1,1]
        self.fc1 = SNLinear(channel // 4 + 1, channel // 8)
        self.fc2 = SNLinear(channel // 8, 1)
        if num_class > 0:
            self.fc2_class = SNLinear(channel // 8, num_class)

        # D^L

    #    self.sub_D = Sub_Discriminator(num_class)
        # CRF
        self.crf = CRF(num_nodes=113, iteration=10)

    # def forward(self, h, h_small, crop_idx):
    def forward(self, h, crop_idx):
        # print(f"layer output shape{h.shape}")
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        h = F.leaky_relu(self.conv7(h), negative_slope=0.2).squeeze()
        # print(h.shape, crop_idx)
        h = torch.cat([h, (crop_idx / 112. * torch.ones((h.size(0), 1))).cuda()], 1)  # 128*7/8
        # print(h.shape, h[1,128]*112)
        # exit(10)
        h = F.leaky_relu(self.fc1(h), negative_slope=0.2)
        h_logit = self.fc2(h)
        if self.num_class > 0:
            h_class_logit = self.fc2_class(h)

            # h_small_logit, h_small_class_logit = self.sub_D(h_small)
            # return (h_logit+ h_small_logit)/2., (h_class_logit+ h_small_class_logit)/2.
            return h_logit, h_class_logit
        else:
            # h_small_logit = self.sub_D(h_small)
            # return (h_logit+ h_small_logit)/2.
            # crf_embedds, labels_embedds = self.embeddings_of_whole_image(whole_images)
            #return h_logit
            # h_crf_logit = self.crf(crf_embedds, labels_embedds)

            #print(f" shape of crf output is: {h_crf_logit.shape},\nlogits of D: {h_logit},\nlogits of crf:{h_crf_logit[:,:,:]}")
            #exit(10)
            return h_logit

'''
This is G_L
class Sub_Generator(nn.Module):
    def __init__(self, channel:int=16):
        super(Sub_Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.tp_conv1 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(8, _c*2)

        self.tp_conv2 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(8, _c)

        self.tp_conv3 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, h):

        h = self.tp_conv1(h)
        h = self.relu(self.bn1(h))

        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))

        h = self.tp_conv3(h)
        h = torch.tanh(h)
        return h
'''


class Generator(nn.Module):
    def __init__(self, mode="train", latent_dim=1024, channel=32, num_class=0):
        super(Generator, self).__init__()
        _c = channel

        self.mode = mode
        self.relu = nn.ReLU()
        self.num_class = num_class

        # G^A and G^H
        self.fc1 = nn.Linear(latent_dim + num_class, 4 * 4 * 4 * _c * 16)

        self.tp_conv1 = nn.Conv3d(_c * 16, _c * 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(8, _c * 16)

        self.tp_conv2 = nn.Conv3d(_c * 16, _c * 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(8, _c * 16)

        self.tp_conv3 = nn.Conv3d(_c * 16, _c * 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.GroupNorm(8, _c * 8)

        self.tp_conv4 = nn.Conv3d(_c * 8, _c * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.GroupNorm(8, _c * 4)

        self.tp_conv5 = nn.Conv3d(_c * 4, _c * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.GroupNorm(8, _c * 2)

        self.tp_conv6 = nn.Conv3d(_c * 2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.GroupNorm(8, _c)

        self.tp_conv7 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # G^L
        # self.sub_G = Sub_Generator(channel=_c//2)

    def forward(self, h, crop_idx=None, class_label=None, crf_need=False):

        # Generate from random noise
        if crop_idx != None or self.mode == 'eval':
            if self.num_class > 0:
                h = torch.cat((h, class_label), dim=1)

            h = self.fc1(h)

            h = h.view(-1, 512, 4, 4, 4)
            h = self.tp_conv1(h)
            h = self.relu(self.bn1(h))

            h = F.interpolate(h, scale_factor=2)
            h = self.tp_conv2(h)
            h = self.relu(self.bn2(h))

            h = F.interpolate(h, scale_factor=2)
            h = self.tp_conv3(h)
            h = self.relu(self.bn3(h))

            h = F.interpolate(h, scale_factor=2)
            h = self.tp_conv4(h)
            h = self.relu(self.bn4(h))

            h = self.tp_conv5(h)
            h_latent = self.relu(self.bn5(h))  # (32, 32, 32), channel:128

            if self.mode == "train":
                # h_small = self.sub_G(h_latent)
                h = h_latent[:, :, crop_idx // 4:crop_idx // 4 + 4, :, :]  # Crop sub-volume, out: (4, 32, 32)
            else:
                h = h_latent
        # Generate from latent feature
        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv6(h)
        h = self.relu(self.bn6(h))  # (64, 64, 64)

        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv7(h)

        h = torch.tanh(h)  # (128, 128, 128)
        # if crop_idx != None and self.mode == "train":
        # return h, h_small
        if crf_need:
            return h, h_latent
        return h
