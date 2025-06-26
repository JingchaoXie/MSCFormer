import argparse

import numpy as np
import torch
from torch import nn

from layers.SelfAttention_Family import AttentionLayer, FullAttention

import torch.nn.functional as F


class Config:
    def __init__(self, args):
        # 将 argparse 对象中的所有属性复制到配置类中
        for key, value in args.__dict__.items():
            setattr(self, key, value)




class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LearnableChannelIndependencePositionalEncoding(nn.Module):

    def __init__(self, channel, dropout=0.1, max_len=1024):
        super(LearnableChannelIndependencePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.empty(channel, max_len))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

    def get_positional_encodings(self):
        return self.pe.permute(1, 0).detach().cpu().numpy()


class MultiScaleConv2d(nn.Module):
    def __init__(self, in_channels, emb_size, config):
        super(MultiScaleConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, emb_size, kernel_size=[1, config.l1], padding='same')
        self.conv2 = nn.Conv2d(in_channels, emb_size, kernel_size=[1, config.l2], padding='same')
        self.conv3 = nn.Conv2d(in_channels, emb_size, kernel_size=[1, config.l3], padding='same')
        self.conv4 = nn.Conv2d(in_channels, emb_size, kernel_size=[1, config.l4], padding='same')

        self.bn = nn.BatchNorm2d(emb_size * 4)
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.bn(x)
        x = self.gelu(x)
        return x


class MSCFormer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config.Data_shape[1], config.Data_shape[2]
        emb_size = config.emb_size
        num_heads = config.num_heads
        dim_ff = config.dim_ff

        self.Fix_pos_encode = config.Fix_pos_encode
        self.Rel_pos_encode = config.Rel_pos_encode


        self.embed_layer = MultiScaleConv2d(1, emb_size, config)

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size * 4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU())
        # self.embed_layer = OptimizedEmbeddingLayer(emb_size, channel_size,num_filter)

        self.learn_Position = LearnableChannelIndependencePositionalEncoding(emb_size, dropout=config.dropout,
                                                                             max_len=seq_len)

        self.feature_attention_layer = AttentionLayer(
            FullAttention(False, attention_dropout=config.dropout, output_attention=True),
            seq_len,
            num_heads
        )

        self.time_attention_layer = AttentionLayer(
            FullAttention(False, attention_dropout=config.dropout, output_attention=True),
            emb_size,
            num_heads
        )

        self.space_LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.space_LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.space_FeedForward = nn.Sequential(
            nn.Conv1d(emb_size, dim_ff, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(dim_ff, emb_size, kernel_size=1),
            nn.Dropout(config.dropout)
        )

        self.avg = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

        # self.out = InceptionNet(1, emb_size, seq_len, num_classes)

    def forward(self, x,return_features=False):

        x = x.unsqueeze(1)

        x_src = self.embed_layer(x)

        x_src = self.embed_layer2(x_src)

        x_src = x_src.squeeze(2)

        x_src = self.learn_Position(x_src)

        feature_enc, feature_attn = self.feature_attention_layer(x_src, x_src, x_src, attn_mask=None, tau=None,
                                                                 delta=None)

        feature_att = x_src + feature_enc

        time_att = feature_att.permute(0, 2, 1)

        time_enc, time_attn = self.time_attention_layer(time_att, time_att, time_att, attn_mask=None, tau=None,
                                                        delta=None)

        time_att = time_att + time_enc
        time_att = self.space_LayerNorm1(time_att)

        time_att = self.space_LayerNorm2(time_att + self.space_FeedForward(time_att.transpose(1, 2)).transpose(1, 2))

        time_att = time_att.permute(0, 2, 1)

        out = self.avg(time_att)
        out = self.flatten(out)

        if return_features:
            return out,feature_attn,time_attn

        out = self.out(out)


        pos_encodings = self.learn_Position.get_positional_encodings()

        return out

