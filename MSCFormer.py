import argparse

import numpy as np
import torch
from torch import nn
from Models_.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from Models_.Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec
from layers.SelfAttention_Family import AttentionLayer, FullAttention

import torch.nn.functional as F


class Config:
    def __init__(self, args):
        # 将 argparse 对象中的所有属性复制到配置类中
        for key, value in args.__dict__.items():
            setattr(self, key, value)


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


# 定义包含 Inception_Block_V2 的网络
class InceptionNet(nn.Module):
    def __init__(self, in_channels, emb_size, seq_len, num_classes=10):
        super(InceptionNet, self).__init__()
        self.inception_block1 = Inception_Block_V2(in_channels=in_channels, out_channels=in_channels,
                                                   )

        self.line = nn.Linear(seq_len, 1)

        self.fc = nn.Linear(in_channels * emb_size, num_classes)

    def forward(self, x):
        x = self.inception_block1(x)
        x = F.relu(x)

        # x_temp = self.line(x_temp)

        # x_temp = x.view(-1, x.size(-1))
        # x_temp = self.avg_pool(x_temp)
        # x = x_temp.view(x.shape[0], x.shape[1], x.shape[2])

        x = self.line(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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
            nn.Conv1d(emb_size, dim_ff, kernel_size=1),  # 使用kernel_size=1实现类似线性变换
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(dim_ff, emb_size, kernel_size=1),  # 使用kernel_size=1实现类似线性变换
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

        # 获取位置编码
        pos_encodings = self.learn_Position.get_positional_encodings()

        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Data_shape', type=tuple, default=(64, 6, 16))
    parser.add_argument('--L', type=int, default=8)
    parser.add_argument('--num_class', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dim_ff', type=int, default=256)
    parser.add_argument('--emb_size', type=int, default=66)
   # parser.add_argument('--num_filter', type=int, default=16, help='Internal dimension of transformer embeddings')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'None'}, default='tAPE',
                        help='Fix Position Embedding')
    parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector', 'None'}, default='eRPE',
                        help='Relative Position Embedding')
    parser.add_argument('--l1', type=int, default=4, help='')
    parser.add_argument('--l2', type=int, default=8, help='')
    parser.add_argument('--l3', type=int, default=8, help='')
    parser.add_argument('--l4', type=int, default=10000, help='')
    args = parser.parse_args([])

    # 创建 Config 实例
    config = Config(args)

    model = MSCFormer(config, 8)
    x = torch.zeros((64, 6, 16))
    out, *res = model(x)
    print(out.shape)
