import torch

from Models_.iTransformer import iTransformer
from Models_.model import ConvTran, CasualConvTran
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--Data_shape', type=tuple, default=(32, 3, 64))
parser.add_argument('--emb_size', type=int, default=128)

parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--num_class', type=int, default=6)

parser.add_argument('--num_heads', type=int, default=8)

parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer')
parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'None'}, default='tAPE',
                    help='Fix Position Embedding')
parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector', 'None'}, default='eRPE',
                    help='Relative Position Embedding')
parser.add_argument('--dropout', type=float, default=0.1)

args = parser.parse_args([])


class Config:
    def __init__(self, args):
        # 将 argparse 对象中的所有属性复制到配置类中
        for key, value in args.__dict__.items():
            setattr(self, key, value)


# 创建 Config 实例
config = Config(args)

x = torch.rand((32, 64, 3))
print(config.Data_shape)
model = iTransformer(config)

print(model(x).shape)
