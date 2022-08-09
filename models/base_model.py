import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.layers import *
import numpy as np


def self_attention(query, key, d):
    key_ = key.transpose(-2, -1)
    scores = torch.matmul(query, key_) / math.sqrt(d)
    # scores = torch.matmul(query, key_)
    p_attn = F.softmax(scores, dim=-1)
    # p_attn = scores
    return p_attn


class Model(nn.Module):
    def __init__(self, units, stack_cnt, time_step, predict_window, blocks, Ks, Kt, device='cuda:0'):
        super(Model, self).__init__()
        self.units = units
        self.stack_cnt = stack_cnt
        self.time_step = time_step
        self.predict_window = predict_window
        self.blocks = blocks
        self.Ks = Ks
        self.Kt = Kt
        self.Ko = self.time_step - 4 * (self.Ks - 1)
        self.weight_key = nn.Parameter(torch.zeros(size=(self.units, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.units, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step, self.units)
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [stemGNN_block(self.time_step, self.units, channels, self.Ks, self.Kt, i) for i, channels in
             enumerate(self.blocks)]
        )
        self.temporal_conv_layer_InputConv2D = torch.nn.Conv2d(self.blocks[1][2], self.blocks[1][2], (1, 1),
                                                               padding='same').to(device)
        self.temporal_conv_layer_GLUConv2D = torch.nn.Conv2d(self.blocks[1][2], 2 * self.blocks[1][2], (1, self.Ko),
                                                             padding='valid').to(device)
        self.temporal_conv_layer_Conv2D = torch.nn.Conv2d(self.blocks[1][2], self.blocks[1][2], (1, 1),
                                                          padding='valid').to(device)
        self.out_conv_layer = torch.nn.Conv2d(self.blocks[1][2], 1, (1, 1), padding='same').to(device)
        self.to(device)

    def attention_conv_layer(self, x, device='cuda:0'):
        # print("Into attention_conv_layer")
        _, T, n, s = x.shape[:]

        x_input = x

        _, time_step_temp, route_temp, channal_temp = x_input.shape[:]

        x_input = torch.reshape(x_input, [-1, time_step_temp, route_temp * channal_temp]).permute(0, 2, 1)

        outputs, mid_state = self.GRU(x_input)
        outputs = outputs.squeeze()

        key = torch.matmul(outputs, self.weight_key)
        query = torch.matmul(outputs, self.weight_query)

        weight = self_attention(query, key, self.units)
        # weight = torch.mean(weight, dim=0)
        weight = torch.sigmoid(weight)

        D = torch.sum(weight, dim=1)
        # v1 = torch.ones(n).to(device)
        # D = D + v1
        D_2 = torch.sqrt(D)
        D_2 = torch.pow(D_2, -1)
        D_2 = torch.diag(D_2)
        # D_2 = torch.diag(torch.pow(torch.sqrt(D), -1))

        L = torch.matmul(weight, D_2)
        L = torch.matmul(D_2, L)

        # L1 = L.detach()
        # e, v = torch.symeig(L, eigenvectors=True)
        e, v = np.linalg.eigh(L.cpu().detach().numpy())
        e, v = torch.Tensor(e).to(device), torch.Tensor(v).to(device)
        # e, v = torch.linalg.eigh(L)
        # e = F.leaky_relu(e)
        # v = F.leaky_relu(v)
        return e, v

    def temporal_conv_layer(self, x, Kt, c_in, c_out, act_func='relu', device='cuda:0'):
        _, T, n, _ = x.shape[:]

        if c_in > c_out:
            x_input = self.temporal_conv_layer_InputConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif c_in < c_out:
            x_input = torch.cat((x, torch.zeros([x.shape[0], T, n, c_out - c_in]).to(device)), dim=3)
        else:
            x_input = x

        _, time_step_temp, route_temp, channel_temp = x_input.shape[:]
        x_input = x_input[:, Kt - 1:T, :, :]

        if act_func == 'GLU':
            x_conv = self.temporal_conv_layer_GLUConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            # return (x_conv[:, :, :, 0:c_out] + x_input) * torch.sigmoid(x_conv[:, :, :, -c_out:])
            return (x_conv[:, :, :, 0:c_out] + x_input) * x_conv[:, :, :, -c_out:]

        else:
            x_conv = self.temporal_conv_layer_Conv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            if act_func == 'sigmoid':
                # return torch.sigmoid(x_conv)
                return x_conv
            elif act_func == 'relu':
                # return torch.relu(x_conv + x_input)
                return x_conv + x_input

    def output_layer(self, x, T, act_func='GLU'):
        _, _, n, channel = x.shape[:]

        x_i = self.temporal_conv_layer(x, T, channel, channel, act_func=act_func)
        x_ln = x_i

        x_o = self.temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')

        x_fc = self.fully_con_layer(x_o, n, channel)
        return x_fc

    def fully_con_layer(self, x, n, channel, device='cuda:0'):
        return self.out_conv_layer(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

    def forward(self, x):
        # print("Model Forward")
        _, T, N, C = x.shape[:]
        e, v = self.attention_conv_layer(x)
        self.Ko = T
        flag = 0
        for i, channels in enumerate(self.blocks):
            if flag == 0:
                x_back = x
                l1 = 0
            x, x_back = self.stock_block[i](x, self.Ks, self.Kt, channels, e, v, flag, x_back)
            flag = 1
            self.Ko -= 2 * (self.Ks - 1)

        if self.Ko > 1:
            y = self.output_layer(x, self.Ko)
        single_pred = y
        return single_pred
