
import pickle
import timeit
import numpy as np
from math import sqrt
import math
import torch
import torch.optim as optim
import os
from torch import nn, einsum
from torch.nn import Parameter
# from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics import R2Score

import optuna

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_num_threads(5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace
        self.softmax = nn.Softmax(-1)
    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * self.softmax(x) + x

class spatt(nn.Module):
    def __init__(self, padding = 3):
        super(spatt, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(2*padding+1),padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xc = x.unsqueeze(1)
        avg = torch.mean(xc, dim=1, keepdim=True)
        max_x, _ = torch.max(xc, dim=1, keepdim=True)
        xt = torch.cat((avg,max_x),dim=1)
        att = self.sigmoid(self.conv1(xt))
        # print(att.squeeze(1).shape)
        return x * (att.squeeze(1))


class CNN_MLP(nn.Module):
    def __init__(self, Affine, patch, channel, output_size, dr, down=False, last=False):
        super(CNN_MLP, self).__init__()
        self.Afine_p1 = Affine(channel)
        self.Afine_p2 = Affine(channel)
        self.Afine_p3 = Affine(channel)
        self.Afine_p4 = Affine(channel)
        self.cross_patch_linear0 = nn.Linear(patch, patch)
        self.cross_patch_linear1 = nn.Linear(patch, patch)
        self.cross_patch_linear = nn.Linear(patch, patch)
        self.attention_patch_linear2 = nn.Linear(patch, patch)
        self.cnn1 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=15, padding=7, groups=patch)
        self.bn1 = nn.BatchNorm1d(patch)
        self.cnn2 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=31, padding=15, groups=patch)
        self.bn2 = nn.BatchNorm1d(patch)
        self.cnn3 = nn.Conv1d(in_channels=patch, out_channels=patch, kernel_size=7, padding=3, groups=patch)
        self.bn3 = nn.BatchNorm1d(patch)
        self.attention_patch_linear2 = nn.Linear(patch, patch)
        self.self_attention = self_attention(channel)
        self.bnp1 = nn.BatchNorm1d(channel)

        self.cross_channel_linear1 = nn.Linear(channel, channel)
        self.cross_channel_linear2 = nn.Linear(channel, channel)

        self.att = spatt(3)
        self.att_sp = spatial_attention(3)
        # self.att = ExternalAtt(patch, channel)
        self.attention_channel_linear2 = nn.Linear(channel, channel)
        self.last_linear = nn.Linear(channel, output_size)
        self.bnp = nn.BatchNorm1d(patch)
        self.act = nn.ReLU()
        self.last = last
        self.dropout = nn.Dropout(0.05)
        self.down = down

    def forward(self, x):
        # print(x.shape)
        x_cp = self.Afine_p1(x).permute(0, 2, 1)
        x_cp = self.act(self.cross_patch_linear0(x_cp))
        x_cp = self.act(self.cross_patch_linear1(x_cp))
        x_cp = self.cross_patch_linear(x_cp).permute(0, 2, 1)
        x_cc = x + self.Afine_p2(x_cp)
        x_cc2 = self.Afine_p3(x_cc)
        x_cc2 = self.act(self.bn1(self.cnn1(x_cc2)))
        x_cc2 = self.act(self.bn2(self.cnn2(x_cc2)))
        x_cc2 = self.act(self.bn3(self.cnn3(x_cc2)))
        x_cc2 = self.Afine_p4(x_cc2)
        # x_cc2 = self.act(x_cc2)
        x_cc2 = self.att(x_cc2)
        # print(atten)
        x_out = x_cc + self.dropout(x_cc2)
        # x_out = x_cc + atten * x_cc2
        if self.last == True:
            x_out = self.last_linear(x_out)
        return x_out


class ResMLP(nn.Module):
    def __init__(self, Affine, patch, channel, output_size,dr, down=False, last=False):
        super(ResMLP, self).__init__()
        self.Afine_p1 = Affine(channel)
        self.Afine_p2 = Affine(channel)
        self.Afine_p3 = Affine(channel)
        self.Afine_p4 = Affine(channel)
        self.cross_patch_linear0 = nn.Linear(patch, patch)
        self.cross_patch_linear1 = nn.Linear(patch, patch)
        self.cross_patch_linear = nn.Linear(patch, patch)
        self.attention_patch_linear2 = nn.Linear(patch, patch)
        self.bnp1 = nn.BatchNorm1d(channel)

        self.cross_channel_linear1 = nn.Linear(channel, channel)
        self.cross_channel_linear2 = nn.Linear(channel, channel)

        self.cross_channel_linear1_down = nn.Linear(channel, channel//5)
        self.cross_channel_linear2_down = nn.Linear(channel//5, channel)

        self.att = spatt(3)
        # self.att = ExternalAtt(patch, channel)
        self.attention_channel_linear2 = nn.Linear(channel, channel)
        self.last_linear = nn.Linear(channel, output_size)
        self.bnp = nn.BatchNorm1d(patch)
        self.activation = nn.ReLU()
        self.last = last
        self.dropout = nn.Dropout(dr)
        self.down = down

    def forward(self, x):
        x_cp = self.Afine_p1(x).permute(0, 2, 1)
        x_cp = self.activation(self.cross_patch_linear0(x_cp))
        x_cp = self.activation(self.cross_patch_linear1(x_cp))
        x_cp = self.cross_patch_linear(x_cp).permute(0, 2, 1)
        x_cc = x + self.Afine_p2(x_cp)
        x_cc2 = self.Afine_p3(x_cc)

        x_cc2 = self.activation(self.cross_channel_linear1(x_cc2))
        x_cc2 = self.cross_channel_linear2(x_cc2)
        x_cc2 = self.Afine_p4(x_cc2)
        x_cc2 = self.activation(x_cc2)
        x_cc2 = self.att(x_cc2)
        # x_cc2 = self.dropout(x_cc2)

        # atten = self.attention_channel_linear2(x_cc2)
        # atten = self.bnp(atten)

        x_out = x_cc + self.dropout(x_cc2)
        # x_out = x_cc + atten * x_cc2
        if self.last == True:
            x_out = self.last_linear(x_out)
        return x_out

class WingLoss(nn.Module):
    def __init__(self, om=10, ep=2):
        super(WingLoss, self).__init__()
        self.om = om
        self.ep = ep

    def forward(self, pred, tar):
        y = tar
        y_hat = pred
        de_y = (y - y_hat).abs()
        de_y1 = de_y[de_y < self.om]
        de_y2 = de_y[de_y >= self.om]
        loss1 = self.om * torch.log(1 + de_y1 / self.ep)
        C = self.om - self.om * math.log(1 + self.om / self.ep)
        loss2 = de_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class RWingLoss(nn.Module):
    def __init__(self, om=10, ep=2, r=0.5):
        super(RWingLoss, self).__init__()
        self.om = om
        self.ep = ep
        self.r = r

    def forward(self, pred, tar):
        y = tar
        y_hat = pred
        de_y = (y - y_hat).abs()
        de_y_0 = de_y[de_y < self.r]
        de_y0 = de_y[de_y >= self.r]
        de_y1 = de_y0[de_y0 < self.om]
        de_y2 = de_y[de_y >= self.om]
        loss0 = 0 * de_y_0
        loss1 = self.om * torch.log(1 + (de_y1-self.r) / self.ep)
        C = self.om - self.om * math.log(1 + (self.om-self.r) / self.ep)
        loss2 = de_y2 - C
        return (loss0.sum() + loss1.sum() + loss2.sum()) / (len(loss0) + len(loss1) + len(loss2))

class atten(nn.Module):
    def __init__(self, padding=3):
        super(atten, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(2 * padding + 1), padding=padding,
                               bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        xc = x.unsqueeze(1)
        xt = torch.cat((xc, self.dropout(xc)), dim=1)
        att = self.sigmoid(self.conv1(xt))
        return att.squeeze(1)



class self_attention(nn.Module):
    def __init__(self, channel):
        super(self_attention, self).__init__()
        self.linear_Q = nn.Linear(channel, channel)
        self.linear_K = nn.Linear(channel, channel)
        self.linear_V = nn.Linear(channel, channel)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        Q = self.linear_Q(xs)
        K = self.linear_K(xs)
        scale = K.size(-1) ** -0.5
        att = self.softmax(Q * scale)
        ys = att * K
        return ys

class linear_attention(nn.Module):
    def __init__(self, channel):
        super(linear_attention, self).__init__()
        self.linear_Q = nn.Linear(channel, channel)
        self.linear_K = nn.Linear(channel, channel)
        self.linear_V = nn.Linear(channel, channel)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        Q = self.linear_Q(xs)
        K = self.linear_K(xs)
        V = self.linear_V(xs)
        weight = torch.matmul(K.permute(0, 2, 1), V)
        scale = K.size(-1) ** -0.5
        attention = self.softmax(weight * scale)
        ys = torch.matmul(Q, attention)
        return ys

class channel_attention(nn.Module):
    def __init__(self, channel):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_channels=channel, out_channels=channel//15, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channels=channel//15, out_channels=channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        avg_ = self.fc2(self.relu(self.fc1(self.avg_pool(xs))))
        max_ = self.fc2(self.relu(self.fc1(self.max_pool(xs))))
        out = avg_ + max_
        return self.sigmoid(out).expand_as(xs)

class se_attention(nn.Module):
    def __init__(self, inputs, reduction=5):
        super(se_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(inputs, inputs//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inputs//reduction, inputs, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg = self.avg_pool(x).squeeze(2)
        att = self.fc(avg)
        return att

class spatial_attention(nn.Module):
    def __init__(self, padding=3):
        super(spatial_attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=(2 * padding + 1), padding=padding,
                               bias=False)
        self.dropout = nn.Dropout(0.1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        xt = torch.cat((avg, max_x), dim=1)
        att = self.Sigmoid((self.conv1(xt)))
        return att.expand_as(x)



class one_hot(nn.Module):
    def __init__(self, input_len, output_shape):
        super(one_hot, self).__init__()
        self.matrix = torch.eye(output_shape)
        self.len = input_len
        self.out = output_shape

    def forward(self, xs):
        out_put = torch.zeros((self.len, self.out), device=device)
        for i in range(self.len):
            out_put[i] = self.matrix[xs[i]]
        return out_put


class Predictor(nn.Module):
    def __init__(self, ResMLP, Affine, dim, window, window2, layer_gnn, layer_cnn, layer_output, dropout):
        super(Predictor, self).__init__()
        # self.embed_fingerprint = nn.Embedding(34, dim)
        self.embed_comg = nn.Embedding(16, 44)
        self.embed_word = nn.Embedding(264, 100)
        self.embed_ss = nn.Embedding(25, 100)
        self.sequence_embedding = sequence_embedding()
        self.embed_com_word = nn.Embedding(65, 100)
        self.layer_gnn = layer_gnn
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)])
        self.W_gat = nn.Parameter(torch.ones(50, 50))

        self.dropout=dropout
        '''
        self.WP_NN1 = CNN_MLP(Affine, 1200, 100, 75, 0, True)
        self.WP_NN3 = CNN_MLP(Affine, 1200, 100, 75, 0, True)
        self.WP_NN4 = CNN_MLP(Affine, 1200, 100, 75, 0, True)
        self.WP_NN2 = CNN_MLP(Affine, 1200, 100, 75, 0, True, True)

        self.WP_FiNN1 = CNN_MLP(Affine, 1200, 100, 75, 0,True)
        self.WP_FiNN3 = CNN_MLP(Affine, 1200, 100, 75, 0, True)
        self.WP_FiNN4 = CNN_MLP(Affine, 1200, 100, 75, 0, True)
        self.WP_FiNN2 = CNN_MLP(Affine, 1200, 100, 75, 0,True, True)
        '''

        self.WC_struc1 = ResMLP(Affine, 64, 75, 75, self.dropout)
        self.WC_struc2 = ResMLP(Affine, 64, 75, 75, self.dropout,False, True) 

        self.WC_struc3 = ResMLP(Affine, 50, 75, 75, 0)
        self.WC_struc4 = ResMLP(Affine, 50, 75, 75, 0,False, True)

        self.WC_fcfp1 = ResMLP(Affine, 1, 2048, 75, self.dropout)
        self.WC_fcfp2 = ResMLP(Affine, 1, 2048, 75, self.dropout,False, True)

        self.WC_finger1 = ResMLP(Affine, 1, 600, 75, self.dropout)
        self.WC_finger2 = ResMLP(Affine, 1, 600, 75, self.dropout,False, True)


        self.WC_words1 = CNN_MLP(Affine, 100, 100, 75, 0)
        self.WC_words2 = CNN_MLP(Affine, 100, 100, 75, 0,False, True)

        self.WPDI_len1 = ResMLP(Affine, 75, 600, 100, 0)

        self.WP_fixfea = CNN_MLP(Affine, 75, 2400, 1200, 0,False, True)


        self.Wpdi_cnn = nn.ModuleList([nn.Conv1d(
            in_channels=75, out_channels=75, kernel_size=2 * window2 + 1,
            stride=1, padding=window2) for _ in range(2)])
        self.Transcnn = nn.Linear((2048 // 16), 75)
        # self.dcnn = nn.Linear(2500, 75)

        self.WC = nn.Linear(75, 75)
        self.WC2 = nn.Linear(75, 75)
        self.WW = nn.Linear(75, 75)
        self.WM = nn.Linear(75, 75)
        self.WF = nn.Linear(75, 75)
        self.WS = nn.Linear(75, 75)
        self.WC3 = nn.Linear(75, 75)
        self.WM3 = nn.Linear(75, 75)
        self.merge_atten = atten(3)
        self.merge_atten2 = atten(3)
        self.down_sample1 = nn.Linear(150, 75)
        self.down_sample2 = nn.Linear(150, 75)


        self.bn = nn.BatchNorm1d(75)
        self.ln = nn.LayerNorm(75)

        self.layer_output = layer_output
        # self.W_out0 = nn.Linear(100 * 75, 10*75)
        '''
        self.Decoder = Mix_Decoder(75, 5, attention_dropout=0.05)
        '''
        self.apha = Parameter(torch.tensor([0.5]))

        self.W_out1_1 = nn.Linear(1251, 1024)
        self.bn_out1 = nn.BatchNorm1d(1)#nn.LayerNorm(1024)
        # nn.init.kaiming_normal(self.W_out1_1.weight,mode='fan_in')
        self.W_out1_2 = nn.Linear(1024, 1024)
        self.bn_out2 = nn.BatchNorm1d(1)
        # nn.init.kaiming_normal(self.W_out1_2.weight, mode='fan_in')
        self.W_out1_3 = nn.Linear(1024, 512)
        self.bn_out3 = nn.BatchNorm1d(1)
        # nn.init.kaiming_normal(self.W_out1_3.weight, mode='fan_in')
        self.W_out2_1 = nn.Linear(75, 128)
        self.W_out2_2 = nn.Linear(128, 128)
        self.W_out2_3 = nn.Linear(128, 128)

        self.dropout1 = nn.Dropout(p=self.dropout)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout1_1 = nn.Dropout(p=0.2)
        self.dropout1_2 = nn.Dropout(p=0.4)
        self.activation = nn.ReLU()
        self.activation2 = nn.GELU()
        self.act_norm = Swish(False)
        self.act_norm2 = Swish(True)
        self.act_norm3 = Swish(True)
        self.act_norm4 = Swish(False)
        self.sigmoid = nn.Sigmoid
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)
        self.softmax3 = nn.Softmax(dim=1)

        self.W_interaction = nn.Linear(152, 152)
        self.W_interaction1 = nn.Linear(152, 1)
        # nn.init.xavier_normal_(self.W_interaction1.weight)
        # nn.init.kaiming_normal(self.W_interaction1.weight, mode='fan_in')
        self.W_interaction2 = nn.Linear(65, 1)


    def gnn(self, xs, A, layer):
        # print(torch.min(xs))
        for i in range(layer):
            hs = self.activation(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return xs

    def attention_PM(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        xs_ = self.activation(self.WC(xs))
        h = self.activation(self.WM(x))
        weights = torch.matmul(xs_, h.permute(0, 2, 1))
        scale = weights.size(1) ** -0.5
        ys = self.softmax1(torch.matmul(weights, h) * scale) * xs
        return ys

    def attention_PM2(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        xs_ = self.activation(self.WC3(xs))
        h = self.activation(self.WM3(x))
        weights = torch.matmul(xs_, h.permute(0, 2, 1))
        scale = weights.size(1) ** -0.5
        ys = self.softmax1(torch.matmul(weights, h) * scale) * xs
        return ys

    def Elem_feature_Fusion_D(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        x_c = self.down_sample1(torch.cat((xs, x), dim=2))
        x_c = self.merge_atten(x_c)
        xs_ = self.activation(self.WC2(xs))
        x_ = self.activation(self.WW(x))
        xs_m = x_c * xs_ + xs
        ones = torch.ones(x_c.shape).to(device)
        x_m = (ones - x_c) * x_ + x
        ys = xs_m + x_m
        return ys

    def Elem_feature_Fusion_P(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        x_c = self.down_sample2(torch.cat((xs, x), dim=2))
        x_c = self.merge_atten2(x_c)
        xs_ = self.activation(self.WF(xs))
        x_ = self.activation(self.WS(x))
        xs_m = x_c * xs_ + xs
        ones = torch.ones(x_c.shape).to(device)
        x_m = (ones - x_c) * x_ + x
        ys = xs_m + x_m
        return ys

    def forward(self, inputs):

        fingerprints, pgraph, padj, graph, morgan, adjacency, size, label, water = inputs
        # print(fingerprints[0].shape)
        N = graph.shape[0]
        L = adjacency.shape[1]

        N2=pgraph.shape[0]
        L2=padj.shape[0]
        """Compound vector with GNN."""
        compound_vector = torch.zeros((N, 64, 75), device=device)
        compound_vec = torch.zeros((graph.shape[0], graph.shape[1], 75), device=device)
        for i in range(N):
            fea = torch.zeros((graph.shape[1], 75), device=device)
            atom_fea = graph[i][:, 0:16]
            p = torch.argmax(atom_fea, dim=1)
            com = self.embed_comg(p)
            oth1 = graph[i][:, 44:75]
            tf = F.normalize(oth1, dim=1)
            fea[:, 0:44] = com
            fea[:, 44:75] = tf
            # com_vec = F.normalize(fea, dim=1)
            compound_vec[i, :, :] = fea
            # print(compound_vec)

        compound_vecs = self.gnn(compound_vec, adjacency, self.layer_gnn)
        t = compound_vecs.shape[1]
        # print(compound_vector.shape)
        compound_vector[:,0:t,:] = compound_vecs

        '''
        microplastics vector with GNN
        '''
        micro_vector = torch.zeros((N2, 64, 75), device=device)
        micro_vec = torch.zeros((pgraph.shape[0], pgraph.shape[1], 75), device=device)
        for i in range(N):
            fea2 = torch.zeros((pgraph.shape[1], 75), device=device)
            atom_fea2 = pgraph[i][:, 0:16]
            p2 = torch.argmax(atom_fea2, dim=1)
            com2 = self.embed_comg(p2)
            oth12 = pgraph[i][:, 44:75]
            tf2 = F.normalize(oth12, dim=1)
            fea2[:, 0:44] = com2
            fea2[:, 44:75] = tf2
            # com_vec = F.normalize(fea, dim=1)
            micro_vec[i, :, :] = fea2
            # print(compound_vec)

        micro_vecs = self.gnn(micro_vec, padj, self.layer_gnn)
        t2 = micro_vecs.shape[1]
        #print(compound_vector.shape)
        micro_vector[:,0:t2,:] = micro_vecs

        '''
        compound fusion
        '''

        compound_vector = compound_vector + self.WC_struc1(compound_vector)
        compound_vector = self.WC_struc2(compound_vector)

        morgan_vector = morgan.view(N, 1, 2048)
        mol_vector = morgan_vector + self.WC_fcfp1(morgan_vector)
        mol_vector = self.WC_fcfp2(mol_vector)

        compound_vectors = self.dropout1(self.attention_PM(compound_vector, mol_vector))  # .permute(0, 2, 1)
        mol_vectors = self.dropout1(self.attention_PM2(mol_vector, compound_vector))

        compound_GNN_att = torch.cat((compound_vector, mol_vector), 1)
        mol_FCFPs_att = torch.cat((compound_vectors, mol_vectors), 1)

        compound_fusion = self.Elem_feature_Fusion_D(compound_GNN_att, mol_FCFPs_att)

        '''
        micro fusion
        '''
        micro_vector = micro_vector + self.WC_struc1(micro_vector)
        micro_vector = self.WC_struc2(micro_vector)

        fingerprint_vector = fingerprints.view(N2, 1, 600)
        finger_vector = fingerprint_vector + self.WC_finger1(fingerprint_vector)
        finger_vector = self.WC_finger2(finger_vector)

        micro_vectors = self.dropout1(self.attention_PM(micro_vector, finger_vector))  # .permute(0, 2, 1)
        finger_vectors = self.dropout1(self.attention_PM2(finger_vector, micro_vector))

        micro_GNN_att = torch.cat((micro_vector, finger_vector), 1)
        finger_FCFPs_att = torch.cat((micro_vectors, finger_vectors), 1)

        micro_fusion = self.Elem_feature_Fusion_D(micro_GNN_att, finger_FCFPs_att)

        

        adjacencys = torch.zeros((N, 64, 64), device=device)
        adjacencys[:, 0:L, 0:L] = adjacency


        ndata=size.shape[0]
        size=size.reshape(ndata,1,1)
        size=size.expand(ndata,65,1)

        water=water.reshape(ndata,1,1)
        water=water.expand(ndata,65,1)
        

        kd=torch.cat((compound_fusion, micro_fusion, size, water),dim=2)
        kd=self.activation(self.W_interaction(kd))
        kd = self.dropout1(self.W_interaction1(kd))  # .squeeze(1)
        kd=torch.squeeze(kd)
        kd=self.dropout1(self.W_interaction2(kd))
        kd=torch.squeeze(kd)

        return kd
        
     


class sequence_embedding(nn.Module):
    def __init__(self,):
        super(sequence_embedding, self).__init__()
        self.pool = nn.MaxPool1d(12)
        self.softmax = nn.Softmax(-1)

    def forward(self, p1, p2):
        protein_matrix = torch.matmul(p1, p2.permute(1, 0))
        scale = protein_matrix.size(-1) ** 0.5
        weight = self.softmax(protein_matrix * scale)
        protein_feture = torch.matmul(weight, p2)
        return protein_feture

from torch.nn.modules.loss import _Loss

class BMSE_Loss(nn.Module):
    def __init__(self, sigma, thr):
        super(BMSE_Loss, self).__init__()
        self.sigma = sigma
        self.thr = thr

    def forward(self, pred, target):
        loss = (pred - target) ** 2
        loss1 = loss[loss <= self.thr]
        loss2 = loss[loss > self.thr]
        loss = (loss1.sum() + loss2) / (len(loss) + len(loss2))
        return loss


def pack(fingerprints, patoms, padjs, atoms,morgans, adjs, sizes, labels, waters, device):
    proteins_len = 1200
    proteinss_len = 1200
    com_words_len = 100

    atoms_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    atoms_new = torch.zeros((N, atoms_len, 75), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1

    patoms_len=0
    N2=len(patoms)
    patom_num=[]
    for patom in patoms:
        patom_num.append(patom.shape[0])
        if patom.shape[0] >= patoms_len:
            patoms_len = patom.shape[0]

    patoms_new=torch.zeros((N2, patoms_len, 75), device=device)
    i=0
    for patom in patoms:
        p_len=patom.shape[0]
        patoms_new[i, :p_len, :]=patom

    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len, device=device)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1

    padjs_new = torch.zeros((N2, patoms_len, patoms_len), device=device)
    i = 0
    for padj in padjs:
        p_len = padj.shape[0]
        padj = padj + torch.eye(p_len, device=device)
        padjs_new[i, :p_len, :p_len] = padj
        i += 1


    morgans_new = torch.zeros((N, 2048, 1), device=device)
    i = 0
    for morgan in morgans:
        morgans_new[i, :, :] = morgan.unsqueeze(1)
        i += 1

    fingerprints_new = torch.zeros((N, 600, 1), device=device)
    i = 0
    for fingerprint in fingerprints:
        fingerprints_new[i, :, :] = fingerprint.unsqueeze(1)
        i += 1

    labels_new = torch.zeros((N,1), device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    sizes_new = torch.zeros((N,1), device=device)
    i = 0
    for size in sizes:
        sizes_new[i] = size
        i += 1

    waters_new = torch.zeros((N,1), device=device)
    i = 0
    for water in waters:
        waters_new[i] = water
        i += 1


    return fingerprints_new, patoms_new, padjs_new, atoms_new ,morgans_new, adjs_new, sizes_new, labels_new, waters_new  # , atom_num, protein_num


def pack_aug(atoms, compounds_words, adjs, morgans, proteins, proteins_sss, labels, res_labels, device):
    proteins_len = 1200
    proteinss_len = 1200
    com_words_len = 100
    atoms_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    atoms_new = torch.zeros((N, atoms_len, 75), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1

    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len, device=device)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1

    compounds_word_new = torch.zeros((N, com_words_len), device=device)
    i = 0
    for compounds_word in compounds_words:
        compounds_word_len = compounds_word.shape[0]
        # print(compounds_word.shape)
        if compounds_word_len <= 100:
            compounds_word_new[i, :compounds_word_len] = compounds_word
        else:
            compounds_word_new[i] = compounds_word[0:100]
        # compounds_word_new[i, :compounds_word_len] = compounds_word
        i += 1

    morgan_new = torch.zeros((N, 2048, 1), device=device)
    i = 0
    for morgan in morgans:
        morgan_new[i, :, :] = morgan.unsqueeze(1)
        i += 1

    proteins_new = torch.zeros((N, proteins_len), device=device)
    i = 0
    for protein in proteins:
        if protein.shape[0] > 1200:
            protein = protein[0:1200]
        a_len = protein.shape[0]
        nums = torch.randint(low=0, high=36, size=(a_len,))
        mask = (nums > 5).to(device)
        proteins_new[i, :a_len] = protein * mask
        i += 1

    proteins_ss_new = torch.zeros((N, proteinss_len), device=device)
    i = 0
    for proteins_ss in proteins_sss:
        if proteins_ss.shape[0] > 1200:
            proteins_ss = proteins_ss[0:1200]
        a_len = proteins_ss.shape[0]
        nums = torch.randint(low=0, high=36, size=(a_len,))
        mask = (nums > 5).to(device)
        proteins_ss_new[i, :a_len] = proteins_ss * mask
        i += 1

    labels_new = torch.zeros(N, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    res_labels_new = torch.zeros((N, 1200), device=device)
    i = 0
    for res_label in res_labels:
        res_labels_new[i] = res_label[0:1200]
        i += 1

    return atoms_new, compounds_word_new, adjs_new, morgan_new, proteins_new, proteins_ss_new, labels_new, res_labels_new




def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / (float(y_obs_sq * y_pred_sq) + 0.00000001)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / (float(sum(y_pred * y_pred)) + 0.00000001)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / (float(down) + 0.00000001))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))



from sklearn import metrics
def get_aupr(Y, P):
    if hasattr(Y, 'A'): Y = Y.A
    if hasattr(P, 'A'): P = P.A
    Y = np.where(Y >= 7, 1, 0)
    P = np.where(P >= 7, 1, 0)
    prec, re, _ = metrics.precision_recall_curve(Y, P)
    aupr = metrics.auc(re, prec)
    return aupr

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

def load_tensor_label(file_name, dtype):
    file=np.load(file_name + '.npy', allow_pickle=True)
    a=file.shape[0]
    return [dtype(d).to(device) for d in file.reshape(a,1)]



def load_tensor2(file_name, dtype):
    domains = np.load(file_name + '.npy', allow_pickle=True)
    Domain = []
    for d in domains:
        domain = []
        for j in d:
            domain.append(dtype(j).to(device))
        Domain.append(domain)
    return Domain


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def dataset(dir_input1, dir_input2):
    print(dir_input1 + 'compounds')
    fingerprints1 = load_tensor(dir_input1 + 'fingerprints', torch.FloatTensor)
    pgraphs1 = load_tensor(dir_input1 + 'pgraph', torch.FloatTensor)
    padjacencies1=load_tensor(dir_input1 + 'padjs', torch.FloatTensor)
    graphs1 = load_tensor(dir_input1 + 'graph', torch.FloatTensor)
    morgans1 = load_tensor(dir_input1 + 'morgan', torch.LongTensor)
    adjacencies1 = load_tensor(dir_input1 + 'adjacencies', torch.FloatTensor)
    size1 = load_tensor_label(dir_input1 + 'size', torch.FloatTensor)
    label1 = load_tensor_label(dir_input1 + 'label', torch.FloatTensor)
    water1 = load_tensor_label(dir_input1 + 'water', torch.FloatTensor)
    train_dataset = list(zip(fingerprints1, pgraphs1, padjacencies1,graphs1, morgans1, adjacencies1, size1, label1, water1))

    train_dataset = shuffle_dataset(train_dataset, 1234)


    print(dir_input2 + 'compounds')
    fingerprints2 = load_tensor(dir_input2 + 'fingerprints', torch.FloatTensor)
    pgraphs2 = load_tensor(dir_input2 + 'pgraph', torch.FloatTensor)
    padjacencies2=load_tensor(dir_input2 + 'padjs', torch.FloatTensor)
    graphs2 = load_tensor(dir_input2 + 'graph', torch.FloatTensor)
    morgans2 = load_tensor(dir_input2 + 'morgan', torch.LongTensor)
    adjacencies2 = load_tensor(dir_input2 + 'adjacencies', torch.FloatTensor)
    size2 = load_tensor_label(dir_input2 + 'size', torch.FloatTensor)
    label2 = load_tensor_label(dir_input2 + 'label', torch.FloatTensor)
    water2 = load_tensor_label(dir_input2 + 'water', torch.FloatTensor)
    test_dataset = list(zip(fingerprints2, pgraphs2, padjacencies2, graphs2, morgans2, adjacencies2, size2, label2, water2))
    test_dataset = shuffle_dataset(test_dataset, 1234)

    # dataset_dev, dataset_test = split_dataset(dataset_, 0.5)
    return train_dataset, test_dataset,morgans1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()
smoothl1_loss_fn=torch.nn.SmoothL1Loss()
r2_score=R2Score().to(device)

DATASET = "D:/microplastics/model/polyDTA"
radius = 2
ngram = 3
dim = 75
layer_gnn = 3
side = 7
side2 = 4
window = (2 * side + 1)
window2 = (2 * side2 + 1)
layer_cnn = 3
layer_output = 2
lr_min = 1e-5
lr_decay = 0.5
decay_interval = 6

iteration = 380

'''
数据集导入准备
'''

dir_input1 = 'D:/microplastics/model/polyDTA/train_input/'
dir_input2 = 'D:/microplastics/model/polyDTA/valid_input/'

dataset_train, dataset_test,label = dataset(dir_input1, dir_input2)

print(dataset_train[0][3].shape[1])



def get_metrics(model, dataset_test, batch):
    valid_outputs = []
    valid_labels = []
    valid_loss = []
    valid_mae_loss = []
    valid_r2=[]
    


    N = len(dataset_test)
    
    i = 0
    fingerprints, pgraphs, padjacencys, graphs,morgans, adjs, sizes, labels, waters = [], [], [], [], [], [], [], [],[]
    for data in dataset_test:
        i = i + 1
        fingerprint, pgraph, padjacency, graph, morgan, adj, size, label, water = data
        fingerprints.append(fingerprint)
        pgraphs.append(pgraph)
        padjacencys.append(padjacency)
        graphs.append(graph)
        morgans.append(morgan)
        adjs.append(adj)
        sizes.append(size)
        labels.append(label)
        waters.append(water)
        if i % batch == 0 or i == N:
            # print(words[0])
            fingerprints1, pgraphs1, padjacencys1, graphs1,morgans1, adjs1, sizes1, labels1, waters1 = pack(fingerprints, 
                pgraphs, padjacencys, graphs,morgans, adjs, sizes, labels, waters,device)
            # print(words.shape)
            data = (fingerprints1, pgraphs1, padjacencys1, graphs1,morgans1, adjs1, sizes1, labels1, waters1)

            outputs=model(data)

            loss = loss_fn(outputs, torch.tensor(labels).to(device).float())
            mae_loss = mae_loss_fn(outputs, torch.tensor(labels).to(device).float())

            outputs=outputs.detach()
            outputs=outputs.numpy()
            outputs=outputs.tolist()
            valid_outputs+=outputs
            '''

            valid_outputs += outputs.cpu().detach().numpy().tolist()
            '''
            valid_loss.append(loss.cpu().detach().numpy())
            valid_mae_loss.append(mae_loss.cpu().detach().numpy())

            valid_labels += labels

            fingerprints, pgraphs, padjacencys, graphs,morgans, adjs, sizes, labels, waters = [], [], [], [], [], [], [], [], []
        else:
                continue
            
    loss = np.mean(np.array(valid_loss).flatten())
    mae_loss = np.mean(np.array(valid_mae_loss).flatten())
    r2=r2_score(torch.tensor(valid_outputs).to(device).float(),torch.tensor(valid_labels).to(device).float()) 

    return loss, mae_loss,r2, valid_outputs,valid_labels

def train(max_epochs, model, optimizer, scheduler, dataset_train, dataset_test, batch):
    best_val_loss = 100
    consecutiveepoch_num=0
    now_train_loss=0
    now_mae=0
    now_r2=0
    best_outputs=[]

    for epoch in range(max_epochs):
        model.train()
        np.random.shuffle(dataset_train)
        N = len(dataset_train)
        running_loss=[]
        running_mae_loss=[]
        loss_total2 = 0
        i = 0

        optimizer.zero_grad()

        fingerprints, pgraphs, padjacencys, graphs,morgans, adjs, sizes, labels, waters = [], [], [], [], [], [], [], [],[]
        for data in dataset_train:
            i = i + 1
            fingerprint, pgraph, padjacency, graph, morgan, adj, size, label, water = data
            fingerprints.append(fingerprint)
            pgraphs.append(pgraph)
            padjacencys.append(padjacency)
            graphs.append(graph)
            morgans.append(morgan)
            adjs.append(adj)
            sizes.append(size)
            labels.append(label)
            waters.append(water)
            if i % batch == 0 or i == N:
                # print(words[0])
                fingerprints1, pgraphs1, padjacencys1, graphs1,morgans1, adjs1, sizes1, labels1, waters1 = pack(fingerprints, 
                    pgraphs, padjacencys, graphs,morgans, adjs, sizes, labels, waters,device)
                # print(words.shape)
                data = (fingerprints1, pgraphs1, padjacencys1, graphs1,morgans1, adjs1, sizes1, labels1, waters1)
                outputs = model(data)
                loss=smoothl1_loss_fn(outputs,torch.tensor(labels).to(device).float())
                mae_loss = mae_loss_fn(outputs, torch.tensor(labels).to(device).float())
                loss.backward()
                optimizer.step()

                running_loss.append(loss.cpu().detach())
                running_mae_loss.append(mae_loss.cpu().detach())

              
                fingerprints, pgraphs, padjacencys, graphs,morgans, adjs, sizes, labels, waters = [], [], [], [], [], [], [], [], []
            else:
                continue

        model.eval()

        val_loss, mae_loss,r2, outputs,labels = get_metrics(model, dataset_test, batch)
        scheduler.step(val_loss)

        with open('D:/microplastics/model/polyDTA/learningloss2.txt','a') as f:
            f.write("train_loss"+str(np.mean(np.array(running_loss)))+'\t'+'val_loss'+str(val_loss)+'\t'+'val_mae_loss'+str(mae_loss)+'\t'+'Val_R2'+str(r2)+'\n')
            f.close()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            consecutiveepoch_num=0
            now_train_loss=str(np.mean(np.array(running_loss)))
            now_mae=mae_loss 
            now_r2=r2
            best_outputs=outputs
            print("Epoch: " + str(epoch + 1) + "  train_loss " + str(np.mean(np.array(running_loss))) + " Val_loss " + str(val_loss) + " MAE Val_loss " + str(mae_loss)+'Val_R2'+str(r2))
            torch.save(model.state_dict(), "D:/microplastics/best_model/"+str(best_val_loss)+".tar")

        else:
            consecutiveepoch_num+=1

        if consecutiveepoch_num>=15:
            break
   

    return best_val_loss
        





    

def main(max_epochs,dataset_train,dataset_test,lr, weight_decay, batch, dropout):
    model = Predictor(ResMLP, Affine, dim, window, window2, layer_gnn, layer_cnn, layer_output, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True)

    val_mse=train(max_epochs,model,optimizer,scheduler,dataset_train,dataset_test, batch)

    return val_mse



main(200,dataset_train,dataset_test,lr=0.000687470637077872,weight_decay=0.00374695039678227,batch=128,dropout=0.01180129237722248)

