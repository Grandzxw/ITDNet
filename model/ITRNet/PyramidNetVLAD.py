import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLADBase(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True):
        super(NetVLADBase, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(
            torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(
            torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(feature_size * cluster_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.bn1 = nn.LayerNorm(cluster_size)  # ✅ 改为 LayerNorm
        else:
            self.cluster_biases = nn.Parameter(torch.randn(cluster_size) * 1 / math.sqrt(feature_size))  
            self.bn1 = None

        self.bn2 = nn.LayerNorm(output_dim)  # ✅ 这里也换成 LayerNorm
        if gating:
            self.context_gating = GatingContext(output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))

        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)  # ✅ 使用 LayerNorm
            activation = activation.view(-1, self.max_samples, self.cluster_size)
        else:
            activation = activation + self.cluster_biases

        activation = self.softmax(activation)
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = activation.transpose(2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = vlad.transpose(2, 1)
        vlad = vlad - a
        vlad = F.normalize(vlad, dim=1, p=2, eps=1e-12).contiguous()
        vlad = vlad.view((-1, self.cluster_size * self.feature_size))

        return vlad


class WPNNetVLAD(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True):
        super(WPNNetVLAD, self).__init__()
        self.vlad0 = NetVLADBase(feature_size[0], max_samples[0], cluster_size[0], output_dim[0], gating, add_batch_norm)
        self.vlad1 = NetVLADBase(feature_size[1], max_samples[1], cluster_size[1], output_dim[1], gating, add_batch_norm)
        self.vlad2 = NetVLADBase(feature_size[2], max_samples[2], cluster_size[2], output_dim[2], gating, add_batch_norm)

        sum_cluster_size = cluster_size[0] + cluster_size[1] + cluster_size[2]
        self.hidden_weights = nn.Parameter(
            torch.randn(feature_size[0] * sum_cluster_size, output_dim[0]) * 1 / math.sqrt(feature_size[0]))
        
        self.bn2 = nn.LayerNorm(output_dim[0])  # ✅ 改为 LayerNorm
        self.gating = gating
        if self.gating:
            self.context_gating = GatingContext(output_dim[0], add_batch_norm=add_batch_norm)

    def forward(self, f0, f1, f2):
        v0 = self.vlad0(f0)
        v1 = self.vlad1(f1)
        v2 = self.vlad2(f2)

        vlad = torch.cat((v0, v1, v2), dim=-1)
        vlad = torch.matmul(vlad, self.hidden_weights)
        vlad = self.bn2(vlad)  # ✅ 使用 LayerNorm

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.bn1 = nn.LayerNorm(dim)  # ✅ 彻底换成 LayerNorm
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)
        if self.add_batch_norm:
            gates = self.bn1(gates)  # ✅ LayerNorm
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)
        activation = x * gates

        return activation




if __name__ == '__main__':    
    x1 = torch.randn(8, 256, 48, 1)
    x2 = torch.randn(8, 256, 24, 1)
    x3 = torch.randn(8, 256, 12, 1)
    
    feature_size = [256,256,256]
    # max_samples = [64,128,256]
    max_samples = [12,24,48]
    cluster_size = [4,16,64]
    output_dim = [256,256,256]
    
    model = WPNNetVLAD(feature_size, max_samples,cluster_size,output_dim)
    r = model(x3,x2,x1)
    print(r.shape)
    