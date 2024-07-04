import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from GCN_model import GraphConvolution
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



class PowerLayer(nn.Module):
    '''
    The power layer: calculates the log-transformed power of the data
    '''
    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        eps = 1e-7
        return torch.log(self.pooling(x.pow(2))+eps)

class MSA(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=4):
        super(MSA, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att

class ADDGNet(nn.Module):
    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate*pool))
        )

    def __init__(self, num_classes = 8, dropout_rate = 0.2):
        super(ADDGNet, self).__init__()

        self.window = [0.5, 0.25, 0.125]
        self.pool = 16
        self.Tout = 16
        self.sampling_rate = 64
        self.pool_step_rate = 0.25
        self.channel = 122
        self.datasize = [1,122,100]
        self.gcnout = 32


        


        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.temporal_learner(self.datasize[0], self.Tout,
                                               (1, int(self.window[0] * self.sampling_rate)),
                                               self.pool, self.pool_step_rate)
        self.Tception2 = self.temporal_learner(self.datasize[0], self.Tout,
                                               (1, int(self.window[1] * self.sampling_rate)),
                                               self.pool, self.pool_step_rate)
        self.Tception3 = self.temporal_learner(self.datasize[0], self.Tout,
                                               (1, int(self.window[2] * self.sampling_rate)),
                                               self.pool, self.pool_step_rate)
        
        self.BN_t = nn.BatchNorm2d(self.Tout)
        self.BN_t_ = nn.BatchNorm2d(self.Tout)

        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(self.Tout, self.Tout, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2)))
        

        size = self.get_size_temporal(self.datasize)

        self.graphpool = [size[-1],16]
        
        
        # diag(W) to assign a weight to each local areas
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)

        # # aggregate function
        # self.aggregate = Aggregator(self.idx)


        # trainable adj weight for global network
        self.global_adj = nn.Parameter(torch.FloatTensor(self.graphpool[1], self.graphpool[1]), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)

        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.channel)
        self.bn_ = nn.BatchNorm1d(self.graphpool[1])

        # learn the global network of networks
        self.GCN = GraphConvolution(size[-1], self.gcnout)


        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.graphpool[1] * self.gcnout), num_classes))
        

        self.prelu = nn.PReLU()

        # DIFFPOOL
        self.lamda = 1
        self.assign_conv_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        for i in range(len(self.graphpool)-1):
            self.assign_conv_modules.append(GraphConvolution(self.graphpool[i], self.graphpool[i+1]))
            self.assign_pred_modules.append(nn.Linear(self.graphpool[i+1], self.graphpool[i+1]))

        self.threshold = nn.Parameter(torch.empty(self.channel,self.channel),requires_grad=True)
        nn.init.constant_(self.threshold, 0.8)

        self.MSA=MSA(16,16,16)
        self.layer_norm = nn.LayerNorm(16)




    def forward(self, x,Sadj):


        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
       
        # adj = self.get_adjacency(out)
        # adj = torch.abs(adj)
        out = self.bn(out)
        x,_,loss = self.Diffpool_forward(out,Sadj,0)



        x = x.permute(0, 2, 1)
        xmsa = self.MSA(x)
        x = self.layer_norm(x + xmsa)
        x = x.permute(0, 2, 1)

        adj = self.get_adj(x)
        # out = self.bn(out)
        out = self.GCN(x, adj)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out,loss
    

    def corrcoef(self,x):
        x_reducemean = x - torch.mean(x, dim=1, keepdim=True)
        numerator = torch.matmul(x_reducemean, x_reducemean.T)
        no = torch.norm(x_reducemean, p=2, dim=1).unsqueeze(1)
        denominator = torch.matmul(no, no.T)
        corrcoef = numerator / denominator
        return corrcoef

    def get_adjacency(self,x):
        res=[]
        for i in range(x.size()[0]):

            corr_matrix = self.corrcoef(x[i]) 
            abs_corr = torch.abs(corr_matrix)
            adj = abs_corr - torch.eye(abs_corr.size()[0]).to(DEVICE)

            res.append(adj)
        return torch.cat(res,dim=0).reshape(x.size()[0],x.size()[1],-1)
    

    def Diffpool_forward(self,x,adj,i):
        # [batch_size x num_nodes x next_lvl_num_nodes]
        self.assign_tensor = self.assign_conv_modules[i](x, adj)
        self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))

        # Link Prediction loss
        threadj = torch.where(adj >= self.threshold, torch.tensor(1).to(DEVICE), torch.tensor(0).to(DEVICE))
        pred_adj = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
        pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
        eps = 1e-7
        self.link_loss = -threadj * torch.log(pred_adj+eps) - (1-threadj) * torch.log(1-pred_adj+eps)
        num_entries = adj.size()[1] * adj.size()[1] * adj.size()[0]
        self.link_loss = torch.sum(self.link_loss) / float(num_entries)

        

        # Entropy Regularization loss
        self.Eloss = (-self.assign_tensor * torch.log2(self.assign_tensor+eps)).sum()
        self.Eloss = self.Eloss/(self.assign_tensor.size()[0]*self.assign_tensor.size()[1]*self.assign_tensor.size()[2])

        # Full Loss
        b = self.assign_tensor.shape[0]
        n = self.assign_tensor.shape[2]

        lsum = torch.sum(self.assign_tensor,dim=1)

        y = torch.where(lsum < 4, (lsum - 4) ** 2, lsum)
        y = torch.where(lsum > 12, (lsum - 12) ** 2, y)
        z = torch.where(torch.logical_and(lsum >= 4, lsum <= 12), torch.tensor(0.0).to(DEVICE), lsum)
        lsum = torch.where(torch.logical_or(lsum < 4, lsum > 12), y, z)
        lsum = torch.sum(lsum,dim=1)
        lsum = torch.sum(lsum,dim=0)
        max_values, _ = torch.max(self.assign_tensor, dim=1, keepdim=False)
        max_values = torch.sum((1-max_values),dim=1)
        max_values = torch.sum(max_values,dim=0)
        self.Floss = lsum / b / n + self.lamda * max_values / b


        # update pooled features and adj matrix
        x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), x)
        adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
        return x,adj,self.link_loss+self.Eloss+self.Floss
    
    
    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = self.prelu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    # def local_filter_fun(self, x, w):
    #     w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
    #     x = F.relu(torch.mul(x, w) - self.local_filter_bias)
    #     return x

    def get_adj(self, x, self_loop=True):
        # x: b, node, feature
        adj = self.self_similarity(x)   # b, n, n
        num_nodes = adj.shape[-1]
        adj = (adj * (self.global_adj + self.global_adj.transpose(1, 0))).to(DEVICE)
        adj = F.relu(adj)
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s

