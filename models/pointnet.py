import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class STNkd(nn.Module):

    def __init__(self, k=3):
        super(STNkd, self).__init__()
        # self.conv1 = torch.nn.Conv1d(k, 64, 1)
        # self.bn1 = nn.BatchNorm1d(64)
        self.seq1 = nn.Sequential(torch.nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU())

        # self.conv2 = torch.nn.Conv1d(64, 128, 1)
        # self.bn2 = nn.BatchNorm1d(128)
        self.seq2 = nn.Sequential(torch.nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())

        # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # self.bn3 = nn.BatchNorm1d(1024)
        self.seq3 = nn.Sequential(torch.nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())

        # self.fc1 = nn.Linear(1024, 512)
        # self.bn4 = nn.BatchNorm1d(512)
        self.seq4 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU())

        # self.fc2 = nn.Linear(512, 256)
        # self.bn5 = nn.BatchNorm1d(256)
        self.seq5 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU())

        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.quant_stubs = nn.ModuleList(
            [torch.quantization.QuantStub() for _ in range(1)]
        )
        self.dequant_stubs = nn.ModuleList(
            [torch.quantization.DeQuantStub() for _ in range(1)]
        )

        self.k = k

    def add_quant_stubs(self):
        self.seq1 = nn.Sequential(torch.quantization.DeQuantStub(), self.seq1, torch.quantization.QuantStub())
        self.seq1[1].qconfig = None
        
    def forward(self, x):
        batchsize = x.size()[0]

        x = self.seq1(x)# F.relu(self.bn1(self.conv1(x)))
        x = self.seq2(x)# F.relu(self.bn2(self.conv2(x)))
        x = self.seq3(x)# F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.seq4(x)# F.relu(self.bn4(self.fc1(x)))
        x = self.seq5(x)# F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=torch.float32).flatten().view(
            1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()

        if self.dequant_stubs != None:
            x = self.dequant_stubs[0](x)
        
        x = x + iden

        if self.quant_stubs != None:
            x = self.quant_stubs[0](x)

        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):

    def __init__(self, feature_len, extra_feature_len=32):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=feature_len)
        # self.fstn = STNkd(k=32 + extra_feature_len)

        # self.conv1 = torch.nn.Conv1d(feature_len, 64, 1)
        # self.bn1 = nn.BatchNorm1d(64)
        self.seq1 = nn.Sequential(torch.nn.Conv1d(feature_len, 64, 1), nn.BatchNorm1d(64), nn.ReLU())

        # self.conv2 = torch.nn.Conv1d(64 + extra_feature_len, 128, 1)
        # self.bn2 = nn.BatchNorm1d(128)
        self.seq2 = nn.Sequential(torch.nn.Conv1d(64 + extra_feature_len, 128, 1), nn.BatchNorm1d(128), nn.ReLU())

        # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # self.bn3 = nn.BatchNorm1d(1024)
        self.seq3 = nn.Sequential(torch.nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.quant_stubs = nn.ModuleList(
            [torch.quantization.QuantStub() for _ in range(2)]
        )
        self.dequant_stubs = nn.ModuleList(
            [torch.quantization.DeQuantStub() for _ in range(2)]
        )

    def add_quant_stubs(self):
        self.stn.add_quant_stubs()

    def forward(self, x):
        # trans pc only and layer 1
        trans = self.stn(x[:, :3])

        if self.dequant_stubs != None:
            x = self.dequant_stubs[0](x)
            trans = self.dequant_stubs[1](trans)

        x_p = torch.bmm(x[:, :3].transpose(2, 1), trans)

        if self.dequant_stubs != None:
            x_p = self.quant_stubs[0](x_p)
            x = self.quant_stubs[1](x)

        x_p = self.seq1(x_p.transpose(2, 1))
        # x_p = self.conv1(x_p.transpose(2, 1))
        # x_p = F.relu(self.bn1(x_p))

        # concat rgbd features
        x = torch.cat([x_p, x[:, 3:]], 1)
        # feature trans and layer 2


        # x = self.conv2(x)
        # x = F.relu(self.bn2(x))

        x = self.seq2(x)
        # feature layer 3
        # x = self.conv3(x)
        # x = self.bn3(x)
        x = self.seq3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x
