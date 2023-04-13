import torch
import torch.nn as nn
import numpy as np
import xlwt
import metrics

CHAR_SMI_SET_LEN = 64
CHAR_SEQ_SET_LEN = 21+4+7
CHAR_PKT_SET_LEN = 21+4+7

class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()


class BasicConv(nn.Module):
    def __init__(self, nIn, nOut, kSize=3, stride=1, dilation=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * dilation  # padding会保持长度不变
        self.bn = nn.BatchNorm1d(nIn)
        self.relu = nn.PReLU()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, dilation=dilation)

    def forward(self, input):
        output = self.conv(self.relu(self.bn(input)))
        return output
    
class ResConv(nn.Module):   
    def __init__(self, nIn, nOut, kSize=3, stride=1, dilation=1):
        super().__init__()
        self.inch = nIn
        self.outch = nOut
        
        self.conv1 = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=int((kSize - 1)/2)*d, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(nOut)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(nOut, nOut, kSize, stride=stride, padding=int((kSize - 1)/2)*2*d, dilation=2*dilation)
        self.bn2 = nn.BatchNorm1d(nOut)
        self.relu2 = nn.PReLU()
        
        if self.inch != self.outch:
            self.conv = nn.Conv1d(nIn, nOut, 1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.bn2(self.conv2(x1))
        if self.inch != self.outch:
            residual = self.conv(residual)
        output = self.relu2(residual+x1)
        return output

class DenseConvLayer(nn.Module):   
    def __init__(self, in_c, out_c, kSize=3, stride=1, dilation=1):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_c)
        self.relu = nn.PReLU()
        self.conv = nn.Conv1d(in_c, out_c, kSize, stride=stride, padding=int((kSize - 1)/2)*dilation, dilation=dilation)  

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        x = torch.cat((x, out), 1)
        return x
    
class WLPPDTA3to1(nn.Module):

    def __init__(self, onehot=False, dilation=True):
        super().__init__()

        self.onehot = onehot
        self.dilation = dilation
        
        smi_embed_size = 128
        pkt_embed_size = 128
        seq_embed_size = 128
        
        smi_oc = 128
        pkt_oc = 128
        seq_oc = 128

        self.seq_embed = nn.Linear(CHAR_SEQ_SET_LEN, seq_embed_size)  
        if self.onehot: # 用linear定义编码层
            self.smi_embed = nn.Linear(CHAR_SMI_SET_LEN, smi_embed_size)  
        else:  # 运行时输入（N * 词索引编号） -> （N * 词数 * 词嵌入向量）
            self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN+1, smi_embed_size)  # 定义时输入词汇表大小(64+1) + 每个词用多少字符表示(128)，
            
        conv_seq = []  
        conv_seq.append(nn.Conv1d(seq_embed_size, 32, kernel_size=3, stride=1, padding=1, dilation=1))  # 序列输入信道64
        conv_seq.append(self._make_dense(32, seq_oc, 4))  
        conv_seq.append(nn.AdaptiveMaxPool1d(1))  
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)  # (N, 128)

        conv_pkt = []
        conv_pkt.append(nn.Conv1d(pkt_embed_size, 32, kernel_size=3, stride=1, padding=1, dilation=1))  # 口袋输入信道64
        conv_pkt.append(self._make_dense(32, pkt_oc, 3))
        conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)  # (N, 64)

        conv_smi = []  
        conv_smi.append(nn.Conv1d(smi_embed_size, 32, kernel_size=3, stride=1, padding=1, dilation=1))  # 化合物输入信道128
        conv_smi.append(self._make_dense(32, smi_oc, 3))            
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N, 64)
        
        self.cat_dropout = nn.Dropout(0.2)  # 失活神经元比例
        
        self.classifier = nn.Sequential(
            nn.Linear(seq_oc+pkt_oc+smi_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1)
        )
        
    def _make_dense(self, nIn, nOut, dense_layer_num, kSize=3, stride=1):
        n = int((nOut-nIn) / dense_layer_num)
        n1 = nOut - nIn - n*(dense_layer_num-1)
        
        layers = []
        for l in range(dense_layer_num-1):
            if self.dilation:
                layers.append(DenseConvLayer(nIn, n, kSize, stride=stride, dilation=pow(2,l)))
            else:
                layers.append(DenseConvLayer(nIn, n, kSize, stride=stride, dilation=1))
            nIn += n
            
        if self.dilation:
            layers += [
                DenseConvLayer(nIn, n1, kSize, stride=stride, dilation=pow(2,dense_layer_num-1)),
                nn.BatchNorm1d(nIn+n1),
                nn.PReLU()
                      ]
        else:
            layers += [
                DenseConvLayer(nIn, n1, kSize, stride=stride, dilation=1),
                nn.BatchNorm1d(nIn+n1),
                nn.PReLU()
                      ]           
        assert nOut == nIn+n1
        
        return nn.Sequential(*layers)

    def _make_conv(self, nIn, nOut, layer_num, kSize=3, stride=1, dilation=1):
        n = int((nOut-nIn) / layer_num)
        
        layers = []
        for l in range(layer_num-1):
            layers.append(BasicConv(nIn, n, kSize, stride=stride, dilation=dilation))
            nIn = n
        layers += [
            BasicConv(nIn, nOut, kSize, stride=stride, dilation=dilation),
            nn.BatchNorm1d(nOut),
            nn.PReLU()
                  ]
        
        return nn.Sequential(*layers)
    
    def forward(self, seq, pkt, smi):
        # assert seq.shape == (N,L,40)
        seq_embed = self.seq_embed(seq)  # (N,L,128)
        seq_embed = torch.transpose(seq_embed, 1, 2)  # (N,128,L)
        seq_conv = self.conv_seq(seq_embed)  # (N,128)

        # assert pkt.shape == (N,L,40)
        pkt_embed = self.seq_embed(pkt)  # (N,L,128)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)  # (N,128)

        # assert smi.shape == (N, L)
        smi_embed = self.smi_embed(smi)  # (N,L,128)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)
        
        cat = torch.cat([seq_conv, pkt_conv, smi_conv], dim=1)  # (N,128*3)
        cat = self.cat_dropout(cat)
        
        output = self.classifier(cat)
        return output
    
    
class WLPPDTA3to2(nn.Module):

    def __init__(self, onehot=False, dilation=True):
        super().__init__()

        self.onehot = onehot
        self.dilation = dilation
        
        smi_embed_size = 128
        pkt_embed_size = 128
        seq_embed_size = 128
        
        smi_oc = 128
        pkt_oc = 128
        seq_oc = 128

        self.seq_embed = nn.Linear(CHAR_SEQ_SET_LEN, seq_embed_size)  
        if self.onehot: # 用linear定义编码层
            self.smi_embed = nn.Linear(CHAR_SMI_SET_LEN, smi_embed_size)  
        else:  # 运行时输入（N * 词索引编号） -> （N * 词数 * 词嵌入向量）
            self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN+1, smi_embed_size)  # 定义时输入词汇表大小(64+1) + 每个词用多少字符表示(128)，
        
        conv_seq = []  
        conv_seq.append(nn.Conv1d(seq_embed_size, 32, kernel_size=3, stride=1, padding=1, dilation=1))  # 序列输入信道64
        conv_seq.append(self._make_dense(32, seq_oc, 4))  
        conv_seq.append(nn.AdaptiveMaxPool1d(1))  
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)  # (N, 128)

        conv_pkt = []
        conv_pkt.append(nn.Conv1d(pkt_embed_size, 32, kernel_size=3, stride=1, padding=1, dilation=1))  # 口袋输入信道64
        conv_pkt.append(self._make_dense(32, pkt_oc, 3))
        conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)  # (N, 128)

        conv_smi = []  
        conv_smi.append(nn.Conv1d(smi_embed_size, 32, kernel_size=3, stride=1, padding=1, dilation=1))  # 化合物输入信道128
        conv_smi.append(self._make_dense(32, smi_oc, 3))            
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N, 128)
        
        self.cat_dropout = nn.Dropout(0.1)  # 失活神经元比例
        
        self.classifier1 = nn.Sequential(
            nn.Linear(seq_oc+smi_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1)
        )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(pkt_oc+smi_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1)
        )
        
    def _make_dense(self, nIn, nOut, dense_layer_num, kSize=3, stride=1):
        n = int((nOut-nIn) / dense_layer_num)
        n1 = nOut - nIn - n*(dense_layer_num-1)
        
        layers = []
        for l in range(dense_layer_num-1):
            if self.dilation:
                layers.append(DenseConvLayer(nIn, n, kSize, stride=stride, dilation=pow(2,l)))
            else:
                layers.append(DenseConvLayer(nIn, n, kSize, stride=stride, dilation=1))
            nIn += n
            
        if self.dilation:
            layers += [
                DenseConvLayer(nIn, n1, kSize, stride=stride, dilation=pow(2,dense_layer_num-1)),
                nn.BatchNorm1d(nIn+n1),
                nn.PReLU()
                      ]
        else:
            layers += [
                DenseConvLayer(nIn, n1, kSize, stride=stride, dilation=1),
                nn.BatchNorm1d(nIn+n1),
                nn.PReLU()
                      ]           
        assert nOut == nIn+n1
        
        return nn.Sequential(*layers)

    def _make_conv(self, nIn, nOut, layer_num, kSize=3, stride=1, dilation=1):
        n = int((nOut-nIn) / layer_num)
        
        layers = []
        for l in range(layer_num-1):
            layers.append(BasicConv(nIn, n, kSize, stride=stride, dilation=dilation))
            nIn = n
        layers += [
            BasicConv(nIn, nOut, kSize, stride=stride, dilation=dilation),
            nn.BatchNorm1d(nOut),
            nn.PReLU()
                  ]
        
        return nn.Sequential(*layers)
    
    def forward(self, seq, pkt, smi):
        # assert seq.shape == (N,L,40)
        seq_embed = self.seq_embed(seq)  # (N,L,128)
        seq_embed = torch.transpose(seq_embed, 1, 2)  # (N,128,L)
        seq_conv = self.conv_seq(seq_embed)  # (N,128)

        # assert pkt.shape == (N,L,40)
        pkt_embed = self.seq_embed(pkt)  # (N,L,128)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)  # (N,128)

        # assert smi.shape == (N, L)
        smi_embed = self.smi_embed(smi)  # (N,L,128)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)
        
        cat1 = torch.cat([seq_conv, smi_conv], dim=1)  # (N,128*2)
        cat1 = self.cat_dropout(cat1)
        output1 = self.classifier1(cat1)

        cat2 = torch.cat([pkt_conv, smi_conv], dim=1)  # (N,128*2)
        cat2 = self.cat_dropout(cat2)
        output2 = self.classifier2(cat2)
        
        return output1, output2

class WLPPDTA2to1(nn.Module):

    def __init__(self, onehot=False, tag=1, dilation=True):  # 1 -> use seq, 0 -> use pkt
        super().__init__()

        self.onehot = onehot
        self.tag = tag
        self.dilation = dilation
        
        smi_embed_size = 128
        pkt_embed_size = 128
        seq_embed_size = 128
        
        smi_oc = 128
        pkt_oc = 128
        seq_oc = 128

        self.seq_embed = nn.Linear(CHAR_SEQ_SET_LEN, seq_embed_size)  
        if self.onehot: # 用linear定义编码层
            self.smi_embed = nn.Linear(CHAR_SMI_SET_LEN, smi_embed_size)  
        else:  # 运行时输入（N * 词索引编号） -> （N * 词数 * 词嵌入向量）
            self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN+1, smi_embed_size)  # 定义时输入词汇表大小(64+1) + 每个词用多少字符表示(128)，
        
        conv_seq = []  
        if self.tag:
            oc = seq_oc
            conv_seq.append(nn.Conv1d(seq_embed_size, 32, kernel_size=3, stride=1, padding=1, dilation=1))  # 序列输入信道64
            conv_seq.append(self._make_dense(32, oc, 4))  # init 8
        else:
            oc = pkt_oc
            conv_seq.append(nn.Conv1d(pkt_embed_size, 32, kernel_size=3, stride=1, padding=1, dilation=1))  # 口袋输入信道64
            conv_seq.append(self._make_dense(32, oc, 3))  # init 4
        conv_seq.append(nn.AdaptiveMaxPool1d(1))  
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)  # (N, 128)

        conv_smi = []  
        conv_smi.append(nn.Conv1d(smi_embed_size, 32, kernel_size=3, stride=1, padding=1, dilation=1))  # 化合物输入信道128
        conv_smi.append(self._make_dense(32, smi_oc, 3))  # init 6     
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N, 64)
        
        self.classifier = nn.Sequential(
            nn.Linear(oc+smi_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1)
        )
        
    def _make_dense(self, nIn, nOut, dense_layer_num, kSize=3, stride=1):
        n = int((nOut-nIn) / dense_layer_num)
        n1 = nOut - nIn - n*(dense_layer_num-1)
        
        layers = []
        for l in range(dense_layer_num-1):
            if self.dilation:
                layers.append(DenseConvLayer(nIn, n, kSize, stride=stride, dilation=pow(2,l)))
            else:
                layers.append(DenseConvLayer(nIn, n, kSize, stride=stride, dilation=1))
            nIn += n
            
        if self.dilation:
            layers += [
                DenseConvLayer(nIn, n1, kSize, stride=stride, dilation=pow(2,dense_layer_num-1)),
                nn.BatchNorm1d(nIn+n1),
                nn.PReLU()
                      ]
        else:
            layers += [
                DenseConvLayer(nIn, n1, kSize, stride=stride, dilation=1),
                nn.BatchNorm1d(nIn+n1),
                nn.PReLU()
                      ]           
        assert nOut == nIn+n1
        
        return nn.Sequential(*layers)

    def _make_conv(self, nIn, nOut, layer_num, kSize=3, stride=1, dilation=1):
        n = int((nOut-nIn) / layer_num)
        
        layers = []
        for l in range(layer_num-1):
            layers.append(BasicConv(nIn, n, kSize, stride=stride, dilation=dilation))
            nIn = n
        layers += [
            BasicConv(nIn, nOut, kSize, stride=stride, dilation=dilation),
            nn.BatchNorm1d(nOut),
            nn.PReLU()
                  ]
        
        return nn.Sequential(*layers)
    
    def forward(self, seq, pkt, smi):
        if self.tag:
            in_seq = seq
        else:
            in_seq = pkt
        # assert seq.shape == (N,L,40)
        seq_embed = self.seq_embed(in_seq)  # (N,L,128)
        seq_embed = torch.transpose(seq_embed, 1, 2)  # (N,128,L)
        seq_conv = self.conv_seq(seq_embed)  # (N,128)

        # assert smi.shape == (N, L)
        smi_embed = self.smi_embed(smi)  # (N,L,128)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)
        
        cat = torch.cat([seq_conv, smi_conv], dim=1)  # (N,128*2)
        output = self.classifier(cat)
        
        return output

############################################# reference model ####################################################

class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d  # padding会保持长度不变
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

class DilatedParllelResidualBlockA(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
#             print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input):
        # reduce
        output1 = self.c1(input)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output    

class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
#             print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input):
        # reduce
        output1 = self.c1(input)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        # merge
        combine = torch.cat([d1, add1, add2, add3], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output


class DeepDTAF(nn.Module):

    def __init__(self):
        super().__init__()

        smi_embed_size = 128
        seq_embed_size = 128
        
        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128

        self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN+1, smi_embed_size)
        self.seq_embed = nn.Linear(CHAR_SEQ_SET_LEN, seq_embed_size)  # (N, *, H_{in}) -> (N, *, H_{out}) 表示蛋白中每个氨基酸独立特征的描述编码

        conv_seq = []
        ic = seq_embed_size  # 序列输入信道128
        for oc in [32, 64, 64, seq_oc]:  # 序列输出信道128
            conv_seq.append(DilatedParllelResidualBlockA(ic, oc))  # 信道变化，连续扩张卷积
            ic = oc
        conv_seq.append(nn.AdaptiveMaxPool1d(1))  # (N, oc)
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)

        conv_pkt = []
        ic = seq_embed_size  # 口袋输入信道128
        for oc in [32, 64, pkt_oc]:  # 口袋输出信道128
            conv_pkt.append(nn.Conv1d(ic, oc, 3))  # (N,C,L)
            conv_pkt.append(nn.BatchNorm1d(oc))
            conv_pkt.append(nn.PReLU())
            ic = oc
        conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)  # (N,oc)

        conv_smi = []
        ic = smi_embed_size  #  化合物输入信道128
        for oc in [32, 64, smi_oc]:  #  化合物输出信道128
            conv_smi.append(DilatedParllelResidualBlockB(ic, oc))
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N,128)
        
        self.cat_dropout = nn.Dropout(0.2)  # 失活神经元比例
        
        self.classifier = nn.Sequential(
            nn.Linear(seq_oc+pkt_oc+smi_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1),
            nn.PReLU())
        

    def forward(self, seq, pkt, smi):
        # assert seq.shape == (N,L,40)
        seq_embed = self.seq_embed(seq)  # (N,L,128)
        seq_embed = torch.transpose(seq_embed, 1, 2)  # (N,32,L)
        seq_conv = self.conv_seq(seq_embed)  # (N,128)

        # assert pkt.shape == (N,L,40)
        pkt_embed = self.seq_embed(pkt)  # (N,L,128)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)  # (N,128)

        # assert smi.shape == (N, L)
        smi_embed = self.smi_embed(smi)  # (N,L,128)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)

        cat = torch.cat([seq_conv, pkt_conv, smi_conv], dim=1)  # (N,128*3)
        cat = self.cat_dropout(cat)
        
        output = self.classifier(cat)
        return output

class DeepDTAF_2to1(nn.Module):

    def __init__(self, tag=1):  # tag: 1,use sequence; 0,use pocket
        super().__init__()  

        self.tag = tag
        
        smi_embed_size = 128
        seq_embed_size = 128
        
        seq_oc = 128
        smi_oc = 128

        self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN+1, smi_embed_size)
        self.seq_embed = nn.Linear(CHAR_SEQ_SET_LEN, seq_embed_size)  # (N, *, H_{in}) -> (N, *, H_{out}) 表示蛋白中每个氨基酸独立特征的描述编码

        if tag:
            conv_seq = []
            ic = seq_embed_size  # 序列输入信道128
            for oc in [32, 64, 64, seq_oc]:  # 序列输出信道128
                conv_seq.append(DilatedParllelResidualBlockA(ic, oc))  # 信道变化，连续扩张卷积
                ic = oc
            conv_seq.append(nn.AdaptiveMaxPool1d(1))  # (N, oc)
            conv_seq.append(Squeeze())
            self.conv_seq = nn.Sequential(*conv_seq)
        else:
            conv_pkt = []
            ic = seq_embed_size  # 口袋输入信道128
            for oc in [32, 64, smi_oc]:  # 口袋输出信道128
                conv_pkt.append(nn.Conv1d(ic, oc, 3))  # (N,C,L)
                conv_pkt.append(nn.BatchNorm1d(oc))
                conv_pkt.append(nn.PReLU())
                ic = oc
            conv_pkt.append(nn.AdaptiveMaxPool1d(1))
            conv_pkt.append(Squeeze())
            self.conv_seq = nn.Sequential(*conv_pkt)  # (N,oc)

        conv_smi = []
        ic = smi_embed_size  #  化合物输入信道128
        for oc in [32, 64, smi_oc]:  #  化合物输出信道128
            conv_smi.append(DilatedParllelResidualBlockB(ic, oc))
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N,128)
                
        self.classifier = nn.Sequential(
            nn.Linear(seq_oc+smi_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1),
            nn.PReLU())
        

    def forward(self, seq, pkt, smi):
        if self.tag:
            # assert seq.shape == (N,L,40)
            seq_embed = self.seq_embed(seq)  # (N,L,128)
            seq_embed = torch.transpose(seq_embed, 1, 2)  # (N,32,L)
            seq_conv = self.conv_seq(seq_embed)  # (N,128)

        else:
            # assert pkt.shape == (N,L,40)
            seq_embed = self.seq_embed(pkt)  # (N,L,128)
            seq_embed = torch.transpose(seq_embed, 1, 2)
            seq_conv = self.conv_seq(seq_embed)  # (N,128)

        # assert smi.shape == (N, L)
        smi_embed = self.smi_embed(smi)  # (N,L,128)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)

        cat = torch.cat([seq_conv, smi_conv], dim=1)  # (N,128*2)        
        output = self.classifier(cat)
        return output
    
class DeepDTAF_3to2(nn.Module):

    def __init__(self):
        super().__init__()

        smi_embed_size = 128
        seq_embed_size = 128
        
        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128

        self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN+1, smi_embed_size)  
        self.seq_embed = nn.Linear(CHAR_SEQ_SET_LEN, seq_embed_size)  # (N, *, H_{in}) -> (N, *, H_{out}) 表示蛋白中每个氨基酸独立特征的描述编码

        conv_seq = []
        ic = seq_embed_size  # 序列输入信道128
        for oc in [32, 64, 64, seq_oc]:  # 序列输出信道128
            conv_seq.append(DilatedParllelResidualBlockA(ic, oc))  # 信道变化，连续扩张卷积
            ic = oc
        conv_seq.append(nn.AdaptiveMaxPool1d(1))  # (N, oc)
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)

        conv_pkt = []
        ic = seq_embed_size  # 口袋输入信道128
        for oc in [32, 64, pkt_oc]:  # 口袋输出信道128
            conv_pkt.append(nn.Conv1d(ic, oc, 3))  # (N,C,L)
            conv_pkt.append(nn.BatchNorm1d(oc))
            conv_pkt.append(nn.PReLU())
            ic = oc
        conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)  # (N,oc)

        conv_smi = []
        ic = smi_embed_size  #  化合物输入信道128
        for oc in [32, 64, smi_oc]:  #  化合物输出信道128
            conv_smi.append(DilatedParllelResidualBlockB(ic, oc))
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N,128)
        
        self.cat_dropout = nn.Dropout(0.1)  # 失活神经元比例
        
        self.classifier1 = nn.Sequential(
            nn.Linear(seq_oc+smi_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1),
            nn.PReLU()
        )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(pkt_oc+smi_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1),
            nn.PReLU()
        )
        
    def forward(self, seq, pkt, smi):
        # assert seq.shape == (N,L,40)
        seq_embed = self.seq_embed(seq)  # (N,L,128)
        seq_embed = torch.transpose(seq_embed, 1, 2)  # (N,32,L)
        seq_conv = self.conv_seq(seq_embed)  # (N,128)

        # assert pkt.shape == (N,L,40)
        pkt_embed = self.seq_embed(pkt)  # (N,L,128)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)  # (N,128)

        # assert smi.shape == (N, L)
        smi_embed = self.smi_embed(smi)  # (N,L,128)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)

        cat1 = torch.cat([seq_conv, smi_conv], dim=1)
        cat1 = self.cat_dropout(cat1)
        output1 = self.classifier1(cat1)

        cat2 = torch.cat([pkt_conv, smi_conv], dim=1)
        cat2 = self.cat_dropout(cat2)
        output2 = self.classifier1(cat2)
        return output1, output2
        
class Loss_fc():
    def __init__(self, tag, func=nn.MSELoss(reduction='sum'), lamb=1):
        self.tag = tag
        self.fn = func
        self.lamb = lamb
        
    def calc_loss(self, output, target):
        if self.tag:  # 使用双通道输出
            loss1 = self.fn((output[0].view(-1) + output[1].view(-1))/2, target.view(-1))
            loss2 = self.fn(output[0].view(-1), output[1].view(-1))
            return loss1 + self.lamb*loss2, loss1, loss2
        else:
            return self.fn(output.view(-1), target.view(-1))
    
def test(model: nn.Module, test_loader, loss_function, device, path=None, save=False):
    model.eval()
    test_loss = 0
    test_diff = 0
    outputs, outputs_ss, outputs_ps = [], [], []
    targets = []
    names = []
    with torch.no_grad():
        for idx, (*x, y, name) in enumerate(test_loader):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)

            y_hat = model(*x)
            
            if isinstance(model, WLPPDTA3to2) or isinstance(model, DeepDTAF_3to2):
                _, loss, diff = loss_function.calc_loss(y_hat, y)
                test_loss += loss.item()
                test_diff += diff.item()
                outputs_ss.append(y_hat[0].cpu().numpy().reshape(-1))
                outputs_ps.append(y_hat[1].cpu().numpy().reshape(-1))
                outputs.append((y_hat[0]/2+y_hat[1]/2).cpu().numpy().reshape(-1))
                targets.append(y.cpu().numpy().reshape(-1))
                
            else:
                test_loss += loss_function.calc_loss(y_hat, y).item()
                outputs.append(y_hat.cpu().numpy().reshape(-1))
                targets.append(y.cpu().numpy().reshape(-1))
            names += name

    if isinstance(model, WLPPDTA3to2) or isinstance(model, DeepDTAF_3to2):
        outputs_ss = np.concatenate(outputs_ss).reshape(-1)
        outputs_ps = np.concatenate(outputs_ps).reshape(-1)        
    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    if save:
        save_xlt(path=path, names=names, targets=targets, outputs=[outputs, outputs_ss, outputs_ps], 
                 multiout=isinstance(model, WLPPDTA3to2) or isinstance(model, DeepDTAF_3to2))
        
    test_loss /= len(test_loader.dataset)
    test_diff /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'diff': test_diff,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation

def save_xlt(path, names, targets, outputs, multiout=False):
    
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('Sheet1',cell_overwrite_ok=True)
    sheet.write(0,0, 'id')  
    sheet.write(0,1, 'target')
    sheet.write(0,2, 'pred')
    if multiout:
        sheet.write(0,3, 'pred_seq+smi')
        sheet.write(0,4, 'pred_pkt+smi')
        for idx,(i,t,p,p_ss,p_ps) in enumerate(zip(names, targets, outputs[0], outputs[1], outputs[2])):
            sheet.write(idx+1,0, i)   
            sheet.write(idx+1,1, float(t))
            sheet.write(idx+1,2, float(p))
            sheet.write(idx+1,3, float(p_ss))
            sheet.write(idx+1,4, float(p_ps))
    else:
        for idx,(i,t,p) in enumerate(zip(names, targets, outputs[0])):
            sheet.write(idx+1,0, i)    
            sheet.write(idx+1,1, float(t))
            sheet.write(idx+1,2, float(p))

    wbk.save(path)