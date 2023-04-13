import random
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from typing import List

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

# 蛋白及口袋序列编码
CHAR_AMI_SET = {"G": 1, "A": 2, "V": 3, "L": 4, "I": 5, 
                "M": 6, "F": 7, "P": 8, "W": 9, "S": 10, 
                "T": 11, "Y": 12, "C": 13, "Q": 14, "N": 15, 
                "D": 16, "E": 17, "K": 18, "R": 19, "H": 20, 
                "X": 21}

# non-polar, polar, acidic, basic 
CHAR_ATT_SET = {"G": 1, "A": 1, "V": 1, "L":1 , "I":1 , 
                "M": 1, "F": 1, "P": 1, "W": 1, "S": 2, 
                "T": 2, "Y": 2, "C": 2, "Q": 2, "N": 2, 
                "D": 3, "E": 3, "K": 4, "R": 4, "H": 4, 
                "X": 0.25}

# seven groups
CHAR_GRO_SET = {"G": 1, "A": 1, "V": 1, "L":2 , "I":2 , 
                "M": 3, "F": 2, "P": 2, "W": 4, "S": 3, 
                "T": 3, "Y": 3, "C": 7, "Q": 4, "N": 4, 
                "D": 6, "E": 6, "K": 5, "R": 5, "H": 4, 
                "X": 0.14285714285714285}

CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)   # smiles字符串字典，实际数目加1，因为有0表示背景
CHAR_SEQ_SET_LEN = 4+7+len(CHAR_AMI_SET)   # 蛋白字符串字典
CHAR_PKT_SET_LEN = 4+7+len(CHAR_AMI_SET)   # pocket字符串字典


def random_start(total_len, select_len):
    if total_len>select_len:
        return random.randint(0, total_len-select_len), True
    else:
        return random.randint(0, select_len-total_len), False

    
def label_sampling(line, max_len, stype, repeat=1, fix=True, pkt_window=None, pkt_stride=None):
    assert stype in ['smi', 'pkt', 'seq']
    X = np.zeros(max_len, dtype=np.int)  # 0本身表示背景
    if stype in ['pkt', 'seq']:
        X_ATT = np.zeros(max_len)  # 0本身表示背景
        X_GRO = np.zeros(max_len)  # 0本身表示背景
        
    X_start, tag = random_start(len(line), max_len)  # tag为True: 要在total中随机截取；False: total不够长，要补0
    
    if repeat == 1 and fix == True:
        X_start = 0  # 不进行平移增强，同时固定位置采样位置
        
    if tag:
        for i, ch in enumerate(line[X_start: X_start+max_len]):
            if stype == 'smi':
                X[i] = CHAR_SMI_SET[ch]
            else:
                X[i] = CHAR_AMI_SET[ch]
                X_ATT[i] = CHAR_ATT_SET[ch]
                X_GRO[i] = CHAR_GRO_SET[ch]                    
    else:
        for i, ch in enumerate(line):
            if stype == 'smi':
                X[X_start+i] = CHAR_SMI_SET[ch]
            else:
                X[X_start+i] = CHAR_AMI_SET[ch]
                X_ATT[X_start+i] = CHAR_ATT_SET[ch]
                X_GRO[X_start+i] = CHAR_GRO_SET[ch]
    
    if stype == 'pkt' and pkt_window is not None and pkt_stride is not None:  # 针对口袋的处理，不太清楚具体含义
        X = np.array(
            [X[i * self.pkt_stride:i * self.pkt_stride + self.pkt_window] 
             for i in range(int(np.ceil((max_len - pkt_window) / pkt_stride)))]
        )
        
    if stype in ['pkt', 'seq']:
        return X, X_ATT, X_GRO
    else:
        return X


class DTIDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_pkt_len, max_smi_len, pkt_window=None, pkt_stride=None, 
                 onehot=False, repeat=1, fix=True, mode=0, input_drop_rate=0.4, select_pkt_rate=0.5):
        # assert phase in ['training', 'validation', 'test', 'test105']
        data_path = Path(data_path)  # '../data/'

        affinity_df = pd.read_csv(data_path / 'affinity.csv')  # dataframe：pdbid, -logKd/Ki
        self.affinity = {i["pdbid"]: i["-logKd/Ki"] for _, i in affinity_df.iterrows()}  # 存储为亲和力字典

        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv")  # training:11906, validation:1000, test:290
        self.smi = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}  # dataframe：pdbid, smiles
        self.max_smi_len = max_smi_len  # 150

        sequences_df = pd.read_csv(data_path / f'{phase}_seq_.csv')
        self.seq = {i["id"]: i["seq"] for _, i in sequences_df.iterrows()}  # dataframe：id, sequences
        self.max_seq_len = max_seq_len  # 1000

        pockets_df = pd.read_csv(data_path / f'{phase}_pocket_.csv')
        self.pkt = {i["id"]: i["seq"] for _, i in pockets_df.iterrows()}  # dataframe：id, sequences
        self.max_pkt_len = max_pkt_len  # 63
        
        self.pdbqt = [i["pdbid"] for _, i in ligands_df.iterrows()]
        
        self.repeat = repeat  # 单个序列平移增强数量
        self.onehot = onehot  # 使用onehot编码
        self.fix = fix  # 增强量为1时，是否固定采样位置
        self.mode = mode  # 输入信息随机失活模式，0:无失活，1:失活口袋，2:失活蛋白，3:口袋蛋白随机失活一个
        self.input_drop_rate = input_drop_rate  # 输入信息失活概率
        self.select_pkt_rate = select_pkt_rate  # 在mode=3并且随机失活时，选择失活口袋的概率
        
        self.pkt_window = pkt_window  # 默认值是None
        self.pkt_stride = pkt_stride  # 默认值是None
        if self.pkt_window is None or self.pkt_stride is None:
            print(f'Dataset {phase}: will not fold pkt')

        assert len(self.seq) == len(self.pkt)
        assert len(self.seq) == len(self.smi)  # 可能序列和化合物数据是一一对应的 

        if 'training' in phase:
            self.length = len(self.pdbqt) * self.repeat
        else:
            self.repeat = 1
            self.fix = True
            self.length = len(self.pdbqt)
      
    def _fill(self, tensor, array, att=None, gro=None):
        row = np.where(tensor>0)[0]
        if array.shape[-1]==CHAR_SMI_SET_LEN:
            array[row, tensor[row]-1] = 1
        else:
            array[row, 4+7+tensor[row]-1] = 1    # fill sequence   
            row_R, row_X = np.where(att>=1)[0], np.where((att<1)*(att>0))[0]
            array[row_R, (4+gro[row_R]-1).astype(int)] = 1    # fill 7 group infomation  
            array[row_R, (att[row_R]-1).astype(int)] = 1    # fill attributes
            if len(row_X):
                array[np.concatenate([row_X]*7), 4+np.repeat(np.arange(7),len(row_X))] = gro[row_X][0] 
                array[np.concatenate([row_X]*4), np.repeat(np.arange(4),len(row_X))] = att[row_X][0]           
        return array
    
    def __getitem__(self, idx):
        idx = idx % len(self.pdbqt)
        
        name = self.pdbqt[idx]
        smi = self.smi[name]
        seq = self.seq[name]
        pkt = self.pkt[name]

        smi_tensor = label_sampling(smi, self.max_smi_len, 'smi', self.repeat, self.fix)
        seq_tensor, seq_tensor_att, seq_tensor_gro = label_sampling(seq, self.max_seq_len, 'seq', self.repeat, self.fix)
        pkt_tensor, pkt_tensor_att, pkt_tensor_gro = label_sampling(pkt, self.max_pkt_len, 'pkt', self.repeat, self.fix)
        
        seq_array = np.zeros((len(seq_tensor), CHAR_SEQ_SET_LEN), dtype=np.float32)
        pkt_array = np.zeros((len(pkt_tensor), CHAR_PKT_SET_LEN), dtype=np.float32)
        
        assert self.mode in [0,1,2,3]
        if self.mode == 0:  # 蛋白和口袋均不置零
            seq_array = self._fill(seq_tensor, seq_array, seq_tensor_att, seq_tensor_gro)
            pkt_array = self._fill(pkt_tensor, pkt_array, pkt_tensor_att, pkt_tensor_gro)
        elif self.mode == 1:  # 口袋概率置零，保留蛋白
            seq_array = self._fill(seq_tensor, seq_array, seq_tensor_att, seq_tensor_gro)
            if np.random.rand()>self.input_drop_rate:  # default 0.4
                pkt_array = self._fill(pkt_tensor, pkt_array, pkt_tensor_att, pkt_tensor_gro)
        elif self.mode == 2:  # 蛋白概率置零，保留口袋
            pkt_array = self._fill(pkt_tensor, pkt_array, pkt_tensor_att, pkt_tensor_gro)
            if np.random.rand()>self.input_drop_rate:
                seq_array = self._fill(seq_tensor, seq_array, seq_tensor_att, seq_tensor_gro)
        elif self.mode == 3:  # 口袋或蛋白概率性置零
            if np.random.rand()>self.input_drop_rate:
                seq_array = self._fill(seq_tensor, seq_array, seq_tensor_att, seq_tensor_gro)
                pkt_array = self._fill(pkt_tensor, pkt_array, pkt_tensor_att, pkt_tensor_gro)
            else:
                if np.random.rand()>self.select_pkt_rate:  # default 0.5
                    pkt_array = self._fill(pkt_tensor, pkt_array, pkt_tensor_att, pkt_tensor_gro)  # 保留口袋
                else:
                    seq_array = self._fill(seq_tensor, seq_array, seq_tensor_att, seq_tensor_gro)  # 保留蛋白
                    
        
        if self.onehot:
            smi_array = np.zeros((len(smi_tensor), CHAR_SMI_SET_LEN), dtype=np.float32)
            return (seq_array, pkt_array, self._fill(smi_tensor, smi_array),
                    np.array(self.affinity[name], dtype=np.float32), name)  # 亲和力标签            
        else:
            return (seq_array, pkt_array, smi_tensor,  # 1000， 63， 150
                    np.array(self.affinity[name], dtype=np.float32), name)  # 亲和力标签

    def __len__(self):
        return self.length

PT_FEATURE_SIZE = 40

def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] - 1

    return X  # 仅取出smiles的前150个，转换成npy

class MyDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_pkt_len, max_smi_len, pkt_window=None, pkt_stride=None):
        data_path = Path(data_path)  # '../data/'

        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')  # dataframe：pdbid, -logKd/Ki
        for _, row in affinity_df.iterrows():  # 第一个空值本来返回编号
            affinity[row[0]] = row[1]  
        self.affinity = affinity  # 存储为亲和力字典

        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv") 
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}  # 蛋白和化合物的作用对，但是内部没有亲和力信息
        self.smi = ligands
        self.max_smi_len = max_smi_len

        seq_path = data_path / phase / 'global'
        self.seq_path = sorted(list(seq_path.glob('*')))
        self.max_seq_len = max_seq_len

        pkt_path = data_path / phase / 'pocket'
        self.pkt_path = sorted(list(pkt_path.glob('*')))
        self.max_pkt_len = max_pkt_len
        
        self.pkt_window = pkt_window  # 默认值是None
        self.pkt_stride = pkt_stride  # 默认值是None
        if self.pkt_window is None or self.pkt_stride is None:
            print(f'Dataset {phase}: will not fold pkt')

        assert len(self.seq_path) == len(self.pkt_path)
        assert len(self.seq_path) == len(self.smi)  # 可能序列和化合物数据是一一对应的 

        self.length = len(self.smi)

    def __getitem__(self, idx):
        seq = self.seq_path[idx]
        pkt = self.pkt_path[idx]
        assert seq.name == pkt.name  # 保护机制，蛋白和口袋一一对应，名字一致

        _seq_tensor = pd.read_csv(seq, index_col=0).drop(['idx'], axis=1).values[:self.max_seq_len]  # 共21位氨基酸残基，8位二级结构类型，11理化性质
        seq_tensor = np.zeros((self.max_seq_len, PT_FEATURE_SIZE))  # 1000 * 40的空数组
        seq_tensor[:len(_seq_tensor)] = _seq_tensor  # 蛋白序列也是，取出头1000个氨基酸，不涉及随机性

        _pkt_tensor = pd.read_csv(pkt, index_col=0).drop(['idx'], axis=1).values[:self.max_pkt_len]
        if self.pkt_window is not None and self.pkt_stride is not None:
            pkt_len = (int(np.ceil((self.max_pkt_len - self.pkt_window) / self.pkt_stride))
                       * self.pkt_stride
                       + self.pkt_window)
            pkt_tensor = np.zeros((pkt_len, PT_FEATURE_SIZE))
            pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor
            pkt_tensor = np.array(
                [pkt_tensor[i * self.pkt_stride:i * self.pkt_stride + self.pkt_window]
                 for i in range(int(np.ceil((self.max_pkt_len - self.pkt_window) / self.pkt_stride)))]
            )
        else:
            pkt_tensor = np.zeros((self.max_pkt_len, PT_FEATURE_SIZE))
            pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor   # 口袋序列取出头63个氨基酸，不涉及随机性

        return (seq_tensor.astype(np.float32),  # 1000 * 40
                pkt_tensor.astype(np.float32),  # 63 * 40
                label_smiles(self.smi[seq.name.split('.')[0]], self.max_smi_len),  # 对应化合物：150
                np.array(self.affinity[seq.name.split('.')[0]], dtype=np.float32), seq.name)  # 亲和力标签

    def __len__(self):
        return self.length
    