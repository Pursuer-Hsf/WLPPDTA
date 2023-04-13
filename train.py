import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import DTIDataset, MyDataset
from model import DeepDTAF, DeepDTAF_3to2, test, WLPPDTA3to1, WLPPDTA3to2, WLPPDTA2to1, Loss_fc
import argparse
import os


DEFAULT_MAX_SEQ_LEN = 904  # 数据预处理使用的值，定义序列长度，蛋白
DEFAULT_MAX_PKT_LEN = 64  # 口袋
DEFAULT_MAX_SMI_LEN = 154  # 化合物
DEFAULT_INPUT_DROP_RATE = 0.75
DEFAULT_SELECT_PKT_RATE = 0.5
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 40
DEFAULT_REPEAT = 80





def main(args):
    # data_path, fold = '../DTI-dataset', 1
    seed = np.random.randint(18885, 18886) # 生成特定范围内的随机数
    path = Path(f'{args.results_dir}/{args.fold}_{datetime.now().strftime("%Y%m%d%H%M%S")}_{seed}')  # 结果文件夹，并时间戳和随机数种子
    path.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)  # or torch.device('cpu')

    # 确保每次使用 cuDNN 时都以确定的方式计算卷积，以便结果始终相同
    torch.backends.cudnn.deterministic = True
    # 禁用 cuDNN 自动寻找最适合当前硬件的卷积算法的功能，以避免结果的随机性
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)  # 让torch和numpy生成特定随机数，可复现结果
    np.random.seed(seed)

    f_param = open(path.joinpath('parameters.txt'), 'w')  # 用于记录超参数

    # print(f'device={device}')
    # print(f'seed={seed}')
    # print(f'write to {path}')
    f_param.write(f'device={device}\n'
                f'seed={seed}\n'
                f'write to {path}\n')
                
    # print(f'max_seq_len={max_seq_len}\n'
    #       f'max_pkt_len={max_pkt_len}\n'
    #       f'max_smi_len={max_smi_len}\n')
    f_param.write(f'max_seq_len={args.max_seq_len}\n'
                f'max_pkt_len={args.max_pkt_len}\n'
                f'max_smi_len={args.max_smi_len}\n')

    # print(f'repeat={repeat}\n'
    #       f'fix={fix}\n'
    #       f'onehot={onehot}\n'
    #       f'mode={mode}\n'
    #       f'input_drop_rate={input_drop_rate}\n'
    #       f'select_pkt_rate={select_pkt_rate}')
    f_param.write(f'repeat={args.repeat}\n'
                f'onehot={args.onehot}\n'
                f'mode={args.mode}\n'
                f'input_drop_rate={args.input_drop_rate}\n'
                f'select_pkt_rate={args.select_pkt_rate}\n')

    assert 0<min_epoch<args.epochs

    model = WLPPDTA3to1(onehot=args.onehot, dilation=args.dilation)
    # model = WLPPDTA3to2(onehot=onehot, dilation=dilation)
    # model = WLPPDTA2to1(onehot=onehot, dilation=dilation)
    # model = DeepDTAF()
    # model = DeepDTAF_3to2()

    model = model.to(device)
    para_quantity = sum([p.data.nelement() for p in model.parameters()])
    print('Parameter quantity:{}'.format(para_quantity))   # 计算参数量
    f_param.write(f'Parameter quantity:{para_quantity} \n')
    f_param.write(str(model)+'\n')
    f_param.close()

    data_loaders = {'training':
                        DataLoader(DTIDataset(
                                        args.data_path, 
                                        f'fold_{args.fold}/training',
                                        args.max_seq_len,
                                        args.max_pkt_len, 
                                        args.max_smi_len, 
                                        pkt_window=None, 
                                        pkt_stride=None, 
                                        onehot=args.onehot, 
                                        repeat=args.repeat, 
                                        fix=args.fix,
                                        mode=args.mode, 
                                        input_drop_rate=args.input_drop_rate, 
                                        select_pkt_rate=args.select_pkt_rate),
                                        batch_size=args.batch_size,
                                        pin_memory=True,
                                        num_workers=4,
                                        shuffle=True),
                    'validation':
                        DataLoader(DTIDataset(
                                        args.data_path, 
                                        f'fold_{args.fold}/validation',
                                        args.max_seq_len, 
                                        args.max_pkt_len, 
                                        args.max_smi_len, 
                                        pkt_window=None, 
                                        pkt_stride=None, 
                                        onehot=args.onehot),
                                        batch_size=args.batch_size,
                                        pin_memory=True,
                                        num_workers=4,
                                        shuffle=False),
                    'test':
                        DataLoader(DTIDataset(args.data_path, f'fold_{args.fold}/test',
                                            args.max_seq_len,
                                            args.max_pkt_len, 
                                            args.max_smi_len, 
                                            pkt_window=None, pkt_stride=None, 
                                            onehot=args.onehot),
                                            batch_size=args.batch_size,
                                            pin_memory=True,
                                            num_workers=4,
                                            shuffle=False)
                }

    train_loader_no_repeat = DataLoader(DTIDataset(args.data_path, f'fold_{args.fold}/training', 
                                                args.max_seq_len, args.max_pkt_len, args.max_smi_len, 
                                                pkt_window=None, pkt_stride=None, 
                                                onehot=args.onehot),
                                        batch_size=args.batch_size,
                                        pin_memory=True,
                                        num_workers=4,
                                        shuffle=False)

    # MultiStepLR, OneCycleLR
    optimizer = optim.AdamW(model.parameters(), lr = 0.005)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, 
                                            epochs=args.epochs,
                                            steps_per_epoch=len(data_loaders['training']))   # 按照一种单周期函数设置学习率
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40, 60, 75], gamma=0.5)   # 按照一种单周期函数设置学习率
            
    tag = isinstance(model, WLPPDTA3to2) or isinstance(model, DeepDTAF_3to2)
    loss_function = Loss_fc(tag=tag)
    print('special loss:', tag)


    start = datetime.now()
    print('start at ', start)
    print('Train epoch: %d, Batch number per epoch: %d'%(args.epochs, len(data_loaders['training'])))

    epoch_train_loss, epoch_val_loss = [], []
    epoch_train_diff, epoch_val_diff = [], []
    epoch_train_c_index, epoch_val_c_index = [], []
    epoch_train_RMSE, epoch_val_RMSE = [], []
    epoch_train_MAE, epoch_val_MAE = [], []
    epoch_train_SD, epoch_val_SD = [], []
    epoch_train_CORR, epoch_val_CORR = [], []
    Lr = []

    best_epoch = -1
    best_val_loss = 100000000
    for epoch in range(1, args.epochs + 1):
        
        batch_num = len(data_loaders['training'])
        for idx, (*x, y, name) in enumerate(data_loaders['training']):
            model.train()
            lr = optimizer.param_groups[0]['lr']
            Lr.append(lr)

            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(*x)
            loss = loss_function.calc_loss(output, y)
            if isinstance(model, WLPPDTA3to2) or isinstance(model, DeepDTAF_3to2):
                loss[0].backward()
                if (idx+1)%100 == 0:
                    print(f'Epoch {epoch}|{args.epochs}, Batch {idx+1}|{batch_num}, Loss:{loss[1].item()}, Diff:{loss[2].item()}')            
            else:
                loss.backward()
                if (idx+1)%100 == 0:
                    print(f'Epoch {epoch}|{args.epochs}, Batch {idx+1}|{batch_num}, Loss:{loss.item()}')
                    
            optimizer.step()
            scheduler.step()
            
        with torch.no_grad():
            for _p in ['training', 'validation']:

                if _p=='training':
                    performance = test(model, train_loader_no_repeat, loss_function, device)  # 是个字典，包括{'loss','c_index'，等各种指标}
                    epoch_train_loss.append(performance['loss'])
                    epoch_train_diff.append(performance['diff'])
                    epoch_train_c_index.append(performance['c_index'])
                    epoch_train_RMSE.append(performance['RMSE'])
                    epoch_train_MAE.append(performance['MAE'])
                    epoch_train_SD.append(performance['SD'])
                    epoch_train_CORR.append(performance['CORR'])

                else:
                    performance = test(model, data_loaders[_p], loss_function, device)  # 是个字典，包括{'loss','c_index'，等各种指标}
                    epoch_val_loss.append(performance['loss'])
                    epoch_val_diff.append(performance['diff'])
                    epoch_val_c_index.append(performance['c_index'])
                    epoch_val_RMSE.append(performance['RMSE'])
                    epoch_val_MAE.append(performance['MAE'])
                    epoch_val_SD.append(performance['SD'])
                    epoch_val_CORR.append(performance['CORR'])

                if _p=='validation' and epoch>=min_epoch and performance['loss']<best_val_loss:
                    best_val_loss = performance['loss']
                    best_epoch = epoch
                    torch.save(model.state_dict(), path / 'best_model.pt')
        
    #     scheduler.step()
        print(f'Epoch {epoch}|{args.epochs}: Lr -- {lr}, Train loss | diff -- {epoch_train_loss[-1]:.3f} | {epoch_train_diff[-1]:.3f}, Val loss | diff -- {epoch_val_loss[-1]:.3f} | {epoch_val_diff[-1]:.3f}') 

    print('training finished')
                
    model.load_state_dict(torch.load(path / 'best_model.pt'))
    with open(path / 'result.txt', 'w') as f:
        f.write(f'best model found at epoch NO.{best_epoch}\n')
        for _p in ['training', 'validation', 'test',]:
            if _p == 'training':
                performance = test(model, train_loader_no_repeat, loss_function, device, path/(_p+'_predictions.xls'), save=True)
            else:
                performance = test(model, data_loaders[_p], loss_function, device, path/(_p+'_predictions.xls'), save=True)
            f.write(f'{_p}:\n')
            print(f'{_p}:')
            for k, v in performance.items():
                f.write(f'{k}: {v}\n')
                print(f'{k}: {v}')
            f.write('\n')
            print()

    print('testing finished')
            
    end = datetime.now()
    print('end at:', end)
    print('time used:', str(end - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')


    parser.add_argument('--max_seq_len', type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument('--max_pkt_len', type=int, default=DEFAULT_MAX_PKT_LEN)
    parser.add_argument('--max_smi_len', type=int, default=DEFAULT_MAX_SMI_LEN)

    parser.add_argument('--repeat', type=int, default=DEFAULT_REPEAT)
    parser.add_argument('--fix', type=bool, default=False)

    parser.add_argument('--onehot', type=bool, default=False) # smiles是否使用one-hot编码
    parser.add_argument('--dilation', type=bool, default=True) # 是否使用扩张卷积
    parser.add_argument('--mode', type=int, default=0) # 输入信息随机失活门控，0:无失活，1:失活口袋，2:失活蛋白，3:口袋蛋白随机失活一个

    parser.add_argument('--input_drop_rate', type=int, default=DEFAULT_INPUT_DROP_RATE)
    parser.add_argument('--select_pkt_rate', type=int, default=DEFAULT_SELECT_PKT_RATE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)

    interrupt = None 
    min_epoch = 9  #  when `min_epoch` is reached and the loss starts to decrease, save best model parameters




    args = parser.parse_args()
    main(args)