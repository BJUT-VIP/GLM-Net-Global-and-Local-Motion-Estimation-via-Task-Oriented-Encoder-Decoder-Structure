import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
# from tqdm import tqdm
from lib import flowlib as f1
import pickle
# from utils.loss import MSELoss as MSELoss

# from eval import eval_net
from unet import *

# from torch.utils.tensorboard import SummaryWriter
from utils.Dataset_local import MotionField_Data
from torch.utils.data import DataLoader, random_split
# from test import out_to_img as Tensor_to_Numpy

from torch.optim import lr_scheduler

class AverageMeter(object):
    """
        Computes the average value
        """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def val_net(model, val_loader, device, val_num, batch_size, scheme):
    model.eval()
    loss_meter = AverageMeter()
    criterion = nn.MSELoss()
    batch_count = 0
    for batch in val_loader:
        batch_count += 1
        Mix = batch['Mix']
        Global = batch['Global']
        Mask = batch['Mask']
        Mix = Mix.to(device=device, dtype=torch.float32, non_blocking=True)
        Global = Global.to(device=device, dtype=torch.float32, non_blocking=True)
        Mask = Mask.to(device=device, dtype=torch.float32, non_blocking=True)

        Local = Mix - Global
        Local_abs = torch.abs(Local)
        Local = torch.where(Local_abs <= 0.1, torch.tensor(0.0, device=device), Local)

        if scheme == 0:
            inputs = torch.cat([Local, Global], dim=1)
        elif scheme == 1:
            inputs = torch.cat([Local, Mix], dim=1)
        elif scheme == 2:
            inputs = Local
        elif scheme == 3:
            inputs = Mix

        with torch.no_grad():
            outputs = model(inputs)
            # 根据mask制作带score区域的global field
            GlobalFieldWithMask = ScoremaskWithGlobal(Mask, Mix, Global)
            # output和上述结果相加
            outputs += GlobalFieldWithMask

            #loss
            loss = criterion(outputs, Mix)
            loss_meter.update(loss.item(), batch_size)
            loss_avg = loss_meter.avg

            print('loss_val:{}  process:{}%'.format(loss_avg, 100 * batch_size * batch_count / val_num))

    return loss_avg

def ScoremaskWithGlobal(Mask, Mix, Global):
    Mask = torch.unsqueeze(Mask, 1)
    Mask = torch.cat([Mask, Mask], dim=1)
    vaild_Mask = torch.sum(Mask)
    GlobalWithScore0 = Global * Mask

    Mask_score = torch.where(Mask == 0.0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    vaild_Mask_score = torch.sum(Mask_score)

    B, C, H, W = Mask.size()
    if vaild_Mask + vaild_Mask_score != B*C*H*W:
        print('Mask Error')
        os._exit(0)

    ScoreField = Mix * Mask_score

    return GlobalWithScore0 + ScoreField




def train_net(model, device, epoch, batch_size, lr, multiGPU, scheme):
    train_data = MotionField_Data(train_Mix_dir, train_Global_dir, train_Mask_dir)
    train_data_num = len(train_data)
    val_data = MotionField_Data(valid_Mix_dir, valid_Global_dir, valid_Mask_dir)
    val_data_num = len(val_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, 20, 0.5)
    criterion = nn.MSELoss()

    model.train()

    loss_list = []
    loss_list_val = []
    epe_val = []
    loss_meter = AverageMeter()
    batch_num = 0
    best_error = 100000
    best_model = model
    best_ep = -1
    for ep in range(epoch):
        ep_now = ep + 1
        for batch in train_loader:
            batch_num += 1
            Mix = batch['Mix']
            Global = batch['Global']
            Mask = batch['Mask']
            Mix = Mix.to(device=device, dtype=torch.float32, non_blocking=True)
            Global = Global.to(device=device, dtype=torch.float32, non_blocking=True)
            Mask = Mask.to(device=device, dtype=torch.float32, non_blocking=True)

            Local = Mix - Global
            Local_abs = torch.abs(Local)
            Local = torch.where(Local_abs <= 0.1, torch.tensor(0.0, device=device), Local)
            # Local = torch.where(Local <= 0.1, torch.tensor(0.0, device=device), Local)

            if scheme == 0:
                inputs = torch.cat([Local, Global], dim=1)
            elif scheme == 1:
                inputs = torch.cat([Local, Mix], dim=1)
            elif scheme == 2:
                inputs = Local
            elif scheme == 3:
                inputs = Mix

            optimizer.zero_grad()
            outputs = model(inputs)
            # 根据mask制作带score区域的global field
            GlobalFieldWithMask = ScoremaskWithGlobal(Mask, Mix, Global)
            # output和上述结果相加
            outputs += GlobalFieldWithMask
            #loss
            loss = criterion(outputs, Mix)
            loss_list.append(loss.item())
            loss_meter.update(loss.item(), batch_size)

            #optim
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch:{}  {}%  Loss_train:{}'.format(ep_now, batch_size * batch_num / train_data_num, loss_meter.avg))
        # print("第%d个epoch的学习率：%f" % (ep, optimizer.param_groups[0]['lr']))


        if ep_now % 10 == 0 or batch_num == 1:
            # 计算val loss
            print('validation:{}'.format(int(ep_now/10)))
            loss_val = val_net(model, val_loader, device, val_data_num, batch_size, scheme)
            if loss_val < best_error:
                best_model = model
                best_error = loss_val
                best_ep = ep_now
                print('saving best model in epoch{}'.format(ep_now))
                torch.save(best_model.state_dict(), saveModel_dir + 'best_model{}.pth'.format(model_name))
                print('finish saving best model')
            loss_list_val.append(loss_val)
            print('finish validation {}'.format(int(ep_now/10)))

        scheduler.step()
        print("第%d个epoch的学习率：%f" % (ep, optimizer.param_groups[0]['lr']))

    # save model and loss_information
    if not multiGPU :
        torch.save(model.state_dict(), saveModel_dir + 'final_model{}.pth'.format(model_name))
    else:
        torch.save(model.module.state_dict(), saveModel_dir + 'final_model{}.pth'.format(model_name))

    with open(saveLoss_dir + 'loss_train{}.pkl'.format(model_name), 'wb') as f:
        pickle.dump(loss_list, f)

    print('Best_model is at epoch : {}'.format(best_ep))

    # with open(saveLoss_dir + 'loss_val.pkl', 'wb') as f:
    #     pickle.dump(loss_list_val, f)


if __name__ == '__main__':
    # 参数设置
    train_Mix_dir = '/disk1/yyc/U-net/data_new/least_square/train/mix_field/'
    train_Global_dir = '/disk1/yyc/U-net/data_new/least_square/train/global/'
    train_Mask_dir = '/disk1/yyc/U-net/data_new/least_square/train/ScoreMask/'
    valid_Mix_dir = '/disk1/yyc/U-net/data_new/least_square/valid/mix_field/'
    valid_Global_dir = '/disk1/yyc/U-net/data_new/least_square/valid/global/'
    valid_Mask_dir = '/disk1/yyc/U-net/data_new/least_square/valid/ScoreMask/'

    saveModel_dir = '/disk1/yyc/AutoEncoder/result_local/'
    saveLoss_dir = '/disk1/yyc/AutoEncoder/result_local/'

    checkpoint_dir = ''

    # model_name = '_WithConcat_multiGPU_Batch32'
    model_name = '_localScoreMask_scheme0_new_epoch200_Batch64_size64'
    with_concat = True
    attention = True
    # 设置相关超参数
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    gpu_ids = [5]

    if len(gpu_ids) > 1:
        multiGPU = True
    else:
        multiGPU = False

    logging.info(f'Using device {device}')

    batch_size = 64
    epoch = 200
    lr = 0.001

    # 不同方案对用不同输入:  0 - 输入为 mix-global concat global   1 - 输入为 mix-global concat mix   2 - 输入为mix-global  3 - 输入为mix
    scheme = 0

    if scheme == 0 or scheme == 1:
        in_channels = 4
    elif scheme == 2 or scheme == 3:
        in_channels = 2
    out_channels = 2
    modelLoad_path = None


    # 加载模型
    if attention:
        model = Attention_U_Net(in_channels, out_channels, upsample=True)
    else:
        if with_concat:
            model = UNet(in_channels, out_channels, bilinear=False)
        else:
            model = UNet_noC(in_channels, out_channels, bilinear=False)

    # if modelLoad_path is not None:
    #     model.load_state_dict()

    # 多GPU
    # model = nn.DataParallel(model, device_ids= gpu_ids)

    # model to GPU
    model.to(device=device)

    print("Use scheme: {}, input_channels: {}".format(scheme, in_channels))

    try:
        train_net(model, device, epoch, batch_size, lr, multiGPU, scheme)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), saveModel_dir + 'INTERRUPTED{}.pth'.format(model_name))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)