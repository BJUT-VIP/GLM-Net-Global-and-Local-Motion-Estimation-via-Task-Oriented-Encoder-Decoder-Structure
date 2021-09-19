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
from utils.loss import MSELoss as MSELoss

# from eval import eval_net
from network.AutoEncoder import AutoEncoder

# from torch.utils.tensorboard import SummaryWriter
from utils.Dataset_mine import MotionField_Data
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

def val_net(model, val_loader, device, val_num, batch_size):
    model.eval()
    loss_meter = AverageMeter()
    criterion = MSELoss()
    batch_count = 0
    for batch in val_loader:
        batch_count += 1
        inputs = batch['image']
        batch_now = inputs.size()
        inputs_vector = inputs.view(-1, 64*64*2)
        targets = inputs
        masks = batch['mask']
        masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
        inputs_vector = inputs_vector.to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            outputs_vector = model(inputs_vector)
            outputs = outputs_vector.view(batch_now[0], batch_now[1], batch_now[2], batch_now[3])
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            #loss
            loss = criterion(outputs, targets, masks)
            loss_meter.update(loss.item(), batch_size)
            loss_avg = loss_meter.avg

            print('loss_val:{}  process:{}%'.format(loss_avg, 100 * batch_size * batch_count / val_num))

    return loss_avg

# 计算val epe
# def val_net2(model, val_data, device):
#     val_num_all = len(val_data)
#     val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=0)
#     val_num = 0
#     epe = 0.0
#     with torch.no_grad():
#         for batch in val_loader:
#             val_num += 1
#             # 每次读取一个数据 [1,2,256,256]
#             input = batch['image']
#             target = batch['label']
#
#             input = input.to(device=device, dtype=torch.float32)
#             output = model(input)
#
#             # [1,2,256,256] to [2,256,256]
#             output = output.squeeze(0)
#             target = target.squeeze(0)
#
#             # 数据变为原始格式(扩大15倍、[2,256,256] to [256,256,2])
#             output = Tensor_to_Numpy(output)
#             target = Tensor_to_Numpy(target)
#
#             # 将输出分成uv维度
#             output_x = output[:, :, 0]
#             output_y = output[:, :, 1]
#
#             target_x = target[:, :, 0]
#             target_y = target[:, :, 1]
#
#             epe += f1.flow_error(target_x, target_y, output_x, output_y)
#
#             print('validation epe process : {}%'.format(100 * val_num / val_num_all))
#
#     mean_epe = epe / val_num
#
#
#     return mean_epe

def train_net(model, device, epoch, batch_size, lr, multiGPU):
    train_data = MotionField_Data(train_img_dir, train_mask_dir)
    train_data_num = len(train_data)
    val_data = MotionField_Data(val_img_dir, val_mask_dir)
    val_data_num = len(val_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, 20, 0.5)
    criterion = MSELoss()

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
            inputs = batch['image']
            batch_now = inputs.size()
            inputs_vector = inputs.view(-1, 64*64*2)
            targets = inputs
            masks = batch['mask']
            masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
            inputs_vector = inputs_vector.to(device=device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad()
            outputs_vector = model(inputs_vector)
            outputs = outputs_vector.view(batch_now[0], batch_now[1], batch_now[2],batch_now[3])
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            #loss
            loss = criterion(outputs, targets, masks)
            loss_list.append(loss.item())
            loss_meter.update(loss.item(), batch_size)

            #optim
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch:{}   {}%     Loss_train:{}'.format(ep_now, batch_size * batch_num / train_data_num, loss_meter.avg))
        print("第%d个epoch的学习率：%f" % (ep, optimizer.param_groups[0]['lr']))


        if ep_now % 10 == 0 or batch_num == 1:
            # 计算val loss
            print('validation:{}'.format(int(ep_now/10)))
            loss_val = val_net(model, val_loader, device, val_data_num, batch_size)
            if loss_val < best_error:
                best_model = model
                best_error = loss_val
                best_ep = ep_now
                print('saving best model in epoch{}'.format(ep_now))
                torch.save(best_model.state_dict(), saveModel_dir + 'best_model{}.pth'.format(model_name))
                print('finish saving best model')
            loss_list_val.append(loss_val)
            print('finish validation {}'.format(int(ep_now/10)))

            # 计算val epe
            # epe = val_net2(model, val_data, device)
            # epe_val.append(epe)
            # print('epe of validation : {}'.format(epe))
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
    # train_img_dir = 'D:/code/OpticalFlow/Global_estimation_by_CNN/Pytorch_UNet/data_new/least_square/train/mix_field/'
    # train_mask_dir = 'D:/code/OpticalFlow/Global_estimation_by_CNN/Pytorch_UNet/data_new/least_square/train/mask_new/'
    #
    # val_img_dir = 'D:/code/OpticalFlow/Global_estimation_by_CNN/Pytorch_UNet/data_new/least_square/val/mix_field/'
    # val_mask_dir = 'D:/code/OpticalFlow/Global_estimation_by_CNN/Pytorch_UNet/data_new/least_square/val/mask_new/'

    ################################NCAA#######################################
    # train_img_dir = '/disk1/yyc/U-net/data_new/least_square/train/mix_field/'
    # train_mask_dir = '/disk1/yyc/U-net/data_new/least_square/train/mask_new/'
    # print('train_mask = {}'.format(train_mask_dir))
    #
    # val_img_dir = '/disk1/yyc/U-net/data_new/least_square/val/mix_field/'
    # val_mask_dir = '/disk1/yyc/U-net/data_new/least_square/val/mask_new/'
    #
    # saveModel_dir = '/disk1/yyc/AutoEncoder/result/'
    # saveLoss_dir = '/disk1/yyc/AutoEncoder/result/'
    ##########################################################################
    #####################################DeepHomography数据集_RE类##################

    # train_img_dir = '/disk4/yyc/DeepHomoGraphy/data/train/PWC_flow/'
    # train_mask_dir = '/disk4/yyc/DeepHomoGraphy/data/train/mask/'
    #
    # print('train_mask = {}'.format(train_mask_dir))
    #
    # val_img_dir = '/disk4/yyc/DeepHomoGraphy/data/valid/PWC_flow/'
    # val_mask_dir = '/disk4/yyc/DeepHomoGraphy/data/valid/mask/'
    #
    # saveModel_dir = '/disk4/yyc/DeepHomoGraphy/result/'
    # saveLoss_dir = '/disk4/yyc/DeepHomoGraphy/result/'

    ###########################################################################
    #####################################DeepHomography数据集_所有类别##################

    train_img_dir = '/data/yyc/DeepHomo/data/train/PWC_flow/'
    train_mask_dir = '/data/yyc/DeepHomo/data/train/mask/'

    print('train_mask = {}'.format(train_mask_dir))

    val_img_dir = '/data/yyc/DeepHomo/data/valid/PWC_flow/'
    val_mask_dir = '/data/yyc/DeepHomo/data/valid/mask/'

    saveModel_dir = '/data/yyc/DeepHomo/data/result/'
    saveLoss_dir = '/data/yyc/DeepHomo/data/result/'

    ###########################################################################

    # saveModel_dir = './result/'
    # saveLoss_dir = './result/'

    checkpoint_dir = ''

    # model_name = '_WithConcat_multiGPU_Batch32'
    # model_name = '_AutoEncoder_charbonnier_MaskNew_deleteData_lastLayerTanh_RELU_1024_128_8_64_epoch200_Batch64_size64'
    model_name = '_AutoEncoder_DeepHomography_AllCati_GuiYi60_middle16_lastLayerTanh_RELU_1024_128_16_64_epoch200_Batch256_size64'
    print('model_name = {}'.format(model_name))

    # 设置相关超参数
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    gpu_ids = [5]

    if len(gpu_ids) > 1:
        multiGPU = True
    else:
        multiGPU = False

    logging.info(f'Using device {device}')

    batch_size = 256
    epoch = 200
    lr = 0.001
    in_channels = 2
    out_channels = 2
    modelLoad_path = None

    # 加载模型
    model = AutoEncoder()

    # if modelLoad_path is not None:
    #     model.load_state_dict()

    # 多GPU
    # model = nn.DataParallel(model, device_ids= gpu_ids)

    # model to GPU
    model.to(device=device)

    try:
        train_net(model, device, epoch, batch_size, lr, multiGPU)
    except KeyboardInterrupt:
        # torch.save(model.state_dict(), saveModel_dir + 'INTERRUPTED{}.pth'.format(model_name))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)