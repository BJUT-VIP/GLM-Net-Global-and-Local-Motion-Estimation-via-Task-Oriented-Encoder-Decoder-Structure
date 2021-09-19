import os
import numpy as np
import torch
from network.AutoEncoder_8 import AutoEncoder
from natsort import natsorted
import glob
import re
from utils.Dataset_mine import MotionField_Data
import cv2
import time
from lib import flowlib as f1
import matplotlib.pyplot as plt
from lib import flowlib_v2 as fl2
import matplotlib.pyplot as pyplot
from utils.flow_error import flow_error


trained_model = '/disk1/yyc/AutoEncoder/result/best_model_AutoEncoder_MaskNew_deleteData_lastLayerTanh_RELU_1024_128_8_64_epoch200_Batch64_size64.pth'

#设置数据集路径和相关参数


test_img_dir = '/disk1/yyc/AutoEncoder/data/flow/'
test_mask_dir = '/disk1/yyc/AutoEncoder/data/annoations/mask/'


test_img_list = natsorted(glob.glob(os.path.join(test_img_dir + '*.npy')))
test_mask_list = natsorted(glob.glob(os.path.join(test_mask_dir + '*.npy')))

if len(test_img_list) != len(test_mask_list):
    print('img and label have diff num!')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

in_channels = 2
out_channels = 2
#加载模型1
print('loading model...')
model = AutoEncoder()
model.load_state_dict(torch.load(trained_model, map_location=device))
model.to(device=device)
model.eval()
print('model loading finish')


test_count = 0
epe = 0.0
epe_refine = 0.0
epe_Attention = 0.0
time_all = 0.0

#将读取的label进行[-15，15]的处理
def process_label(label):
    label = cv2.resize(label, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
    # label_with_mask = label * mask
    # label = np.clip(label, -15, 15)
    return label

#将网络输出的[-1,1]之间的CHW Tensor转换为与label一致的格式（与preprocess逻辑相反）
def out_to_img(output):
    output = output.cpu().numpy()
    output = output * 20
    output = output.transpose((1, 2, 0))
    # output = cv2.resize(output, dsize=(img_width, img_height), interpolation=cv2.INTER_LINEAR)
    return output

# refine方案
def refine_x(mat):
    width = mat.shape[0]
    height = mat.shape[1]

    mat_new = np.zeros((width, height))

    for index in range(width):
        data = mat[:,index]
        data_sort = np.sort(data, axis=0)
        data_mean = np.mean(data_sort[22:42], axis=0)
        # data_mean = np.mean(data_sort[89:168], axis=0)
        mat_new[:,index] = data_mean

    return mat_new

def refine_y(mat):
    width = mat.shape[0]
    height = mat.shape[1]

    mat_new = np.zeros((width, height))

    for index in range(height):
        data = mat[index,:]
        data_sort = np.sort(data, axis=0)
        data_mean = np.mean(data_sort[22:42], axis=0)
        # data_mean = np.mean(data_sort[89:168], axis=0)
        mat_new[index,:] = data_mean

    return mat_new

def view(img, output_global, maxrad, mask):
    output_global = cv2.resize(output_global, dsize=(img_width, img_height), interpolation=cv2.INTER_LINEAR)
    mask_origin = cv2.resize(mask, dsize=(img_width, img_height), interpolation=cv2.INTER_NEAREST)
    mask_origin = mask_origin[:, :, np.newaxis]
    label = img * mask_origin

    img_label_color = fl2.flow_to_image(label, maxrad)
    output_global_color = fl2.flow_to_image(output_global, maxrad)


    f1.visualize_flow(img)

    plt.imshow(img_label_color)
    plt.show()

    plt.imshow(output_global_color)
    plt.show()




hash_epe = {}
hash_vaild_region = {}
test_count2 = 0
epe2 = 0.0
if __name__ == '__main__':
    for index in range(len(test_img_list)):
        test_count += 1
        test_count2 += 1

        print(test_count)
        test_img_path_now = test_img_list[index]
        test_mask_path_now = test_mask_list[index]
        test_img_num = re.sub("\D", "", test_img_path_now)
        test_mask_num = re.sub("\D", "", test_mask_path_now)
        
        #如果读取的img和label id不一致则报错
        if test_img_num != test_mask_num:
            print('wrong match between img and label')
            break

        #读取img和label
        img_now = np.load(test_img_path_now).astype(np.float32)
        mask_now = np.load(test_mask_path_now).astype(np.float32)
        vaild_region = np.sum(mask_now) / (360 * 490)
        mask_now = cv2.resize(mask_now, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
        # label_local_now = np.load(test_label_local_path_now).astype(np.float32)

        u = img_now[:, :, 0]
        v = img_now[:, :, 1]
        rad = np.sqrt(u ** 2 + v ** 2)  # 光流场方向
        maxrad = max(-1, np.max(rad))

        start1 = time.time()

        #将img HWC改为CHW，并归一化
        input = MotionField_Data.preprocess(img_now)
        label = process_label(img_now)
        # label = img_now
        size = input.size()
        input_vector = input.contiguous().view(-1, 64*64*2)
        input_vector = input_vector.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output_vector = model(input_vector)
            output = output_vector.view(size[0], size[1], size[2])
            output = out_to_img(output)
            
        end1 = time.time()

        time_now = end1 - start1
        # print('inference time{}'.format(time_now))

        #将输出分成uv维度
        output_x = output[:, :, 0]
        output_y = output[:, :, 1]


        label_x = label[:, :, 0]
        label_y = label[:, :, 1]

        #计算误差和时间
        epe_now = flow_error(label_x, label_y, output_x, output_y, mask_now)

        epe += epe_now
        epe2 += epe_now

        
        time_all += time_now

        # view the optical field
        output_x_color = output_x[:, :, np.newaxis]
        output_y_color = output_y[:, :, np.newaxis]
        output_global = np.concatenate((output_x_color, output_y_color), axis=2)

        # view(img_now, output_global, maxrad, mask_now)

        print('process: {}%'.format(100 * test_count/len(test_img_list)))

    print('averageEPE: {}  averageTIME: {}'.format(epe/(test_count), time_all/test_count))




