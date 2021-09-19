import matplotlib.pyplot as plt
import cv2
import numpy as np
from natsort import natsorted
import glob
import os
from lib import flowlib_v2 as fl2
from lib import flowlib as fl

img_width = 490
img_height = 360

def view(img, maxrad, mask):
    mask_origin = mask
    mask_origin = mask_origin[:, :, np.newaxis]
    label = img * mask_origin

    img_label_color = fl2.flow_to_image(label, maxrad)

    fl.visualize_flow(img)

    plt.imshow(img_label_color)
    plt.show()


# 设置flow和mask的dir
mask_dir = '/disk3/yyc/U-net/data_new/least_square/test/mask/'
flow_dir = '/disk3/yyc/U-net/data_new/least_square/test/mix_field/'

flow_list = natsorted(glob.glob(os.path.join(flow_dir + '*.npy')))
mask_list = natsorted(glob.glob(os.path.join(mask_dir + '*.npy')))




if __name__ == '__main__':
    flo_count = 0
    for index in range(len(flow_list)):
        flo_count += 1

        # if flo_count < 305:
        #     flo_count += 1
        #     continue


        flow_path_now = flow_list[index]
        mask_path_now = mask_list[index]

        flow_now = np.load(flow_path_now).astype(np.float32)
        mask_now = np.load(mask_path_now).astype(np.float32)

        u = flow_now[:, :, 0]
        v = flow_now[:, :, 1]
        rad = np.sqrt(u ** 2 + v ** 2)  # 光流场方向
        maxrad = max(-1, np.max(rad))

        view(flow_now, maxrad, mask_now)

