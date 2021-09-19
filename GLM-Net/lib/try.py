import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import time
import glob
from lib import flowlib as fl
from lib import flowlib_v2 as fl2
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':

    print(np.arctan2(np.array([0]), np.array([-1])) / np.pi)

    flow_path = r'D:\flow_multi_mode\flow\0aWfrZAM6Q8\3-pointer-failure\306225.742\1.flo'
    flow = fl.read_flow(flow_path)
    # seg = fl.segment_flow(flow).astype(np.uint8)
    # img = cv2.imread(r'D:\flow_multi_mode\rgb\306225.742\1.jpg', 0)
    # classmap_hist_feature1 = cv2.calcHist(seg, [0], None, [9], [0.0, 8.0]).reshape(1, -1)
    #
    # flow_path = r'D:\flow_multi_mode\flow\0aWfrZAM6Q8\3-pointer-failure\306225.742\7.flo'
    # flow = fl.read_flow(flow_path)
    # seg = fl.segment_flow(flow).astype(np.uint8)
    # classmap_hist_feature2 = cv2.calcHist(seg, [0], None, [9], [0.0, 8.0]).reshape(1, -1)
    #
    # c = cosine_similarity(np.array(classmap_hist_feature1), np.array(classmap_hist_feature2))

    # print(1e8)
    #
    frame_path = r'D:\flow_multi_mode\rgb\306225.742\2.jpg'
    frame = cv2.imread(frame_path)
    warp_img = fl.warp_image(frame, flow)
    #
    cv2.imshow(cv2.namedWindow('1'), warp_img)
    cv2.waitKey()




    # idx = np.repeat(m[:, :, np.newaxis], 3, axis=1)
    # print(idx)

    # vector1 = np.array([0, 1, 0])  # y 方向
    # vector2 = np.array([-0.2, -1, -1])  # x 方向

    # Lx = np.sqrt(vector1.dot(vector1))
    # Ly = np.sqrt(vector2.dot(vector2))
    # cos_angle = vector1.dot(vector2) / (Lx * Ly)
    # angle_cos = np.arccos(cos_angle)

    # angle = np.arctan2(vector1, vector2) / np.pi
    # print(angle)
    # print(angle_cos)

    # a = np.arange(0, 100).reshape((10, 10))
    # a[a <= 50] = 1
    # print(a)

    # instance = [5, 6, 7, 8, 9, 10]
    # i = list(iter([(4, 2), (3, 2)]))
    #
    # a = list(map(lambda value, mm: (value - mm[0]) / (mm[0] - mm[1]) if mm[0] - mm[1] > 0 else 0, instance, i))
    # print(a)

    # start = time.time()
    # with open(r'D:\flow_multi_mode\smooth_v3_pkl\5w4QpuGJLj4\steal-success\234078.524\data.pkl', "rb") as f:
    #     data = pickle.load(f)
    #     for sub in data:
    #         img = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
    #         img_r = cv2.resize(img, (200, 200))
    #         # cv2.imshow(cv2.namedWindow("1"), img)
    #         # cv2.waitKey()
    # end = time.time()
    # print(end - start)
    #
    # img_list = glob.glob(r'D:\flow_multi_mode\smooth_v3\5w4QpuGJLj4\steal-success\234078.524\*.png')
    # start = time.time()
    # for img_path in img_list:
    #     img = cv2.imread(img_path, )
    #     img_r = cv2.resize(img, (200, 200))
    # end = time.time()
    # print(end - start)