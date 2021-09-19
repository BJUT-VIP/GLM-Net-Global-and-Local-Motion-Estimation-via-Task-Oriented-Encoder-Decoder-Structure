import os
import shutil
import re
from utils.DeleteData import *


class Shear():

    def __init__(self):
        # 源目录
        self.src = '/disk1/yyc/U-net/data_new/least_square/val/mask_new/'
        # 目标目录
        self.dest = '/disk1/yyc/U-net/data_new/least_square/val/mask_new_less4/'

    def start_shear(self, a_filepath):
        obj = DeleteData()
        for a_item in os.listdir(a_filepath):  # 遍历源目录下的所有文件
            a_absolute_path = os.path.join(a_filepath, a_item)  # 新的目录为传入目录+传入目录里面的文件夹名称
            num = re.sub("\D", "", a_item)
            num = int(num)
            if num in obj.validdrop:
                shutil.move(self.src + a_item,
                            self.dest)  # shutil.move(full_path,destpath) 这里移动文件的源文件要完全路径，否则会报找不到文件的错
                # os.rename(a_item,self.dest)
            print(num)





if __name__ == "__main__":
    shear = Shear()
    shear.start_shear(shear.src)
