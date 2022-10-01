'''
Author: Yihan Liu
Date: 2022-10-01 21:14:31
LastEditTime: 2022-10-01 22:06:28
Email: 117010177@link.cuhk.edu.cn
'''
import numpy as np
import Poisson as ps
import cv2
import sys

# Prepare the input images 
argvs = sys.argv
src_name = argvs[1]
mask_name = argvs[2]
target_name = argvs[3]
res_in = "../res/Input/"
pic_type = ".jpeg"
src_path =  res_in + src_name + pic_type
mask_path = res_in + mask_name +pic_type
target_path = res_in + target_name + pic_type

# Prepare the output file
res_out = "../res/Output/"
out_path = res_out + "result.jpeg"

# load images
src = np.array(cv2.imread(src_path, 1)/255.0, dtype=np.float32)
target = np.array(cv2.imread(target_path, 1)/255.0, dtype=np.float32)
mask = np.array(cv2.imread(mask_path, 0), dtype=np.uint8)
ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
mask = mask/255.0

