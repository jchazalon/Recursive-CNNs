import numpy as np
import cv2

import utils
import os
import math
import sys 


DATA_DIR = "../../4pointdataw4"
DATA_DIR2= "../..//DataGenerator/multipleBackgroundsNew"

GT_DIR = DATA_DIR + "/gt.csv"
GT_DIR2 = DATA_DIR2+"/gt.csv"
VALIDATION_PERCENTAGE = .2
TEST_PERCENTAGE = .01
Debug = True
size = (32,32)

limi = -1
image_list1, gt_list1, file_name = utils.load_data_4(DATA_DIR, GT_DIR, limit=limi, size=size,remove_background=int(sys.argv[1]))
image_list2, gt_list2, file_name_2 = utils.load_data_4(DATA_DIR2, GT_DIR2, limit=limi, size=size)

gt_list= np.array(np.append(gt_list1, gt_list2,axis=0))
image_list= np.array(np.append(image_list1, image_list2, axis=0))

image_list, gt_list = utils.unison_shuffled_copies(image_list, gt_list)


print len(image_list)


if (Debug):
    print ("(Image_list_len, gt_list_len)", (len(image_list), len(gt_list)))
train_image = image_list[0:max(1, int(len(image_list) * (1 - VALIDATION_PERCENTAGE)))]
validate_image = image_list[int(len(image_list) * (1 - VALIDATION_PERCENTAGE)):len(image_list) - 1]

train_gt = gt_list[0:max(1, int(len(image_list) * (1 - VALIDATION_PERCENTAGE)))]
validate_gt = gt_list[int(len(image_list) * (1 - VALIDATION_PERCENTAGE)):len(image_list) - 1]
if (Debug):
    print ("(Train_Image_len, Train_gt_len)", (len(train_image), len(train_gt)))
    print ("(Validate_Image_len, Validate_gt_len)", (len(validate_image), len(validate_gt)))
for a in range(0, 10):
    temp_image=np.copy(image_list[a])
    for b in range(0, 4):
        cv2.circle(temp_image, (gt_list[a][b*2], gt_list[a][b*2+1]), 2, (255, 0, 0), 4)
    cv2.imwrite("../temp"+str(a)+".jpg", temp_image)

np.save("../train_gt_bg"+str(sys.argv[1]), train_gt)
np.save("../train_image_bg"+str(sys.argv[1]), train_image)
np.save("../validate_gt_bg"+str(sys.argv[1]), validate_gt)
np.save("../validate_image_bg"+str(sys.argv[1]), validate_image)
# 0/0
