import tensorflow as tf
import numpy as np
import os
import scipy.misc as misc
import TRAIN
import Inference

train = []
test = []

images = []

image_ids = open("/content/drive/My Drive/IP - 7th sem/CUB_200_2011/images.txt", "r")
test_train = open("/content/drive/My Drive/IP - 7th sem/CUB_200_2011/train_test_split.txt", "r")

for l in image_ids:
  path = l.split(" ")[1]
  images.append(path.split("/")[1][:-1])

for x in test_train:
  image_id = int(x.split(" ")[0]) - 1
  if(int(x.split(" ")[1]) == 0):
    test.append(images[image_id])
  else:
    train.append(images[image_id])

print(len(test))
print(len(train))

# print(train)

# TRAIN.fun_main(train)

Inference.fun_main(test)