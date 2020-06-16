# Author LGF
# GreatTime 2020/5/30
# Description: simple introduction of the code
import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


file = 'image_data/image'
flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]

mkfile('image_data/train')
for cla in flower_class:
    mkfile('image_data/train/'+cla)

mkfile('image_data/val')
for cla in flower_class:
    mkfile('image_data/val/'+cla)

split_rate = 0.2
for cla in flower_class:
    cla_path = file + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'image_data/val/' + cla
            copy(image_path, new_path)
        else:
            image_path = cla_path + image
            new_path = 'image_data/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    print()

print("processing done!")