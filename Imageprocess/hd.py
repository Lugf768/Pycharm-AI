# Author LGF
# GreatTime 2020/5/19
# Description: simple introduction of the code
from PIL import Image
import os

input_dir = 'data/1/'
out_dir = 'data/2/'
a = os.listdir(input_dir)

for i in a:
    print(i)
    I = Image.open(input_dir + i)
    L = I.convert('L')
    L.save(out_dir + i)
