# Author LGF
# GreatTime 2020/6/5
# Description: simple introduction of the code
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from Test.test.model import resnet34

data_transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224),
     transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img = Image.open("../pic/17.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = resnet34()
model_weight_path = "./resNet34.pth"
model.load_state_dict(torch.load(model_weight_path))
# 保证BN用全部训练数据的均值和方差
model.eval()
# 不对我们的损失梯度跟踪
with torch.no_grad():
    #squeeze压缩它的batch维度
    output = torch.squeeze(model(img))
    #softmax处理得到概率分布
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].numpy())
plt.show()