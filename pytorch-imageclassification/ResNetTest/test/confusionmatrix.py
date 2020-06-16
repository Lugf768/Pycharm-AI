# Author LGF
# GreatTime 2020/6/7
# Description: simple introduction of the code
import torch
from torchvision import transforms, datasets
import json
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from Test.test.model import resnet34


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        # 初始化混淆矩阵
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, Fscore
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Fscore"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            # TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3)
            Recall = round(TP / (TP + FN), 3)
            Fscore = round((2*Precision*Recall) / (Precision + Recall), 3)
            table.add_row([self.labels[i], Precision, Recall, Fscore])
        print(table)

    def plot(self):
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=90)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_path ="../data_set/image_data/"
    validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                            transform=data_transform)

    batch_size = 16
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    net = resnet34()
    model_weight_path = "./resNet34.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=22, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()