# Author LGF
# GreatTime 2020/6/5
# Description: simple introduction of the code
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import torch.optim as optim
import matplotlib.pyplot as plt
from Test.test.model import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                               transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

image_path ="../data_set/image_data/"
train_dataset = datasets.ImageFolder(root=image_path+"train", transform=data_transform["train"])
train_num = len(train_dataset)
image_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in image_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
validate_dataset = datasets.ImageFolder(root=image_path + "val", transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

net = resnet34()
print(net)
net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './resNet34.pth'
Loss_list = []
Accuracy_list = []
for epoch in range(50):
    t1 = time.perf_counter()
    # train和eval能够控制Batch Normalization的状态
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad() # 将所有参数的梯度都置零
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward() #误差反向传播计算参数梯度
        optimizer.step() # 通过梯度做一步参数更新

        running_loss += loss.item()
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()

    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
        print(time.perf_counter() - t1)

        Loss_list.append(running_loss / step)
        Accuracy_list.append(val_accurate)


x1 = range(0, 50)
x2 = range(0, 50)
y1 = Accuracy_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test accuracy')
plt.ylabel('accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Train loss')
plt.ylabel('loss')
plt.savefig("accuracy_loss.jpg")
plt.show()

print('Finished Training')