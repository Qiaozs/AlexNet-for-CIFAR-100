import torch
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import AlexNet
from torch import nn
from torch.optim import lr_scheduler
import os
import copy
import pandas as pd

root_train = '/root/autodl-tmp/AlexNet/CIFAR-100/train'
root_val = '/root/autodl-tmp/AlexNet/CIFAR-100/val'
# root_train = '/root/autodl-tmp/AlexNet/chest_xray/chest_xray/train'
# root_val = '/root/autodl-tmp/AlexNet/chest_xray/chest_xray/val'
# root_path = '/root/autodl-tmp/AlexNet/model_save'

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def datalode(image_size,root_train, root_val,batch_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    val_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.ImageFolder(root=root_train, transform = train_transform)
    val_dataset = datasets.ImageFolder(root=root_val, transform = val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size,shuffle=True)

    print(train_dataset.classes)
    print("训练集共有图像{}张".format(len(train_dataset.imgs)))
    print(val_dataset.classes)
    print("验证集共有图像{}张".format(len(val_dataset.imgs)))
    return train_loader,val_loader,len(train_dataset),len(val_dataset)

def whole_train(model,learning_rate,epoch_num,train_loader,val_loader,pth_name,root_path):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cup')
    # 损失函数，使用交叉熵函数
    Loss = nn.CrossEntropyLoss()
    # 优化器
    # optimizer = torch.optim.SGD(model.parameters(),learning_rate)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # 模型指认到设备中
    model.to(device)
    # Copy current model param
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 保存当前时间
    Time_Start = time.time()

    for epoch in range(epoch_num):
        print("epoch{}/{}".format(epoch, epoch_num - 1))

        # 参数初始化
        train_loss = 0.0  # 训练损失
        train_acc = 0.0  # 训练准确率
        val_loss = 0.0  # 验证损失
        val_acc = 0.0  # 验证准确率
        train_num = 0  # 训练数
        val_num = 0  # 验证数

        # 训练
        for batch,(x,y) in enumerate(train_loader):
            inputs, labels = x.to(device), y.to(device)
            model.train()
            # 前向传播
            # 使用模型得到一个batch大小的inputs得到的outputs
            outputs = model(inputs)
            loss = Loss(outputs, labels)
            predict = torch.argmax(outputs, dim=1)

            # 清空梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(predict == labels.data)
            train_num += inputs.size(0)


            # print train process
            rate = (batch + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            # print("Training:\n")
            print("\rTraining {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")


        # 验证
        model.eval()
        with torch.no_grad():
            for batch, (x,y) in enumerate(val_loader):
                inputs, labels = x.to(device), y.to(device)
                outputs = model(inputs)
                loss = Loss(outputs, labels)
                predict = torch.argmax(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_acc += torch.sum(predict == labels.data)
                val_num += inputs.size(0)


                # print
                rate = (batch + 1) / len(val_loader)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                # print("Training:\n")
                print("\rTraining {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_acc.double().item() / train_num)
        val_acc_all.append(val_acc.double().item() / val_num)
        print('{} Train Loss: {:.4f} Train acc:{:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss:{:.4f} Val acc:{:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            # 保持当前参数
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - Time_Start
        print('训练耗时{:.1f}min{:.1f}s/epoch'.format(time_use // 60, time_use % 60))

    # 选择最优模型保存-加载最高准确率下的参数
    model.load_state_dict(best_model_wts)
    torch.save(obj=best_model_wts, f="/root/autodl-tmp/AlexNet/save_model_cifar/best_model_weights_0524.pth")

    train_process = pd.DataFrame(data={'epoch': range(epoch_num),
                                       'train_loss_all': train_loss_all,
                                       'val_loss_all': val_loss_all,
                                       'train_acc_all': train_acc_all,
                                       'val_acc_all': val_loss_all, })
    return train_process


def plot(train_process, Title, Save_Path):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, 'ro-', label='train loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('train-loss')

    plt.subplot(2, 2, 3)
    plt.plot(train_process['epoch'], train_process.train_loss_all, 'bs-', label='Val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('val-loss')

    plt.subplot(2, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, 'ro-', label='train acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('train-acc')

    plt.subplot(2, 2, 4)
    plt.plot(train_process['epoch'], train_process.train_acc_all, 'bs-', label='val acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('val-acc')

    # plt.title(Title)
    # plt.show()
    plt.savefig(Save_Path)


if __name__ == "__main__":
    # 0.获取项目文件路径
    root_path = os.getcwd()
    # 1.模型实例化
    net = AlexNet()
    # 2.加载数据集
    train_dataloader, val_dataloader, train_num, val_num = datalode(
        image_size=224,root_train=root_train,root_val=root_val,batch_size=64)
    train_process = whole_train(model=net, learning_rate=0.0001,epoch_num=30,
                                train_loader=train_dataloader,val_loader=val_dataloader,
                                pth_name='AlexNet.pth',root_path=root_path)
    plot(train_process=train_process, Title='lr=0.0001 epoch=30',
         Save_Path='plt_cifar/VTest_0524.png')
