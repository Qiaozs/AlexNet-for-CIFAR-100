import torch
from torch import nn
import torch.nn.functional as F

# 定义AlexNet网络模型
class AlexNet(nn.Module):
    def __init__(self):
        # super：引入父类的初始化方法给子类进行初始化
        super(AlexNet, self).__init__()

        # 第一卷积层，输入大小为224×224，通道数为3，步长为4，卷积核大小为11×11，padding为2
        # 输出大小为（（224+4）-11）/4 +1 = 54.25，取整为55
        # 1： 224，224，3 -- 55，55，96
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

        # 使用ReLU作为激活函数，跟在卷积运算之后
        self.ReLU = nn.ReLU()

        # 池化层，池化核大小为3×3，步长为2
        # 输出大小为（55-3）/2 +1 = 27
        # 2： 55，55，96 -- 27，27，96
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 第二卷积层，输入大小为27×27，通道数96，步长为1，卷积核大小5×5，padding2
        # 输出大小为 （（27+4-5）\1)+1 = 27
        # 3. 27，27，96 -- 27，27，256
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)

        # 最大池化层，输入大小为27*27，输出大小为13*13，输入通道为256，输出为256，池化核为3，步长为2
        # 输出（27-3）/2 + 1 = 13
        # 3: 27,27,256 -- 13，13，256
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为256，输出为384，卷积核为3，扩充边缘为1，步长为1
        # 输出（13+1*2-3）/1 + 1 = 13
        # 4：13，13，256 -- 13,13,384
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)

        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为384，输出为384，卷积核为3，扩充边缘为1，步长为1
        # 输出（13+1*2-3）/1 + 1 = 13
        # 4：13，13，384 -- 13,13,384
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)

        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为384，输出为256，卷积核为3，扩充边缘为1，步长为1
        # 输出（13+1*2-3）/1 + 1 = 13
        # 5：13，13，384 -- 13,13,256
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)

        # 最大池化层，输入大小为13*13，输出大小为6*6，输入通道为256，输出为256，池化核为3，步长为2
        # 输出（13-3）/2 + 1 = 6
        # 6: 13,13,256 -- 6，6，256
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()

        # 全连接层，输入大小为6*6*256，输出大小为4096
        self.f6 = nn.Linear(6 * 6 * 256, 4096) # 9216,4096
        self.f7 = nn.Linear(4096, 4096)
        # self.f8 = nn.Linear(4096, 1000)
        self.f8 = nn.Linear(4096, 100)



    def forward(self,x):
        x= self.ReLU(self.conv1(x))
        x = self.pool1(x)
        x= self.ReLU(self.conv2(x))
        x = self.pool2(x)
        x= self.ReLU(self.conv3(x))
        x= self.ReLU(self.conv4(x))
        x= self.ReLU(self.conv5(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.ReLU(x)
        x = F.dropout(x,p=0.5)
        x = self.f7(x)
        x = self.ReLU(x)
        x = F.dropout(x,p=0.5)
        x = self.f8(x)
        return x

if __name__ == "__main__":
    x = torch.rand([1, 3, 224, 224])
    model = AlexNet()
    y=model(x)
    print(y.shape)
