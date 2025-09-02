import torch.nn as nn

class simplecnn(nn.Module):
    def __init__(self,num_class): # num_class是我们的分类数
        super().__init__()
        self.features = nn.Sequential( # 做特征提取
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1), # 保持图像大小不变 16* 224*224
            nn.ReLU(), # 卷积之后接上激活函数 增加非线特征
            nn.MaxPool2d(kernel_size=2,stride=2), # 池化之后变为 16*112*112
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1), # 保持图像大小不变 32* 112 *112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # 图像大小变为 32*56*56
        )
        # 定义全连接层 做分类
        self.classifier = nn.Sequential(
            nn.Linear(32*56*56,128),
            nn.ReLU(),
            nn.Linear(128,num_class) # num_class为分类的个数
        )

    def forward(self,x):
        # 前向传播部分
        x = self.features(x) # 先将图像进行特征提取
        x = x.view(x.size(0),-1) # 展平 x.size(0) 为batch
        x = self.classifier(x)
        return x

