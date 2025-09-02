import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.models as models
from matplotlib import image as mpimg, pyplot as plt
# from torchviz import make_dot
import torch
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)

        return out

# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         # 主路径
#         self.conv1 = nn.Conv1d(
#             in_channels, out_channels,
#             kernel_size=3, stride=stride,  # 关键修改：应用stride到下采样
#             padding=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(
#             out_channels, out_channels,
#             kernel_size=3, stride=1,
#             padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm1d(out_channels)
#
#         # 残差捷径
#         self.downsample = None
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv1d(
#                     in_channels, out_channels,
#                     kernel_size=1, stride=stride,  # 关键修改：应用相同的stride
#                     bias=False
#                 ),
#                 nn.BatchNorm1d(out_channels)
#             )
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity  # 此时形状必须一致
#         out = self.relu(out)
#         return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=2, include_top=True, dropout_prob=0.5):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channels = 4

        # 基础特征提取层（网页7的预训练层设计思想）
        self.base_features = nn.Sequential(
            nn.Conv1d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 残差模块堆叠（网页1的残差块堆叠逻辑）
        self.res_layers = nn.Sequential(
            self._make_layer(block, 4, blocks_num[0], stride=2),
            self._make_layer(block, 8, blocks_num[1], stride=2),
            self._make_layer(block, 16, blocks_num[2], stride=2)
        )

        # 分类头设计（网页6的层次化学习率策略）
        if self.include_top:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout_prob),
                nn.Linear(16 * block.expansion, num_classes)
            )
        else:
            self.classifier = None

        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        # 维度匹配策略（网页2的残差连接实现）
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base_features(x)  # [BS,4,L//2]
        x = self.res_layers(x)     # [BS,16,L//8]
        if self.include_top:
            x = self.classifier(x) # [BS,2]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def resnet(num_classes=2, include_top=True):
    model = ResNet(BasicBlock, [1, 1, 1], num_classes=num_classes, include_top=include_top)
    return model


def create_model(model_name: str,
                 num_classes: int = 2,
                 include_top: bool = True) -> torch.nn.Module:
    if model_name == 'resnet':
        model = ResNet(BasicBlock, [1, 1, 1], num_classes=num_classes, include_top=include_top)
        return model
    else:
        raise ValueError(f"不支持的模型类型: {model_name}. 当前仅支持: 'resnet'")


from torchsummary import summary  # 新增

# 示例用法
if __name__ == "__main__":
    # Create the model instance
    model = resnet(num_classes=2)

    # Print model summary for debugging
    print("Model Summary:")
    summary(model, input_size=(1, 1824))  # Adjust input size

    # Generate a random input tensor
    dummy_input = torch.randn(1, 1, 1824)

    # Forward pass through the model
    output = model(dummy_input)

