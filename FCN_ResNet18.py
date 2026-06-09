import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    ResNet18 基础残差块
    """
    expansion = 1  # 通道扩张倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 下采样连接（用于匹配通道数和分辨率）
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，对identity进行处理
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


class ResNet18Backbone(nn.Module):
    """
    自定义 ResNet18 特征提取网络，用于 FCN 系列模型。
    返回三个中间特征图：
        - layer2: 下采样8倍（对应原VGG的pool3）
        - layer3: 下采样16倍（对应原VGG的pool4）
        - layer4: 下采样32倍（对应原VGG的conv7）
    """

    def __init__(self):
        super().__init__()
        self.in_channels = 64  # 初始通道数

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1/4

        # 四个残差层
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)  # 1/4 -> 1/4
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # 1/4 -> 1/8
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # 1/8 -> 1/16
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  # 1/16 -> 1/32

        # 输出通道数
        self.out_channels = [128, 256, 512]  # layer2, layer3, layer4

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        创建一个残差层，包含多个残差块
        """
        downsample = None

        # 需要下采样或通道数改变时，创建下采样层
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        权重初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始卷积和池化 -> 1/4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 各残差块
        x = self.layer1(x)  # 1/4 -> 1/4
        x = self.layer2(x)  # 1/4 -> 1/8
        layer2_out = x  # 下采样8倍
        x = self.layer3(x)  # 1/8 -> 1/16
        layer3_out = x  # 下采样16倍
        x = self.layer4(x)  # 1/16 -> 1/32
        layer4_out = x  # 下采样32倍

        # 返回三个特征图，对应FCN需要的三个尺度
        return layer2_out, layer3_out, layer4_out


class FCN_ResNet18(nn.Module):
    """
    基于 ResNet18 主干的全卷积网络（Fully Convolutional Network）用于语义分割。
    参数：
        num_classes: 输出类别数（含背景），默认21（PASCAL VOC）。
        version: 模型版本，可选 '32s', '16s', '8s'。
    """

    def __init__(self, num_classes=21, version='8s'):
        super().__init__()
        self.version = version

        # 使用自定义的 ResNet18 主干
        self.backbone = ResNet18Backbone()

        # ResNet18 输出通道：layer2=128, layer3=256, layer4=512
        pool3_ch, pool4_ch, conv7_ch = self.backbone.out_channels

        # 1x1 卷积用于通道降维
        self.score_fr = nn.Conv2d(conv7_ch, num_classes, 1)

        if version == '16s' or version == '8s':
            self.score_pool4 = nn.Conv2d(pool4_ch, num_classes, 1)
            # 2× 上采样（score_fr → pool4 尺寸）
            self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1, bias=False)

        if version == '8s':
            self.score_pool3 = nn.Conv2d(pool3_ch, num_classes, 1)
            # 2× 上采样（fuse16 → pool3 尺寸）
            self.upscore2_pool4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1, bias=False)

        # 最终上采样层
        if version == '32s':
            self.upscore32 = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, padding=16, bias=False)
        elif version == '16s':
            self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, 32, stride=16, padding=8, bias=False)
        elif version == '8s':
            self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4, bias=False)

        # 权重初始化（主干已初始化，仅初始化新增层）
        self._initialize_additional_layers()

    def _initialize_additional_layers(self):
        """
        初始化新增的 score 和 upscore 层
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m not in self.backbone.modules():
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pool3, pool4, conv7 = self.backbone(x)

        score = self.score_fr(conv7)

        if self.version == '32s':
            out = self.upscore32(score)

        elif self.version == '16s':
            up_score = self.upscore2(score)
            pool4_score = self.score_pool4(pool4)
            fuse = up_score + pool4_score
            out = self.upscore16(fuse)

        elif self.version == '8s':
            up_score2 = self.upscore2(score)
            pool4_score = self.score_pool4(pool4)
            fuse16 = up_score2 + pool4_score

            up_fuse16 = self.upscore2_pool4(fuse16)
            pool3_score = self.score_pool3(pool3)
            fuse8 = up_fuse16 + pool3_score

            out = self.upscore8(fuse8)
        else:
            raise ValueError("version must be '32s', '16s' or '8s'")

        return out



if __name__ == "__main__":
    # 参数设置
    num_classes = 21  # PASCAL VOC 类别数（含背景）
    batch_size = 1
    input_h, input_w = 224, 224

    # 创建随机输入张量 (B, C, H, W)
    x = torch.randn(batch_size, 3, input_h, input_w)
    print(f"输入尺寸: {x.shape}")

    # 测试 VGG16 主干的 FCN
    print("\n=== Testing FCN with ResNet18 Backbone ===")
    for version in ['32s', '16s', '8s']:
        print(f"\n--- Testing FCN-{version} (ResNet18) ---")
        model = FCN_ResNet18(num_classes=num_classes, version=version)
        model.eval()  # 测试模式

        with torch.no_grad():
            out = model(x)

        print(f"输出尺寸: {out.shape}")

    pass
