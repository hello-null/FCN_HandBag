import torch
import torch.nn as nn


class VGG16Backbone(nn.Module):
    """
    VGG16 特征提取网络，用于 FCN 系列模型。
    返回三个中间特征图：
        - pool3: 第3个池化层输出（下采样8倍）
        - pool4: 第4个池化层输出（下采样16倍）
        - conv7: 卷积化后的 fc7 输出（下采样32倍）
    """

    @staticmethod
    def _init_weights(module):
        """
        权重初始化函数：
        - 卷积层使用 He 初始化（kaiming_normal）
        - 偏置项初始化为 0
        - BN 层权重初始化为 1，偏置初始化为 0
        """
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)  # 1/2
        )
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)  # 1/4
        )
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)  # 1/8  -> pool3
        )
        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)  # 1/16 -> pool4
        )
        # Block 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)  # 1/32
        )
        # 将全连接层 fc6、fc7 转换为卷积层
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3), nn.BatchNorm2d(4096), nn.ReLU(inplace=False), nn.Dropout(p=0.2)
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1), nn.BatchNorm2d(4096), nn.ReLU(inplace=False), nn.Dropout(p=0.2)
        )

        # 权重初始化
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        pool3 = x  # 下采样 8倍  torch.Size([1, 256, 28, 28])
        x = self.conv4(x)
        pool4 = x  # 下采样16倍 torch.Size([1, 512, 14, 14])
        x = self.conv5(x)  # torch.Size([1, 512, 7, 7])
        x = self.fc6(x)  # torch.Size([1, 4096, 7, 7])
        x = self.fc7(x)  # torch.Size([1, 4096, 7, 7])
        conv7 = x  # 下采样32倍   torch.Size([1, 4096, 7, 7])
        return pool3, pool4, conv7  # torch.Size([1, 256, 28, 28])   torch.Size([1, 512, 14, 14])    torch.Size([1, 4096, 7, 7])



class FCN_Vgg16(nn.Module):
    """
    全卷积网络（Fully Convolutional Network）用于语义分割。
    参数：
        num_classes: 输出类别数（含背景），默认21（PASCAL VOC）。
        version: 模型版本，可选 '32s', '16s', '8s'。
    """

    @staticmethod
    def _init_weights(module):
        """
        权重初始化函数：
        - 卷积层使用 He 初始化（kaiming_normal）
        - 偏置项初始化为 0
        - BN 层权重初始化为 1，偏置初始化为 0
        """
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def __init__(self, num_classes=21, version='32s'):
        super().__init__()
        self.version = version
        self.backbone = VGG16Backbone()

        self.score_fr = nn.Conv2d(4096, num_classes, 1)

        if version == '16s' or version == '8s':
            self.score_pool4 = nn.Conv2d(512, num_classes, 1)
            # 2× 上采样（score_fr → pool4 尺寸）
            self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1, bias=False)

        if version == '8s':
            self.score_pool3 = nn.Conv2d(256, num_classes, 1)
            # 2× 上采样（fuse16 → pool3 尺寸）
            self.upscore2_pool4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1, bias=False)

        # 最终上采样层
        if version == '32s':
            self.upscore32 = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, padding=16, bias=False)
        elif version == '16s':
            self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, 32, stride=16, padding=8, bias=False)
        elif version == '8s':
            self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4, bias=False)

        # 权重初始化
        self.apply(self._init_weights)

    def forward(self, x):
        pool3, pool4, conv7 = self.backbone(x)
        # torch.Size([1, 256, 28, 28]) torch.Size([1, 512, 14, 14]) torch.Size([1, 4096, 7, 7])
        # print(pool3.shape, pool4.shape, conv7.shape)
        # exit()

        score = self.score_fr(conv7)  # torch.Size([1, num_class, 7, 7])

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
    print("\n=== Testing FCN with VGG16 Backbone ===")
    for version in ['32s', '16s', '8s']:
        print(f"\n--- Testing FCN-{version} (VGG16) ---")
        model = FCN_Vgg16(num_classes=num_classes, version=version)
        model.eval()  # 测试模式

        with torch.no_grad():
            out = model(x)

        print(f"输出尺寸: {out.shape}")
    pass