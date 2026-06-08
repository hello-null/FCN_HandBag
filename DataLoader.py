import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
from torchvision import transforms
from typing import List
from torch.utils.data import DataLoader



def batch_tensor_to_pil(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> List[Image.Image]:
    """
    将 [B, 3, H, W] 归一化张量逆变换为 PIL 图像列表。

    参数:
        tensor: 形状 (B, 3, H, W)，值域为归一化后的范围（通常均值附近）
        mean:   预处理时使用的均值，默认 ImageNet 标准值
        std:    预处理时使用的标准差，默认 ImageNet 标准值

    返回:
        List[PIL.Image] 每张图像的模式均为 'RGB'
    """
    # 反归一化: x = x * std + mean
    device = tensor.device
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    tensor = tensor * std + mean                # 还原到 [0, 1] 区间
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # 转为 PIL 列表（ToPILImage 要求单张 [C, H, W]）
    pil_images = []
    for i in range(tensor.size(0)):
        img_t = tensor[i].cpu()                 # [3, H, W]
        pil_img = transforms.ToPILImage()(img_t)
        pil_images.append(pil_img)

    return pil_images


def batch_gray_tensor_to_pil(tensor):
    """
    将灰度图张量转换为 PIL 图像列表。
    支持形状 [N, 1, H, W] 或 [N, H, W]。
    自动处理浮点 [0,1] 和 uint8 [0,255] 情况。
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(1)  # [N, H, W]
    elif tensor.dim() != 3:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")

    pil_images = []
    for i in range(tensor.size(0)):
        img_t = tensor[i].detach().cpu()
        if img_t.dtype != torch.uint8:
            # 假设为浮点 [0,1] 或任意范围，先 clamp 再乘 255
            img_t = (img_t.clamp(0, 1) * 255).to(torch.uint8)
        # 转为 numpy (H, W)
        arr = img_t.numpy()
        pil_img = Image.fromarray(arr, mode='L')
        pil_images.append(pil_img)
    return pil_images


class LetterboxResize:
    """等比缩放图像到目标尺寸，空白区域用指定颜色填充"""

    def __init__(self, w: int, h: int, fill_color: tuple = (0, 0, 0)):
        """
        参数:
            w: 目标宽度
            h: 目标高度
            fill_color: 填充颜色，RGB 三元组，范围 0-255，默认黑色
        """
        self.w = w
        self.h = h
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        对输入图像进行等比缩放并填充至目标尺寸
        参数:
            img: PIL 图像 (模式为 RGB 或 RGBA，建议统一为 RGB)
        返回:
            处理后的 PIL 图像，尺寸为 (self.w, self.h)
        """
        # 计算缩放比例，确保内容完整可见
        w_orig, h_orig = img.size
        scale = min(self.w / w_orig, self.h / h_orig)

        # 按比例缩放图像
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

        # 创建背景画布并居中粘贴缩放后的图像
        canvas = Image.new('RGB', (self.w, self.h), self.fill_color)
        paste_x = (self.w - new_w) // 2
        paste_y = (self.h - new_h) // 2
        canvas.paste(img_resized, (paste_x, paste_y))

        return canvas



class ColorToBinaryMask:
    """
    将 PIL 彩色图像转换为二值掩码张量。
    处理流程：灰度化 -> 黑白互换 -> 阈值 >100 处设为 1，其余为 0。
    返回形状 [1, H, W] 的浮点张量，值仅为 0.0 或 1.0。
    """
    def __init__(self):
        pass

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1. 转为灰度图像（模式 'L'）
        gray = img.convert('L')

        # 2. 黑白互换（像素值反转：255 - 原值）
        inverted = Image.eval(gray, lambda x: 255 - x)

        # 3. 根据反转后的灰度进行二值化：>100 -> 1，否则 0
        binary = Image.eval(inverted, lambda x: 1 if x >= 100 else 0)

        # 4. 转为张量 [1, H, W]，保留 0/1 值
        arr = np.array(binary, dtype=np.uint8)          # (H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0)      # (1, H, W)
        return tensor.float()                           # 转为 float，值为 0.0 或 1.0


class HandbagSegDataset(Dataset):
    """手提包语义分割数据集加载器

    数据集目录结构：
        root/
        ├── imgs/        # 原始手提包图像（.jpg）
        └── labels/      # 对应二值分割标签（.jpg），黑色(0)为包，白色(255)为背景

    返回：
        image: 经过预处理的 RGB 图像张量
        mask:  归一化后的单通道标签（0=背景，1=手提包），这里要注意，重新规定1=前景，0=背景，和.jpg相反了。
    """
    def __init__(self, root, img_transform=None, mask_transform=None):
        """
        Args:
            root (str): 数据集根目录，下面包含 imgs/ 和 labels/ 两个文件夹
            img_transform (callable, optional): 仅作用于图像的变换（如 Normalize）
            mask_transform (callable, optional): 仅作用于 mask 的变换（如 ToTensor）
        """
        self.root = root
        self.img_dir = os.path.join(root, 'imgs')
        self.label_dir = os.path.join(root, 'labels')

        # 获取 imgs 文件夹下所有 .jpg 文件名，并验证 labels 中也存在同名文件
        self.img_names = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith('.jpg') and os.path.exists(os.path.join(self.label_dir, f))
        ])

        if len(self.img_names) == 0:
            raise RuntimeError(f"在 {self.img_dir} 和 {self.label_dir} 中未找到匹配的 .jpg 文件")

        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.label_dir, img_name)

        # 用 PIL 读取图像
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 灰度图，0~255
        width, height = image.size

        # 分别应用特定的图像和 mask 变换
        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask, width, height


# ----------------------- 测试代码 -----------------------
def main():
    # 假设你的数据集解压到当前目录下的 data/handbag 文件夹
    dataset_root = r'F:\datasets\HandBag\train'   # 请根据实际情况修改

    # 图像预处理
    img_transform = transforms.Compose([
        LetterboxResize(w=224,h=224,fill_color=(0,0,0)),
        transforms.ToTensor(),                     # [0,255] -> [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_transform = transforms.Compose([
        LetterboxResize(w=224,h=224,fill_color=(255,255,255)),
        ColorToBinaryMask(),
    ])

    # 创建数据集实例
    dataset = HandbagSegDataset(
        root=dataset_root,
        img_transform=img_transform,
        mask_transform=label_transform,
    )
    train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

    print(f"数据集大小: {len(dataset)}")

    for img, mask, w, h in train_dataloader:
        print(img.shape,mask.shape,w.shape,h.shape) # torch.Size([16, 3, 224, 224]) torch.Size([16, 1, 224, 224]) torch.Size([16]) torch.Size([16])

        # 逆变换并使用plt显示
        lst_pils = batch_tensor_to_pil(img)
        for pil in lst_pils:
            np_img = np.array(pil)
            plt.imshow(np_img)
            plt.show()

        lst_pils = batch_gray_tensor_to_pil(mask)
        for pil in lst_pils:
            np_img = np.array(pil)
            plt.imshow(np_img,cmap='gray')
            plt.show()

    pass

if __name__ == '__main__':
    main()