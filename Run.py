import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms


# 导入之前实现的网络和数据加载类（假设它们在当前目录下）
from Model import FCN                          # FCN 模型（8s/16s/32s）
from DataLoader import HandbagSegDataset, LetterboxResize, ColorToBinaryMask, batch_tensor_to_pil, batch_gray_tensor_to_pil     # 数据集类



def save_visualization(model, loader, device, epoch, save_dir='vis', num_samples=4):
    """
    从数据集中取 num_samples 张图像，模型推理后，保存原图、真值、预测的拼接图。
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    lst_imgs = []   # [i]=H W C
    lst_gt = []     # [i]=H W
    lst_pred = []   # [i]=H W
    with torch.no_grad():
        for img, mask, w, h in loader: # [8, 3, 224, 224] [8, 1, 224, 224 | 0~1] [8] [8]
            # print(img.shape, mask.shape, w.shape, h.shape)
            # print(torch.unique(mask))
            # print(mask[0][0])
            # exit()

            img = img.to(device)
            pred = model(img)            # [8, num_classes, 224, 224]
            pred = torch.argmax(pred, dim=1)   # [8, 224, 224] 0~1
            # print(pred[0])
            # exit()

            lst_img_np = [np.array(a) for a in batch_tensor_to_pil(img.cpu())]
            lst_imgs.extend(lst_img_np)

            lst_gt_np = (torch.squeeze(mask,dim=1).cpu() * 255).to(torch.uint8).numpy()
            lst_gt.extend(lst_gt_np)

            lst_pred_np = (pred.cpu() * 255).to(torch.uint8).numpy()
            lst_pred.extend(lst_pred_np)

            if len(lst_imgs) > num_samples:
                break

    # print(len(lst_imgs),lst_imgs[0].shape,lst_imgs[1].shape)
    # print(len(lst_gt),lst_gt[0].shape,lst_gt[1].shape)
    # print(len(lst_pred),lst_pred[0].shape,lst_pred[1].shape)
    # exit()

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    for i in range(num_samples):
        # 原图
        axes[i][0].imshow(lst_imgs[i])
        axes[i][0].set_title('Image')
        axes[i][0].axis('off')
        # 真值 mask
        axes[i][1].imshow(lst_gt[i], cmap='gray')
        axes[i][1].set_title('Ground Truth')
        axes[i][1].axis('off')
        # 预测 mask
        axes[i][2].imshow(lst_pred[i], cmap='gray')
        axes[i][2].set_title('Prediction')
        axes[i][2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'))
    plt.close()
    model.train()


# ------------------ 训练主函数 ------------------
def run_train(epochs, lr, train_root, test_root, num_classes, loss_txt_path,
              version='8s', batch_size=8, save_interval=10,
              resume=None, vis_dir='vis', device='cuda', save_dir='./checkpoints/'):
    """
    训练 FCN 模型，支持断点续训、日志记录、可视化。

    Args:
        epochs: 总训练轮次
        lr: 学习率
        train_root / test_root: 训练 / 测试 数据集根目录（含 imgs/ 和 labels/）
        num_classes: 分割类别数（含背景）
        loss_txt_path: 损失记录文件路径（每行 "epoch: i, loss: xxxx"）
        version: FCN 版本，可选 '32s', '16s', '8s'
        batch_size: 批大小
        save_interval: 每隔多少 epoch 保存一次权重（文件名 fcn_{epoch}.pth）
        resume: 断点续训的 checkpoint 路径（None 表示从头训练）
        vis_dir: 可视化结果保存目录
        device: 'cuda' 或 'cpu'
        save_dir: 模型权重保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ---------- 1. 数据集与数据加载 ----------
    img_transform = transforms.Compose([
        LetterboxResize(w=224,h=224,fill_color=(0,0,0)),
        transforms.ToTensor(),  # [0,255] -> [0,1]  HWC -> CHW
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_transform = transforms.Compose([
        LetterboxResize(w=224,h=224,fill_color=(255,255,255)), # 真实label是前景黑色背景白色
        ColorToBinaryMask(), # 彩色转灰度  0~255 -> 0~1
    ])

    # 创建数据集实例
    train_dataset = HandbagSegDataset(
        root=train_root,
        img_transform=img_transform,
        mask_transform=label_transform,
    )
    test_dataset = HandbagSegDataset(
        root=test_root,
        img_transform=img_transform,
        mask_transform=label_transform,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

    # ---------- 2. 模型、损失、优化器 ----------
    model = FCN(num_classes=num_classes, version=version).to(device)
    criterion = nn.CrossEntropyLoss()  # 自动对预测做 softmax
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # 也可使用 Adam：optimizer = optim.Adam(model.parameters(), lr=lr)

    # 测试save_visualization
    # save_visualization(model,test_dataloader,device,1,num_samples=10)
    # exit()

    start_epoch = 0
    if resume and os.path.exists(resume):
        print(f"加载断点权重：{resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        start_epoch = checkpoint['epoch'] + 1
        print(f"从 epoch {start_epoch} 继续训练")

    # ---------- 3. 训练循环 ----------
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for imgs, masks, w, h in pbar: # [B, 3, 224, 224] [B, 1, 224, 224] [B] [B]
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs) # torch.Size([B, num_class, 224, 224])

            targets = masks.squeeze(1).long()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_dataloader)

        current_lr = optimizer.param_groups[0]['lr']
        
        # 写入损失文件（追加模式）
        with open(loss_txt_path, 'a', encoding='utf-8') as f:
            f.write(f"epoch: {epoch+1}, loss: {epoch_loss:.6f}, lr: {current_lr:.6f}\n")

        print(f"Epoch {epoch+1} 完成，平均损失: {epoch_loss:.6f}，学习率: {current_lr:.6f}")

        # ---------- 4. 保存可视化结果 ----------
        save_visualization(model, test_dataloader, device, epoch+1, save_dir=vis_dir, num_samples=6)

        # ---------- 5. 定期保存完整 checkpoint ----------
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"fcn_{version}_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, save_path)
            print(f"权重已保存至 {save_path}")

    # 训练结束保存最终权重
    final_path = os.path.join(save_dir, f"fcn_{version}_final.pth")
    torch.save({
        'epoch': epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"训练完成，最终权重已保存至 {final_path}")


# ------------------ 测试集推理函数 ------------------
def run_inference(pth_path, test_root, num_classes, version='8s', device='cuda'):
    """
    加载测试集进行推理，显示原图、ground truth 和预测结果。
    
    Args:
        pth_path: 预训练模型权重路径
        test_root: 测试数据集根目录（含 imgs/ 和 labels/）
        num_classes: 分割类别数（含背景）
        version: FCN 版本，可选 '32s', '16s', '8s'
        device: 'cuda' 或 'cpu'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 数据预处理
    img_transform = transforms.Compose([
        LetterboxResize(w=224, h=224, fill_color=(0, 0, 0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_transform = transforms.Compose([
        LetterboxResize(w=224, h=224, fill_color=(255, 255, 255)),
        ColorToBinaryMask(),
    ])
    
    # 创建测试数据集和数据加载器
    test_dataset = HandbagSegDataset(root=test_root, img_transform=img_transform, mask_transform=label_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 加载模型
    model = FCN(num_classes=num_classes, version=version).to(device)
    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 推理并显示
    with torch.no_grad():
        for img_tensor, mask, w, h in test_loader: # [1,3,H,W] [1,1,H,W]
            img_tensor = img_tensor.to(device)
            
            # 推理
            output = model(img_tensor) # [1,2,H,W]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() # [H,W]
            
            # 转换为可显示格式
            img_pil = batch_tensor_to_pil(img_tensor.cpu())[0]
            gt_np = (mask.squeeze(1).cpu().squeeze(0) * 255).numpy()
            
            # 显示原图、ground truth、预测
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(img_pil); ax1.set_title('Original'); ax1.axis('off')
            ax2.imshow(gt_np, cmap='gray'); ax2.set_title('Ground Truth'); ax2.axis('off')
            ax3.imshow(pred, cmap='gray'); ax3.set_title('Prediction'); ax3.axis('off')
            plt.tight_layout(); plt.show()


# ------------------ 使用示例 ------------------
if __name__ == "__main__":

    # 训练示例
    # run_train(
    #     epochs=100,
    #     lr=0.01,
    #     train_root=r'F:\datasets\HandBag\train',          # 修改为你的数据集根目录
    #     test_root=r'F:\datasets\HandBag\test',
    #     num_classes=2,                             # 背景+手提包
    #     loss_txt_path='train_log.txt',
    #     version='8s',                              # FCN-8s
    #     batch_size=4,
    #     save_interval=5,                          # 每5个epoch保存一次
    #     resume='./checkpoints/fcn_8s_60.pth',     # 如需断点续训：resume='fcn_8s_20.pth'
    #     vis_dir='./vis_output/',                  # 每个epoch保存的预测图路径，方便主观判断效果
    #     device='cuda',
    #     save_dir='./checkpoints/'                  # 模型权重保存目录
    # )


    # 推理示例
    run_inference(
        pth_path='./checkpoints/fcn_8s_final.pth',  # 预训练模型路径
        test_root=r'F:\datasets\HandBag\test',      # 测试数据集根目录
        num_classes=2,                              # 背景+手提包
        version='8s',                               # FCN-8s
        device='cuda'                               # 设备
    )