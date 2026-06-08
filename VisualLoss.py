import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
import re

def parse_log(filepath):
    """解析训练日志，返回 epochs, losses, lrs"""
    epochs, losses, lrs = [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            # 匹配 epoch, loss, lr 数值
            match = re.match(r'epoch:\s*(\d+),\s*loss:\s*([0-9.]+),\s*lr:\s*([0-9.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                lrs.append(float(match.group(3)))
    return epochs, losses, lrs

def plot_log(filepath='./train_log.txt'):
    epochs, losses, lrs = parse_log(filepath)
    if not epochs:
        print("未找到有效日志行，请检查文件格式。")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Loss 图
    ax1.plot(epochs, losses, 'b-', marker='o', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    # Learning Rate 图
    ax2.plot(epochs, lrs, 'r-', marker='s', label='Learning Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_log()