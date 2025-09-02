import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from Get_data import get_source_loaders
from model.resnet import create_model

# 参数设置
parser = argparse.ArgumentParser(description='PyTorch 模型训练')
parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--viz_freq', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--data_path', type=str, default="dataset/data_source.xlsx")
parser.add_argument('--exp_path', type=str, default='./PretrainArgs.exp_path')
args = parser.parse_args()

# 可视化类
class JournalVisualizer:
    @classmethod
    def init_font(cls):
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['DejaVu Serif', 'Liberation Serif'],
            'mathtext.fontset': 'stix',
            'axes.formatter.use_mathtext': True
        })

    @classmethod
    def set_journal_style(cls):
        plt.rcParams.update({
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'lines.linewidth': 1.2,
            'axes.linewidth': 0.6,
            'grid.linewidth': 0.4,
            'savefig.dpi': 600,
            'savefig.format': 'tiff',
            'pdf.fonttype': 42
        })

# 训练跟踪器
class TrainingTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epochs = []
        self.epoch_times = []

    def update(self, epoch, train_loss, val_loss, train_accuracy, val_accuracy, epoch_time):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)
        self.epoch_times.append(epoch_time)

# 可视化函数
def plot_training_progress(tracker, exp_path):
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    epochs = tracker.epochs
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_palette = sns.color_palette("colorblind")
    ax1.plot(epochs, tracker.train_losses, label='Train Loss', color=color_palette[0])
    ax1.plot(epochs, tracker.val_losses, label='Val Loss', color=color_palette[1])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(epochs, tracker.train_accuracies, label='Train Acc', color=color_palette[2], linestyle='--')
    ax2.plot(epochs, tracker.val_accuracies, label='Val Acc', color=color_palette[3], linestyle='--')
    ax2.set_ylabel('Accuracy')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=2)

    plt.title('Training Progress')
    plt.tight_layout()
    os.makedirs(exp_path, exist_ok=True)
    plt.savefig(os.path.join(exp_path, 'training_progress.png'))
    plt.close()

    df = pd.DataFrame({
        'Epoch': tracker.epochs,
        'Train Loss': tracker.train_losses,
        'Val Loss': tracker.val_losses,
        'Train Acc': tracker.train_accuracies,
        'Val Acc': tracker.val_accuracies,
        'Time': tracker.epoch_times
    })
    df.to_csv(os.path.join(exp_path, 'training_metrics.csv'), index=False)

# 单个 epoch 训练
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (output.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

# 验证函数
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

# 总训练流程
def train(model, optimizer, train_loader, val_loader, scheduler, criterion, exp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tracker = TrainingTracker()

    best_val_loss = float('inf')
    best_model_weights = None

    for epoch in range(args.epoch):
        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        end = time.time()
        tracker.update(epoch + 1, train_loss, val_loss, train_acc, val_acc, end - start)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, Time={end-start:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()

    torch.save(best_model_weights, os.path.join(exp_path, 'best_model.pth'))
    plot_training_progress(tracker, exp_path)

# 微调函数
import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

def fine_tune(model, source_train_loader, target_train_loader, target_val_loader, args):
    params_group = [
        {'params': model.base_features.parameters(), 'lr': args.lr * 0.1},
        {'params': model.res_layers.parameters(), 'lr': args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr * 5}
    ]

    optimizer = optim.SGD(params_group, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    tracker = TrainingTracker()  # 用来追踪微调过程中的数据

    # 获取迁移学习前的源域和目标域特征
    source_features_before, source_labels = get_features_and_labels(source_train_loader, model)
    target_features_before, target_labels = get_features_and_labels(target_train_loader, model)

    # 可视化迁移学习前的特征分布
    plot_feature_distribution(source_features_before, source_labels, target_features_before, target_labels, args.exp_path, "Before Fine-tuning")

    # 计算迁移学习前的MMD
    mmd_before = compute_mmd(source_features_before, target_features_before)
    print(f"Maximum Mean Discrepancy (MMD) before fine-tuning: {mmd_before:.4f}")

    # 进行微调
    for epoch in range(args.epoch):
        start_time = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in source_train_loader:  # 使用源域训练集
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in target_val_loader:  # 使用目标域验证集
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        scheduler.step()

        epoch_time = time.time() - start_time
        tracker.update(epoch + 1, train_loss, val_loss, train_acc, val_acc, epoch_time)

        print(f"Epoch {epoch + 1}/{args.epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")

        # 计算并输出每个epoch的MMD
        source_features, _ = get_features_and_labels(source_train_loader, model)  # 使用源域训练集
        target_features, _ = get_features_and_labels(target_train_loader, model)  # 使用目标域训练集

        mmd_after_epoch = compute_mmd(source_features, target_features)
        print(f"Maximum Mean Discrepancy (MMD) after epoch {epoch + 1}: {mmd_after_epoch:.4f}")

        # 每个epoch结束后可视化特征分布
        plot_feature_distribution(source_features, source_labels, target_features, target_labels, args.exp_path, f"Epoch {epoch + 1} Fine-tuning")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{args.exp_path}/best_finetune_model.pth")

    # 微调后计算MMD
    mmd_after = compute_mmd(source_features, target_features)
    print(f"Final Maximum Mean Discrepancy (MMD) after fine-tuning: {mmd_after:.4f}")

    # 最终的特征分布可视化
    plot_feature_distribution(source_features, source_labels, target_features, target_labels, args.exp_path, "After Fine-tuning")
    plot_training_progress(tracker, args.exp_path)



def get_features_and_labels(loader, model):
    features = []
    labels = []
    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 禁用梯度计算
        for inputs, targets in loader:
            outputs = model(inputs)  # 获取模型的输出
            features.append(outputs.cpu().numpy())  # 将输出移到CPU并转换为NumPy数组
            labels.append(targets.cpu().numpy())  # 获取对应的标签

    features = np.concatenate(features, axis=0)  # 合并所有批次的特征
    labels = np.concatenate(labels, axis=0)  # 合并所有批次的标签
    return features, labels


def plot_feature_distribution(source_features, source_labels, target_features, target_labels, exp_path, title):
    # 使用t-SNE将特征降到二维
    tsne = TSNE(n_components=2, random_state=0)
    source_tsne = tsne.fit_transform(source_features)  # 降维源域特征
    target_tsne = tsne.fit_transform(target_features)  # 降维目标域特征

    # 绘制t-SNE图
    plt.figure(figsize=(10, 6))
    plt.scatter(source_tsne[:, 0], source_tsne[:, 1], c=source_labels, label='Source Domain', alpha=0.5, cmap='coolwarm')
    plt.scatter(target_tsne[:, 0], target_tsne[:, 1], c=target_labels, label='Target Domain', alpha=0.5, cmap='coolwarm')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.title(f't-SNE Visualization of Feature Distribution ({title})')

    # 保存图像
    plt.savefig(f'{exp_path}/feature_distribution_{title}.png')
    plt.show()  # 关闭当前图像，释放内存


def compute_mmd(source_features, target_features):
    # 使用L2距离计算最大均值差异（MMD）
    source_distances = pairwise_distances(source_features, source_features, metric='euclidean')
    target_distances = pairwise_distances(target_features, target_features, metric='euclidean')
    cross_distances = pairwise_distances(source_features, target_features, metric='euclidean')

    # 计算MMD值：源域内、目标域内、源域和目标域之间的距离的均值
    mmd_value = np.mean(source_distances) + np.mean(target_distances) - 2 * np.mean(cross_distances)
    return mmd_value





# 主程序入口
if __name__ == '__main__':
    visualizer = JournalVisualizer()
    visualizer.init_font()
    visualizer.set_journal_style()

    (source_train, source_val, source_test), _, _ = get_source_loaders(
        args.data_path,
        batch_size=args.batch_size,
        augment=True
    )

    source_model = create_model('resnet', num_classes=2)
    optimizer = optim.Adam(source_model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0)

    train(
        model=source_model,
        optimizer=optimizer,
        train_loader=source_train,
        val_loader=source_val,
        scheduler=scheduler,
        criterion=criterion,
        exp_path=args.exp_path
    )

    (target_train, target_val, target_test), _, _ = get_source_loaders(
        "./dataset/data_domain.xlsx",
        batch_size=16,
        augment=False
    )

    fine_tune_model = create_model('resnet', num_classes=2)
    fine_tune_model.load_state_dict(torch.load(f"{args.exp_path}/best_model.pth"))

    fine_tune(
        model=fine_tune_model,
        source_train_loader=source_train,
        target_train_loader=target_train,
        target_val_loader=target_val,
        args=args
    )


