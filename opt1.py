import torch.optim as optim
from net3 import *
from data2 import *
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# 创建结果文件夹
RESULT_DIR = 'result1'
os.makedirs(RESULT_DIR, exist_ok=True)


# 训练配置
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CombinedLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.tensor(0.7))  # 可学习权重

    def dice_loss(self, preds, target):
        smooth = 1.
        # 计算Dice系数
        intersection = (preds * target).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def BCE_loss(self, preds, target):
        eps = self.eps
        loss1 = -torch.log(preds + eps) * target
        loss2 = -torch.log(1 - preds + eps) * (1 - target)
        loss = loss1 + loss2
        return loss.mean()

    def focal_loss(self, preds, target, alpha=0.25, gamma=2):
        eps = self.eps
        loss_1 = -1 * alpha * torch.pow((1 - preds), gamma) * torch.log(preds + eps) * target
        loss_0 = -1 * (1 - alpha) * torch.pow(preds, gamma) * torch.log(1 - preds + eps) * (1 - target)
        loss = loss_0 + loss_1
        return torch.mean(loss)

    def forward(self, preds, target):
        preds = F.sigmoid(preds)
        ce = self.BCE_loss(preds, target)
        dice = self.dice_loss(preds, target)
        # focal = self.focal_loss(preds, target)
        return self.weight * ce + (1 - self.weight) * dice


loss_fn = CombinedLoss()


def calculate_metrics(preds, targets):
    """指标计算函数，支持批量处理"""
    conf_matrix = torch.bincount(
        2 * targets + preds,
        minlength=2 ** 2
    ).reshape(2, 2)
    return conf_matrix


def evaluate_phase(model, dataloader, criterion, device):
    """统一验证/测试阶段处理流程"""
    model.eval()
    total_loss = 0.0
    conf_matrix = torch.zeros((2, 2), device=device)

    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs, masks = inputs.to(device), masks.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            outputs = F.sigmoid(outputs)
            preds = (outputs > 0.5).long()
            targets = masks.squeeze(1).long()

            # 更新混淆矩阵
            batch_conf = calculate_metrics(preds.flatten(), targets.flatten())
            conf_matrix += batch_conf

    # 指标计算
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    return {
        'loss': total_loss / len(dataloader),
        'precision': precision[1].item(),
        'recall': recall[1].item(),
        'dice': dice[1].item(),
        'conf_matrix': conf_matrix.cpu().numpy()
    }


# 初始化训练组件
model = UNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=0, factor=0.8)  # 调整学习率
criterion = loss_fn.to(DEVICE)

# 训练监控数据
training_log = {
    'train_loss': [],
    'val_metrics': [],
    'best_dice': 0.0,
    'start_time': time.time()
}

total_params = sum(p.numel() for p in model.parameters())
print(f"模型的总参数数量为: {total_params}")

# 训练主循环
print(f"开始训练，设备: {DEVICE}, 初始学习率: {LEARNING_RATE}")
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    epoch_loss = 0.0
    batch_counter = 0

    for inputs, masks in train_loader:
        inputs, masks = inputs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_counter += 1

        # 每50个batch输出进度
        if batch_counter % 50 == 0:
            avg_loss = epoch_loss / batch_counter
            print(f"Epoch {epoch + 1}/{EPOCHS} | Batch {batch_counter}/{len(train_loader)}"
                  f" | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # 验证阶段
    val_results = evaluate_phase(model, val_loader, criterion, DEVICE)
    training_log['val_metrics'].append(val_results)
    training_log['train_loss'].append(epoch_loss / len(train_loader))

    # 学习率调整
    scheduler.step(val_results['dice'])

    # 保存最佳模型
    if val_results['dice'] > training_log['best_dice']:
        training_log['best_dice'] = val_results['dice']
        model_path = os.path.join(RESULT_DIR, 'best_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"保存新最佳模型到 {model_path}，验证集dice: {val_results['dice']:.4f}")

    # 打印阶段报告
    epoch_time = time.time() - training_log['start_time']
    print(f"\nEpoch {epoch + 1} 完成 | 耗时: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")
    print(f"训练损失: {training_log['train_loss'][-1]:.4f}")
    print(f"验证结果 - 损失: {val_results['loss']:.4f} | 精确率: {val_results['precision']:.4f}"
          f" | 召回率: {val_results['recall']:.4f} | dice: {val_results['dice']:.4f}\n")

# 最终测试
model_path = os.path.join(RESULT_DIR, 'best_model.pth')
model.load_state_dict(torch.load(model_path))
test_results = evaluate_phase(model, test_loader, criterion, DEVICE)

print("\n最终测试结果:")
print(f"测试集dice: {test_results['dice']:.4f}")
print(f"混淆矩阵:\n{test_results['conf_matrix']}")

# 可视化训练过程
plt.figure(figsize=(12, 6))
plt.plot(training_log['train_loss'], label='Training Loss')
plt.plot([m['loss'] for m in training_log['val_metrics']], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
loss_curve_path = os.path.join(RESULT_DIR, 'training_curves.png')
plt.savefig(loss_curve_path)
print(f"已保存损失曲线到 {loss_curve_path}")

plt.figure(figsize=(12, 6))
plt.plot([m['precision'] for m in training_log['val_metrics']], label='precision')
plt.plot([m['recall'] for m in training_log['val_metrics']], label='recall')
plt.plot([m['dice'] for m in training_log['val_metrics']], label='Dice')
plt.title('metrics Trend')
plt.xlabel('Epoch')
plt.ylabel('metrics')
plt.legend()
metrics_path = os.path.join(RESULT_DIR, 'metrics.png')
plt.savefig(metrics_path)
print(f"已保存指标曲线到 {metrics_path}")


# 可视化示例预测
def visualize_prediction(model, dataset, device, save_dir=RESULT_DIR, num_samples=5):
    """可视化多个预测结果并保存到指定目录"""
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

    model.eval()

    for i in range(num_samples):
        sample_idx = np.random.randint(len(dataset))
        image, mask = dataset[sample_idx]

        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).cpu().squeeze()
            pred = F.sigmoid(pred)
            pred = (pred > 0.5).long()

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title('Input Image')

        plt.subplot(1, 3, 2)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.title('Ground Truth')

        plt.subplot(1, 3, 3)
        plt.imshow(pred.numpy(), cmap='gray')
        plt.title('Prediction')

        save_path = os.path.join(save_dir, 'predictions', f'prediction_{i}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"已保存预测示例到 {save_path}")


# 保存多个预测结果
visualize_prediction(model, test_dataset, DEVICE)

# 保存测试结果到文本文件
with open(os.path.join(RESULT_DIR, 'test_results.txt'), 'w') as f:
    f.write("最终测试结果:\n")
    f.write(f"测试集dice: {test_results['dice']:.4f}\n")
    f.write(f"精确率: {test_results['precision']:.4f}\n")
    f.write(f"召回率: {test_results['recall']:.4f}\n")
    f.write(f"混淆矩阵:\n{test_results['conf_matrix']}\n")

print(f"已保存测试结果到 {os.path.join(RESULT_DIR, 'test_results.txt')}")