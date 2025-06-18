from net1 import *
import numpy as np
import matplotlib.pyplot as plt


def dice_loss(preds, target):
    smooth = 1.
    # 计算Dice系数
    intersection = (preds * target).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
def BCE_loss( preds, target):
    eps = 1e-7
    loss1 = -torch.log(preds + eps) * target
    loss2 = -torch.log(1 - preds + eps) * (1 - target)
    loss = loss1 + loss2
    return loss.mean()
def focal_loss( preds, target, alpha=0.25, gamma=2):
    eps = 1e-7
    loss_1 = -1 * alpha * torch.pow((1 - preds), gamma) * torch.log(preds + eps) * target
    loss_0 = -1 * (1 - alpha) * torch.pow(preds, gamma) * torch.log(1 - preds + eps) * (1 - target)
    loss = loss_0 + loss_1
    return torch.mean(loss)

model = UNet()

model.load_state_dict(torch.load("./result7/best_model.pth"))
x = np.load("processed_data/x_data.npy")
y = np.load("processed_data/liver.npy")

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

a = x[1408].unsqueeze(0)
b = y[1408]

print(b.shape)


pred = model(a)
a = a.squeeze()
# pred = torch.argmax(pred, dim=1)
# pred = pred.squeeze()
pred = torch.sigmoid(pred)
pred = (pred > 0.5).long()

b = b.unsqueeze(0)
loss1 = dice_loss(pred, b)
print(loss1)
loss2 = BCE_loss(pred, b)
print(loss2)
loss3 = focal_loss(pred, b)
print(loss3)

b = b.squeeze()

pred = pred.squeeze()
print(pred.shape)


plt.imshow(a, cmap="gray")
plt.show()

plt.imshow(b, cmap="gray")
plt.show()

plt.imshow(pred, cmap="gray")
plt.show()

