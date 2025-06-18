import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import os



x = np.load("processed_data/x_data.npy")
y = np.load("processed_data/liver.npy")

x_train = torch.from_numpy(x[0:1300])
y_train = torch.from_numpy(y[0:1300]).long()

x_val = torch.from_numpy(x[1300:1450])
y_val = torch.from_numpy(y[1300:1450]).long()

x_test = torch.from_numpy(x[1450:])
y_test = torch.from_numpy(y[1450:]).long()

# 定义训练集的数据增强管道
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5, interpolation=cv2.INTER_LINEAR),
    A.ElasticTransform(p=0.2, alpha=30, sigma=12,
                       interpolation=cv2.INTER_LINEAR),
    A.GridDistortion(p=0.2, num_steps=5),
], additional_targets={'mask': 'mask'})


# 修改 Dataset 类以支持 Albumentations 转换
class MedicalDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x = x_data
        self.y = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 转换为 HxW 格式的 numpy array
        image = self.x[idx].squeeze(0).numpy().astype(np.float32)
        mask = self.y[idx].squeeze(0).numpy().astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # 添加通道维度并转换为 tensor
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).long()

        return image, mask


# 创建数据集实例
train_dataset = MedicalDataset(x_train, y_train, transform=train_transform)
val_dataset = MedicalDataset(x_val, y_val)
test_dataset = MedicalDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)



if __name__ == '__main__':
    test_sample, test_mask = test_dataset[1]
    plt.imshow(test_sample.squeeze(), cmap='gray')

    plt.show()
    plt.imshow(test_mask.squeeze(), cmap='gray')
    plt.show()


    path = './ARRAY_FORMAT'

    items = [item for item in os.listdir(path)
             if os.path.isfile(os.path.join(path, item)) and item.endswith('.npy')]
    count = 0
    f = None
    for item in items:
        if count == 1301:
            f = item
            break
        count += 1
    full_path = os.path.join(path, f)
    loaded_data = np.load(full_path, allow_pickle=True)
    data = loaded_data.item()
    image = data['image']
    structures = data['structures']
    liver = structures['liver']
    if image.ndim == 3:
        image = image/255.0
        image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.imshow(liver, cmap='gray')
    plt.show()
