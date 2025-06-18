import numpy as np
import os
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch.nn.functional as F

path = './ARRAY_FORMAT'

items = [item for item in os.listdir(path)
         if os.path.isfile(os.path.join(path, item)) and item.endswith('.npy')]

x = np.zeros((1588,1,192,256))

artery_data = np.zeros((1588,1,192,256),dtype='uint8')
liver_data = np.zeros((1588,1,192,256),dtype='uint8')
stomach_data = np.zeros((1588,1,192,256),dtype='uint8')
vein_data = np.zeros((1588,1,192,256),dtype='uint8')
count = 0



for item in items:
    full_path = os.path.join(path, item)
    loaded_data = np.load(full_path, allow_pickle=True)
    data = loaded_data.item()

    image = data['image']
    if image.ndim == 3:
        image = image/255.0
        image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

    image = resize(image, (192,256), anti_aliasing=True)
    x[count,0,:,:] = image

    structures = data['structures']
    artery = structures['artery']
    liver = structures['liver']
    stomach = structures['stomach']
    vein = structures['vein']

    artery = artery.unsqueeze(0).unsqueeze(0)
    liver = liver.unsqueeze(0).unsqueeze(0)
    stomach = stomach.unsqueeze(0).unsqueeze(0)
    vein = vein.unsqueeze(0).unsqueeze(0)

    artery = F.max_pool2d(artery, kernel_size=4)
    liver = F.max_pool2d(liver, kernel_size=4)
    stomach = F.max_pool2d(stomach, kernel_size=4)
    vein = F.max_pool2d(vein, kernel_size=4)

    artery = artery.squeeze(0).squeeze(0)
    liver = liver.squeeze(0).squeeze(0)
    stomach = stomach.squeeze(0).squeeze(0)
    vein = vein.squeeze(0).squeeze(0)

    artery_data[count,0,:,:] = artery
    liver_data[count,0,:,:] = liver
    stomach_data[count,0,:,:] = stomach
    vein_data[count,0,:,:] = vein
    print(count)
    count += 1

a1 = liver_data[100,0,:,:]
plt.imshow(a1, cmap='gray')
plt.show()

save_path = './processed_data'  # 保存目录（可自定义）
os.makedirs(save_path, exist_ok=True)  # 创建目录（若不存在）


np.save(os.path.join(save_path, 'x_data.npy'), x)

np.save('processed_data/artery.npy', artery_data)
np.save('processed_data/liver.npy', liver_data)
np.save('processed_data/stomach.npy', stomach_data)
np.save('processed_data/vein.npy', vein_data)