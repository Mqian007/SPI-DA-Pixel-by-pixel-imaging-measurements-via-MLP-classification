import numpy as np
import os
from scipy.io import loadmat


data = loadmat('xxxx.mat')
data1 = data['data1']
train_y = data['train_y']


base_folder = 'fangzhen'
if not os.path.exists(base_folder):
    os.makedirs(base_folder)


num_rows = min(data1.shape[1], 14290)

for col_idx in range(num_rows):
    column_data = data1[:, col_idx]
    folder_name = str(int(train_y[col_idx]))
    folder_path = os.path.join(base_folder, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    first_value = column_data[0]
    if first_value == 0:
        raise ValueError(f" {col_idx} 错误")

    normalized_data = column_data / first_value


    sum_normalized = np.sum(normalized_data)
    scaled_data = normalized_data * (100 / sum_normalized)


    file_path = os.path.join(folder_path, f'd{col_idx}.txt')
    np.savetxt(file_path, scaled_data, fmt='%.16f')

print("数据处理完成。")
