import glob
import os
import pandas as pd
import random
import shutil

os.makedirs('DR/Train', exist_ok=True)
os.makedirs('DR/Test', exist_ok=True)

labels_name = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

## 创建新file，split train test
for label in labels_name:
    
    train_loc = f'DR/Train/{label}'
    test_loc = f'DR/Test/{label}'

    os.makedirs(test_loc, exist_ok=True)
    os.makedirs(train_loc, exist_ok=True)

    image_list = glob.glob(f'C:\\Users\\User\\Documents\\Pytorch\\DR\\colored_images\\{label}\\*.png')
    test_ratio = 0.15
    num_test = int(test_ratio * len(image_list))

    random.shuffle(image_list)

    train_list = image_list[num_test:]
    test_list = image_list[:num_test]

    for filepath in test_list:
        filename = os.path.basename(filepath)
        shutil.copy(filepath, os.path.join(test_loc, filename))
    
    for filepath in train_list:
        filename = os.path.basename(filepath)
        shutil.copy(filepath, os.path.join(train_loc, filename))
    
    print(f'{label} -> Train: {len(train_list)}, Test: {len(test_list)}')