## 提取softmax值，出来form csv
## csv有了，就正常build xgboost model
## if condition - 5个model
## 先过efficientnet， net forward判断结果，然后选sub-class model，同时提取他的softmax值

import torch
from PIL import Image
from DataPreparation_1 import train_transform
from EfficientNet import create_efficientnet_dr
import torch.nn.functional as F
import glob
import os
import pandas as pd

label_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
file_list = ['Train', 'Test']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_efficientnet_dr()
model.load_state_dict(torch.load("DR/models/best_acc.pth", map_location=device))
model.to(device)
model.eval()

for label in label_names:
    df = pd.read_csv(f'DR\\class_wise_subclasses\\{label}\\{label}_cluster_mapping.csv')
    df_name_list = df['filename'].tolist()
    
    for loc in file_list:
        results = []
        os.makedirs(f'DR/softmax_results/{loc}', exist_ok=True)
        img_list = glob.glob(f'DR\\{loc}\\{label}\\*.png')
        
        for img in img_list:
            img_name = os.path.splitext(os.path.basename(img))[0]

            if img_name in df_name_list:
                cluster = df['cluster'][df_name_list.index(img_name)]
                img_data = Image.open(img).convert('RGB')
                img_data = train_transform(img_data)

                img_data = img_data.to(device)
                img_data = img_data.unsqueeze(0)

                with torch.no_grad():  # 推理时禁用梯度计算
                    output = model(img_data)
                    probabilities = F.softmax(output, dim=1)  # 指定维度为1（类别维度）

                    result = {
                        'label': img_name,
                        '0': probabilities[0][0].item(),
                        '1': probabilities[0][1].item(),
                        '2': probabilities[0][2].item(),
                        '3': probabilities[0][3].item(),
                        '4': probabilities[0][4].item(),
                        'cluster': cluster
                    }

                    results.append(result)

        df_write = pd.DataFrame(results)
        df_write.to_csv(f'DR/softmax_results/{loc}/{label}.csv', index=False)
        print(f"结果已保存到 {loc}-{label} csv file，共处理了 {len(results)} 张图片")


# df = pd.read_csv(f'DR\\class_wise_subclasses\\Mild\\Mild_cluster_mapping.csv')
# df_name_list = df['filename'].tolist()
# print(df_name_list[:15])
# if '0dce95217626' in df_name_list:
#     print(df_name_list.index('0dce95217626'))
#     print(df['cluster'][df_name_list.index('0dce95217626')])