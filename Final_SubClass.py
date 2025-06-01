import pandas as pd
import numpy as np
import torch
import xgboost as xgb
import torch.nn.functional as F
from EfficientNet import create_efficientnet_dr
from DataPreparation_1 import train_transform
from PIL import Image

label_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
label_dic = {i: label for i, label in enumerate(label_names)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_efficientnet_dr()
model.load_state_dict(torch.load("DR/models/best_acc.pth", map_location=device))
model.to(device)
model.eval()

img = 'DR\\Test\\Moderate\\15cd5f52d300.png'
img_data = Image.open(img).convert('RGB')
img_data = train_transform(img_data)

img_data = img_data.to(device)
img_data = img_data.unsqueeze(0)

with torch.no_grad():
    output = model(img_data)
    softmax = F.softmax(output, dim=1)
    label = label_dic[softmax.argmax(dim=1).item()]

loaded_model = xgb.XGBClassifier()

if label == 'No_DR':
    print(f'Class: {label}')
elif label == 'Mild':
    loaded_model.load_model(f'DR/subclass_model/Mild_xgb_model.json')
    y_pred = loaded_model.predict(softmax.cpu().numpy())
    print(f'Class: {label}, Sub-Class: clsuster {y_pred[0]}')
elif label == 'Moderate':
    loaded_model.load_model(f'DR/subclass_model/Moderate_xgb_model.json')
    y_pred = loaded_model.predict(softmax.cpu().numpy())
    print(f'Class: {label}, Sub-Class: clsuster {y_pred[0]}')
elif label == 'Severe':
    loaded_model.load_model(f'DR/subclass_model/Severe_xgb_model.json')
    y_pred = loaded_model.predict(softmax.cpu().numpy())
    print(f'Class: {label}, Sub-Class: clsuster {y_pred[0]}')
elif label == 'Proliferate_DR':
    print(f'Class: {label}')