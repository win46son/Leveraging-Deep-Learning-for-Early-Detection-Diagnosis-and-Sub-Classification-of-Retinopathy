import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

label_names = ['Mild', 'Moderate', 'Severe']
file_list = ['Train', 'Test']
os.makedirs('DR/subclass_model', exist_ok=True)

for label in label_names:
    for loc in file_list:
        df = pd.read_csv(f'DR\\softmax_results\\{loc}\\{label}.csv')

        if loc == 'Train':
            X_train = df.drop(['label', 'cluster'], axis=1)
            y_train = df['cluster']
        else:
            X_test = df.drop(['label', 'cluster'], axis=1)
            y_test = df['cluster']

    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1
    
    print('='*60)
    print(f'{label} Classification')
    print('='*60)
    print('Value Counts')
    print(y_test.value_counts())
    
    xgb_model = xgb.XGBClassifier(
        # 基本参数
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        
        # 采样参数
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        
        # 正则化参数
        reg_alpha=0,
        reg_lambda=1,
        gamma=0,
        min_child_weight=1,
        
        # 二分类专用参数
        eval_metric='logloss',          # 评估指标
        scale_pos_weight=scale_pos_weight,  # 处理类别不平衡
        
        # 其他参数
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False  # 避免警告
    )

    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    print(f'Acc{acc}')
    sns.heatmap(cm, annot=True, fmt='d', cbar=True)
    plt.title(f'Confusion Matrix of Class {label}')
    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')
    plt.show()

    xgb_model.save_model(f'DR/subclass_model/{label}_xgb_model.json')
