import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 8),
            nn.ReLU(),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class EfficientNet(nn.Module):
    def __init__(self, variant='b0', num_classes=5, dropout_rate=0.3):
        super(EfficientNet, self).__init__()
        
        # 初始化EfficientNet基础模型
        if variant == 'b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            feature_dim = 1280
        elif variant == 'b1':
            self.model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
            feature_dim = 1280
        elif variant == 'b2':
            self.model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            feature_dim = 1408
        elif variant == 'b3':
            self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            feature_dim = 1536
        elif variant == 'b4':
            self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            feature_dim = 1792
        elif variant == 'b5':
            self.model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
            feature_dim = 2048
        elif variant == 'b6':
            self.model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.DEFAULT)
            feature_dim = 2304
        elif variant == 'b7':
            self.model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
            feature_dim = 2560
        
        # 添加注意力层
        self.attention = AttentionBlock(feature_dim)
        
        # 重要：先将分类器设为Identity
        self.model.classifier = nn.Identity()
        
        # 然后再设置新的分类器
        self.model.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.model.features(x)
        features = self.model.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.attention(features)
        return self.model.classifier(features)
    
def create_efficientnet_dr():
    return EfficientNet()
