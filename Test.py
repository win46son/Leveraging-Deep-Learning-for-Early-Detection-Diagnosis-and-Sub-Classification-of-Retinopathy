import torch
import torch.nn.functional as F
import glob
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from EfficientNet import create_efficientnet_dr
from tqdm import tqdm
from DataPreparation_1 import ClaheMedian
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

tta_transform = [
    transforms.Compose([
        ClaheMedian(clip_limit=2.0, tile_grid_size=(8,8), median_kernel_size=5),
        transforms.ToTensor(),
        transforms.Normalize([0.46238312125205994, 0.26547566056251526, 0.11903924494981766],[0.2862863838672638, 0.16006295382976532, 0.09059497714042664])
    ]),

    # transforms.Compose([
    #     ClaheMedian(clip_limit=2.0, tile_grid_size=(8,8), median_kernel_size=5),
    #     transforms.RandomRotation(30),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.46238312125205994, 0.26547566056251526, 0.11903924494981766],[0.2862863838672638, 0.16006295382976532, 0.09059497714042664])
    # ]),

    # transforms.Compose([
    #     ClaheMedian(clip_limit=2.0, tile_grid_size=(8,8), median_kernel_size=5),
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.46238312125205994, 0.26547566056251526, 0.11903924494981766],[0.2862863838672638, 0.16006295382976532, 0.09059497714042664])
    # ]),

    # transforms.Compose([
    #     ClaheMedian(clip_limit=2.0, tile_grid_size=(8,8), median_kernel_size=5),
    #     transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
]

def tta_predict(model, image, device):
    model.eval()
    pred = []

    with torch.no_grad():
        for tform in tta_transform:
            augmented = tform(image).unsqueeze(0).to(device)
            output = model(augmented)
            prob = F.softmax(output, dim=1)
            pred.append(prob.cpu().numpy())
        
    mean_preb = np.mean(pred, axis=0)
    return mean_preb.argmax()

def plot_confusion_matrix(y_true, y_pred, label_names, save_pth=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matricx - TTA Test Results', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_pth:
        plt.savefig(save_pth, dpi=300, bbox_inches='tight')
        print(f'Confusion matrix saved to: {save_pth}')
    
    plt.show()

def plot_normalized_confusion_matrix(y_true, y_pred, label_names, save_pth=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matricx - TTA Test Results', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_pth:
        plt.savefig(save_pth, dpi=300, bbox_inches='tight')
        print(f'Normalized confusion matrix saved to: {save_pth}')
    
    plt.show()

def print_detailed_report(y_true, y_pred, label_names):
    print('\n'+'='*80)
    print('DETAILED CLASSIFICATION REPORT')
    print('='*80)

    total_samples = len(y_true)
    correct_predictions = np.sum(y_true==y_pred)
    accuracy = correct_predictions / total_samples * 100

    print(f'Total Samples: {total_samples}')
    print(f'correct Predictions: {correct_predictions}')
    print(f'Overall Accuracy: {accuracy:.2f}%')

    print('\nCLASSIFICATION REPORT:')
    print('-'*60)
    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    print(report)

    print("\nPER-CLASS DETAILED STATISTICS:")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred)
    
    for i, class_name in enumerate(label_names):
        true_positives = cm[i, i]
        false_positives = cm[:, i].sum() - true_positives
        false_negatives = cm[i, :].sum() - true_positives
        true_negatives = cm.sum() - true_positives - false_positives - false_negatives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_total = cm[i, :].sum()
        class_accuracy = true_positives / class_total * 100 if class_total > 0 else 0
        
        print(f"{class_name}:")
        print(f"  Total samples: {class_total}")
        print(f"  Correctly classified: {true_positives}")
        print(f"  Class accuracy: {class_accuracy:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print()

def display_error_samples(selected_error_samples, label_names):
    print("\nDisplaying Error Samples ...")

    for class_name, samples in selected_error_samples.items():
        if len(samples) > 0:
            print(f'\n--- Class: {class_name} ---')

            n_samples = len(samples)
            if n_samples <= 3:
                rows, cols = 1, n_samples
            else:
                rows = 2
                cols = (n_samples+1 // 2)
            
            plt.figure(figsize=(cols*4, rows*4))
            plt.suptitle(f'Error Samples for Class: {class_name}', fontsize=16, fontweight='bold')

            for i, (img_path, pred_label, true_label) in enumerate(samples):
                img_name = img_path.split('\\')[-1]

                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    plt.subplot(rows, cols, i+1)
                    plt.imshow(img_rgb)
                    plt.title(f'{img_name}\nTrue: {label_names[true_label]}\nPred: {label_names[pred_label]}', fontsize=10, pad=10)
                    plt.axis('off')
                else:
                    print(f'Could not load image: {img_path}')
            
            plt.tight_layout()
            plt.show()

            input(f'Press Enter to continue to next class ...')
    print('Finish displaying all error samples.')

def get_random_error_samples(error_samples_by_class, label_names, samples_per_class=5):
    selected_error = {}

    for class_idx, class_name in enumerate(label_names):
        if class_idx in error_samples_by_class and error_samples_by_class[class_idx]:
            sampels_to_select = random.sample(error_samples_by_class[class_idx],
                                              min(samples_per_class, len(error_samples_by_class[class_idx])))
            selected_error[class_name] = sampels_to_select
        else:
            selected_error[class_name] = []
    return selected_error

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_efficientnet_dr()
    net.load_state_dict(torch.load("DR\\models\\best_acc_n.pth"))
    net.to(device)

    img_list = glob.glob('DR/Test/*/*.png')
    np.random.shuffle(img_list)
    
    correct = 0
    total = len(img_list)

    print(f'Running TTA inference on {total} test samples ...')
    
    label_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
    label_dic = {label: i for i, label in enumerate(label_names)}

    error_samples_by_class = {i: [] for i in range(len(label_names))}

    all_predictions = []
    all_true_labels = []

    for i in tqdm(range(total)):
        img_path = img_list[i]
        img_label = label_dic[img_list[i].split('\\')[-2]]
        image = Image.open(img_path).convert('RGB')
        pred = tta_predict(net, image, device)

        # 收集预测结果和真实标签
        all_predictions.append(pred)
        all_true_labels.append(img_label)

        if pred == img_label:
            correct += 1
        else:
            error_samples_by_class[img_label].append((img_path, pred, img_label))
        
    acc = correct * 100.0 / total
    print(f'TTA Test Accuracy: {acc:.2f}%')

    # 转换为numpy数组
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_predictions)

    # 生成并显示详细报告
    print_detailed_report(y_true, y_pred, label_names)

    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred, label_names, save_pth=None)
    
    # 绘制归一化混淆矩阵
    plot_normalized_confusion_matrix(y_true, y_pred, label_names, save_pth=None)

    # 获取随机错误样本（每个类别5张）
    print("\nSelecting random error samples...")
    selected_error_samples = get_random_error_samples(error_samples_by_class, label_names, samples_per_class=5)
    
    # 打印每个类别的错误统计
    print("\nError Statistics:")
    print("=" * 50)
    for class_idx, class_name in enumerate(label_names):
        error_count = len(error_samples_by_class[class_idx])
        selected_count = len(selected_error_samples[class_name])
        print(f"{class_name}: {error_count} total errors, {selected_count} selected for display")
    
    # 显示错误样本
    display_error_samples(selected_error_samples, label_names)
    
    return selected_error_samples, y_true, y_pred

if __name__ == '__main__':
    selected_errors, true_labels, predictions = main()
    print("\nTesting completed!")