import torch
import torch.multiprocessing
import torch.nn as nn
import os
import torchvision
from torch.utils.tensorboard import SummaryWriter
from DataPreparation_1 import test_loader, train_loader, test_dataset, train_dataset
from EfficientNet import create_efficientnet_dr
from torch.amp import autocast, GradScaler
import multiprocessing
from tqdm import tqdm
from collections import Counter

def compute_class_weights(dataloader):
    """计算每个类别的权重"""
    class_counts = Counter()
    print("Calculating labels distribution...")
    
    # 遍历数据集统计每个类别的样本数
    for _, labels in dataloader:
        for label in labels.numpy():
            class_counts[label] += 1
    
    # 打印类别分布
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    print(f"Label distribution: {dict(class_counts)}")
    
    # 计算类别权重 (样本数的倒数，归一化使平均为1)
    weights = []
    for i in range(num_classes):
        count = max(1, class_counts.get(i, 0))  # 避免除以零
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    # 归一化权重
    weights = torch.FloatTensor(weights)
    weights = weights / weights.mean()
    
    print(f"Calculated label weights: {weights}")
    return weights

def main():

    print(f"Train: {len(train_dataset)}")
    print(f'Test: {len(test_dataset)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch = 100
    lr = 0.05
    net = create_efficientnet_dr().to(torch.device(device))
    global_step = 0

    class_weights = compute_class_weights(train_loader)
    class_weights = class_weights.to(device)

    loss_func = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.8
    )

    os.makedirs('DR/log_n', exist_ok=True)
    os.makedirs('DR/models', exist_ok=True)

    scaler = GradScaler()
    best_accuracy = 0.0
    best_loss = torch.inf
    writer = SummaryWriter('DR/log_n')

    for e in tqdm(range(epoch)):
        print(f'Epoch {e+1}/{epoch}')
        net.train()
        train_loss = 0.0
        train_correct = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(torch.device(device)), labels.to(torch.device(device))

            with autocast(device):
                outputs = net(inputs)
                loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)

            scaler.step(optimizer)
            scaler.update()

            _, pred = torch.max(outputs, dim=1)
            correct = pred.eq(labels).sum().item()
            train_correct += correct 
            train_loss += loss.item()

            if i%10 == 0:
                print(f'Epoch {e+1}, Batch {i}, Loss: {loss.item():.4f}, Accuracy: {100.0*correct/inputs.size(0):.2f}%')
                writer.add_scalar('train/batch_loss', loss.item(), global_step=e*len(train_loader)+i)
                writer.add_scalar('train/batch_acc', 100.0*correct/inputs.size(0), global_step=e*len(train_loader)+i)

                im_grid = torchvision.utils.make_grid(inputs[:100])
                writer.add_image('train/image', im_grid, e*len(train_loader)+i)

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct * 100.0 / len(train_loader.dataset)
        writer.add_scalar('train/epoch_loss', train_loss, global_step=e)
        writer.add_scalar('train/epoch_acc', train_acc, global_step=e)

        print(f'Training: Loss={train_loss:.4f}, Accuracy={train_acc:.2f}%')
        print(f'current learning rate: {optimizer.param_groups[0]['lr']:.6f}')

        scheduler.step()

        net.eval()
        sum_loss = 0.0
        sum_correct = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(torch.device(device)), labels.to(torch.device(device))

                with autocast(device):
                    outputs = net(inputs)
                    loss = loss_func(outputs, labels)

                _, pred = torch.max(outputs, dim=1)
                correct = pred.eq(labels).sum().item()

                sum_loss += loss.item()
                sum_correct += correct

                writer.add_scalar('test/batch_loss', loss.item(), global_step=e*len(test_loader)+i)
                writer.add_scalar('test/batch_acc', 100.0*correct/inputs.size(0), global_step=e*len(test_loader)+i)

                if i == 0:
                    im_grid = torchvision.utils.make_grid(inputs[:100])
                    writer.add_image('test/image', im_grid, global_step=e)

            test_loss = sum_loss / len(test_loader)
            test_acc = sum_correct * 100.0 / len(test_loader.dataset)

            writer.add_scalar('test/epoch_loss', test_loss, global_step=e)
            writer.add_scalar('test/epoch_acc', test_acc, global_step=e)
            print(f'Testing: Loss={test_loss:.4f}, Accuracy={test_acc:.2f}%')

            if test_acc > best_accuracy:
                best_acc_i = e+1
                best_accuracy = test_acc
                print(f'Save new acc model on epoch {e+1} ...')
                torch.save(net.state_dict(), 'DR/models/best_acc_n.pth')

            if test_loss < best_loss:
                best_loss_i = e+1
                best_loss = test_loss
                print(f'Save new loss model on epoch {e+1} ...')
                torch.save(net.state_dict(), 'DR/models/best_loss_n.pth')

    print(f'Best Acc Model: Epoch {best_acc_i}, Accuracy {best_accuracy}')
    print(f'Best Loss Model: Epoch {best_loss_i}, Accuracy {best_loss}')
    writer.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()