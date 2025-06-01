import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import glob
from PIL import Image
from torchvision import transforms
from EfficientNet import create_efficientnet_dr
from DataPreparation_1 import train_dataset, test_dataset, label_names, ClaheMedian

class OriginalDataset:
    """创建基于原始数据的数据集，只包含基本transform"""
    def __init__(self, data_dir, use_augmented=False):
        self.use_augmented = use_augmented
        self.image_paths = []
        self.labels = []
        files = ['Train', 'Test']
        # 基本transform（保持与训练时一致）
        self.basic_transform = transforms.Compose([
            ClaheMedian(clip_limit=2.0, tile_grid_size=(8,8), median_kernel_size=5),
            transforms.ToTensor(),
            transforms.Normalize([0.46238312125205994, 0.26547566056251526, 0.11903924494981766],[0.2862863838672638, 0.16006295382976532, 0.09059497714042664])
        ])
        
        # 收集图片路径和标签
        # for dir in data_dir:
        for class_idx, class_name in enumerate(label_names):
            for file in files:
                class_dir = os.path.join(data_dir, file, class_name)
                if os.path.exists(class_dir):
                    image_files = glob.glob(os.path.join(class_dir, '*.png'))
                    
                    if not use_augmented:
                        # 过滤掉augmentation生成的图片
                        original_files = []
                        for img_file in image_files:
                            filename = os.path.basename(img_file)
                            # 如果文件名包含augmentation标识，则跳过
                            if not any(aug_tag in filename for aug_tag in ['_flip', '_rot', '_bc', '_trans', '_zoom', '_persp']):
                                original_files.append(img_file)
                        image_files = original_files
                    
                    self.image_paths.extend(image_files)
                    self.labels.extend([class_idx] * len(image_files))
        
        print(f"Dataset created with {len(self.image_paths)} images")
        print("Class distribution:")
        for class_idx, class_name in enumerate(label_names):
            count = self.labels.count(class_idx)
            print(f"  {class_name}: {count}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.basic_transform:
            image = self.basic_transform(image)
        
        return image, label

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = []
        self.hook_registered = False
        
    def hook_fn(self, module, input, output):
        """Hook函数，用于捕获中间层的输出"""
        self.features.append(output.detach().cpu().numpy())
    
    def register_hook(self, target_layer_name="model.classifier.10"):
        """注册hook到指定层"""
        print("Model structure:")
        for name, module in self.model.named_modules():
            if 'classifier' in name:
                print(f"  {name}: {module}")
        
        found = False
        for name, module in self.model.named_modules():
            if target_layer_name in name:
                print(f"Registering hook to layer: {name}")
                module.register_forward_hook(self.hook_fn)
                found = True
                self.hook_registered = True
                break
        
        if not found:
            print(f"Layer {target_layer_name} not found!")
            return False
        return True
    
    def extract_features_for_class(self, dataset, target_class_idx, device, batch_size=64):
        """为特定类别提取特征"""
        self.model.eval()
        self.features = []  # 清空特征
        
        # 找到属于目标类别的所有样本索引
        class_indices = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label == target_class_idx:
                class_indices.append(i)
        
        print(f"Found {len(class_indices)} samples for class '{label_names[target_class_idx]}'")
        
        if len(class_indices) == 0:
            return None, None, None, None, None
        
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            # 按批次处理该类别的样本
            for start_idx in tqdm(range(0, len(class_indices), batch_size), 
                                desc=f"Processing {label_names[target_class_idx]}"):
                end_idx = min(start_idx + batch_size, len(class_indices))
                
                batch_inputs = []
                batch_labels = []
                
                for i in range(start_idx, end_idx):
                    sample_idx = class_indices[i]
                    img, label = dataset[sample_idx]
                    batch_inputs.append(img.unsqueeze(0))
                    batch_labels.append(label)
                
                # 转换为tensor并forward
                batch_inputs = torch.cat(batch_inputs, dim=0).to(device)
                batch_labels = torch.tensor(batch_labels).to(device)
                
                outputs = self.model(batch_inputs)
                
                all_labels.append(batch_labels.cpu().numpy())
                all_predictions.append(outputs.cpu().numpy())
        
        # 合并结果
        if self.features:
            extracted_features = np.vstack(self.features)
            all_labels = np.hstack(all_labels)
            all_predictions = np.vstack(all_predictions)
            
            # 获取对应的图片路径
            image_paths = [dataset.image_paths[idx] for idx in class_indices]
            
            print(f"Extracted {extracted_features.shape[0]} features of dimension {extracted_features.shape[1]}")
            return extracted_features, all_labels, all_predictions, class_indices, image_paths
        else:
            return None, None, None, None, None

def find_optimal_clusters_for_class(features, class_name, method='both', max_clusters=8, min_silhouette=0.3):
    """为单个类别找最优聚类数和方法，允许不分组"""
    print(f"\nFinding optimal clustering for {class_name}...")
    print(f"Total samples: {len(features)}")
    
    # 检查样本数是否足够聚类
    if len(features) < 10:  # 样本太少，不分组
        print(f"Too few samples ({len(features)}) for clustering. Keeping as single subclass.")
        return 'single', {'n_clusters': 1}, None
    
    max_clusters = min(max_clusters, len(features) // 5)  # 每组至少5个样本
    max_clusters = max(2, max_clusters)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 检查特征的方差 - 如果特征太相似，不需要分组
    feature_std = np.std(features_scaled, axis=0)
    mean_std = np.mean(feature_std)
    print(f"Feature variance (mean std): {mean_std:.4f}")
    
    if mean_std < 0.1:  # 特征太相似
        print(f"Features too similar (std={mean_std:.4f}). Keeping as single subclass.")
        return 'single', {'n_clusters': 1}, scaler
    
    results = {}
    
    if method in ['kmeans', 'both']:
        print("Testing K-Means...")
        kmeans_results = test_kmeans_with_threshold(features_scaled, max_clusters, class_name, min_silhouette)
        results['kmeans'] = kmeans_results
    
    if method in ['dbscan', 'both']:
        print("Testing DBSCAN...")
        dbscan_results = test_dbscan_with_threshold(features_scaled, class_name, min_silhouette)
        results['dbscan'] = dbscan_results
    
    # 选择最佳方法，或者决定不分组
    best_method, best_params = choose_best_clustering_or_single(results, class_name, min_silhouette)
    
    return best_method, best_params, scaler

def test_kmeans_with_threshold(features_scaled, max_clusters, class_name, min_silhouette=0.3):
    """测试K-Means，如果效果不好就建议不分组"""
    silhouette_scores = []
    inertias = []
    valid_k_range = []
    
    K_range = range(2, max_clusters + 1)  # 从2开始，因为1个cluster没意义
    
    for k in K_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # 检查是否所有点都分到了同一个cluster
            unique_labels = np.unique(cluster_labels)
            if len(unique_labels) < 2:
                print(f"  K-Means k={k}: All points in same cluster, skipping")
                continue
            
            sil_score = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append(sil_score)
            inertias.append(kmeans.inertia_)
            valid_k_range.append(k)
            
            print(f"  K-Means k={k}, silhouette={sil_score:.4f}")
            
        except Exception as e:
            print(f"  K-Means k={k}: Error - {e}")
            continue
    
    if not silhouette_scores:
        return {'best_silhouette': -1, 'optimal_k': 1, 'should_cluster': False}
    
    best_idx = np.argmax(silhouette_scores)
    optimal_k = valid_k_range[best_idx]
    best_score = silhouette_scores[best_idx]
    
    # 如果最好的silhouette score还是很低，建议不分组
    if best_score < min_silhouette:
        print(f"  K-Means best silhouette ({best_score:.4f}) below threshold ({min_silhouette})")
        return {'best_silhouette': best_score, 'optimal_k': 1, 'should_cluster': False}
    
    return {
        'optimal_k': optimal_k,
        'best_silhouette': best_score,
        'all_scores': silhouette_scores,
        'inertias': inertias,
        'k_range': valid_k_range,
        'should_cluster': True
    }

def test_dbscan_with_threshold(features_scaled, class_name, min_silhouette=0.3):
    """测试DBSCAN，如果效果不好就建议不分组"""
    eps_values = np.arange(0.3, 2.1, 0.3)
    min_samples_values = [3, 5, 8]
    
    best_score = -1
    best_params = None
    best_n_clusters = 0
    valid_results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(features_scaled)
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                
                # 检查聚类质量
                if n_clusters < 2:
                    continue
                if n_noise > len(features_scaled) * 0.6:  # 噪声点太多
                    continue
                
                sil_score = silhouette_score(features_scaled, cluster_labels)
                
                valid_results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette': sil_score
                })
                
                print(f"  DBSCAN eps={eps:.1f}, min_samples={min_samples}, "
                      f"clusters={n_clusters}, noise={n_noise}, silhouette={sil_score:.4f}")
                
                if sil_score > best_score:
                    best_score = sil_score
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    best_n_clusters = n_clusters
                    
            except Exception as e:
                continue
    
    if best_score < min_silhouette:
        print(f"  DBSCAN best silhouette ({best_score:.4f}) below threshold ({min_silhouette})")
        return {'best_silhouette': best_score, 'should_cluster': False}
    
    return {
        'best_params': best_params,
        'best_silhouette': best_score,
        'best_n_clusters': best_n_clusters,
        'all_results': valid_results,
        'should_cluster': True
    }

def choose_best_clustering_or_single(results, class_name, min_silhouette=0.3):
    """选择最佳聚类方法，或者决定保持单一类别"""
    best_method = 'single'
    best_params = {'n_clusters': 1}
    best_score = -1
    
    print(f"\nEvaluating clustering options for {class_name}:")
    
    # 检查K-Means结果
    if 'kmeans' in results:
        kmeans_result = results['kmeans']
        if kmeans_result.get('should_cluster', False) and kmeans_result['best_silhouette'] > best_score:
            best_score = kmeans_result['best_silhouette']
            best_method = 'kmeans'
            best_params = {'n_clusters': kmeans_result['optimal_k']}
            print(f"  K-Means: k={kmeans_result['optimal_k']}, silhouette={best_score:.4f} ✓")
        else:
            print(f"  K-Means: silhouette={kmeans_result['best_silhouette']:.4f} (below threshold)")
    
    # 检查DBSCAN结果
    if 'dbscan' in results:
        dbscan_result = results['dbscan']
        if dbscan_result.get('should_cluster', False) and dbscan_result['best_silhouette'] > best_score:
            best_score = dbscan_result['best_silhouette']
            best_method = 'dbscan'
            best_params = dbscan_result['best_params']
            print(f"  DBSCAN: {best_params}, silhouette={best_score:.4f} ✓")
        else:
            print(f"  DBSCAN: silhouette={dbscan_result['best_silhouette']:.4f} (below threshold)")
    
    # 最终决策
    if best_method == 'single':
        print(f"  → Decision: Keep as SINGLE subclass (clustering not beneficial)")
    else:
        print(f"  → Decision: Use {best_method.upper()} with {best_params} (silhouette={best_score:.4f})")
    
    return best_method, best_params

def perform_clustering_for_class(features, class_name, method, params, scaler):
    """为单个类别执行聚类，包括单一类别的情况"""
    if method == 'single':
        # 不分组，所有样本都是同一个子类
        cluster_labels = np.zeros(len(features), dtype=int)
        clusterer = None
        print(f"Keeping {class_name} as single subclass:")
        print(f"  Subclass 0: {len(features)} samples")
        
    else:
        features_scaled = scaler.transform(features)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=params['n_clusters'], random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(features_scaled)
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
            cluster_labels = clusterer.fit_predict(features_scaled)
        
        print(f"Clustering results for {class_name} using {method.upper()}:")
        unique, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            if cluster_id == -1:
                print(f"  Noise points: {count} samples")
            else:
                print(f"  Subclass {cluster_id}: {count} samples")
    
    return cluster_labels, clusterer

def visualize_class_clusters(features, cluster_labels, class_name, save_dir=None):
    """可视化单个类别的聚类结果"""
    print(f"Creating visualization for {class_name}...")
    
    # PCA降维到2D
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Subclass Label')
    plt.title(f'Subclass Clustering for {class_name}\n'
              f'PCA Explained Variance: {pca.explained_variance_ratio_.sum():.3f}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{class_name}_clustering.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()

def save_class_results(features, cluster_labels, class_name, class_idx, 
                      sample_indices, image_paths, method, params, save_dir='DR/class_wise_subclasses'):
    """保存单个类别的聚类结果，包括单一类别的情况"""
    class_dir = os.path.join(save_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # 1. 保存特征 + 子类标签
    feature_columns = [f'feature_{i}' for i in range(features.shape[1])]
    df_features = pd.DataFrame(features, columns=feature_columns)
    df_features['original_class'] = class_name
    df_features['original_label'] = class_idx
    df_features['subclass_label'] = cluster_labels
    df_features['sample_index'] = sample_indices
    df_features['clustering_method'] = method
    
    features_file = os.path.join(class_dir, f'{class_name}_features_128d.csv')
    df_features.to_csv(features_file, index=False)
    print(f"Features saved: {features_file}")
    
    # 2. 保存子类统计
    unique_labels = np.unique(cluster_labels)
    subclass_stats = pd.DataFrame({
        'subclass_id': unique_labels,
        'count': [np.sum(cluster_labels == i) for i in unique_labels],
        'percentage': [np.sum(cluster_labels == i) / len(cluster_labels) * 100 
                      for i in unique_labels]
    })
    
    # 添加方法信息
    subclass_stats['clustering_method'] = method
    if method == 'kmeans':
        subclass_stats['n_clusters'] = params['n_clusters']
    elif method == 'dbscan':
        subclass_stats['eps'] = params.get('eps', 'N/A')
        subclass_stats['min_samples'] = params.get('min_samples', 'N/A')
    elif method == 'single':
        subclass_stats['n_clusters'] = 1
        subclass_stats['reason'] = 'Single subclass (no clustering needed)'
    
    stats_file = os.path.join(class_dir, f'{class_name}_subclass_stats.csv')
    subclass_stats.to_csv(stats_file, index=False)
    print(f"Statistics saved: {stats_file}")
    
    # 3. 新增：保存简单的文件名到cluster映射 
    cluster_mapping = []
    
    for i, img_path in enumerate(image_paths):
        # 提取文件名（不包含路径和扩展名）
        filename = os.path.splitext(os.path.basename(img_path))[0]
        cluster_id = cluster_labels[i]
        
        cluster_mapping.append({
            'filename': filename,
            'cluster': cluster_id
        })
    
    # 保存简单映射文件
    cluster_df = pd.DataFrame(cluster_mapping)
    cluster_file = os.path.join(class_dir, f'{class_name}_cluster_mapping.csv')
    cluster_df.to_csv(cluster_file, index=False)
    print(f"Cluster mapping saved: {cluster_file}")
    
    return df_features, subclass_stats

def process_all_classes(extractor, dataset, device, clustering_method='both', 
                       save_dir='DR/class_wise_subclasses', min_silhouette=0.3):
    """处理所有DR类别，允许某些类别保持单一子类"""
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}
    summary_stats = []
    
    # 对每个类别分别处理
    for class_idx, class_name in enumerate(label_names):
        print(f"\n{'='*60}")
        print(f"PROCESSING CLASS: {class_name} (index: {class_idx})")
        print(f"{'='*60}")
        
        try:
            # 1. 提取该类别的特征
            features, labels, predictions, sample_indices, image_paths = extractor.extract_features_for_class(
                dataset, class_idx, device
            )
            
            if features is None:
                print(f"No samples found for class {class_name}, skipping...")
                continue
            
            # 2. 寻找最优聚类方法和参数
            best_method, best_params, scaler = find_optimal_clusters_for_class(
                features, class_name, method=clustering_method, min_silhouette=min_silhouette
            )
            
            # 3. 执行聚类（或保持单一）
            cluster_labels, clusterer = perform_clustering_for_class(
                features, class_name, best_method, best_params, scaler
            )
            
            # 4. 可视化（即使是单一类别也可以看分布）
            visualize_class_clusters(features, cluster_labels, class_name, save_dir)
            
            # 5. 保存结果
            df_features, subclass_stats = save_class_results(
                features, cluster_labels, class_name, class_idx, sample_indices, 
                image_paths, best_method, best_params, save_dir
            )
            
            # 6. 收集统计信息
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1) if -1 in cluster_labels else 0
            
            all_results[class_name] = {
                'features': features,
                'cluster_labels': cluster_labels,
                'method': best_method,
                'params': best_params,
                'sample_count': len(features),
                'subclass_stats': subclass_stats
            }
            
            summary_stats.append({
                'class_name': class_name,
                'class_index': class_idx,
                'total_samples': len(features),
                'clustering_method': best_method,
                'num_subclasses': max(1, n_clusters),  # 至少1个子类
                'noise_points': n_noise,
                'best_params': str(best_params),
                'clustered': 'Yes' if best_method != 'single' else 'No'
            })
            
        except Exception as e:
            print(f"Error processing {class_name}: {e}")
            # 即使出错也要记录
            summary_stats.append({
                'class_name': class_name,
                'class_index': class_idx,
                'total_samples': 0,
                'clustering_method': 'error',
                'num_subclasses': 1,
                'noise_points': 0,
                'best_params': 'N/A',
                'clustered': 'No'
            })
            continue
    
    # 保存总体统计
    summary_df = pd.DataFrame(summary_stats)
    summary_file = os.path.join(save_dir, 'overall_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\nOverall summary saved: {summary_file}")
    
    # 打印总结
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    for _, row in summary_df.iterrows():
        status = "CLUSTERED" if row['clustered'] == 'Yes' else "SINGLE"
        print(f"{row['class_name']:>15} | {row['total_samples']:>4} samples | "
              f"{row['clustering_method']:>8} | {row['num_subclasses']:>2} subclasses | {status}")
    
    return all_results, summary_df

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 选择数据集类型
    use_original_data = True  # 暂时设为False，使用train_dataset
    clustering_method = 'both'  # 'kmeans', 'dbscan', 或 'both'
    min_silhouette = 0.5  # 聚类质量阈值，可以调整
    
    if use_original_data:
        print("Creating dataset from original images (no augmentation)...")
        dataset = OriginalDataset('C:\\Users\\User\\Documents\\Pytorch\\DR', use_augmented=False)
        
        # 测试dataset
        print("Testing dataset access...")
        try:
            test_img, test_label = dataset[0]
            print(f"Dataset test successful: label={test_label}, shape={test_img.shape}")
        except Exception as e:
            print(f"Dataset test failed: {e}")
            return
    else:
        print("Using existing transformed dataset...")
        dataset = train_dataset
    
    # 加载模型
    print("Loading trained model...")
    model = create_efficientnet_dr()
    model.load_state_dict(torch.load("DR/models/best_acc.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # 创建特征提取器并注册hook
    extractor = FeatureExtractor(model)
    
    # 尝试注册hook
    potential_targets = ["model.classifier.10", "model.classifier.9", "model.classifier.8"]
    success = False
    for target in potential_targets:
        print(f"\nTrying to register hook to: {target}")
        if extractor.register_hook(target):
            success = True
            break
    
    if not success:
        print("Failed to register hook!")
        return
    
    # 处理所有类别
    print(f"\nStarting class-wise subclass generation...")
    print(f"Clustering method: {clustering_method}")
    print(f"Min silhouette threshold: {min_silhouette}")
    print(f"Using original data: {use_original_data}")
    
    all_results, summary_df = process_all_classes(
        extractor, dataset, device, 
        clustering_method=clustering_method,
        min_silhouette=min_silhouette
    )
    
    print("\nClass-wise unsupervised labeling completed!")
    print(f"Results saved in: DR/class_wise_subclasses/")
    
    return all_results, summary_df

if __name__ == '__main__':
    results, summary = main()