#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ファインチューニング済みGLIPモデルを使用した45カテゴリーの階層的クラスタリング

使用方法:
python finetuned_glip_hierarchical_clustering_45cat.py --model_path ./1st_OUTPUT/20250411共有_all_item_valid/full/all_shot/ft_task_1/model_best.pth
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

# GLIPディレクトリに移動してからパスを追加
original_dir = os.getcwd()
glip_dir = os.path.join(original_dir, 'GLIP')
if os.path.exists(glip_dir):
    os.chdir(glip_dir)
    sys.path.append('.')
    print(f"Changed directory to: {os.getcwd()}")
else:
    print(f"Warning: GLIP directory not found at {glip_dir}")
    sys.path.append('./GLIP')

# クラスタリング関連のインポート
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# GLIPの関連モジュールをインポート
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.data import make_data_loader

class FineTunedGLIPFeatureExtractor:
    """ファインチューニング済みGLIPモデルから特徴量を抽出するクラス"""
    
    def __init__(self, config_file, model_path, device='cuda'):
        self.device = torch.device(device)
        self.config_file = config_file
        self.model_path = model_path
        
        # モデルの初期化
        self._load_model()
        
    def _load_model(self):
        """ファインチューニング済みモデルをロード"""
        print(f"Loading fine-tuned model from: {self.model_path}")
        
        # 設定の読み込み
        cfg.merge_from_file(self.config_file)
        cfg.MODEL.DEVICE = self.device.type
        
        # モデルの構築
        self.model = build_detection_model(cfg)
        self.model.to(self.device)
        
        # ファインチューニング済みウェイトの読み込み
        checkpointer = DetectronCheckpointer(cfg, self.model)
        checkpointer.load(self.model_path, force = True)
        
        self.model.eval()
        print("Fine-tuned model loaded successfully!")
        
    def extract_features(self, images, categories=None):
        """画像から特徴量を抽出"""
        if not isinstance(images, list):
            images = [images]
            
        features = {}
        
        with torch.no_grad():
            # 画像をテンソルに変換
            image_tensors = []
            for img in images:
                if isinstance(img, Image.Image):
                    # PIL ImageをTensorに変換
                    transform = transforms.Compose([
                        transforms.Resize((800, 1333)),  # GLIPの標準入力サイズ
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    img_tensor = transform(img)
                    image_tensors.append(img_tensor)
                else:
                    image_tensors.append(img)
            
            # バッチ作成
            image_list = to_image_list(image_tensors).to(self.device)
            
            # 特徴量抽出
            # Backboneからの特徴量
            backbone_features = self.model.backbone(image_list.tensors)
            
            # backbone_featuresの型を確認して適切に処理
            if isinstance(backbone_features, dict):
                # 辞書形式の場合（通常のケース）
                for level, feature in backbone_features.items():
                    if feature is not None:
                        # Global Average Pooling
                        pooled = torch.nn.functional.adaptive_avg_pool2d(feature, (1, 1))
                        pooled = pooled.view(pooled.size(0), -1)
                        features[f'backbone_{level}'] = pooled.cpu().numpy()
            elif isinstance(backbone_features, (list, tuple)):
                # リストまたはタプル形式の場合
                for i, feature in enumerate(backbone_features):
                    if feature is not None:
                        # Global Average Pooling
                        pooled = torch.nn.functional.adaptive_avg_pool2d(feature, (1, 1))
                        pooled = pooled.view(pooled.size(0), -1)
                        features[f'backbone_level_{i}'] = pooled.cpu().numpy()
            else:
                # 単一のテンソルの場合
                if backbone_features is not None:
                    pooled = torch.nn.functional.adaptive_avg_pool2d(backbone_features, (1, 1))
                    pooled = pooled.view(pooled.size(0), -1)
                    features['backbone_single'] = pooled.cpu().numpy()
            
            # Language backboneからの特徴量（カテゴリ情報がある場合）
            if categories is not None and hasattr(self.model, 'language_backbone'):
                try:
                    lang_features = self.model.language_backbone(categories)
                    if lang_features is not None:
                        features['language'] = lang_features.cpu().numpy()
                except:
                    pass
                    
        return features
    
    def get_category_features(self, images, categories):
        """カテゴリごとの特徴量を取得"""
        category_features = {}
        
        for i, (img, cat) in enumerate(zip(images, categories)):
            features = self.extract_features([img], [cat])
            
            if cat not in category_features:
                category_features[cat] = []
                
            # 各レベルの特徴量を結合
            combined_feature = []
            for level_name, level_features in features.items():
                if level_features is not None:
                    combined_feature.append(level_features[0])  # バッチサイズ1なので[0]
            
            if combined_feature:
                category_features[cat].append(np.concatenate(combined_feature))
                
        return category_features

class CategoryHierarchicalClusterer:
    """45カテゴリーの階層的クラスタリングを実行するクラス"""
    
    def __init__(self):
        self.clustering_results = {}
        self.category_mappings = {}
        
    def load_category_data(self, json_file_path=None):
        """JSONアノテーションファイルから直接カテゴリ情報を読み込み"""
        if json_file_path is None:
            # デフォルトのパスを試行（GLIPディレクトリから相対パス）
            possible_paths = [
                '../../../bridge_data/json_data/20250411共有_all_item_test.json',
                '../../../bridge_data/json_data/20250411共有_all_item_valid.json',
                '../../bridge_data/json_data/20250411共有_all_item_test.json',
                '../../bridge_data/json_data/20250411共有_all_item_valid.json',
                '../bridge_data/json_data/20250411共有_all_item_test.json',
                '../bridge_data/json_data/20250411共有_all_item_valid.json'
            ]
            
            json_file_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    json_file_path = path
                    print(f"Found potential file: {path}")
                    # ファイルサイズをチェック
                    file_size = os.path.getsize(path)
                    print(f"File size: {file_size} bytes")
                    if file_size > 0:
                        json_file_path = path
                        break
                    else:
                        print(f"File {path} is empty, skipping...")
                        
            if json_file_path is None:
                # 直接1st_OUTPUTディレクトリからconfig.ymlを探してカテゴリ情報を取得
                print("JSONファイルが見つかりません。config.ymlからカテゴリ情報を取得します...")
                return self._load_categories_from_config()
        
        print(f"Loading categories from: {json_file_path}")
        
        try:
            # ファイルの最初の部分を確認
            with open(json_file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                print(f"First line of file: {first_line[:100]}...")
                
            # ファイル全体を読み込み
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            print("Trying to load categories from config files...")
            return self._load_categories_from_config()
        except Exception as e:
            print(f"File reading error: {e}")
            return self._load_categories_from_config()
        
        if 'categories' not in data:
            print("No 'categories' key found in JSON. Trying config files...")
            return self._load_categories_from_config()
            
        categories = [(cat['id'], cat['name']) for cat in data['categories']]
        self.category_mappings = {cat_id: cat_name for cat_id, cat_name in categories}
        
        print(f"Loaded {len(categories)} categories:")
        for cat_id, cat_name in categories[:10]:  # 最初の10個を表示
            print(f"  {cat_id}: {cat_name}")
        if len(categories) > 10:
            print(f"  ... and {len(categories) - 10} more categories")
            
        return categories
    
    def _load_categories_from_config(self):
        """1st_OUTPUTディレクトリのconfig.ymlからカテゴリ情報を読み込み"""
        config_paths = [
            '../1st_OUTPUT/20250411共有_all_item_valid/full/all_shot/ft_task_1/config.yml',
            '../1st_OUTPUT/20250411共有_all_item_without_cable_components_valid/full/all_shot/ft_task_1/config.yml',
            '../1st_OUTPUT/*/full/all_shot/ft_task_1/config.yml'
        ]
        
        import glob
        
        for pattern in config_paths:
            paths = glob.glob(pattern)
            for config_path in paths:
                if os.path.exists(config_path):
                    print(f"Trying to load categories from config: {config_path}")
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = yaml.safe_load(f)
                        
                        # DATASETSセクションからカテゴリ情報を取得
                        if 'DATASETS' in config_data and 'OVERRIDE_CATEGORY' in config_data['DATASETS']:
                            categories_list = config_data['DATASETS']['OVERRIDE_CATEGORY']
                            if isinstance(categories_list, list) and len(categories_list) > 0:
                                categories = []
                                for i, cat in enumerate(categories_list):
                                    if isinstance(cat, dict) and 'name' in cat:
                                        cat_id = cat.get('id', i + 1)
                                        cat_name = cat['name']
                                        categories.append((cat_id, cat_name))
                                
                                if len(categories) > 0:
                                    self.category_mappings = {cat_id: cat_name for cat_id, cat_name in categories}
                                    print(f"Loaded {len(categories)} categories from config:")
                                    for cat_id, cat_name in categories[:10]:
                                        print(f"  {cat_id}: {cat_name}")
                                    if len(categories) > 10:
                                        print(f"  ... and {len(categories) - 10} more categories")
                                    return categories
                    except Exception as e:
                        print(f"Error reading config {config_path}: {e}")
                        continue
        
        # フォールバック: 45カテゴリのダミーデータを作成
        print("No valid category data found. Creating dummy 45 categories...")
        categories = [(i+1, f"Category_{i+1}") for i in range(45)]
        self.category_mappings = {cat_id: cat_name for cat_id, cat_name in categories}
        print(f"Created {len(categories)} dummy categories")
        return categories
    
    def extract_category_representations(self, feature_extractor, image_dir, categories, max_images_per_category=50):
        """各カテゴリの代表特徴量を抽出"""
        category_representations = {}
        
        print("Extracting category representations...")
        
        for cat_id, cat_name in tqdm(categories, desc="Processing categories"):
            # カテゴリ名に対応する画像を検索（簡易実装）
            category_images = []
            
            # 画像ディレクトリからサンプル画像を取得
            image_paths = list(Path(image_dir).glob('*.jpg')) + \
                         list(Path(image_dir).glob('*.jpeg')) + \
                         list(Path(image_dir).glob('*.png'))
            
            # カテゴリ名がファイル名に含まれる画像を検索
            for img_path in image_paths[:max_images_per_category]:
                try:
                    img = Image.open(img_path).convert('RGB')
                    category_images.append(img)
                except:
                    continue
                    
                if len(category_images) >= max_images_per_category:
                    break
            
            if len(category_images) > 0:
                # カテゴリの特徴量を抽出
                features = feature_extractor.extract_features(category_images, [cat_name] * len(category_images))
                
                # 各レベルの特徴量を平均化
                avg_features = []
                for level_name, level_features in features.items():
                    if level_features is not None:
                        avg_feature = np.mean(level_features, axis=0)
                        avg_features.append(avg_feature)
                
                if avg_features:
                    category_representations[cat_name] = np.concatenate(avg_features)
                    
        print(f"Extracted representations for {len(category_representations)} categories")
        return category_representations
    
    def perform_hierarchical_clustering(self, category_representations, method='ward', 
                                      n_clusters_range=(2, 15)):
        """階層的クラスタリングを実行"""
        print("Performing hierarchical clustering...")
        
        # カテゴリ名と特徴量を分離
        category_names = list(category_representations.keys())
        features = np.array(list(category_representations.values()))
        
        print(f"Number of categories: {len(category_names)}")
        print(f"Feature shape: {features.shape}")
        
        if len(features) < 2:
            raise ValueError("Need at least 2 categories for clustering")
        
        # 特徴量の正規化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 特徴量の分散をチェック
        if np.var(features_scaled) < 1e-10:
            print("Warning: Features have very low variance. Adding small noise...")
            features_scaled += np.random.normal(0, 1e-6, features_scaled.shape)
        
        # 階層的クラスタリング
        linkage_matrix = linkage(features_scaled, method=method)
        
        # 最適なクラスタ数を決定
        best_score = -1
        best_n_clusters = 2
        best_labels = None
        
        silhouette_scores = []
        cluster_numbers = []
        
        max_clusters = min(n_clusters_range[1], len(features) - 1)
        
        for n_clusters in range(n_clusters_range[0], max_clusters + 1):
            try:
                labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                # ラベルの有効性をチェック
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1 and len(unique_labels) <= len(features):
                    score = silhouette_score(features_scaled, labels)
                    silhouette_scores.append(score)
                    cluster_numbers.append(n_clusters)
                    
                    print(f"  n_clusters={n_clusters}, silhouette_score={score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        best_labels = labels
                else:
                    print(f"  n_clusters={n_clusters}: Invalid clustering result")
                    
            except Exception as e:
                print(f"  n_clusters={n_clusters}: Error - {e}")
                continue
        
        # best_labelsがNoneの場合の処理
        if best_labels is None:
            print("Warning: Could not find valid clustering. Using 2 clusters as fallback.")
            best_n_clusters = min(2, len(features))
            best_labels = fcluster(linkage_matrix, best_n_clusters, criterion='maxclust')
            best_score = 0.0
            
            # それでもNoneの場合は、すべて同じクラスタに割り当て
            if best_labels is None:
                best_labels = np.ones(len(features), dtype=int)
                best_n_clusters = 1
        
        print(f"Final clustering: {best_n_clusters} clusters, silhouette score: {best_score:.4f}")
        
        # クラスタリング結果
        cluster_assignment = {}
        for i, cat_name in enumerate(category_names):
            cluster_assignment[cat_name] = {
                'cluster_id': int(best_labels[i]),
                'category_id': i,
                'features': features[i]
            }
        
        results = {
            'linkage_matrix': linkage_matrix,
            'best_n_clusters': best_n_clusters,
            'best_score': best_score,
            'best_labels': best_labels,
            'cluster_assignment': cluster_assignment,
            'category_names': category_names,
            'features': features,
            'features_scaled': features_scaled,
            'scaler': scaler,
            'silhouette_scores': silhouette_scores,
            'cluster_numbers': cluster_numbers
        }
        
        self.clustering_results = results
        return results
    
    def analyze_clusters(self, results):
        """クラスタ分析結果を表示"""
        print("\n" + "="*80)
        print("CLUSTER ANALYSIS RESULTS")
        print("="*80)
        
        print(f"Optimal number of clusters: {results['best_n_clusters']}")
        print(f"Best silhouette score: {results['best_score']:.4f}")
        
        # クラスタごとのカテゴリを表示
        clusters = {}
        for cat_name, info in results['cluster_assignment'].items():
            cluster_id = info['cluster_id']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(cat_name)
        
        print(f"\nCluster composition:")
        print("-" * 40)
        for cluster_id in sorted(clusters.keys()):
            categories = clusters[cluster_id]
            print(f"Cluster {cluster_id} ({len(categories)} categories):")
            for cat in sorted(categories):
                print(f"  - {cat}")
            print()
    
    def visualize_results(self, results, output_dir):
        """結果を可視化"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # デンドログラムの描画
        plt.figure(figsize=(20, 10))
        dendrogram(results['linkage_matrix'], 
                  labels=results['category_names'],
                  leaf_rotation=90,
                  leaf_font_size=8)
        plt.title('Hierarchical Clustering Dendrogram (45 Categories)', fontsize=16)
        plt.xlabel('Categories', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / 'dendrogram_45categories.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # シルエット係数の推移
        if results['silhouette_scores']:
            plt.figure(figsize=(10, 6))
            plt.plot(results['cluster_numbers'], results['silhouette_scores'], 'bo-')
            plt.axvline(x=results['best_n_clusters'], color='red', linestyle='--', 
                       label=f'Optimal clusters: {results["best_n_clusters"]}')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score vs Number of Clusters')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'silhouette_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # PCAで2次元可視化
        if results['features'].shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(results['features_scaled'])
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=results['best_labels'], cmap='tab20', s=100, alpha=0.7)
            
            # カテゴリ名をラベルとして追加
            for i, cat_name in enumerate(results['category_names']):
                plt.annotate(cat_name, (features_2d[i, 0], features_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
            plt.title('Category Clustering Results (PCA Visualization)')
            plt.colorbar(scatter, label='Cluster ID')
            plt.tight_layout()
            plt.savefig(output_path / 'clusters_pca_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # t-SNEで可視化
        if len(results['features']) > 30:  # t-SNEは多くのサンプルがある場合に有効
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(results['features'])-1))
            features_tsne = tsne.fit_transform(results['features_scaled'])
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                                c=results['best_labels'], cmap='tab20', s=100, alpha=0.7)
            
            for i, cat_name in enumerate(results['category_names']):
                plt.annotate(cat_name, (features_tsne[i, 0], features_tsne[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.title('Category Clustering Results (t-SNE Visualization)')
            plt.colorbar(scatter, label='Cluster ID')
            plt.tight_layout()
            plt.savefig(output_path / 'clusters_tsne_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {output_path}")
    
    def save_results(self, results, output_dir):
        """結果をファイルに保存"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Pickleで保存
        with open(output_path / 'clustering_results_45categories.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # CSVで保存
        cluster_data = []
        for cat_name, info in results['cluster_assignment'].items():
            cluster_data.append({
                'category_name': cat_name,
                'cluster_id': info['cluster_id'],
                'category_index': info['category_id']
            })
        
        df = pd.DataFrame(cluster_data)
        df.to_csv(output_path / 'cluster_assignments_45categories.csv', index=False, encoding='utf-8')
        
        # クラスタ統計の保存
        cluster_stats = {}
        for cluster_id in range(1, results['best_n_clusters'] + 1):
            cluster_categories = [cat for cat, info in results['cluster_assignment'].items() 
                                if info['cluster_id'] == cluster_id]
            cluster_stats[f'cluster_{cluster_id}'] = {
                'size': len(cluster_categories),
                'categories': cluster_categories
            }
        
        with open(output_path / 'cluster_statistics_45categories.json', 'w', encoding='utf-8') as f:
            json.dump(cluster_stats, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tuned GLIP Hierarchical Clustering for 45 Categories')
    parser.add_argument('--config', type=str, 
                       default='./configs/pretrain/glip_Swin_L.yaml',
                       help='GLIP config file path')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned model (e.g., model_best.pth)')
    parser.add_argument('--task_config', type=str,
                       default='../../bridge_data/configs/20250411共有_all_item_test.yaml',
                       help='Task configuration file with category definitions')
    parser.add_argument('--image_dir', type=str,
                       default='../../bridge_data/20250411共有',
                       help='Directory containing sample images')
    parser.add_argument('--output_dir', type=str, 
                       default='./../clustering_results_45categories',
                       help='Output directory for results')
    parser.add_argument('--max_images_per_category', type=int, default=20,
                       help='Maximum number of images to sample per category')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("45-Category Hierarchical Clustering with Fine-tuned GLIP")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Model: {args.model_path}")
    print(f"Task config: {args.task_config}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print()
    
    try:
        # 1. ファインチューニング済みモデルをロード
        print("Loading fine-tuned GLIP model...")
        feature_extractor = FineTunedGLIPFeatureExtractor(args.config, args.model_path, args.device)
        
        # 2. クラスタリング器を初期化
        clusterer = CategoryHierarchicalClusterer()
        
        # 3. カテゴリ情報を読み込み
        print("Loading category information...")
        categories = clusterer.load_category_data(args.task_config)
        
        # 4. カテゴリ表現を抽出
        print("Extracting category representations...")
        category_representations = clusterer.extract_category_representations(
            feature_extractor, args.image_dir, categories, args.max_images_per_category
        )
        
        if len(category_representations) < 2:
            print("Error: Need at least 2 categories for clustering")
            return
        
        # 5. 階層的クラスタリングを実行
        print("Performing hierarchical clustering...")
        results = clusterer.perform_hierarchical_clustering(category_representations)
        
        # 6. 結果を分析
        clusterer.analyze_clusters(results)
        
        # 7. 結果を可視化
        print("Creating visualizations...")
        clusterer.visualize_results(results, args.output_dir)
        
        # 8. 結果を保存
        print("Saving results...")
        clusterer.save_results(results, args.output_dir)
        
        print("\n" + "="*70)
        print("CLUSTERING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Results saved to: {args.output_dir}")
        print(f"Total categories processed: {len(category_representations)}")
        print(f"Optimal number of clusters: {results['best_n_clusters']}")
        print(f"Best silhouette score: {results['best_score']:.4f}")
        
    except Exception as e:
        print(f"Error during clustering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()