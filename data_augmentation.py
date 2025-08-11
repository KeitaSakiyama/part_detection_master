import os
import random
import torch
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from torchvision import transforms

class RandomApplyColorJitter:
    """ColorJitterを適用するクラス"""
    def __init__(self, brightness=0.2, contrast=0.2, p=0.1):
        self.color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast)
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.color_jitter(img)
        return img

class RandomGaussianBlur:
    """ランダムにガウシアンブラーを適用するクラス"""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        img = transforms.ToPILImage()(img)

        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
        return transforms.ToTensor()(img)

class AddGaussianNoise:
    """ランダムにガウシアンノイズを追加するクラス"""
    def __init__(self, mean=0.0, std=1.0, p=0.1):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor

# データ拡張のパイプラインを定義
trans_aug = transforms.Compose([
    transforms.ToTensor(),                                                               # PIL画像をTensorに変換
    RandomApplyColorJitter(brightness=0.2, contrast=0.2, p=0.1),                         # 10%の確率で輝度やコントラストをランダムに変化
    RandomGaussianBlur(p=0.1),                                                           # 10%の確率でぼかしを適用
    AddGaussianNoise(mean=0.0, std=1.0, p=0.1),                                          # 10%の確率でガウシアンノイズを追加
    # transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=(0.5,)), # 10%の確率でランダムに領域を消す（擬似的遮蔽）
    transforms.ToPILImage(),                                                             # TensorをPIL画像に戻す
])

def apply_transforms_to_folder(input_folder, output_folder, transform):
    """フォルダ内の画像にデータ拡張を適用して保存する"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(input_folder, filename)
            image = Image.open(file_path)

            # データ拡張を適用
            augmented_image = transform(image)

            # 処理後の画像を保存
            output_path = os.path.join(output_folder, filename)
            augmented_image.save(output_path)
    print(f"Processed {len(os.listdir(input_folder))} images and saved to {output_folder}")

# 使用例
input_folder = "../bridge_data/20250411共有"  # 入力画像フォルダのパス
output_folder = "../bridge_data/20250411共有_data_augmentation"  # 出力画像フォルダのパス
apply_transforms_to_folder(input_folder, output_folder, trans_aug)
