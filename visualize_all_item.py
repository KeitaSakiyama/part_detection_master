import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- 設定 ---
json_path = '../bridge_data/json_data/20250411共有_all_item_test.json'
image_folder = '../bridge_data/20250411共有'  # ← 画像ファイルのディレクトリを指定
output_dir = 'visualized_output_images_all_item/'
os.makedirs(output_dir, exist_ok=True)  # 保存ディレクトリを作成

# カラーパレット（6色を繰り返し）
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# JSON読み込み
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data)

# カテゴリ辞書を作成（id→name）
categories = {cat["id"]: cat["name"] for cat in data["categories"]}
print(f"カテゴリ数: {len(categories)}")
print(f"画像数: {len(data['images'])}")
print(f"アノテーション数: {len(data['annotations'])}")
# アノテーションを画像IDごとにまとめる
annotations_per_image = {}
for ann in data.get("annotations", []):
    img_id = ann["image_id"]
    annotations_per_image.setdefault(img_id, []).append(ann)

# 見える化関数
def plot_results(pil_img, boxes, labels, scores, figsize_input):
    font_size = 12
    np_image = np.array(pil_img)

    fig, ax = plt.subplots(figsize=figsize_input)
    ax.imshow(np_image)

    for i, ((xmin, ymin, w, h), l, s) in enumerate(zip(boxes, labels, scores)):
        if w <= 0 or h <= 0:
            print(f"無効なbbox: {xmin}, {ymin}, {w}, {h}")
            continue
        color = COLORS[i % len(COLORS)]
        ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
                                   fill=False, color=color, linewidth=2))
        text = f'{l}: {s:.2f}' if s is not None else l
        ax.text(xmin, ymin - (font_size + 2), text, fontsize=font_size,
                bbox=dict(facecolor='white', alpha=0.8), color='black')
    ax.axis('off')

# def plot_results(pil_img, boxes, labels, scores, figsize_input):
#     font_size = 12
#     plt.figure(figsize=figsize_input)
#     np_image = np.array(pil_img)
#     ax = plt.gca()
#     for i, ((xmin, ymin, w, h), l, s) in enumerate(zip(boxes, labels, scores)):
#         color = COLORS[i % len(COLORS)]
#         xmin, ymin, w, h = map(int, [xmin, ymin, w, h])
#         ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
#                                    fill=False, color=color, linewidth=2))
#         text = f'{l}: {s:.2f}' if isinstance(s, (float, int)) else l
#         ax.text(xmin, ymin - (font_size + 2), text, fontsize=font_size,
#                 bbox=dict(facecolor='white', alpha=0.8))
#     plt.imshow(np_image)
#     plt.axis('off')
# def plot_results(pil_img, boxes, labels, scores, figsize_input):
#     font_size = 12
#     plt.figure(figsize=figsize_input)
#     np_image = np.array(pil_img)
#     ax = plt.gca()
#     colors = COLORS * 100
#     for (xmin, ymin, w, h), l, s, c in zip(boxes, labels, scores, colors):
#         ax.add_patch(plt.Rectangle((xmin, ymin), w, h,
#                                    fill=False, color=c, linewidth=2))
#         text = f'{l}: {s:.2f}' if s is not None else l
#         ax.text(xmin, ymin - (font_size + 2), text, fontsize=font_size,
#                 bbox=dict(facecolor='white', alpha=0.8))
#     plt.imshow(np_image)
#     plt.axis('off')

# 各画像に対して可視化＆保存
for img_info in data["images"]:
    # print(f"Processing image ID: {img_info}")
    img_id = img_info["id"]
    # print(f"Processing image ID: {img_id}")
    file_name = img_info["file_name"]
    img_path = os.path.join(image_folder, file_name)
    
    if not os.path.exists(img_path):
        print(f"画像が見つかりません: {img_path}")
        continue

    image = Image.open(img_path).convert("RGB")
    anns = annotations_per_image.get(img_id, [])

    boxes = [ann["bbox"] for ann in anns]
    labels = [categories[ann["category_id"]] for ann in anns]
    scores = [ann.get("score", None) for ann in anns]  # 推論結果ならscoreがある

    figsize = (img_info["width"]/100, img_info["height"]/100)
    plot_results(image, boxes, labels, scores, figsize)
    
    output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_vis.jpg")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=100)  # 画像を保存
    print(f"画像を保存しました: {output_path}")  # 保存確認のログを追加
    plt.close()

