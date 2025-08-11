import json
import torch
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime

# --- 設定項目 ---
# ご自身の環境に合わせてパスを修正してください。
# log.txtから特定した正解アノテーションファイルへのパス
GROUND_TRUTH_JSON_PATH = '../bridge_data/json_data/20250411共有_all_item_test.json'
# GLIPの推論結果が保存されているpredictions.pthファイルへのパス
PREDICTIONS_PATH = './2nd_RESULT/20250411共有_all_item_valid/result_test/full/all_shot/predictions.pth'

# IoUのマッチングに使用する閾値
IOU_THRESHOLD = 0.5

# log.txtの DATASETS.OVERRIDE_CATEGORY から取得したカテゴリ情報
# モデルの出力ラベルは1から始まるため、リストのインデックス+1に対応します。
CATEGORIES = [
    {'id': 1, 'name': 'Main Cable'},
    {'id': 2, 'name': 'Main Cable (Multiple Cables)'},
    {'id': 3, 'name': 'Hanger Rope'},
    {'id': 4, 'name': 'Stays'},
    {'id': 5, 'name': 'Stay Support'},
    {'id': 6, 'name': 'Parallel Wire Strand'},
    {'id': 7, 'name': 'Strand Rope'},
    {'id': 8, 'name': 'Spiral Rope'},
    {'id': 9, 'name': 'Locked Coil Rope'},
    {'id': 10, 'name': 'PC Steel Wire'},
    {'id': 11, 'name': 'Semi-Parallel Wire Strand'},
    {'id': 12, 'name': 'Zinc-Plated Steel Wire'},
    {'id': 13, 'name': 'Cable Band'},
    {'id': 14, 'name': 'Saddle Cover'},
    {'id': 15, 'name': 'Cable Cover'},
    {'id': 16, 'name': 'Socket Cover'},
    {'id': 17, 'name': 'Anchor Cover'},
    {'id': 18, 'name': 'Tower Saddle'},
    {'id': 19, 'name': 'Spray Saddle'},
    {'id': 20, 'name': 'Intersection Fitting'},
    {'id': 21, 'name': 'Stay Grip Fitting'},
    {'id': 22, 'name': 'Socket (Open Type)'},
    {'id': 23, 'name': 'Socket (Rod Anchor Type)'},
    {'id': 24, 'name': 'Socket (Pressure Anchor Type)'},
    {'id': 25, 'name': 'End Clamp (SG Socket)'},
    {'id': 26, 'name': 'End Clamp (Screw Type)'},
    {'id': 27, 'name': 'Crimp Anchor'},
    {'id': 28, 'name': 'Wire Clip Anchor'},
    {'id': 29, 'name': 'Embedded Anchor'},
    {'id': 30, 'name': 'Cable Anchor'},
    {'id': 31, 'name': 'Socket Anchor'},
    {'id': 32, 'name': 'Rod Anchor'},
    {'id': 33, 'name': 'Anchor Rod'},
    {'id': 34, 'name': 'Rod Thread Part'},
    {'id': 35, 'name': 'Rod Anchor Nut'},
    {'id': 36, 'name': 'Fixing Bolt'},
    {'id': 37, 'name': 'Anchor Piece'},
    {'id': 38, 'name': 'Shackle'},
    {'id': 39, 'name': 'Turnbuckle'},
    {'id': 40, 'name': 'Wire Clip'},
    {'id': 41, 'name': 'Connecting Fitting'},
    {'id': 42, 'name': 'Wire Seeging'},
    {'id': 43, 'name': 'Rubber Boot'},
    {'id': 44, 'name': 'Stainless Band'},
    {'id': 45, 'name': 'Grout Injection Port'},
]

# 出力ディレクトリの設定
OUTPUT_DIR = './evaluation_results'
# --- 設定項目ここまで ---

def calculate_iou(box1, box2):
    """2つのバウンディングボックスのIoUを計算する"""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def plot_confusion_matrix(cm, class_names, output_filename='confusion_matrix.png'):
    """
    コンフュージョンマトリックスをプロットして画像として保存する。

    Args:
        cm (np.array): コンフュージョンマトリックス (True x Pred)
        class_names (list): クラス名のリスト
        output_filename (str): 保存するファイル名
    """
    # 個数をそのまま表示するDataFrameを作成
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    # 描画エリアのサイズをクラス数に応じて調整
    fig_size = max(15, len(class_names) // 2)
    plt.figure(figsize=(fig_size, fig_size))
    
    # 日本語フォントの設定 (環境に応じて変更してください)
    try:
        # 例: 'IPAexGothic', 'Noto Sans CJK JP', 'Yu Gothic' など
        plt.rcParams['font.family'] = 'IPAexGothic' 
    except RuntimeError:
        print("警告: IPAexGothicフォントが見つかりません。日本語が文字化けする可能性があります。")
        plt.rcParams['font.family'] = 'sans-serif'

    # 個数を表示、色は対数スケール、枠線付き
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", 
                          norm=plt.cm.colors.LogNorm(vmin=1, vmax=cm.max()),
                          linewidths=0.5, linecolor='black')
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('confusion matrix')
    plt.tight_layout()
    
    plt.savefig(output_filename)
    print(f"\nコンフュージョンマトリックスを {output_filename} に保存しました。")
    plt.close()


def save_evaluation_results(stats, cat_id_to_name, precision, recall, f1_score, all_tp, all_fp, all_fn, output_dir):
    """評価結果をファイルに保存する"""
    # 全体結果の保存
    overall_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'tp': int(all_tp),
            'fp': int(all_fp),
            'fn': int(all_fn)
        }
    }
    
    with open(os.path.join(output_dir, 'overall_results.json'), 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2)
    
    # クラス別結果の保存
    class_results = []
    sorted_cat_ids = sorted(stats.keys(), key=lambda cid: cat_id_to_name.get(cid, ''))
    
    for cat_id in sorted_cat_ids:
        s = stats[cat_id]
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        cat_name = cat_id_to_name.get(cat_id, f"Unknown ID: {cat_id}")
        
        class_results.append({
            'category_id': int(cat_id),
            'category_name': cat_name,
            'precision': float(p),
            'recall': float(r),
            'f1_score': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        })
    
    # JSON形式で保存
    with open(os.path.join(output_dir, 'class_results.json'), 'w', encoding='utf-8') as f:
        json.dump(class_results, f, ensure_ascii=False, indent=2)
    
    # CSV形式でも保存
    df = pd.DataFrame(class_results)
    df.to_csv(os.path.join(output_dir, 'class_results.csv'), index=False, encoding='utf-8')
    
    print(f"評価結果を {output_dir} に保存しました。")

def save_misclassification_analysis(misclassification_records, output_dir):
    """誤分類分析結果をファイルに保存する"""
    if not misclassification_records:
        return
    
    # numpy型をPython型に変換
    converted_records = []
    for record in misclassification_records:
        converted_record = {
            'image_id': int(record['image_id']),
            'image_filename': record['image_filename'],
            'true_class': record['true_class'],
            'predicted_class': record['predicted_class'],
            'confidence': float(record['confidence']),
            'iou': float(record['iou'])
        }
        converted_records.append(converted_record)
    
    # 詳細な誤分類記録をCSV形式で保存
    df_records = pd.DataFrame(converted_records)
    df_records.to_csv(os.path.join(output_dir, 'misclassification_details.csv'), index=False, encoding='utf-8')
    
    # 画像別誤分類統計
    image_stats = defaultdict(lambda: {'count': 0, 'errors': []})
    for record in converted_records:
        image_name = record['image_filename']
        image_stats[image_name]['count'] += 1
        image_stats[image_name]['errors'].append({
            'true_class': record['true_class'],
            'predicted_class': record['predicted_class'],
            'confidence': record['confidence'],
            'iou': record['iou']
        })
    
    # 画像別統計をJSONで保存
    image_stats_list = []
    for image_name, stats in image_stats.items():
        image_stats_list.append({
            'image_filename': image_name,
            'error_count': stats['count'],
            'errors': stats['errors']
        })
    
    # エラー数でソート
    image_stats_list.sort(key=lambda x: x['error_count'], reverse=True)
    
    with open(os.path.join(output_dir, 'image_error_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(image_stats_list, f, ensure_ascii=False, indent=2)
    
    # 誤り傾向分析
    error_pairs = defaultdict(int)
    for record in converted_records:
        pair = (record['true_class'], record['predicted_class'])
        error_pairs[pair] += 1
    
    # 双方向誤りの検出
    bidirectional_errors = []
    checked_pairs = set()
    
    for (true_class, pred_class), count in error_pairs.items():
        if (true_class, pred_class) in checked_pairs:
            continue
            
        reverse_pair = (pred_class, true_class)
        if reverse_pair in error_pairs:
            reverse_count = error_pairs[reverse_pair]
            bidirectional_errors.append({
                'class_a': true_class,
                'class_b': pred_class,
                'a_to_b_errors': count,
                'b_to_a_errors': reverse_count,
                'total_errors': count + reverse_count
            })
            checked_pairs.add((true_class, pred_class))
            checked_pairs.add(reverse_pair)
    
    # 双方向誤りを総エラー数でソート
    bidirectional_errors.sort(key=lambda x: x['total_errors'], reverse=True)
    
    # 双方向誤りをCSVで保存
    if bidirectional_errors:
        df_bidirectional = pd.DataFrame(bidirectional_errors)
        df_bidirectional.to_csv(os.path.join(output_dir, 'bidirectional_errors.csv'), index=False, encoding='utf-8')
    
    print(f"誤分類分析結果を {output_dir} に保存しました。")

def main():
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. データの読み込み ---
    print("データを読み込んでいます...")
    # 正解データを読み込み
    with open(GROUND_TRUTH_JSON_PATH, 'r') as f:
        gt_data = json.load(f)

    # 予測結果を読み込み
    try:
        predictions = torch.load(PREDICTIONS_PATH, map_location=torch.device("cpu"))
    except Exception as e:
        print(f"予測ファイルの読み込みに失敗しました: {e}")
        print("maskrcnn-benchmarkの環境でスクリプトを実行しているか確認してください。")
        return

    # --- 2. データの前処理 ---
    # カテゴリIDと名前のマッピングを作成
    cat_id_to_name = {cat['id']: cat['name'] for cat in CATEGORIES}
    
    # 除外されたカテゴリIDのセット - 新しい連続番号では全カテゴリを有効にする
    excluded_cat_ids = set()  # 空のセットにすることで全45カテゴリを使用
    
    # GLIPで使用される正しいマッピングを作成
    # CATEGORIESリストの全カテゴリを使用（除外なし）
    valid_categories = CATEGORIES  # gt_data['categories']の代わりにCATEGORIESを直接使用
    
    # 予測ラベル（1-indexed）から元のカテゴリIDへのマッピング
    pred_label_to_cat_id = {}
    for i, cat in enumerate(valid_categories):
        pred_label_to_cat_id[i + 1] = cat['id']  # 1スタートで連続したマッピング
    
    print(f"予測ラベルから正解カテゴリIDへのマッピング:")
    for pred_label, cat_id in sorted(pred_label_to_cat_id.items()):
        cat_name = cat_id_to_name.get(cat_id, f"Unknown ID: {cat_id}")
        print(f"  予測ラベル {pred_label} -> カテゴリID {cat_id} ({cat_name})")
    
    # フィルタされたカテゴリIDからマトリックスインデックスへのマッピング
    cat_id_to_matrix_idx = {cat['id']: i for i, cat in enumerate(CATEGORIES)}

    # 画像IDをキーとする正解アノテーションの辞書を作成
    gt_annotations = defaultdict(list)
    for ann in gt_data['annotations']:
        # 除外されたカテゴリはスキップ
        if ann['category_id'] in excluded_cat_ids:
            continue
        # bboxを [x1, y1, x2, y2] 形式に変換
        x, y, w, h = ann['bbox']
        bbox = [x, y, x + w, y + h]
        gt_annotations[ann['image_id']].append({
            'bbox': bbox,
            'category_id': ann['category_id']
        })

    # 画像IDとファイル名のマッピング
    image_id_to_filename = {img['id']: img['file_name'] for img in gt_data['images']}
    
    # 予測結果はテストデータセットの画像順に対応していると仮定
    # gt_data['images']の順序とpredictionsの順序が一致している必要がある
    sorted_images = sorted(gt_data['images'], key=lambda x: x['id'])
    
    if len(sorted_images) != len(predictions):
        print(f"エラー: 正解画像の数 ({len(sorted_images)}) と予測の数 ({len(predictions)}) が一致しません。")
        return

    # --- 3. 評価の実行 ---
    print("評価を実行中...")
    stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    num_classes = len(CATEGORIES)
    # コンフュージョンマトリックスの初期化 (行:正解ラベル, 列:予測ラベル)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # 部品同士の誤りを記録する辞書
    misclassification_records = []

    for i, image_info in enumerate(sorted_images):
        image_id = image_info['id']
        image_filename = image_info['file_name']
        
        pred_boxlist = predictions[i]
        pred_boxes = pred_boxlist.bbox.numpy()
        pred_labels = pred_boxlist.get_field('labels').numpy()
        pred_scores = pred_boxlist.get_field('scores').numpy()

        gt_anns_for_image = gt_annotations.get(image_id, [])
        
        # この画像に正解アノテーションがない場合、すべての予測はFP
        if not gt_anns_for_image:
            for pred_label in pred_labels:
                pred_cat_id = pred_label_to_cat_id.get(pred_label)
                if pred_cat_id is not None and pred_cat_id not in excluded_cat_ids:
                    stats[pred_cat_id]['fp'] += 1
            continue

        gt_boxes = np.array([ann['bbox'] for ann in gt_anns_for_image])
        gt_cat_ids = np.array([ann['category_id'] for ann in gt_anns_for_image])
        
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        sorted_indices = np.argsort(-pred_scores)

        # 各予測について、最もIoUが高い正解ボックスを探す
        for pred_idx in sorted_indices:
            pred_box = pred_boxes[pred_idx]
            pred_label = pred_labels[pred_idx]
            pred_cat_id = pred_label_to_cat_id.get(pred_label)

            if pred_cat_id is None or pred_cat_id in excluded_cat_ids:
                continue

            best_iou = 0
            best_gt_idx = -1

            # クラスに関係なく、最もIoUが高い正解ボックスを見つける
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= IOU_THRESHOLD:
                true_cat_id = gt_cat_ids[best_gt_idx]
                
                if not gt_matched[best_gt_idx]:
                    # この正解ボックスに対する初めての有効なマッチ
                    gt_matched[best_gt_idx] = True
                    
                    # コンフュージョンマトリックスの更新（マトリックスインデックスを使用）
                    true_matrix_idx = cat_id_to_matrix_idx[true_cat_id]
                    pred_matrix_idx = cat_id_to_matrix_idx[pred_cat_id]
                    confusion_matrix[true_matrix_idx, pred_matrix_idx] += 1
                    
                    if true_cat_id == pred_cat_id:
                        # 正解: TP
                        stats[true_cat_id]['tp'] += 1
                    else:
                        # クラス誤り: 予測クラスにとってはFP
                        stats[pred_cat_id]['fp'] += 1
                        
                        # 部品同士の誤りを記録
                        misclassification_records.append({
                            'image_id': image_id,
                            'image_filename': image_filename,
                            'true_class': cat_id_to_name.get(true_cat_id, f"Unknown ID: {true_cat_id}"),
                            'predicted_class': cat_id_to_name.get(pred_cat_id, f"Unknown ID: {pred_cat_id}"),
                            'confidence': pred_scores[pred_idx],
                            'iou': best_iou
                        })
                else:
                    # 重複検出: 既にマッチ済みの正解ボックスに対する予測 -> FP
                    stats[pred_cat_id]['fp'] += 1

            else:
                # 背景への誤検出 (どの正解ボックスとも十分に重ならない) -> FP
                stats[pred_cat_id]['fp'] += 1

        # 全ての予測を処理した後、マッチしなかった正解ボックスはFN
        for gt_idx, matched in enumerate(gt_matched):
            if not matched:
                true_cat_id = gt_cat_ids[gt_idx]
                stats[true_cat_id]['fn'] += 1

    # --- 4. 結果の集計と表示 ---
    print("\n--- 評価結果 ---")
    all_tp = sum(s['tp'] for s in stats.values())
    all_fp = sum(s['fp'] for s in stats.values())
    all_fn = sum(s['fn'] for s in stats.values())

    # Micro Average (全体での計算)
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- 全体 (Micro Average) ---")
    print(f"適合率 (Precision): {precision:.4f}")
    print(f"再現率 (Recall):    {recall:.4f}")
    print(f"F1スコア:           {f1_score:.4f}")
    print(f"(TP: {all_tp}, FP: {all_fp}, FN: {all_fn})")

    print("\n--- クラス別 ---")
    print(f"{'クラス名':<30} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'TP':>5} | {'FP':>5} | {'FN':>5}")
    print("-" * 90)

    sorted_cat_ids = sorted(stats.keys(), key=lambda cid: cat_id_to_name.get(cid, ''))
    
    for cat_id in sorted_cat_ids:
        s = stats[cat_id]
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        cat_name = cat_id_to_name.get(cat_id, f"Unknown ID: {cat_id}")
        print(f"{cat_name:<30} | {p:10.4f} | {r:10.4f} | {f1:10.4f} | {tp:5} | {fp:5} | {fn:5}")

    # --- 5. 部品同士の誤りの詳細出力 ---
    print("\n--- 部品同士の誤り詳細 ---")
    if misclassification_records:
        print(f"部品同士の誤りが検出された画像数: {len(set(record['image_filename'] for record in misclassification_records))}")
        print(f"部品同士の誤りの総数: {len(misclassification_records)}")
        print("\n詳細:")
        print(f"{'画像ファイル名':<40} | {'正解クラス':<30} | {'予測クラス':<30} | {'信頼度':>8} | {'IoU':>6}")
        print("-" * 125)
        
        # 画像ファイル名でソート
        sorted_records = sorted(misclassification_records, key=lambda x: x['image_filename'])
        
        for record in sorted_records:
            print(f"{record['image_filename']:<40} | {record['true_class']:<30} | {record['predicted_class']:<30} | {record['confidence']:8.4f} | {record['iou']:6.3f}")
        
        # 誤り傾向の分析
        print("\n--- 誤り傾向の分析 ---")
        error_pairs = defaultdict(int)
        for record in misclassification_records:
            pair = (record['true_class'], record['predicted_class'])
            error_pairs[pair] += 1
        
        # 双方向誤りの検出と表示
        print("\n--- 双方向誤り分析 ---")
        bidirectional_errors = []
        checked_pairs = set()
        
        for (true_class, pred_class), count in error_pairs.items():
            if (true_class, pred_class) in checked_pairs:
                continue
                
            reverse_pair = (pred_class, true_class)
            if reverse_pair in error_pairs:
                reverse_count = error_pairs[reverse_pair]
                bidirectional_errors.append({
                    'class_a': true_class,
                    'class_b': pred_class,
                    'a_to_b': count,
                    'b_to_a': reverse_count,
                    'total': count + reverse_count
                })
                checked_pairs.add((true_class, pred_class))
                checked_pairs.add(reverse_pair)
        
        if bidirectional_errors:
            # 総エラー数でソート
            bidirectional_errors.sort(key=lambda x: x['total'], reverse=True)
            print(f"双方向誤りペア数: {len(bidirectional_errors)}")
            print(f"{'クラスA':<30} | {'クラスB':<30} | {'A→B':>5} | {'B→A':>5} | {'合計':>5}")
            print("-" * 100)
            
            for error in bidirectional_errors:
                print(f"{error['class_a']:<30} | {error['class_b']:<30} | {error['a_to_b']:5} | {error['b_to_a']:5} | {error['total']:5}")
        else:
            print("双方向誤りは検出されませんでした。")
        
        # 頻度の高い誤りペア上位10個を表示
        sorted_error_pairs = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n頻度の高い誤りペア (上位10個):")
        print(f"{'正解クラス':<30} | {'予測クラス':<30} | {'回数':>4}")
        print("-" * 70)
        for (true_class, pred_class), count in sorted_error_pairs:
            print(f"{true_class:<30} | {pred_class:<30} | {count:4}")
    else:
        print("部品同士の誤りは検出されませんでした。")

    # --- 6. 結果をファイルに保存 ---
    save_evaluation_results(stats, cat_id_to_name, precision, recall, f1_score, all_tp, all_fp, all_fn, OUTPUT_DIR)
    save_misclassification_analysis(misclassification_records, OUTPUT_DIR)

    # --- 7. コンフュージョンマトリックスのプロット ---
    class_names = [cat['name'] for cat in CATEGORIES]
    confusion_matrix_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(confusion_matrix, class_names, confusion_matrix_path)
    
    # コンフュージョンマトリックスの数値データも保存
    cm_df = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix.csv'), encoding='utf-8')
    
    print(f"\n--- 全ての結果が {OUTPUT_DIR} に保存されました ---")
    print("保存されたファイル:")
    print("- overall_results.json: 全体の評価結果")
    print("- class_results.json/csv: クラス別の評価結果")
    print("- misclassification_details.csv: 誤分類の詳細")
    print("- image_error_stats.json: 画像別エラー統計")
    print("- error_patterns.csv: 誤り傾向分析")
    print("- bidirectional_errors.csv: 双方向誤り分析")
    print("- confusion_matrix.png/csv: コンフュージョンマトリックス")


if __name__ == '__main__':
    main()