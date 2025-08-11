#!/usr/bin/env python3
"""
ディレクトリ比較スクリプト
image_output/20250411共有_testとimage_output/20250411共有_5cat_testを比較し、
同じファイル名のものを調べ上げるPythonコード
"""

import os
from pathlib import Path
import pandas as pd
from datetime import datetime

def get_files_in_directory(directory_path):
    """
    指定されたディレクトリ内のすべてのファイルを取得する
    
    Args:
        directory_path (str): ディレクトリのパス
    
    Returns:
        dict: ファイル名をキー、フルパスを値とする辞書
    """
    files_dict = {}
    
    if not os.path.exists(directory_path):
        print(f"警告: ディレクトリが見つかりません: {directory_path}")
        return files_dict
    
    # ディレクトリ内のすべてのファイルを再帰的に取得
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # 隠しファイルを除外
            if not file.startswith('.'):
                full_path = os.path.join(root, file)
                # 相対パスを保存（ディレクトリ構造を保持）
                relative_path = os.path.relpath(full_path, directory_path)
                files_dict[file] = {
                    'full_path': full_path,
                    'relative_path': relative_path,
                    'size': os.path.getsize(full_path)
                }
    
    return files_dict

def compare_directories(dir1_path, dir2_path):
    """
    2つのディレクトリを比較し、同じファイル名のものを見つける
    
    Args:
        dir1_path (str): 1つ目のディレクトリのパス
        dir2_path (str): 2つ目のディレクトリのパス
    
    Returns:
        tuple: (共通ファイル, dir1のみのファイル, dir2のみのファイル)
    """
    print(f"ディレクトリ1を分析中: {dir1_path}")
    files1 = get_files_in_directory(dir1_path)
    
    print(f"ディレクトリ2を分析中: {dir2_path}")
    files2 = get_files_in_directory(dir2_path)
    
    # ファイル名のセットを作成
    filenames1 = set(files1.keys())
    filenames2 = set(files2.keys())
    
    # 共通するファイル名を見つける
    common_files = filenames1.intersection(filenames2)
    
    # 各ディレクトリのみにあるファイル名を見つける
    only_in_dir1 = filenames1 - filenames2
    only_in_dir2 = filenames2 - filenames1
    
    return common_files, only_in_dir1, only_in_dir2, files1, files2

def save_results_to_csv(common_files, only_in_dir1, only_in_dir2, files1, files2, dir1_name, dir2_name):
    """
    結果をCSVファイルに保存する
    
    Args:
        common_files (set): 共通するファイル名のセット
        only_in_dir1 (set): dir1のみにあるファイル名のセット
        only_in_dir2 (set): dir2のみにあるファイル名のセット
        files1 (dict): dir1のファイル情報
        files2 (dict): dir2のファイル情報
        dir1_name (str): dir1の名前
        dir2_name (str): dir2の名前
    """
    # 共通ファイルの詳細情報
    common_data = []
    for filename in sorted(common_files):
        common_data.append({
            'ファイル名': filename,
            f'{dir1_name}_相対パス': files1[filename]['relative_path'],
            f'{dir1_name}_サイズ(bytes)': files1[filename]['size'],
            f'{dir2_name}_相対パス': files2[filename]['relative_path'],
            f'{dir2_name}_サイズ(bytes)': files2[filename]['size'],
            'サイズ一致': files1[filename]['size'] == files2[filename]['size']
        })
    
    # DataFrameを作成してCSVに保存
    if common_data:
        df_common = pd.DataFrame(common_data)
        df_common.to_csv('共通ファイル一覧.csv', index=False, encoding='utf-8-sig')
        print(f"共通ファイル一覧を '共通ファイル一覧.csv' に保存しました")
    
    # dir1のみのファイル
    if only_in_dir1:
        only_dir1_data = []
        for filename in sorted(only_in_dir1):
            only_dir1_data.append({
                'ファイル名': filename,
                '相対パス': files1[filename]['relative_path'],
                'サイズ(bytes)': files1[filename]['size']
            })
        df_only_dir1 = pd.DataFrame(only_dir1_data)
        df_only_dir1.to_csv(f'{dir1_name}_のみのファイル一覧.csv', index=False, encoding='utf-8-sig')
        print(f"{dir1_name}のみのファイル一覧を '{dir1_name}_のみのファイル一覧.csv' に保存しました")
    
    # dir2のみのファイル
    if only_in_dir2:
        only_dir2_data = []
        for filename in sorted(only_in_dir2):
            only_dir2_data.append({
                'ファイル名': filename,
                '相対パス': files2[filename]['relative_path'],
                'サイズ(bytes)': files2[filename]['size']
            })
        df_only_dir2 = pd.DataFrame(only_dir2_data)
        df_only_dir2.to_csv(f'{dir2_name}_のみのファイル一覧.csv', index=False, encoding='utf-8-sig')
        print(f"{dir2_name}のみのファイル一覧を '{dir2_name}_のみのファイル一覧.csv' に保存しました")

def main():
    """
    メイン関数
    """
    # 比較対象のディレクトリパス
    dir1_path = "image_output/20250411共有_test"
    dir2_path = "image_output/20250411共有_5cat_test"
    
    # ディレクトリ名（CSVファイル名用）
    dir1_name = "test"
    dir2_name = "5cat_test"
    
    print("=" * 60)
    print("ディレクトリ比較スクリプト")
    print("=" * 60)
    print(f"比較対象:")
    print(f"  ディレクトリ1: {dir1_path}")
    print(f"  ディレクトリ2: {dir2_path}")
    print("-" * 60)
    
    # ディレクトリを比較
    common_files, only_in_dir1, only_in_dir2, files1, files2 = compare_directories(dir1_path, dir2_path)
    
    # 結果を表示
    print(f"\n結果:")
    print(f"  {dir1_name}内のファイル数: {len(files1)}")
    print(f"  {dir2_name}内のファイル数: {len(files2)}")
    print(f"  共通するファイル数: {len(common_files)}")
    print(f"  {dir1_name}のみのファイル数: {len(only_in_dir1)}")
    print(f"  {dir2_name}のみのファイル数: {len(only_in_dir2)}")
    
    # 共通ファイルを表示（最初の10個）
    if common_files:
        print(f"\n共通するファイル名（最初の10個）:")
        for i, filename in enumerate(sorted(common_files)[:10]):
            print(f"  {i+1:2d}. {filename}")
        if len(common_files) > 10:
            print(f"  ... （他{len(common_files)-10}個）")
    
    # 各ディレクトリのみのファイルを表示（最初の5個）
    if only_in_dir1:
        print(f"\n{dir1_name}のみのファイル（最初の5個）:")
        for i, filename in enumerate(sorted(only_in_dir1)[:5]):
            print(f"  {i+1}. {filename}")
        if len(only_in_dir1) > 5:
            print(f"  ... （他{len(only_in_dir1)-5}個）")
    
    if only_in_dir2:
        print(f"\n{dir2_name}のみのファイル（最初の5個）:")
        for i, filename in enumerate(sorted(only_in_dir2)[:5]):
            print(f"  {i+1}. {filename}")
        if len(only_in_dir2) > 5:
            print(f"  ... （他{len(only_in_dir2)-5}個）")
    
    # 結果をCSVファイルに保存
    print(f"\n結果をCSVファイルに保存中...")
    save_results_to_csv(common_files, only_in_dir1, only_in_dir2, files1, files2, dir1_name, dir2_name)
    
    print(f"\n分析完了!")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()