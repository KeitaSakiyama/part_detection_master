# 部品検出マスター（part_detection-master）

このリポジトリは、部品検出に関する各種スクリプトやツールをまとめたものです。

## 主なファイル一覧
- 1st_finetuning_0526.py / 1st_finetuning.py：ファインチューニング用スクリプト
- 2nd_validation_0526.py / 2nd_validation.py：バリデーション用スクリプト
- auto_git_pull_monitor.sh：Gitリポジトリの自動監視・更新
- compare_directories.py：ディレクトリ比較ツール
- data_augmentation.py：データ拡張スクリプト
- evaluate_glip.py：GLIPモデル評価
- finetuned_glip_hierarchical_clustering_45cat.py：階層クラスタリング
- plot_ap_results_5cat.py / plot_ap_results_all_item.py：AP結果の可視化
- visualize.py / visualize_all_item.py：検出結果の可視化
- その他、学習・検証・可視化・自動化関連のシェルスクリプト

## 使い方
1. 必要なPythonパッケージをインストールしてください。
2. 各スクリプトはコマンドラインから実行できます。
3. 詳細な使い方は各スクリプト内のコメントやREADME.md（英語版）を参照してください。

## 注意事項
- Python 3.xが必要です。
- データセットやモデルのパスは適宜修正してください。
- 実行環境によっては追加のライブラリが必要な場合があります。


