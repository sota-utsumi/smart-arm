# 雑草検出ツール (Weed Detector)

OpenCV ベースの雑草分類・領域検出ツールです。

- `learn.py` が学習を担当します
- `weed_detector.py` と `weed_detector.cpp` は推論専用です
- Python と C++ は同じ OpenCV SVM モデルを共有します

## ディレクトリ構成

```text
mediapipe/
├── learn.py               # 学習用 CLI
├── weed_model.py          # 共有特徴量抽出 / モデル入出力
├── weed_detector.py       # Python 推論 CLI
├── weed_detector.cpp      # C++ 推論 CLI
├── requirements.txt       # Python 依存パッケージ
├── Makefile               # C++ ビルド
├── images/
│   ├── train/
│   │   ├── weed/          # 雑草画像
│   │   └── not_weed/      # 非雑草画像
│   └── source/            # 推論対象画像
├── model/
│   ├── weed_classifier.yml
│   └── weed_scaler.yml
└── output/
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 学習

学習画像を以下に配置します。

- `images/train/weed/`
- `images/train/not_weed/`

その後、学習を実行します。

```bash
python learn.py
```

必要なら出力先を変えられます。

```bash
python learn.py --model-path model/custom_classifier.yml --scaler-path model/custom_scaler.yml
```

学習後は以下の 2 ファイルが生成されます。

- `model/weed_classifier.yml`
- `model/weed_scaler.yml`

## Python で推論

静止画推論:

```bash
python weed_detector.py --detect
```

カメラ推論:

```bash
python weed_detector.py --camera-detect --config config.txt
```

## C++ で推論

ビルド:

```bash
make
```

静止画推論:

```bash
build/weed_detector_cpp --detect
```

カメラ推論:

```bash
build/weed_detector_cpp --camera-detect --config config.txt
```

## モデルの中身

学習では画像から以下の特徴量を作り、OpenCV SVM に入力しています。

- HSV 3 チャンネルのヒストグラム
- 緑色画素の割合
- Sobel エッジ強度の統計量

## 調整ポイント

主な閾値は [weed_model.py](./weed_model.py) にあります。

- `HSV_LOWER`
- `HSV_UPPER`
- `MIN_CONTOUR_AREA`

## 補足

- `weed_detector.py` / `weed_detector.cpp` は学習を行いません
- Python と C++ は同じ `.yml` モデルを読み込みます
