# 雑草検出スクリプト (Weed Detector)

MediaPipe Model Maker を使った雑草検出・サイズ計測ツールです。

## ディレクトリ構成

```
mediapipe/
├── weed_detector.py        # メインスクリプト
├── requirements.txt        # 依存パッケージ
├── images/
│   ├── train/
│   │   ├── weed/           # 雑草画像（学習用）
│   │   └── not_weed/       # 雑草でない画像（学習用）
│   └── source/             # 推論対象の画像
├── model/                  # 学習済みモデルの保存先
└── output/                 # 検出結果の出力先
```

## セットアップ

```bash
pip install -r requirements.txt
```

> **注意**: `mediapipe-model-maker` は Python 3.9〜3.11 で動作します。  
> TensorFlow 2.x が必要です。

## 使い方

### 1. 学習データを準備する

- `images/train/weed/` に **雑草の画像** を配置
- `images/train/not_weed/` に **雑草でない画像**（芝生、土、コンクリートなど）を配置
- 各クラス最低 **10〜20枚**（多いほど精度向上）

### 2. モデルを学習する

```bash
python weed_detector.py --train
```

学習済みモデルは `model/` フォルダに `.tflite` として保存されます。

### 3. 推論を実行する

`images/source/` に検出対象の画像を配置してから:

```bash
python weed_detector.py --detect
```

結果は `output/output_1.jpg`, `output/output_2.jpg`, ... として保存されます。

### 4. 学習→推論を一括実行

```bash
python weed_detector.py --all
```

## 出力画像の見方

| 表示内容 | 説明 |
|---|---|
| ヘッダー (黄色文字) | 画像分類結果（weed / not_weed）と信頼度 |
| 緑色バウンディングボックス | 検出された雑草領域 |
| 半透明の緑マスク | 雑草と判定されたピクセル領域 |
| 各領域のラベル | 領域番号、幅×高さ(px)、面積(px)、画像全体に対する割合(%) |
| フッター (黄色文字) | 検出領域数と雑草面積の合計 |

## パラメータ調整

[weed_detector.py](weed_detector.py) 内の設定値を変更できます：

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `EPOCHS` | 10 | 学習エポック数 |
| `BATCH_SIZE` | 8 | バッチサイズ |
| `HSV_LOWER` | [25, 40, 40] | 緑色検出の下限 (HSV) |
| `HSV_UPPER` | [95, 255, 255] | 緑色検出の上限 (HSV) |
| `MIN_CONTOUR_AREA` | 500 | 最小輪郭面積 (ノイズ除去) |

## トラブルシューティング

- **緑以外の雑草が検出されない**: `HSV_LOWER` / `HSV_UPPER` の値を調整
- **精度が低い**: 学習画像を各クラス 50枚以上に増やす、`EPOCHS` を増やす
- **小さい雑草が検出されない**: `MIN_CONTOUR_AREA` を小さくする
