from ultralytics import YOLO
import requests

# 1. ネット上の画像のURL（例：犬と自転車）
image_url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"

# 2. 画像をダウンロードして保存
response = requests.get(image_url)
with open("input.jpg", "wb") as f:
    f.write(response.content)

# 3. 学習済みモデル(YOLOv8n)をロード
model = YOLO("yolov8n.pt") 

# 4. 画像認識（推論）を実行
results = model.predict(source="input.jpg", save=True)

print("画像認識が完了しました！ 'runs/detect/predict' フォルダを確認してください。")