import cv2
import requests
from ultralytics import YOLO

VIDEO_PATH = "video.mp4"
API_ENDPOINT = "http://localhost:8080/api/detections"

model_coco = YOLO("yolov8n.pt")
model_fire = YOLO("fire_model.pt")

def extract_detections(results, model_names):
    detections = []
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model_names.get(cls_id, "unknown")
        detections.append({
            "x": round(x1, 2),
            "y": round(y1, 2),
            "width": round(x2 - x1, 2),
            "height": round(y2 - y1, 2),
            "class_name": class_name,
            "confidence": round(conf, 3)
        })
    return detections

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Nie udało się otworzyć pliku wideo: {VIDEO_PATH}")

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    print(f"\n🎞️ Klatka {frame_number}")

    results_coco = model_coco(frame)
    detections_coco = extract_detections(results_coco, model_coco.names)

    results_fire = model_fire(frame)
    detections_fire = extract_detections(results_fire, model_fire.names)

    all_detections = detections_coco + detections_fire

    print(all_detections)

    payload = {
        "frame": frame_number,
        "detections": all_detections
    }

    try:
        r = requests.post(API_ENDPOINT, json=payload, timeout=1)
    except requests.exceptions.RequestException as e:
        print(f"Nie udało się wysłać danych: {e}")
