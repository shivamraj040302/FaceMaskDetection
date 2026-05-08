import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

yolo_face = YOLO("yolov8nfa.pt")
yolo_face.verbose = False

mask_model = load_model("face_mask_model.keras")


def preprocess(face_img):
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def classify_face(face_crop):
    processed = preprocess(face_crop)
    pred = mask_model.predict(processed, verbose=0)[0]
    pred = float(pred[0])
    if pred > 0.5:
        return "Without Mask", pred
    else:
        return "With Mask", 1 - pred


def detect_and_annotate(frame):
    annotated = frame.copy()
    results = yolo_face(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Add padding to the face crop for better classification
            h, w = frame.shape[:2]
            pad = 10
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)

            face_crop = frame[y1p:y2p, x1p:x2p]

            if face_crop.size == 0:
                continue

            mask_label, conf = classify_face(face_crop)

            color = (0, 255, 0) if mask_label == "With Mask" else (0, 0, 255)
            label_text = f"{mask_label} ({conf:.2f})"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)

            # Draw label text
            cv2.putText(annotated, label_text, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated