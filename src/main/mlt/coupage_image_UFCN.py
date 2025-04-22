import cv2
import numpy as np
import sys
import json
import base64
import torch

from doc_ufcn.main import DocUFCN
from huggingface_hub import hf_hub_download

def detect_lines_UFNC(image_path):
    # Lecture et conversion en RGB
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Sélection du device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Téléchargement et chargement du modèle
    model_filename = hf_hub_download(
        repo_id="Teklia/doc-ufcn-generic-historical-line",
        filename="model.pth"
    )
    nb_of_classes = 2   # 1 classe ligne + 1 arrière‑plan
    input_size = 600
    model = DocUFCN(nb_of_classes, input_size, device)
    model.load(model_filename, mean=[0, 0, 0], std=[1, 1, 1], mode="eval")

    # Prédiction brute
    detected_polygons, _, mask, overlap = model.predict(
        image,
        raw_output=True,
        mask_output=True,
        overlap_output=True
    )

    # Seuils de filtrage
    conf_threshold = 0.85
    min_width = 500
    min_height = 20
    min_area = min_width * min_height
    max_white_ratio = 0.8

    rects = []
    for class_id, items in detected_polygons.items():
        for item in items:
            if item['confidence'] < conf_threshold:
                continue
            coords = np.array(item['polygon'], dtype=np.int32).reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(coords)

            # Filtrage par zone et ratio de blanc
            if w < min_width or h < min_height or (w * h) < min_area:
                continue
            roi = image[y:y+h, x:x+w]
            white_pixels = np.sum(np.all(roi >= 250, axis=2))
            if white_pixels / float(w * h) > max_white_ratio:
                continue

            rects.append((x, y, w, h))

    # Application de l'offset et extraction des crops
    crops = []
    pad_left, pad_top = 70, 10
    pad_right, pad_bottom = 0, 10
    for x, y, w, h in rects:
        x1 = max(0, x - pad_left)
        y1 = max(0, y - pad_top)
        x2 = min(width, x + w + pad_right)
        y2 = min(height, y + h + pad_bottom)
        crop = image[y1:y2, x1:x2]
        crops.append(crop)

    return crops


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        detected_crops = detect_lines_UFNC(image_path)

        # Encodage des crops en Base64
        b64_list = []
        for crop in detected_crops:
            _, buffer = cv2.imencode('.png', crop)
            b64 = base64.b64encode(buffer).decode('utf-8')
            b64_list.append(b64)

        print(json.dumps({"detectedRects": b64_list}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
