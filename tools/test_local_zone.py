"""
tools/test_local_zone.py
Prueba sin dispositivo del pipeline de zona local usando imágenes estáticas de logs/
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from attack.deployment_strategy import detect_village_boundary
from vision.yolo_detector import get_detector


def test_static_zone(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: No existe {image_path}")
        return False

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar imagen {image_path}")
        return False

    detector = get_detector()
    dets = detector.detect(img) if detector else []
    print(f"Detecciones YOLO: {len(dets)}")

    red_cnt, green_cnt = detect_village_boundary(img, detections=dets, mode="high_res_contour")
    print(
        f"Vértices Red: {len(red_cnt) if red_cnt is not None else 0} | "
        f"Green: {len(green_cnt) if green_cnt is not None else 0}"
    )

    dbg = img.copy()
    if red_cnt is not None:
        cv2.drawContours(dbg, [red_cnt], -1, (0, 0, 255), 2)
    if green_cnt is not None:
        cv2.drawContours(dbg, [green_cnt], -1, (0, 255, 0), 2)

    out_dir = os.path.join("logs", "yolo")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "test_zone_output.png")
    cv2.imwrite(out, dbg)
    print(f"Resultado guardado en {out}")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "logs/yolo/attack_20261109_094640.png"
    if not os.path.exists(target):
        target = "logs/yolo/quota_attack_debug.png"
    test_static_zone(target)
