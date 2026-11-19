"""
tools/test_canonical_diamond.py
Visualizador del Rombo Canónico y trazado de vectores de drop sin dispositivo.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from attack.deployment_strategy import (
    CANONICAL_CENTER,
    CANONICAL_RX_BUILDABLE,
    CANONICAL_RY_BUILDABLE,
    CANONICAL_RX_GRASS,
    CANONICAL_RY_GRASS,
    compute_canonical_diamond_drop
)
from vision.yolo_detector import get_detector

def test_on_image(image_path: str):
    if not os.path.exists(image_path):
        print(f"No existe: {image_path}")
        return

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    dbg = img.copy()

    cx, cy = CANONICAL_CENTER

    # 1. Dibujar Rombo Construible (Rojo / Exclusión)
    pts_bld = np.array([
        [cx, cy - CANONICAL_RY_BUILDABLE],
        [cx + CANONICAL_RX_BUILDABLE, cy],
        [cx, cy + CANONICAL_RY_BUILDABLE],
        [cx - CANONICAL_RX_BUILDABLE, cy]
    ], dtype=np.int32)
    cv2.polylines(dbg, [pts_bld], isClosed=True, color=(0, 0, 255), thickness=2)

    # 2. Dibujar Rombo Césped Legal (Verde / Despliegue)
    pts_grass = np.array([
        [cx, cy - CANONICAL_RY_GRASS],
        [cx + CANONICAL_RX_GRASS, cy],
        [cx, cy + CANONICAL_RY_GRASS],
        [cx - CANONICAL_RX_GRASS, cy]
    ], dtype=np.int32)
    cv2.polylines(dbg, [pts_grass], isClosed=True, color=(0, 255, 0), thickness=3)

    # 3. Detectar edificios con YOLO y trazar vectores de drop
    detector = get_detector()
    if detector:
        dets = detector.detect(img)
        print(f"Edificios detectados: {len(dets)}")
        for d in dets:
            bx, by = d.center
            dx, dy = compute_canonical_diamond_drop(d.center)
            # Dibujar edificio
            cv2.circle(dbg, (bx, by), 5, (255, 255, 0), -1)
            # Dibujar drop
            cv2.circle(dbg, (dx, dy), 7, (255, 0, 255), -1)
            # Flecha del drop al edificio
            cv2.arrowedLine(dbg, (dx, dy), (bx, by), (255, 200, 0), 2, tipLength=0.15)

    os.makedirs("logs/yolo", exist_ok=True)
    out_path = "logs/yolo/test_canonical_diamond_result.png"
    cv2.imwrite(out_path, dbg)
    print(f"✅ Resultado guardado en: {out_path}")

if __name__ == "__main__":
    import glob
    candidates = glob.glob("logs/yolo/attack_*.png") + glob.glob("datasets/coc/images/raw/*.png")
    if candidates:
        test_on_image(candidates[0])
    else:
        print("No se encontraron capturas para probar.")
