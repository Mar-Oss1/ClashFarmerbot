"""
tools/test_jump_overlay.py
Genera overlays visuales con círculos de radio R sobre las capturas de ataque
utilizando el nuevo algoritmo de intersección de muros y máxima cobertura.
"""
import os
import sys
import json
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.yolo_detector import YoloDetector
from attack.deployment_strategy import compute_adaptive_jump_targets_from_detections
from core.config import SETTINGS, PROJECT_ROOT
from core.logger import get_logger

log = get_logger("test_jump_overlay")

CLASS_COLORS = {
    "storage_gold": (0, 215, 255),      # Dorado
    "storage_elixir": (255, 0, 255),    # Rosa / Magenta
    "storage_dark": (0, 140, 255),      # Naranja oscuro / Oscuro
    "town_hall": (255, 120, 0),         # Azul / Cyan
    "collector_gold": (0, 255, 255),    # Amarillo
    "collector_elixir": (220, 150, 255),# Rosa suave
    "collector_dark": (50, 50, 180),    # Granate / Oscuro
}


def run_overlay_test(images: list[str], out_dir: str = "logs/yolo") -> list[str]:
    attack_cfg = SETTINGS.get("attack", {})
    radius = int(attack_cfg.get("jump_spell_radius_px", 125))
    detector = YoloDetector(conf=float(attack_cfg.get("yolo_conf", 0.40)), imgsz=int(attack_cfg.get("yolo_imgsz", 1280)))

    os.makedirs(out_dir, exist_ok=True)
    generated_overlays = []

    for img_path in images:
        if not os.path.exists(img_path):
            log.warning("Imagen no encontrada: %s", img_path)
            continue

        base_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 1. Inferencia YOLO
        detections = detector.detect(img)
        log.info("Procesando %s: %d detecciones YOLO", base_name, len(detections))

        # 2. Cálculo con nuevo algoritmo de intersecciones
        targets = compute_adaptive_jump_targets_from_detections(detections, (w, h))

        # 3. Dibujar overlay
        vis = img.copy()

        # Dibujar edificios detectados
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            color = CLASS_COLORS.get(d.cls_name, (200, 200, 200))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis, d.center, 4, color, -1)
            label = f"{d.cls_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Dibujar cada objetivo de salto con círculo de radio R
        for idx, pt in enumerate(targets, 1):
            jx, jy = int(pt[0]), int(pt[1])

            # Círculo semitransparente del radio de efecto
            overlay_layer = vis.copy()
            cv2.circle(overlay_layer, (jx, jy), radius, (0, 255, 0), -1)
            cv2.addWeighted(overlay_layer, 0.20, vis, 0.80, 0, vis)

            # Borde del círculo
            cv2.circle(vis, (jx, jy), radius, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(vis, (jx, jy), 5, (0, 0, 255), -1)

            # Etiqueta
            tag = f"SALTO #{idx} ({jx}, {jy}) [R={radius}px]"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (jx - tw // 2 - 4, jy - radius - th - 8), (jx + tw // 2 + 4, jy - radius), (0, 0, 0), -1)
            cv2.putText(vis, tag, (jx - tw // 2, jy - radius - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Líneas guía hacia edificios cubiertos por este salto
            for d in detections:
                dist_d = np.hypot(d.center[0] - jx, d.center[1] - jy)
                if dist_d <= radius:
                    cv2.line(vis, (jx, jy), d.center, (0, 255, 255), 1, cv2.LINE_AA)

        # Banner superior
        banner_text = f"NUEVO ALGORITMO: Intersecciones de Muros | Saltos: {len(targets)} | R={radius}px"
        cv2.rectangle(vis, (0, 0), (w, 38), (15, 15, 15), -1)
        cv2.putText(vis, banner_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        out_name = f"overlay_{base_name}"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, vis)
        log.info("Overlay guardado en: %s", out_path)
        generated_overlays.append(out_path)

    return generated_overlays


if __name__ == "__main__":
    targets = [
        "logs/yolo/attack_20261105_203506.png",
        "logs/yolo/attack_20261105_183131.png",
        "logs/yolo/attack_20261105_183319.png",
        "logs/yolo/attack_20261105_183459.png",
    ]
    run_overlay_test(targets)
