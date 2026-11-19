"""
tools/test_jump_from_labels.py
Valida la estrategia de Hechizos de Salto SIN necesidad de tener el modelo entrenado.
Usa las etiquetas manuales (.txt) como 'detecciones perfectas' y alimenta directamente
compute_adaptive_jump_targets_from_detections() para graficar los objetivos de salto,
el centroide de la base y los vectores de proyección sobre las capturas reales.
"""
import argparse
import glob
import os
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.deployment_strategy import compute_adaptive_jump_targets_from_detections
from vision.yolo_detector import Detection
from core.logger import get_logger

log = get_logger("test_jumps")

CLASS_MAP = {
    0: "storage_gold",
    1: "storage_elixir",
    2: "storage_dark",
    3: "town_hall",
    4: "collector_gold",
    5: "collector_elixir",
    6: "collector_dark",
}

CLASS_COLORS = {
    "storage_gold": (0, 215, 255),
    "storage_elixir": (255, 0, 255),
    "storage_dark": (128, 0, 128),
    "town_hall": (255, 100, 0),
    "collector_gold": (0, 255, 255),
    "collector_elixir": (200, 120, 255),
    "collector_dark": (80, 20, 120),
}


def load_detections_from_txt(txt_path: str, img_w: int, img_h: int) -> list[Detection]:
    detections = []
    if not os.path.exists(txt_path):
        return detections

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx_norm, cy_norm = float(parts[1]), float(parts[2])
                w_norm, h_norm = float(parts[3]), float(parts[4])

                cls_name = CLASS_MAP.get(cls_id, f"class_{cls_id}")

                cx = int(round(cx_norm * img_w))
                cy = int(round(cy_norm * img_h))
                bw = int(round(w_norm * img_w))
                bh = int(round(h_norm * img_h))

                x1 = max(0, min(img_w, cx - bw // 2))
                y1 = max(0, min(img_h, cy - bh // 2))
                x2 = max(0, min(img_w, cx + bw // 2))
                y2 = max(0, min(img_h, cy + bh // 2))

                detections.append(Detection(
                    cls_name=cls_name,
                    confidence=1.0,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center=(cx, cy)
                ))
    return detections


def render_jump_analysis(img: np.ndarray, detections: list[Detection], jump_targets: list[tuple[int, int]]) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]

    # 1. Dibujar edificios etiquetados
    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        color = CLASS_COLORS.get(d.cls_name, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, d.center, 3, color, -1)
        label = f"{d.cls_name}"
        cv2.putText(vis, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # 2. Centroide de la base
    if detections:
        all_pts = np.array([d.center for d in detections], dtype=np.float32)
        if len(all_pts) >= 3:
            hull = cv2.convexHull(all_pts)
            cv2.polylines(vis, [hull.astype(np.int32)], isClosed=True, color=(100, 100, 100), thickness=1, lineType=cv2.LINE_AA)
            M = cv2.moments(hull)
            if M["m00"] > 0:
                base_cx = int(M["m10"] / M["m00"])
                base_cy = int(M["m01"] / M["m00"])
            else:
                base_cx = int(np.mean(all_pts[:, 0]))
                base_cy = int(np.mean(all_pts[:, 1]))
        else:
            base_cx = int(np.mean(all_pts[:, 0]))
            base_cy = int(np.mean(all_pts[:, 1]))

        cv2.drawMarker(vis, (base_cx, base_cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(vis, "CENTRO BASE", (base_cx + 8, base_cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # 3. Dibujar objetivos calculados de Hechizo de Salto
    for idx, pt in enumerate(jump_targets, 1):
        jx, jy = int(pt[0]), int(pt[1])
        # Círculo del radio de efecto del salto
        cv2.circle(vis, (jx, jy), 28, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(vis, (jx, jy), 4, (0, 255, 0), -1)
        # Línea guía desde centroide de base al salto
        if detections:
            cv2.line(vis, (base_cx, base_cy), (jx, jy), (0, 255, 0), 1, cv2.LINE_AA)
        tag = f"SALTO #{idx} ({jx}, {jy})"
        cv2.putText(vis, tag, (jx - 30, jy - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Banner superior
    banner_text = f"Detecciones GT: {len(detections)} | Saltos calculados: {len(jump_targets)}"
    cv2.rectangle(vis, (0, 0), (w, 35), (20, 20, 20), -1)
    cv2.putText(vis, banner_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Prueba visual de lógica de Hechizos de Salto a partir de etiquetas")
    parser.add_argument("--image", type=str, default="", help="Ruta de una imagen específica (opcional)")
    parser.add_argument("--images-dir", type=str, default="datasets/coc/images/raw", help="Directorio de imágenes")
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="datasets/coc/labels/raw",
        help="Directorio de etiquetas .txt (o datasets/coc/labels_pre)"
    )
    parser.add_argument("--out-dir", type=str, default="logs/yolo/test_jumps", help="Directorio de salida para visualizaciones")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.image:
        img_paths = [args.image]
    else:
        img_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")) + glob.glob(os.path.join(args.images_dir, "*.jpg")))

    if not img_paths:
        log.error("No se encontraron imágenes en: %s", args.images_dir)
        sys.exit(1)

    log.info("Analizando lógica de saltos con etiquetas en %s...", args.labels_dir)
    processed = 0

    for p in img_paths:
        stem = Path(p).stem
        label_file = os.path.join(args.labels_dir, f"{stem}.txt")
        # Fallback a labels_pre si no está en labels/raw
        if not os.path.exists(label_file):
            alt = os.path.join("datasets/coc/labels_pre", f"{stem}.txt")
            if os.path.exists(alt):
                label_file = alt

        if not os.path.exists(label_file):
            continue

        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]

        dets = load_detections_from_txt(label_file, w, h)
        if not dets:
            continue

        targets = compute_adaptive_jump_targets_from_detections(dets, (w, h))

        vis_img = render_jump_analysis(img, dets, targets)
        out_path = os.path.join(args.out_dir, f"jump_{stem}.png")
        cv2.imwrite(out_path, vis_img)
        log.info("✅ [%-25s] %d edificios -> %d saltos calculados -> %s", stem, len(dets), len(targets), out_path)
        processed += 1

    if processed == 0:
        log.warning(
            "No se encontraron archivos .txt coincidentes en '%s'. "
            "Asegúrate de tener anotaciones guardadas (con review_prelabels.py o prelabel_with_model.py).",
            args.labels_dir
        )
    else:
        log.info("=" * 60)
        log.info("Prueba completada: %d imágenes renderizadas en: %s", processed, os.path.abspath(args.out_dir))
        log.info("=" * 60)


if __name__ == "__main__":
    main()
