"""
tools/prelabel_with_model.py
Script para pre-etiquetado asistido por modelo YOLO (Ronda 2 del workflow).
Carga un modelo entrenado (ej: runs/coc_v1/weights/best.pt) e infiere sobre las capturas
en 'datasets/coc/images/raw', guardando los archivos .txt en formato YOLO para su
revisión y corrección rápida en LabelImg, Label Studio, CVAT o tools/review_prelabels.py.
"""
import argparse
import glob
import os
import sys
from pathlib import Path
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import get_logger

log = get_logger("prelabel_model")


def main():
    parser = argparse.ArgumentParser(description="Pre-etiquetar imágenes con modelo YOLOv8 entrenado")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/coc_v1/weights/best.pt",
        help="Ruta a los pesos del modelo YOLO (default: runs/coc_v1/weights/best.pt)"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="datasets/coc/images/raw",
        help="Directorio con imágenes crudas (default: datasets/coc/images/raw)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="datasets/coc/labels_pre",
        help="Directorio destino para etiquetas .txt (default: datasets/coc/labels_pre)"
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Tamaño de imagen para inferencia (default: 1280)")
    parser.add_argument("--conf", type=float, default=0.20, help="Umbral de confianza para pre-etiquetar (default: 0.20)")
    parser.add_argument("--iou", type=float, default=0.50, help="Umbral IoU para NMS (default: 0.50)")
    parser.add_argument(
        "--filter-hud",
        action="store_true",
        default=True,
        help="Filtrar detecciones fuera del área de juego (150 < y < 900) para evitar falsos positivos en HUD"
    )
    parser.add_argument("--no-filter-hud", dest="filter_hud", action="store_false")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        log.critical("Modelo no encontrado en: %s", args.model)
        log.info(
            "Recuerda que para la Ronda 1 debes etiquetar a mano ~12 imágenes y entrenar un modelo rápido con:"
            "\n  python tools/train_yolo.py --epochs 40 --batch 8 --imgsz 1280 --name coc_v1"
        )
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        log.critical("No se pudo importar 'ultralytics': %s", e)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    images = sorted(glob.glob(os.path.join(args.src, "*.png")) + glob.glob(os.path.join(args.src, "*.jpg")))

    if not images:
        log.warning("No se encontraron imágenes en: %s", args.src)
        sys.exit(0)

    log.info("Cargando modelo: %s...", args.model)
    model = YOLO(args.model)

    log.info(
        "Procesando %d imágenes con imgsz=%d, conf=%.2f, iou=%.2f, filter_hud=%s...",
        len(images), args.imgsz, args.conf, args.iou, args.filter_hud
    )

    total_boxes = 0
    class_counts = {}

    for idx, p in enumerate(images, 1):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]

        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]

        lines = []
        for (a, b, c, d), k in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            cls_id = int(k)
            cx_px = (a + c) / 2.0
            cy_px = (b + d) / 2.0
            bw_px = c - a
            bh_px = d - b

            # Filtro de área de juego (HUD superior e inferior)
            if args.filter_hud:
                # Proporcional a la altura de la imagen (~150 a ~900 px en 1080p)
                min_y = 0.138 * h
                max_y = 0.833 * h
                if cy_px < min_y or cy_px > max_y:
                    continue

            cx_norm = cx_px / w
            cy_norm = cy_px / h
            w_norm = bw_px / w
            h_norm = bh_px / h

            lines.append(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            total_boxes += 1

        base_name = os.path.splitext(os.path.basename(p))[0]
        out_file = os.path.join(args.out, f"{base_name}.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        log.info("  [%d/%d] %-35s -> %d cajas pre-etiquetadas", idx, len(images), os.path.basename(p), len(lines))

    log.info("=" * 60)
    log.info("PRE-ETIQUETADO COMPLETADO:")
    log.info("  Total imágenes procesadas : %d", len(images))
    log.info("  Total cajas generadas     : %d", total_boxes)
    log.info("  Desglose por clase ID     : %s", class_counts)
    log.info("  Directorio de etiquetas   : %s", os.path.abspath(args.out))
    log.info("=" * 60)
    log.info("Siguiente paso (Ronda 2): Abre estas etiquetas con tools/review_prelabels.py o LabelImg/Label Studio para corregir cajas.")


if __name__ == "__main__":
    main()
