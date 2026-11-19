"""
tools/prelabel_hsv.py
Script CLI para pre-anotar imágenes crudas con formato YOLO v8 utilizando
la detección por colorimetría HSV (StorageDetector.detect_boxes).
"""
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Asegurar importación del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.storage_detector import StorageDetector
from core.logger import get_logger

log = get_logger("prelabel_hsv")

CLASS_TO_ID = {
    "storage_gold": 0,
    "storage_elixir": 1,
    "storage_dark": 2,
}

CLASS_COLORS = {
    "storage_gold": (0, 215, 255),    # Dorado / BGR
    "storage_elixir": (255, 0, 255),  # Magenta
    "storage_dark": (128, 0, 128),    # Púrpura oscuro
}


def draw_debug_overlay(img: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Dibuja rectángulos, etiquetas y banner superior en la imagen."""
    debug_img = img.copy()
    h, w = debug_img.shape[:2]

    counts = {"storage_gold": 0, "storage_elixir": 0, "storage_dark": 0}

    for b in boxes:
        cls_name = b["cls"]
        x1, y1, x2, y2 = b["bbox_xyxy"]
        conf = b["confidence"]
        counts[cls_name] = counts.get(cls_name, 0) + 1

        color = CLASS_COLORS.get(cls_name, (0, 255, 0))
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)

        # Etiqueta con fondo
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bg_y1 = max(0, y1 - th - 6)
        cv2.rectangle(debug_img, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(debug_img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Banner superior
    banner_text = f"gold:{counts['storage_gold']}  elixir:{counts['storage_elixir']}  dark:{counts['storage_dark']}  |  total:{len(boxes)}"
    banner_h = 40
    overlay = debug_img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, debug_img, 0.25, 0, debug_img)
    cv2.putText(debug_img, banner_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return debug_img


def prelabel_images(input_dir: str, output_dir: str, debug_dir: str, min_conf: float = 0.30, overwrite: bool = False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    debug_path = Path(debug_dir) if debug_dir else None

    if not input_path.exists():
        log.error("El directorio de entrada no existe: %s", input_path)
        return

    output_path.mkdir(parents=True, exist_ok=True)
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)

    # Buscar imágenes
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    img_files = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions])

    if not img_files:
        log.warning("No se encontraron imágenes en: %s", input_path)
        return

    log.info("Iniciando pre-etiquetado HSV en %d imágenes...", len(img_files))
    detector = StorageDetector()

    total_processed = 0
    total_skipped = 0
    class_totals = {"storage_gold": 0, "storage_elixir": 0, "storage_dark": 0}
    zero_detection_images = []

    for img_file in img_files:
        label_file = output_path / f"{img_file.stem}.txt"
        if label_file.exists() and not overwrite:
            log.debug("Omitiendo %s (label ya existe y --overwrite no fue activado)", img_file.name)
            total_skipped += 1
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            log.warning("No se pudo leer imagen: %s", img_file)
            continue

        h, w = img.shape[:2]
        boxes = detector.detect_boxes(img, min_conf=min_conf)

        # Generar líneas YOLO: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
        yolo_lines = []
        counts = {"storage_gold": 0, "storage_elixir": 0, "storage_dark": 0}

        for b in boxes:
            cls_name = b["cls"]
            if cls_name not in CLASS_TO_ID:
                continue
            cls_id = CLASS_TO_ID[cls_name]
            counts[cls_name] += 1
            class_totals[cls_name] += 1

            x1, y1, x2, y2 = b["bbox_xyxy"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = (x2 - x1)
            bh = (y2 - y1)

            cx_norm = cx / w
            cy_norm = cy / h
            w_norm = bw / w
            h_norm = bh / h

            yolo_lines.append(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Guardar archivo de etiquetas (incluso si está vacío)
        with open(label_file, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

        # Guardar imagen debug si fue solicitada
        if debug_path:
            debug_img = draw_debug_overlay(img, boxes)
            debug_out_file = debug_path / f"{img_file.stem}.png"
            cv2.imwrite(str(debug_out_file), debug_img)

        total_processed += 1
        if not boxes:
            zero_detection_images.append(img_file.name)
            log.info("  [-] %-30s | 0 detecciones (requiere revisión manual)", img_file.name)
        else:
            log.info("  [+] %-30s | %2d detecciones (gold=%d, elixir=%d, dark=%d)",
                     img_file.name, len(boxes), counts["storage_gold"], counts["storage_elixir"], counts["storage_dark"])

    # Resumen final
    log.info("==================================================")
    log.info("RESUMEN DE PRE-ETIQUETADO HSV:")
    log.info("  Imágenes procesadas : %d", total_processed)
    log.info("  Imágenes omitidas   : %d", total_skipped)
    log.info("  Total storage_gold  : %d", class_totals["storage_gold"])
    log.info("  Total storage_elixir: %d", class_totals["storage_elixir"])
    log.info("  Total storage_dark  : %d", class_totals["storage_dark"])
    log.info("  Total almacenes     : %d", sum(class_totals.values()))
    log.info("  Imágenes sin cajas  : %d (%.1f%%)", len(zero_detection_images),
             (len(zero_detection_images) / total_processed * 100) if total_processed > 0 else 0)
    if zero_detection_images:
        log.info("  Lista sin cajas: %s", ", ".join(zero_detection_images[:10]) + ("..." if len(zero_detection_images) > 10 else ""))
    log.info("==================================================")


def main():
    parser = argparse.ArgumentParser(description="Pre-etiquetador HSV de almacenes para dataset YOLO")
    parser.add_argument("--input", type=str, default="datasets/coc/images/raw", help="Directorio con imágenes crudas")
    parser.add_argument("--output", type=str, default="datasets/coc/labels/raw", help="Directorio destino para etiquetas .txt")
    parser.add_argument("--debug", type=str, default="datasets/coc/images/raw_debug", help="Directorio destino para imágenes debug anotadas")
    parser.add_argument("--min-conf", type=float, default=0.30, help="Confianza mínima heurística para filtrar cajas")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir etiquetas existentes")
    args = parser.parse_args()

    prelabel_images(
        input_dir=args.input,
        output_dir=args.output,
        debug_dir=args.debug,
        min_conf=args.min_conf,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
