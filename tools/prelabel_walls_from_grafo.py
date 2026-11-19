"""
tools/prelabel_walls_from_grafo.py
Script CLI standalone para auto-preetiquetado de muros en Clash of Clans
generando un dataset en formato YOLOv8 Segmentation (orientado a fine-tuning
con SAM 2 / X-AnyLabeling y entrenamiento en ultralytics).

Pipeline:
1) Enmascara el HUD de batalla (y >= 888 y bordes superiores/laterales).
2) Usa get_deploy_zone_detector() (models/deploy_zone_finder.onnx) si está
   disponible para suprimir el césped exterior (mask == 1 mantiene la aldea).
3) Usa get_detector() (models/yolov8s.onnx) para borrar el interior de las cajas
   de edificios (storage_*, collector_*, town_hall, erosionadas 5 px) y eliminar
   falsos bordes de techos.
4) Extrae los segmentos isométricos (36° y 144°) combinando gradiente en espacio
   de color Lab (canales L y b, resistente a aldeas de nieve y muros oscuros) con
   cierre morfológico direccional para unir tramos conectados (no piezas sueltas de 1x1).

Salidas:
- datasets/coc_walls/images/train/ y datasets/coc_walls/labels/train/
- datasets/coc_walls/images/val/ y datasets/coc_walls/labels/val/ (si val existe)
- datasets/coc_walls/data.yaml (nc: 1, names: ['wall'])
- scratch/wall_prelabels_preview/ (imágenes con polígonos dibujados)
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

# Configurar path raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import get_logger
from vision.wall_compartments import get_battle_hud_mask

log = get_logger("prelabel_walls")


def _angle_diff_180(ang1: float, ang2: float) -> float:
    """Diferencia angular mínima en el rango [0, 180)."""
    diff = abs((ang1 - ang2) % 180.0)
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def make_line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """
    Crea un kernel lineal binario orientado al ángulo especificado.
    En coordenadas de imagen, y crece hacia abajo.
    """
    rad = math.radians(angle_deg)
    half = length // 2
    size = max(length, 3)
    if size % 2 == 0:
        size += 1
    cx = size // 2
    cy = size // 2
    kernel = np.zeros((size, size), dtype=np.uint8)
    dx = math.cos(rad) * half
    dy = math.sin(rad) * half
    pt1 = (int(round(cx - dx)), int(round(cy - dy)))
    pt2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(kernel, pt1, pt2, 1, thickness=1)
    return kernel


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    """Carga cajas de edificios en píxeles (x1, y1, x2, y2) desde label YOLO si existe."""
    boxes: List[Tuple[int, int, int, int]] = []
    if not label_path.is_file():
        return boxes
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, bw, bh = (
                        float(parts[1]) * img_w,
                        float(parts[2]) * img_h,
                        float(parts[3]) * img_w,
                        float(parts[4]) * img_h,
                    )
                    x1 = int(round(xc - bw / 2.0))
                    y1 = int(round(yc - bh / 2.0))
                    x2 = int(round(xc + bw / 2.0))
                    y2 = int(round(yc + bh / 2.0))
                    boxes.append((x1, y1, x2, y2))
    except Exception as e:
        log.warning("No se pudo leer archivo de etiquetas %s: %s", label_path, e)
    return boxes


def extract_wall_polygons_from_image(
    img: np.ndarray,
    dz_detector: Optional[object] = None,
    yolo_detector: Optional[object] = None,
    gt_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    min_area: float = 120.0,
    epsilon: float = 2.0,
    iso_tol: float = 7.0,
    line_thickness: int = 6,
    kernel_len: int = 25,
    hough_thresh: int = 25,
    hough_min_len: int = 25,
    hough_max_gap: int = 12,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Ejecuta el pipeline completo de extracción limpia de muros sobre una captura.
    Retorna (lista_de_poligonos_en_pixeles, mascara_binaria_de_muros).
    """
    h, w = img.shape[:2]

    # 1. Gradientes en espacio de color Lab (canales L y b)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L = cv2.GaussianBlur(lab[:, :, 0], (3, 3), 0)
    b = cv2.GaussianBlur(lab[:, :, 2], (3, 3), 0)

    edges_L = cv2.Canny(L, 50, 150)
    edges_b = cv2.Canny(b, 30, 90)
    edges_lab = cv2.bitwise_or(edges_L, edges_b)

    # 2. Paso 1: Enmascarar HUD de batalla (y >= 888 y bordes superiores/laterales)
    hud_mask = get_battle_hud_mask(h, w)
    y_hud_limit = int(888.0 * (h / 1080.0))
    hud_mask[y_hud_limit:, :] = 255
    hud_mask[: int(70.0 * (h / 1080.0)), :] = 255
    hud_mask[:, : int(40.0 * (w / 2400.0))] = 255
    hud_mask[:, int((2400.0 - 40.0) * (w / 2400.0)) :] = 255
    edges_lab[hud_mask > 0] = 0

    # 3. Paso 2: Usar get_deploy_zone_detector() para suprimir césped exterior (mask == 1)
    if dz_detector is not None and getattr(dz_detector, "is_ready", lambda: False)():
        try:
            dz_mask = dz_detector.predict_mask(img)
            if dz_mask is not None:
                # El interior es 1, el césped exterior es 0. Suprimir donde no sea aldea.
                edges_lab[dz_mask != 1] = 0
        except Exception as e:
            log.warning("Fallo al predecir zona de despliegue: %s", e)

    # 4. Paso 3: Usar get_detector() para borrar interior de edificios (erodados 5px)
    building_boxes: List[Tuple[int, int, int, int]] = []
    if yolo_detector is not None and getattr(yolo_detector, "is_ready", lambda: False)():
        try:
            dets = yolo_detector.detect(img)
            for d in dets:
                cls_name = getattr(d, "cls_name", "")
                if any(
                    cls_name.startswith(p)
                    for p in ("storage_", "collector_", "town_hall", "class_")
                ):
                    building_boxes.append(d.bbox_xyxy)
        except Exception as e:
            log.warning("Fallo en detección YOLO: %s", e)

    # Añadir también cajas GT si existen
    if gt_boxes:
        building_boxes.extend(gt_boxes)

    for bx1, by1, bx2, by2 in building_boxes:
        ex1, ey1 = max(0, bx1 + 5), max(0, by1 + 5)
        ex2, ey2 = min(w, bx2 - 5), min(h, by2 - 5)
        if ex2 > ex1 and ey2 > ey1:
            edges_lab[ey1:ey2, ex1:ex2] = 0

    # 5. Paso 4: Extraer segmentos isométricos (36° y 144°)
    lines = cv2.HoughLinesP(
        edges_lab,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=hough_min_len,
        maxLineGap=hough_max_gap,
    )

    mask_36 = np.zeros((h, w), dtype=np.uint8)
    mask_144 = np.zeros((h, w), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
            d1 = _angle_diff_180(deg, 36.0)
            d2 = _angle_diff_180(deg, 144.0)

            if d1 <= iso_tol:
                cv2.line(mask_36, (x1, y1), (x2, y2), 255, thickness=int(line_thickness))
            elif d2 <= iso_tol:
                cv2.line(mask_144, (x1, y1), (x2, y2), 255, thickness=int(line_thickness))

    # Cierre morfológico direccional para unir tramos conectados
    k_36 = make_line_kernel(kernel_len, 36.0)
    k_144 = make_line_kernel(kernel_len, 144.0)
    closed_36 = cv2.morphologyEx(mask_36, cv2.MORPH_CLOSE, k_36)
    closed_144 = cv2.morphologyEx(mask_144, cv2.MORPH_CLOSE, k_144)

    # Extraer contornos separados en 36° y 144° para preservar tramos lineales independientes
    cnts_36, _ = cv2.findContours(closed_36, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_144, _ = cv2.findContours(closed_144, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_polygons: List[np.ndarray] = []
    for cnt in list(cnts_36) + list(cnts_144):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        poly = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
        if len(poly) < 3:
            continue
        if cv2.contourArea(poly) < min_area:
            continue
        valid_polygons.append(poly)

    wall_mask_combined = cv2.bitwise_or(closed_36, closed_144)
    return valid_polygons, wall_mask_combined


def polygon_to_yolo_seg_line(poly: np.ndarray, w: int, h: int, class_id: int = 0) -> str:
    """Convierte un contorno aproximado en una línea normalizada YOLOv8 Seg: '0 x1 y1 x2 y2 ...'"""
    coords: List[str] = []
    for pt in poly[:, 0, :]:
        x_norm = max(0.0, min(1.0, float(pt[0]) / float(w)))
        y_norm = max(0.0, min(1.0, float(pt[1]) / float(h)))
        coords.append(f"{x_norm:.6f} {y_norm:.6f}")
    return f"{class_id} " + " ".join(coords)


def render_preview_image(img: np.ndarray, polygons: List[np.ndarray]) -> np.ndarray:
    """Dibuja los polígonos sobre una copia de la imagen original para verificación visual."""
    preview = img.copy()
    overlay = img.copy()

    for poly in polygons:
        cv2.fillPoly(overlay, [poly], color=(0, 255, 0))
        cv2.polylines(preview, [poly], isClosed=True, color=(0, 230, 255), thickness=2)

    cv2.addWeighted(overlay, 0.25, preview, 0.75, 0, preview)
    return preview


def create_dataset_yaml(output_dir: Path) -> Path:
    """Genera datasets/coc_walls/data.yaml compatible con ultralytics."""
    yaml_path = output_dir / "data.yaml"
    # Usar path relativo respecto al directorio de trabajo o dataset
    content = (
        "path: ./datasets/coc_walls\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['wall']\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info("Archivo data.yaml generado en %s", yaml_path)
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-preetiquetado de muros de Clash of Clans en formato YOLOv8 Segmentation."
    )
    parser.add_argument(
        "--train-images",
        type=str,
        default="datasets/coc/images/train",
        help="Directorio de imágenes de entrenamiento.",
    )
    parser.add_argument(
        "--val-images",
        type=str,
        default="datasets/coc/images/val",
        help="Directorio de imágenes de validación (opcional).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/coc_walls",
        help="Directorio destino del dataset generado.",
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        default="scratch/wall_prelabels_preview",
        help="Directorio donde guardar imágenes de verificación.",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="datasets/coc/labels",
        help="Directorio de etiquetas GT de edificios existentes.",
    )
    parser.add_argument(
        "--unseen-dir",
        type=str,
        default="logs/yolo/test_split_unseen",
        help="Directorio del conjunto ciego a ignorar estrictamente.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=120.0,
        help="Área mínima en píxeles para conservar un polígono.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
        help="Tolerancia epsilon para cv2.approxPolyDP.",
    )
    parser.add_argument(
        "--iso-tol",
        type=float,
        default=7.0,
        help="Tolerancia angular en grados alrededor de 36° y 144°.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=6,
        help="Grosor en píxeles al dibujar los segmentos de muro.",
    )
    parser.add_argument(
        "--kernel-len",
        type=int,
        default=25,
        help="Longitud del kernel direccional para cierre morfológico.",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Omitir generación del split de validación.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t_start = time.time()

    train_dir = PROJECT_ROOT / args.train_images
    val_dir = PROJECT_ROOT / args.val_images
    out_dir = PROJECT_ROOT / args.output_dir
    preview_dir = PROJECT_ROOT / args.preview_dir
    labels_dir = PROJECT_ROOT / args.labels_dir
    unseen_dir = PROJECT_ROOT / args.unseen_dir

    if not train_dir.is_dir():
        log.error("Directorio de imágenes de entrenamiento no existe: %s", train_dir)
        return 1

    # RESTRICCIÓN CRÍTICA: Identificar archivos del conjunto ciego para NUNCA procesarlos
    unseen_filenames: Set[str] = set()
    if unseen_dir.is_dir():
        try:
            unseen_filenames = {p.name for p in unseen_dir.iterdir() if p.is_file()}
            log.info("Conjunto ciego protegido detectado: %d archivos registrados para exclusión estricta.", len(unseen_filenames))
        except Exception as e:
            log.warning("No se pudo leer directorio de conjunto ciego: %s", e)

    # Preparar directorios de salida
    preview_dir.mkdir(parents=True, exist_ok=True)
    images_out_train = out_dir / "images" / "train"
    labels_out_train = out_dir / "labels" / "train"
    images_out_val = out_dir / "images" / "val"
    labels_out_val = out_dir / "labels" / "val"

    images_out_train.mkdir(parents=True, exist_ok=True)
    labels_out_train.mkdir(parents=True, exist_ok=True)
    if not args.skip_val and val_dir.is_dir():
        images_out_val.mkdir(parents=True, exist_ok=True)
        labels_out_val.mkdir(parents=True, exist_ok=True)

    # Cargar detectores una sola vez
    log.info("Cargando detectores ONNX (DeployZoneDetector y YoloDetector)...")
    from vision.deploy_zone_detector import get_deploy_zone_detector
    from vision.yolo_detector import get_detector

    dz_detector = get_deploy_zone_detector()
    yolo_detector = get_detector()

    # Splits a procesar
    splits_to_process: List[Tuple[str, Path, Path, Path]] = [
        ("train", train_dir, images_out_train, labels_out_train),
    ]
    if not args.skip_val and val_dir.is_dir():
        splits_to_process.append(("val", val_dir, images_out_val, labels_out_val))

    total_images_processed = 0
    total_polygons_generated = 0
    stats_per_split: dict[str, dict] = {}

    for split_name, in_split_dir, out_img_dir, out_lbl_dir in splits_to_process:
        img_files = sorted(
            [p for p in in_split_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        log.info("Iniciando procesamiento split '%s' (%d imágenes encontradas)...", split_name, len(img_files))

        split_img_count = 0
        split_poly_count = 0

        for img_path in img_files:
            # RESTRICCIÓN CRÍTICA: Saltear si está en el conjunto ciego
            if img_path.name in unseen_filenames:
                log.warning("RESTRICCIÓN CIEGA: Omitiendo %s presente en test_split_unseen.", img_path.name)
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("No se pudo cargar la imagen: %s", img_path)
                continue

            h, w = img.shape[:2]

            # Cargar cajas GT de edificios si existen en datasets/coc/labels/<split>/
            gt_file = labels_dir / split_name / f"{img_path.stem}.txt"
            gt_boxes = load_gt_boxes(gt_file, w, h) if gt_file.is_file() else []

            # Extraer polígonos de muro
            polygons, wall_mask = extract_wall_polygons_from_image(
                img=img,
                dz_detector=dz_detector,
                yolo_detector=yolo_detector,
                gt_boxes=gt_boxes,
                min_area=args.min_area,
                epsilon=args.epsilon,
                iso_tol=args.iso_tol,
                line_thickness=args.line_thickness,
                kernel_len=args.kernel_len,
            )

            # Copiar imagen a datasets/coc_walls/images/<split>/
            dest_img = out_img_dir / img_path.name
            shutil.copy2(img_path, dest_img)

            # Escribir archivo de etiquetas normalizado YOLOv8 Seg
            dest_lbl = out_lbl_dir / f"{img_path.stem}.txt"
            with open(dest_lbl, "w", encoding="utf-8") as f:
                for poly in polygons:
                    line = polygon_to_yolo_seg_line(poly, w, h, class_id=0)
                    f.write(line + "\n")

            # Generar preview de verificación en scratch/
            if preview_dir is not None:
                preview = render_preview_image(img, polygons)
                preview_path = preview_dir / f"{img_path.stem}_preview.png"
                cv2.imwrite(str(preview_path), preview)

            split_img_count += 1
            split_poly_count += len(polygons)

            log.info(
                "[%s] %s: %d polígonos de muro generados (res: %dx%d)",
                split_name,
                img_path.name,
                len(polygons),
                w,
                h,
            )

        stats_per_split[split_name] = {
            "images": split_img_count,
            "polygons": split_poly_count,
            "avg_per_image": round(split_poly_count / max(1, split_img_count), 1),
        }
        total_images_processed += split_img_count
        total_polygons_generated += split_poly_count

    # Generar data.yaml
    create_dataset_yaml(out_dir)

    elapsed = time.time() - t_start
    log.info("=================================================================")
    log.info("Auto-preetiquetado de muros completado exitosamente en %.1f s", elapsed)
    log.info("Total imágenes procesadas: %d", total_images_processed)
    log.info("Total polígonos de muro generados: %d", total_polygons_generated)
    for s_name, s_data in stats_per_split.items():
        log.info(
            "  - Split '%s': %d imágenes, %d polígonos (promedio %.1f por imagen)",
            s_name,
            s_data["images"],
            s_data["polygons"],
            s_data["avg_per_image"],
        )
    log.info("Dataset generado en: %s", out_dir)
    log.info("Previsualizaciones guardadas en: %s", preview_dir)
    log.info("=================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
