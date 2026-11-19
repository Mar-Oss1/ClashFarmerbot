"""
tools/calibrate_jump_footprint.py
Genera comparaciones visuales lado a lado sobre aldeas reales de Clash of Clans:
- Círculo R=90px (definido en config/settings.json, radio equivalente de área)
- Círculo R=125px (fallback de depuración en _save_yolo_attack_debug y test_jump_overlay)
- Elipse isométrica Rx=115px, Ry=65px (huella real proyectada en elevación ~30°-35°)

Guarda overlays detallados y paneles comparativos en logs/yolo/calibration_footprint_*.png.
"""
import argparse
import glob
import math
import os
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Configurar path raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import SETTINGS
from core.logger import get_logger
from vision.yolo_detector import YoloDetector
from vision.wall_raycast import detect_global_wall_segments, estimate_iso_angles
from attack.deployment_strategy import compute_adaptive_jump_targets_from_detections

log = get_logger("calibrate_jump_footprint")

# Colores BGR para cada huella
COLOR_CIRCLE_90 = (0, 0, 255)       # Rojo puro para 90px
COLOR_CIRCLE_125 = (0, 255, 255)    # Amarillo brillante para 125px
COLOR_ELLIPSE_ISO = (0, 255, 0)     # Verde lima brillante para Elipse Isométrica
COLOR_WALL = (160, 160, 160)        # Gris para muros
COLOR_TARGET = (255, 120, 0)        # Naranja/Cyan para centro de salto


def calculate_footprint_metrics(
    r_circle_small: float = 90.0,
    r_circle_large: float = 125.0,
    rx_ellipse: float = 115.0,
    ry_ellipse: float = 65.0
) -> dict:
    """
    Calcula áreas y métricas geométricas comparativas de las tres huellas.
    """
    area_90 = math.pi * (r_circle_small ** 2)
    area_125 = math.pi * (r_circle_large ** 2)
    area_ellipse = math.pi * rx_ellipse * ry_ellipse
    equiv_radius_ellipse = math.sqrt(rx_ellipse * ry_ellipse)
    aspect_ratio = rx_ellipse / ry_ellipse if ry_ellipse > 0 else 1.0

    return {
        "r_circle_small": r_circle_small,
        "area_circle_small_px2": area_90,
        "r_circle_large": r_circle_large,
        "area_circle_large_px2": area_125,
        "rx_ellipse": rx_ellipse,
        "ry_ellipse": ry_ellipse,
        "area_ellipse_px2": area_ellipse,
        "equiv_radius_ellipse": equiv_radius_ellipse,
        "aspect_ratio": aspect_ratio,
        "area_ratio_125_vs_ellipse": area_125 / area_ellipse if area_ellipse > 0 else 0.0,
        "area_ratio_90_vs_ellipse": area_90 / area_ellipse if area_ellipse > 0 else 0.0,
    }


def draw_legend(
    img: np.ndarray,
    metrics: dict,
    start_x: int = 40,
    start_y: int = 60
) -> None:
    """Dibuja una leyenda explicativa con métricas geométricas en la imagen."""
    box_w = 700
    box_h = 210
    overlay = img.copy()
    cv2.rectangle(overlay, (start_x - 15, start_y - 35), (start_x + box_w, start_y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (start_x - 15, start_y - 35), (start_x + box_w, start_y + box_h), (120, 120, 120), 1)

    cv2.putText(
        img,
        "CALIBRACION DE HUELLA DE HECHIZO DE SALTO (FASE 1.2)",
        (start_x, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    items = [
        (
            COLOR_CIRCLE_90,
            f"Circulo R={int(metrics['r_circle_small'])}px (settings.json default) | Area: {int(metrics['area_circle_small_px2']):,} px2 (Radio equiv: {metrics['r_circle_small']:.1f}px)"
        ),
        (
            COLOR_CIRCLE_125,
            f"Circulo R={int(metrics['r_circle_large'])}px (Debug fallback) | Area: {int(metrics['area_circle_large_px2']):,} px2 (+{((metrics['area_ratio_125_vs_ellipse'] - 1) * 100):.1f}% vs real)"
        ),
        (
            COLOR_ELLIPSE_ISO,
            f"Elipse Isometrica Rx={int(metrics['rx_ellipse'])}px, Ry={int(metrics['ry_ellipse'])}px (Aspect={metrics['aspect_ratio']:.2f}) | Area: {int(metrics['area_ellipse_px2']):,} px2 (Req={metrics['equiv_radius_ellipse']:.1f}px)"
        ),
    ]

    for i, (col, text) in enumerate(items):
        y_pos = start_y + 40 + i * 40
        cv2.line(img, (start_x, y_pos - 5), (start_x + 35, y_pos - 5), col, 3)
        cv2.circle(img, (start_x + 17, y_pos - 5), 4, col, -1)
        cv2.putText(
            img,
            text,
            (start_x + 50, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )

    note = "Nota: En vista isometrica (30 deg elev), el circulo en suelo se proyecta como elipse horizontal."
    cv2.putText(
        img,
        note,
        (start_x, start_y + 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def generate_side_by_side_comparison(
    image_path: str,
    output_dir: str = "logs/yolo",
    detector: Optional[YoloDetector] = None,
    rx_px: float = 115.0,
    ry_px: float = 65.0,
    r_small_px: float = 90.0,
    r_large_px: float = 125.0
) -> Optional[str]:
    """
    Genera una imagen comparativa lado a lado para una aldea específica:
    - Panel Izquierdo: Vista global con muros, detecciones y los 3 overlays superpuestos.
    - Panel Derecho: 3 sub-paneles con zoom detallado en el punto de salto principal:
      1. R=90px (muestra brechas horizontales sin cubrir)
      2. R=125px (muestra falso positivo en dirección vertical)
      3. Elipse Isométrica (cobertura conforme precisa)
    """
    if not os.path.exists(image_path):
        log.warning("Imagen no encontrada: %s", image_path)
        return None

    img = cv2.imread(image_path)
    if img is None:
        log.warning("Fallo al decodificar imagen: %s", image_path)
        return None

    h, w = img.shape[:2]
    parent_dir = os.path.basename(os.path.dirname(image_path))
    raw_base = os.path.splitext(os.path.basename(image_path))[0]
    if raw_base == "screen_raw":
        base_name = f"{parent_dir}_{raw_base}"
    else:
        base_name = raw_base
    os.makedirs(output_dir, exist_ok=True)

    metrics = calculate_footprint_metrics(r_small_px, r_large_px, rx_px, ry_px)

    # 1. Detección YOLO
    detections = []
    if detector is not None:
        try:
            detections = detector.detect(img)
        except Exception as e:
            log.warning("Error en inferencia YOLO para %s: %s", base_name, e)

    # 2. Detección de muros globales
    _, wall_segments = detect_global_wall_segments(img, force_refresh=True)

    # 3. Puntos de salto candidatos
    jump_points = []
    if detections:
        try:
            raw_targets = compute_adaptive_jump_targets_from_detections(detections, (w, h), screen=img)
            for t in raw_targets:
                pt = t[1] if (isinstance(t, (tuple, list)) and len(t) == 2 and isinstance(t[0], str)) else t
                jump_points.append((int(round(pt[0])), int(round(pt[1]))))
        except Exception as e:
            log.warning("Error en compute_adaptive_jump_targets: %s", e)

    # Fallback si no hay puntos por YOLO
    if not jump_points:
        if wall_segments:
            # Seleccionar el punto medio del segmento más cercano al centro
            center_x, center_y = w // 2, h // 2
            dists = [
                ((s[0] + s[2]) // 2 - center_x) ** 2 + ((s[1] + s[3]) // 2 - center_y) ** 2
                for s in wall_segments
            ]
            best_idx = int(np.argmin(dists))
            s = wall_segments[best_idx]
            jump_points.append(((s[0] + s[2]) // 2, (s[1] + s[3]) // 2))
        else:
            jump_points.append((w // 2, h // 2))

    # Limitar a máximo 3 saltos para el overlay principal
    primary_jumps = jump_points[:3]
    primary_pt = primary_jumps[0]
    pjx, pjy = primary_pt

    # Construir Panel Izquierdo: Vista global con los 3 overlays
    panel_left = img.copy()

    # Dibujar segmentos de muro en gris tenue
    for s in wall_segments:
        cv2.line(panel_left, (s[0], s[1]), (s[2], s[3]), COLOR_WALL, 1, cv2.LINE_AA)

    # Dibujar huellas en los puntos de salto
    for jx, jy in primary_jumps:
        # Círculo 90px
        cv2.circle(panel_left, (jx, jy), int(r_small_px), COLOR_CIRCLE_90, 2, cv2.LINE_AA)
        # Círculo 125px
        cv2.circle(panel_left, (jx, jy), int(r_large_px), COLOR_CIRCLE_125, 2, cv2.LINE_AA)
        # Elipse Isométrica (major axis horizontal = 0 deg)
        cv2.ellipse(
            panel_left,
            (jx, jy),
            (int(rx_px), int(ry_px)),
            0.0,
            0.0,
            360.0,
            COLOR_ELLIPSE_ISO,
            2,
            cv2.LINE_AA
        )
        # Centro
        cv2.circle(panel_left, (jx, jy), 4, COLOR_TARGET, -1)
        cv2.circle(panel_left, (jx, jy), 6, (0, 0, 0), 1)

    # Dibuja leyenda
    draw_legend(panel_left, metrics)

    # Construir Panel Derecho: Zoom Comparativo
    # Región de interés alrededor del salto principal
    crop_radius = int(r_large_px + 70)
    x1 = max(0, pjx - crop_radius)
    y1 = max(0, pjy - crop_radius)
    x2 = min(w, pjx + crop_radius)
    y2 = min(h, pjy + crop_radius)

    crop_base = img[y1:y2, x1:x2].copy()
    crop_walls = [
        (s[0] - x1, s[1] - y1, s[2] - x1, s[3] - y1)
        for s in wall_segments
        if (x1 <= s[0] <= x2 or x1 <= s[2] <= x2) and (y1 <= s[1] <= y2 or y1 <= s[3] <= y2)
    ]
    rel_jx = pjx - x1
    rel_jy = pjy - y1

    # Sub-panel 1: Círculo 90px
    crop_90 = crop_base.copy()
    for s in crop_walls:
        cv2.line(crop_90, (s[0], s[1]), (s[2], s[3]), COLOR_WALL, 2, cv2.LINE_AA)
    cv2.circle(crop_90, (rel_jx, rel_jy), int(r_small_px), COLOR_CIRCLE_90, 3, cv2.LINE_AA)
    cv2.circle(crop_90, (rel_jx, rel_jy), 5, COLOR_TARGET, -1)
    cv2.putText(
        crop_90,
        f"A: Circulo R={int(r_small_px)}px (Subestima horizontal)",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_CIRCLE_90,
        2,
        cv2.LINE_AA,
    )

    # Sub-panel 2: Círculo 125px
    crop_125 = crop_base.copy()
    for s in crop_walls:
        cv2.line(crop_125, (s[0], s[1]), (s[2], s[3]), COLOR_WALL, 2, cv2.LINE_AA)
    cv2.circle(crop_125, (rel_jx, rel_jy), int(r_large_px), COLOR_CIRCLE_125, 3, cv2.LINE_AA)
    cv2.circle(crop_125, (rel_jx, rel_jy), 5, COLOR_TARGET, -1)
    cv2.putText(
        crop_125,
        f"B: Circulo R={int(r_large_px)}px (Sobreestima vertical +93%)",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_CIRCLE_125,
        2,
        cv2.LINE_AA,
    )

    # Sub-panel 3: Elipse Isométrica
    crop_ellipse = crop_base.copy()
    for s in crop_walls:
        cv2.line(crop_ellipse, (s[0], s[1]), (s[2], s[3]), COLOR_WALL, 2, cv2.LINE_AA)
    cv2.ellipse(
        crop_ellipse,
        (rel_jx, rel_jy),
        (int(rx_px), int(ry_px)),
        0.0,
        0.0,
        360.0,
        COLOR_ELLIPSE_ISO,
        3,
        cv2.LINE_AA,
    )
    cv2.circle(crop_ellipse, (rel_jx, rel_jy), 5, COLOR_TARGET, -1)
    cv2.putText(
        crop_ellipse,
        f"C: Elipse Rx={int(rx_px)}px, Ry={int(ry_px)}px (Conforme exacta)",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_ELLIPSE_ISO,
        2,
        cv2.LINE_AA,
    )

    # Redimensionar crops para apilarlos verticalmente en el panel derecho
    target_crop_w = int(w * 0.40)
    target_crop_h = int(h / 3.0)

    c90_res = cv2.resize(crop_90, (target_crop_w, target_crop_h), interpolation=cv2.INTER_AREA)
    c125_res = cv2.resize(crop_125, (target_crop_w, target_crop_h), interpolation=cv2.INTER_AREA)
    cel_res = cv2.resize(crop_ellipse, (target_crop_w, target_crop_h), interpolation=cv2.INTER_AREA)

    panel_right = np.vstack([c90_res, c125_res, cel_res])
    # Ajustar altura de panel derecho si hay discrepancia por redondeo
    if panel_right.shape[0] != h:
        panel_right = cv2.resize(panel_right, (target_crop_w, h), interpolation=cv2.INTER_AREA)

    # Línea divisoria vertical
    separator = np.zeros((h, 4, 3), dtype=np.uint8)
    separator[:] = (100, 100, 100)

    # Componer imagen final lado a lado
    composite = np.hstack([panel_left, separator, panel_right])

    out_path = os.path.join(output_dir, f"calibration_footprint_{base_name}.png")
    cv2.imwrite(out_path, composite)
    log.info("Overlay de calibracion guardado en: %s", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Calibración de huella de hechizo de salto (Fase 1.2)")
    parser.add_argument("--samples", type=int, default=5, help="Número de aldeas de muestra a calibrar")
    parser.add_argument("--all", action="store_true", help="Procesar todas las aldeas disponibles")
    parser.add_argument("--output-dir", type=str, default="logs/yolo", help="Directorio destino para overlays")
    parser.add_argument("--rx", type=float, default=115.0, help="Radio semi-mayor horizontal en px")
    parser.add_argument("--ry", type=float, default=65.0, help="Radio semi-menor vertical en px")
    parser.add_argument("--r-small", type=float, default=90.0, help="Radio de círculo pequeño en px")
    parser.add_argument("--r-large", type=float, default=125.0, help="Radio de círculo grande en px")
    args = parser.parse_args()

    raw_dir = os.path.join(PROJECT_ROOT, "datasets", "coc", "images", "raw")
    attack_raw_pattern = os.path.join(PROJECT_ROOT, "logs", "yolo", "attack_*", "screen_raw.png")

    raw_images = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
    attack_images = sorted(glob.glob(attack_raw_pattern))
    all_images = raw_images + attack_images

    log.info("Imágenes encontradas: %d en raw, %d en attack logs (Total: %d)", len(raw_images), len(attack_images), len(all_images))

    if not all_images:
        log.error("No se encontraron imágenes para calibrar.")
        sys.exit(1)

    # Selección de muestras representativas si no es --all
    if args.all:
        selected_images = all_images
    else:
        step = max(1, len(all_images) // args.samples)
        selected_images = all_images[::step][:args.samples]

    log.info("Procesando %d aldeas seleccionadas para calibración...", len(selected_images))

    # Inicializar detector YOLO
    detector = None
    try:
        attack_cfg = SETTINGS.get("attack", {})
        conf = float(attack_cfg.get("yolo_conf", 0.40))
        imgsz = int(attack_cfg.get("yolo_imgsz", 1280))
        detector = YoloDetector(conf=conf, imgsz=imgsz)
        log.info("YoloDetector inicializado exitosamente.")
    except Exception as e:
        log.warning("No se pudo inicializar YoloDetector: %s (se utilizará fallback geométrico)", e)

    metrics = calculate_footprint_metrics(args.r_small, args.r_large, args.rx, args.ry)
    log.info("==================================================")
    log.info("METRICAS DE CALIBRACION DE HUELLA:")
    log.info("  Circulo 90px: Area = %.1f px2, Radio equiv = %.1f px", metrics["area_circle_small_px2"], metrics["r_circle_small"])
    log.info("  Circulo 125px: Area = %.1f px2 (+%.1f%% vs elipse)", metrics["area_circle_large_px2"], (metrics["area_ratio_125_vs_ellipse"] - 1) * 100)
    log.info("  Elipse Isométrica: Rx = %.1f px, Ry = %.1f px, Aspect = %.2f, Area = %.1f px2, Req = %.1f px",
             metrics["rx_ellipse"], metrics["ry_ellipse"], metrics["aspect_ratio"], metrics["area_ellipse_px2"], metrics["equiv_radius_ellipse"])
    log.info("==================================================")

    generated = []
    for img_path in selected_images:
        out_path = generate_side_by_side_comparison(
            image_path=img_path,
            output_dir=args.output_dir,
            detector=detector,
            rx_px=args.rx,
            ry_px=args.ry,
            r_small_px=args.r_small,
            r_large_px=args.r_large
        )
        if out_path:
            generated.append(out_path)

    log.info("Calibración completada. %d overlays generados en %s.", len(generated), args.output_dir)


if __name__ == "__main__":
    main()
