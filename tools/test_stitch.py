#!/usr/bin/env python3
"""
tools/test_stitch.py
Herramienta de diagnóstico y auditoría para la Costura Panorámica Milimétrica (Template Registration)
y el cálculo de puntos legales de despliegue radial sobre el mapa completo.

Uso:
  python tools/test_stitch.py --offline
  python tools/test_stitch.py --offline --center logs/pan_test_4_center.png --south logs/pan_test_1_south.png --north logs/pan_test_3_north.png
  python tools/test_stitch.py (con teléfono conectado)
"""
import os
import sys
import argparse
import time
import cv2
import numpy as np

# Asegurar path de importación
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.logger import get_logger
from vision.yolo_detector import get_detector, Detection
from attack.deployment_strategy import (
    DeploymentStrategy,
    detect_village_boundary,
    get_battle_hud_mask,
)

log = get_logger("test_stitch")


def run_stitching(center_img: np.ndarray, south_img: np.ndarray | None, north_img: np.ndarray | None):
    h, w = center_img.shape[:2]
    log.info("🖼️ Dimensiones base: %dx%d", w, h)

    detector = get_detector()

    # 1. Calcular desplazamientos exactos
    shift_south = None
    shift_north = None

    if south_img is not None:
        shift_south = DeploymentStrategy._compute_exact_camera_shift(center_img, south_img)
        if shift_south:
            log.info("📐 Desplazamiento Sur exacto calculado: dx=%d, dy=%d", shift_south[0], shift_south[1])
        else:
            log.warning("⚠️ No se pudo calcular desplazamiento para Flanco Sur.")

    if north_img is not None:
        shift_north = DeploymentStrategy._compute_exact_camera_shift(center_img, north_img)
        if shift_north:
            log.info("📐 Desplazamiento Norte exacto calculado: dx=%d, dy=%d", shift_north[0], shift_north[1])
        else:
            log.warning("⚠️ No se pudo calcular desplazamiento para Flanco Norte.")

    # 2. Lienzo Expandido (2400 x 1600)
    canvas_h, canvas_w = 1600, w
    center_y_offset = (canvas_h - h) // 2  # 260 px de margen superior
    center_x_offset = 0

    stitched = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Pegar franjas del Norte (si existe y tiene shift)
    if north_img is not None and shift_north:
        ndx, ndy = shift_north
        # coord en canvas = coord_en_center + center_y_offset = coord_en_north + ndy + center_y_offset
        # Solo incrustar la mitad superior de north_img (excluyendo HUD)
        ny1_src = int(h * 0.12)
        ny2_src = int(h * 0.50)
        ny1_dst = ny1_src + ndy + center_y_offset
        ny2_dst = ny2_src + ndy + center_y_offset

        if 0 <= ny1_dst < canvas_h and 0 <= ny2_dst <= canvas_h:
            stitched[ny1_dst:ny2_dst, :] = north_img[ny1_src:ny2_src, :]

    # Pegar franjas del Sur (si existe y tiene shift)
    if south_img is not None and shift_south:
        sdx, sdy = shift_south
        sy1_src = int(h * 0.50)
        sy2_src = int(h * 0.86)
        sy1_dst = sy1_src + sdy + center_y_offset
        sy2_dst = sy2_src + sdy + center_y_offset

        if 0 <= sy1_dst < canvas_h and 0 <= sy2_dst <= canvas_h:
            stitched[sy1_dst:sy2_dst, :] = south_img[sy1_src:sy2_src, :]

    # Pegar vista central en el medio (área jugable)
    cy1_dst = center_y_offset
    cy2_dst = center_y_offset + h
    stitched[cy1_dst:cy2_dst, :] = center_img

    # 3. Detecciones en Vista Central
    all_detections: list[Detection] = []
    if detector and detector.is_ready():
        raw_dets = detector.detect(center_img)
        h_mask = get_battle_hud_mask(h, w)
        for d in raw_dets:
            cx, cy = d.center
            if 0 <= cy < h and 0 <= cx < w and h_mask[cy, cx] == 0:
                all_detections.append(d)

    # 4. Sondeo de Línea Roja sobre Vista Central
    red_c, green_c = detect_village_boundary(center_img, detections=all_detections, mode="high_res_contour")

    # 5. Dibujar sobre el lienzo expandido
    dbg_full = stitched.copy()

    # Dibujar contorno verde en coordenadas de canvas
    if green_c is not None and len(green_c) >= 3:
        green_canvas = green_c.reshape(-1, 2).copy()
        green_canvas[:, 1] += center_y_offset
        cv2.polylines(dbg_full, [green_canvas.reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 0), 2)

    # Instancia mock/dummy de DeploymentStrategy para invocar _get_drop_point_on_contour
    dummy_deployer = DeploymentStrategy.__new__(DeploymentStrategy)

    drop_count = 0
    for d in all_detections:
        bc_x, bc_y = d.center
        x1, y1, x2, y2 = d.bbox_xyxy

        # En canvas
        cbx1, cby1 = x1, y1 + center_y_offset
        cbx2, cby2 = x2, y2 + center_y_offset
        cv2.rectangle(dbg_full, (cbx1, cby1), (cbx2, cby2), (255, 200, 0), 2)

        if green_c is not None and len(green_c) >= 3:
            dx, dy = dummy_deployer._get_drop_point_on_contour((bc_x, bc_y), green_c, extra_margin_px=32.0)
            cdx, cdy = dx, dy + center_y_offset

            # Magenta circle en drop point
            cv2.circle(dbg_full, (cdx, cdy), 7, (255, 0, 255), -1)
            # Cian arrow hacia el centro del edificio
            cv2.arrowedLine(dbg_full, (cdx, cdy), (bc_x, bc_y + center_y_offset), (255, 255, 0), 2, tipLength=0.15)
            drop_count += 1

    # Guardar resultado
    os.makedirs("logs/yolo", exist_ok=True)
    out_path = "logs/yolo/test_stitched_full_map.png"
    cv2.imwrite(out_path, dbg_full)
    log.info("✅ Mapa completo guardado en: %s (estructuras=%d, drops=%d)", out_path, len(all_detections), drop_count)


def main():
    parser = argparse.ArgumentParser(description="Test Stitching y Despliegue Milimétrico")
    parser.add_argument("--offline", action="store_true", help="Usa imágenes locales en lugar del dispositivo ADB")
    parser.add_argument("--center", default="logs/pan_test_4_center.png", help="Ruta de la captura central")
    parser.add_argument("--south", default="logs/pan_test_1_south.png", help="Ruta de la captura sur")
    parser.add_argument("--north", default="logs/pan_test_3_north.png", help="Ruta de la captura norte")
    args = parser.parse_args()

    if args.offline:
        log.info("📂 Modo Offline: Cargando capturas locales...")
        center_path = os.path.join(PROJECT_ROOT, args.center)
        south_path = os.path.join(PROJECT_ROOT, args.south)
        north_path = os.path.join(PROJECT_ROOT, args.north)

        center_img = cv2.imread(center_path)
        south_img = cv2.imread(south_path) if os.path.exists(south_path) else None
        north_img = cv2.imread(north_path) if os.path.exists(north_path) else None

        if center_img is None:
            # Fallback a screen_searching o screen_now3
            alt_centers = ["logs/screen_searching.png", "logs/screen_now3.png", "logs/test_screen.png"]
            for alt in alt_centers:
                alt_full = os.path.join(PROJECT_ROOT, alt)
                if os.path.exists(alt_full):
                    center_img = cv2.imread(alt_full)
                    log.info("Usando captura alternativa central: %s", alt)
                    break

        if center_img is None:
            log.error("❌ No se encontró ninguna imagen central válida para modo offline.")
            sys.exit(1)

        run_stitching(center_img, south_img, north_img)
    else:
        log.info("📱 Conectando al dispositivo ADB...")
        from core.device import Device
        device = Device()
        log.info("Dispositivo conectado: %s (%dx%d)", device.serial, device.width, device.height)
        device.zoom_out()
        time.sleep(0.6)
        center_img = device.screenshot()

        log.info("Haciendo swipe al Sur...")
        device.d.shell("input swipe 1200 760 1200 260 450")
        time.sleep(0.65)
        south_img = device.screenshot()

        log.info("Haciendo swipe al Norte...")
        device.d.shell("input swipe 1200 240 1200 820 500")
        time.sleep(0.65)
        north_img = device.screenshot()

        device.zoom_out()
        time.sleep(0.5)

        run_stitching(center_img, south_img, north_img)


if __name__ == "__main__":
    main()
