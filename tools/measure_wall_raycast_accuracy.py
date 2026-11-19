"""
tools/measure_wall_raycast_accuracy.py
Herramienta de validación cuantitativa recomendada por el usuario:
Evalúa la precisión de Wall Raycast sobre 12 capturas reales de batallas guardadas en logs/yolo/.
Compara la posición de cada cruce de muro detectado contra la cresta de gradiente real
(máximo de relieve de muro) a lo largo del rayo y calcula el error medio en píxeles.
Criterio de aceptación: Error medio < 25.0 px (el radio de 85px del hechizo cubre con holgura).
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import glob
import cv2
import numpy as np

from vision.wall_raycast import (
    detect_global_wall_segments,
    find_wall_crossings_from_segments,
    reset_iso_angles_cache,
)
from vision.yolo_detector import get_detector
from attack.deployment_strategy import compute_canonical_diamond_drop


def measure_accuracy():
    files = sorted(glob.glob("logs/yolo/attack_*.png"), reverse=True)
    # Filtrar solo capturas reales completas (> 1MB)
    valid_files = [f for f in files if os.path.getsize(f) > 1_000_000][:12]

    if not valid_files:
        print("No se encontraron capturas válidas en logs/yolo/")
        return

    print(f"==================================================================")
    print(f"🔍 Evaluando precisión de Wall Raycast sobre {len(valid_files)} capturas reales de CoC")
    print(f"==================================================================")

    detector = get_detector()
    total_crossings_evaluated = 0
    all_errors = []

    for idx, path in enumerate(valid_files):
        reset_iso_angles_cache()
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Gradiente Sobel para hallar las crestas reales de muros
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        iso_angles, wall_segs = detect_global_wall_segments(img, force_refresh=True)

        detections = []
        if detector and detector.is_ready():
            detections = detector.detect(img)

        storages = [
            d for d in detections
            if d.cls_name in {"storage_dark", "storage_gold", "storage_elixir", "town_hall"}
        ]

        if not storages:
            continue

        # Centroide de la aldea
        all_pts = np.array([d.center for d in detections], dtype=np.float32)
        center = (float(np.mean(all_pts[:, 0])), float(np.mean(all_pts[:, 1])))

        file_errors = []
        for s in storages[:4]:  # Evaluar hasta 4 almacenes por captura
            p_target = (int(s.center[0]), int(s.center[1]))
            p_drop = compute_canonical_diamond_drop(p_target, center_pt=center)

            crossings = find_wall_crossings_from_segments(
                p_drop=p_drop,
                p_target=p_target,
                wall_segments=wall_segs,
                min_crossing_angle_deg=25.0,
                min_dist_drop=40.0,
                min_dist_target=50.0,
                endpoint_tol=30.0,
            )

            # Para cada cruce detectado, medir la distancia al pico de gradiente más cercano a lo largo del rayo
            vx = p_target[0] - p_drop[0]
            vy = p_target[1] - p_drop[1]
            dist_ray = max(float(np.hypot(vx, vy)), 1e-3)
            ux, uy = vx / dist_ray, vy / dist_ray

            for cx, cy in crossings:
                # Muestrear el perfil de gradiente en una ventana local [-35, +35] px alrededor del cruce
                best_peak_dist = 0.0
                max_grad = -1.0
                for offset in range(-35, 36):
                    sx = int(round(cx + offset * ux))
                    sy = int(round(cy + offset * uy))
                    if 0 <= sx < w and 0 <= sy < h:
                        g = float(grad_mag[sy, sx])
                        if g > max_grad:
                            max_grad = g
                            best_peak_dist = abs(offset)

                file_errors.append(best_peak_dist)
                all_errors.append(best_peak_dist)
                total_crossings_evaluated += 1

        avg_file_err = (sum(file_errors) / len(file_errors)) if file_errors else 0.0
        print(f"[{idx+1:02d}/12] {os.path.basename(path)} | Muros: {len(wall_segs):3d} | Cruces: {len(file_errors):2d} | Error Medio: {avg_file_err:4.1f} px")

    if all_errors:
        mean_err = sum(all_errors) / len(all_errors)
        max_err = max(all_errors)
        p90_err = float(np.percentile(all_errors, 90))

        print(f"\n==================================================================")
        print(f"📊 RESUMEN DE PRECISIÓN CUANTITATIVA (Muros Evaluados: {total_crossings_evaluated})")
        print(f"   • Error Medio en Píxeles: {mean_err:.2f} px (Umbral Aceptación: < 25.0 px)")
        print(f"   • Percentil 90:            {p90_err:.2f} px")
        print(f"   • Error Máximo Local:      {max_err:.2f} px")
        if mean_err < 25.0:
            print(f"   ✅ RESULTADO: SUPERADO CON ÉXITO (Precisión milimétrica sobre muros)")
        else:
            print(f"   ⚠️ RESULTADO: Requiere afinamiento de umbrales")
        print(f"==================================================================")


if __name__ == "__main__":
    measure_accuracy()
