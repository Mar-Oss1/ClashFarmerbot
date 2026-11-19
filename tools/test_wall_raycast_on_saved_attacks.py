"""
Herramienta de verificación visual para evaluar la detección física de muros
sobre las capturas de batallas reales guardadas en logs/yolo/.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import glob
import cv2
from vision.wall_raycast import (
    estimate_iso_angles,
    find_wall_crossings,
    draw_wall_raycast_debug,
    reset_iso_angles_cache,
)
from vision.yolo_detector import get_detector
from attack.deployment_strategy import compute_adaptive_jump_targets_from_detections


def test_attacks():
    attack_imgs = glob.glob("logs/yolo/attack_*.png")
    if not attack_imgs:
        print("No se encontraron capturas attack_*.png en logs/yolo/")
        return

    detector = get_detector()

    for idx, img_path in enumerate(sorted(attack_imgs, reverse=True)[:3]):
        reset_iso_angles_cache()
        img = cv2.imread(img_path)
        if img is None:
            continue

        print(f"\n==========================================")
        print(f"Evaluando captura real: {img_path}")
        iso1, iso2 = estimate_iso_angles(img, force_refresh=True)
        print(f"Ángulos isométricos detectados: {iso1:.1f}°, {iso2:.1f}°")

        detections = []
        if detector and detector.is_ready():
            detections = detector.detect(img)
            print(f"Detecciones YOLO: {len(detections)}")

        targets = compute_adaptive_jump_targets_from_detections(
            detections, (img.shape[1], img.shape[0]), screen=img
        )
        print(f"🎯 Saltos finales sobre muros: {targets}")

        out_path = f"logs/yolo/wall_raycast_verified_atk{idx+1}.png"
        if os.path.exists("logs/yolo/wall_raycast_debug.png"):
            import shutil
            shutil.copy("logs/yolo/wall_raycast_debug.png", out_path)
            print(f"📸 Debug guardado en: {out_path}")


if __name__ == "__main__":
    test_attacks()
