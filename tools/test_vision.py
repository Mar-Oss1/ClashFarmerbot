"""
tools/test_vision.py
Prueba en seco: valida conexión ADB, ROIs de recursos y detección de botones sin hacer clic.
"""
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.device import Device
from core.vision import Vision
from vision.resource_reader import ResourceReader
from core.config import SETTINGS, PROJECT_ROOT


def main():
    dev = Device()
    if not dev.connect():
        return

    vis = Vision(threshold=0.80)
    vis.preload_templates("assets")
    reader = ResourceReader(vis)

    print("📸 Tomando captura en vivo del teléfono...")
    screen = dev.screenshot()
    h, w = screen.shape[:2]

    # 1. Probar lectura de recursos en ambas zonas
    own_rss = reader.read_own_resources(screen)
    enemy_loot = reader.read_enemy_loot(screen)

    print(f"🏦 Lectura Recursos Propios : {own_rss}")
    print(f"🔎 Lectura Botín Enemigo    : {enemy_loot}")

    # 2. Dibujar los rectángulos (ROIs) sobre la captura para verificar puntería
    debug_img = screen.copy()
    for _, (ry1, ry2, rx1, rx2) in SETTINGS["rois"]["own_resources"].items():
        cv2.rectangle(debug_img, (int(w * rx1), int(h * ry1)), (int(w * rx2), int(h * ry2)), (0, 255, 0), 2)

    for _, (ry1, ry2, rx1, rx2) in SETTINGS["rois"]["enemy_loot"].items():
        cv2.rectangle(debug_img, (int(w * rx1), int(h * ry1)), (int(w * rx2), int(h * ry2)), (0, 165, 255), 2)

    out_file = os.path.join(PROJECT_ROOT, "logs", "debug_rois.png")
    cv2.imwrite(out_file, debug_img)
    print(f"✅ Imagen de calibración guardada en: {out_file}")


if __name__ == "__main__":
    main()