import os
import sys
import time
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.device import Device
from core.vision import Vision
from vision.army_checker import ArmyChecker
from core.config import PROJECT_ROOT
from attack.deployment_strategy import (
    DeploymentStrategy,
    SWEEP_SUR,
    SWEEP_NORTE,
    PAN_CENTER_TO_SOUTH,
    PAN_SOUTH_TO_NORTH,
    PAN_NORTH_TO_SOUTH,
    interpolate_catmull_rom,
)


def draw_lines(img, sweep_dict, color):
    h, w = img.shape[:2]
    out = img.copy()
    for lado, pts in sweep_dict.items():
        raw_coords = [(int(rx * w), int(ry * h)) for rx, ry in pts]
        smooth_coords = interpolate_catmull_rom(raw_coords, density=6)
        coords = np.array(smooth_coords, np.int32)
        cv2.polylines(out, [coords], isClosed=False, color=color, thickness=3)
        for x, y in raw_coords:
            cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(out, (x, y), 7, (255, 255, 255), 1)
    return out


def main():
    out_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(out_dir, exist_ok=True)

    dev = Device()
    if not dev.connect():
        print("⚠️ Dispositivo no detectado en ADB. Generando proyecciones offline calibradas...")
        sur_ref = os.path.join(out_dir, "sur_region.png")
        norte_ref = os.path.join(out_dir, "norte_region.png")
        zoom_ref = os.path.join(out_dir, "zoom.png")
        m1_path = os.path.join(out_dir, "movi1.jpeg")
        m2_path = os.path.join(out_dir, "movi2.jpeg")

        if os.path.exists(sur_ref):
            m1 = cv2.imread(sur_ref)
            img1 = draw_lines(m1, SWEEP_SUR, color=(0, 255, 0))
            cv2.imwrite(os.path.join(out_dir, "debug_sweep1_sur.png"), img1)
        elif os.path.exists(m1_path):
            m1 = cv2.imread(m1_path)
            img1 = draw_lines(m1, SWEEP_SUR, color=(0, 255, 0))
            cv2.imwrite(os.path.join(out_dir, "debug_sweep1_sur.png"), img1)

        if os.path.exists(norte_ref):
            m2 = cv2.imread(norte_ref)
            img2 = draw_lines(m2, SWEEP_NORTE, color=(255, 255, 0))
            cv2.imwrite(os.path.join(out_dir, "debug_sweep2_norte.png"), img2)
        elif os.path.exists(m2_path):
            m2 = cv2.imread(m2_path)
            img2 = draw_lines(m2, SWEEP_NORTE, color=(255, 255, 0))
            cv2.imwrite(os.path.join(out_dir, "debug_sweep2_norte.png"), img2)

        if os.path.exists(zoom_ref):
            mz = cv2.imread(zoom_ref)
            img_all = draw_lines(mz, SWEEP_SUR, color=(0, 255, 0))
            img_all = draw_lines(img_all, SWEEP_NORTE, color=(255, 255, 0))
            cv2.imwrite(os.path.join(out_dir, "debug_sweep_all.png"), img_all)
            cv2.imwrite(os.path.join(out_dir, "test_calibrated_all.png"), img_all)

        print("✅ Proyecciones calibradas generadas en logs/debug_sweep1_sur.png, logs/debug_sweep2_norte.png y logs/debug_sweep_all.png")
        return

    if "--deploy" in sys.argv:
        print("=== MODO DESPLIEGUE EN VIVO (--deploy) ===")
        vis = Vision()
        vis.preload_templates("assets")
        army = ArmyChecker(vis)
        strat = DeploymentStrategy(dev, vis, army)
        print("Ejecutando despliegue de ataque completo...")
        res = strat.execute_attack()
        print(f"Resultado del ataque: {res}")
        return

    out_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(out_dir, exist_ok=True)

    print("1. Zoom out obligatorio...")
    dev.zoom_out()
    time.sleep(0.8)

    print("2. Moviendo cámara del Centro al Sector Sur (PAN_CENTER_TO_SOUTH)...")
    dev.pan_camera(*PAN_CENTER_TO_SOUTH, duration=0.9)
    dev.wait_camera_settled(timeout=2.5)
    time.sleep(0.6)

    print("3. Capturando y proyectando sector Sur...")
    img1 = dev.screenshot()
    img1_sur = draw_lines(img1, SWEEP_SUR, color=(0, 255, 0))
    cv2.imwrite(os.path.join(out_dir, "debug_sweep1_sur.png"), img1_sur)

    # Proyección combinada sobre vista base
    img_all = draw_lines(img1_sur, SWEEP_NORTE, color=(255, 255, 0))
    cv2.imwrite(os.path.join(out_dir, "debug_sweep_all.png"), img_all)
    cv2.imwrite(os.path.join(out_dir, "test_calibrated_all.png"), img_all)

    print("4. Moviendo cámara al otro extremo: Sector Norte (PAN_SOUTH_TO_NORTH)...")
    dev.pan_camera(*PAN_SOUTH_TO_NORTH, duration=1.1)
    dev.wait_camera_settled(timeout=2.5)
    time.sleep(0.6)

    print("5. Capturando y proyectando sector Norte...")
    img2 = dev.screenshot()
    img2 = draw_lines(img2, SWEEP_NORTE, color=(255, 255, 0))
    cv2.imwrite(os.path.join(out_dir, "debug_sweep2_norte.png"), img2)

    time.sleep(0.6)
    print("6. Regresando cámara al otro extremo: Sector Sur (PAN_NORTH_TO_SOUTH)...")
    dev.pan_camera(*PAN_NORTH_TO_SOUTH, duration=1.1)
    dev.wait_camera_settled(timeout=2.5)
    print("Prueba finalizada. Archivos generados en logs/")
    print("TIP: Para probar el despliegue real de tropas en una batalla en vivo, ejecuta: python tools/test_sweep.py --deploy")


if __name__ == "__main__":
    main()
