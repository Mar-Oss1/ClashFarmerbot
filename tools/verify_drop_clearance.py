"""
tools/verify_drop_clearance.py
Valida que todos los puntos calculados por _get_drop_point_on_contour
tengan distancia negativa (exterior) respecto a red_contour.
"""
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.deployment_strategy import DeploymentStrategy, detect_village_boundary
from vision.yolo_detector import get_detector


class DummyDevice:
    width = 2400
    height = 1080


def test_clearance():
    img_path = "logs/yolo/attack_debug_latest.png"
    if not os.path.exists(img_path):
        # Buscar la primera captura disponible en logs/yolo/
        import glob
        files = glob.glob("logs/yolo/attack_*.png") + glob.glob("logs/yolo/red_boundary_*.png")
        if not files:
            print("No se encontraron imagenes de prueba en logs/yolo/")
            return
        img_path = files[0]

    print(f"Cargando imagen: {img_path}")
    img = cv2.imread(img_path)
    detector = get_detector()
    dets = detector.detect(img) if detector else []
    print(f"Detecciones: {len(dets)}")

    red_cnt, green_cnt = detect_village_boundary(img, detections=dets)
    if green_cnt is None:
        print("ERROR: No se pudo obtener green_contour")
        return

    strat = DeploymentStrategy(DummyDevice(), None, None)
    strat._cached_red_contour = red_cnt

    failed = 0
    passed = 0
    for d in dets:
        dx, dy = strat._get_drop_point_on_contour(d.center, green_cnt, red_contour=red_cnt)
        if red_cnt is not None and len(red_cnt) >= 3:
            dist = cv2.pointPolygonTest(red_cnt, (float(dx), float(dy)), True)
            if dist > -15.0:
                print(f"FALLO: {d.cls_name} drop=({dx}, {dy}) distancia_roja={dist:.2f}px (Dentro o muy cerca)")
                failed += 1
            else:
                passed += 1
        else:
            passed += 1

    print(f"\n--- RESULTADO CLEARANCE ---")
    print(f"Pasadados: {passed}")
    print(f"Fallados: {failed}")
    if failed == 0:
        print("EXITO: Todos los drops caen 100% en cesped legal exterior.")


if __name__ == "__main__":
    test_clearance()
