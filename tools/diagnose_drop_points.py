"""
tools/diagnose_drop_points.py
Herramienta de diagnóstico offline para validar puntos de drop de duendes (Fase 1.2).
"""
import os
import glob
import argparse
import csv
import cv2
import numpy as np

from core.config import PROJECT_ROOT, SETTINGS
from core.logger import get_logger
from vision.deploy_zone_detector import get_deploy_zone_detector
from vision.yolo_detector import get_detector
from attack.deployment_strategy import DeploymentStrategy

log = get_logger("diagnose_drop_points")


def get_battle_hud_mask(h: int, w: int) -> np.ndarray:
    """Filtra HUD superior e inferior."""
    mask = np.ones((h, w), dtype=np.uint8)
    mask[:138, :] = 0  # HUD superior
    mask[845:, :] = 0  # Barra inferior de tropas
    mask[:, :120] = 0  # Margen lateral izquierdo
    mask[:, 2280:] = 0  # Margen lateral derecho
    return mask


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico offline de puntos de drop")
    parser.add_argument("--images", default="logs/yolo", help="Carpeta de imágenes a evaluar")
    parser.add_argument("--out", default="logs/diagnostics/drops", help="Carpeta de salida")
    parser.add_argument("--limit", type=int, default=20, help="Límite de imágenes")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    images_dir = os.path.join(PROJECT_ROOT, args.images) if not os.path.isabs(args.images) else args.images
    img_files = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))

    if not img_files:
        log.warning("No se encontraron imágenes en: %s", images_dir)
        return

    img_files = img_files[:args.limit]
    log.info("Analizando %d capturas...", len(img_files))

    dz = get_deploy_zone_detector()
    yolo = get_detector()
    strat = DeploymentStrategy(None, None)

    results = []
    dilations = [0, 15, 25, 35, 50]

    for idx_img, fpath in enumerate(img_files, 1):
        fname = os.path.basename(fpath)
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]

        raw_dets = yolo.detect(img) if yolo and yolo.is_ready() else []
        hud_mask = get_battle_hud_mask(h, w)
        dets = [d for d in raw_dets if hud_mask[int(d.center[1]), int(d.center[0])] == 1]

        mask = dz.predict_mask(img) if dz and dz.is_ready() else None
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)

        # Componentes conexos de la máscara
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # El mayor componente (excluyendo fondo label 0)
        largest_comp_idx = -1
        max_area = 0
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] > max_area:
                max_area = stats[lbl, cv2.CC_STAT_AREA]
                largest_comp_idx = lbl

        # Contornos principales
        pts_bld, pts_grs = dz.get_boundary_polygons(img, mask=mask)
        red_contour = pts_bld

        # Overlay de debug
        overlay = img.copy()
        # Máscara en magenta traslúcido
        magenta_layer = np.zeros_like(img)
        magenta_layer[mask > 0] = (255, 0, 255)
        overlay = cv2.addWeighted(overlay, 0.75, magenta_layer, 0.25, 0)

        for det in dets:
            cx, cy = int(round(det.center[0])), int(round(det.center[1]))
            bx1, by1, bx2, by2 = det.bbox_xyxy

            mask_at_center = int(mask[cy, cx]) if 0 <= cy < h and 0 <= cx < w else 0
            deployable_at_center = dz.is_point_deployable(cx, cy, safety_margin_px=0, mask=mask) if dz else True

            # Fracción de bbox prohibido
            sub_mask = mask[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
            frac_forbidden = float(np.mean(sub_mask > 0)) if sub_mask.size > 0 else 0.0

            # Componente conexo
            center_label = labels[cy, cx] if 0 <= cy < h and 0 <= cx < w else 0
            isolated_blob = bool(center_label > 0 and center_label != largest_comp_idx)

            # Drop actual calculado por DeploymentStrategy
            drop_x, drop_y = strat._get_drop_point_on_contour(
                det.center, green_contour=pts_grs, red_contour=pts_bld
            )

            drop_in_mask = bool(mask[drop_y, drop_x] > 0) if 0 <= drop_y < h and 0 <= drop_x < w else False

            # Comprobación de invasión de BBox con varias dilataciones
            veto_by_dil = {}
            for dil in dilations:
                invaded = False
                for other in dets:
                    obx1, oby1, obx2, oby2 = other.bbox_xyxy
                    if (obx1 - dil) <= drop_x <= (obx2 + dil) and (oby1 - dil) <= drop_y <= (oby2 + dil):
                        invaded = True
                        break
                veto_by_dil[dil] = invaded

            drop_inside_red = bool(cv2.pointPolygonTest(red_contour, (float(drop_x), float(drop_y)), False) >= 0) if red_contour is not None and len(red_contour) >= 3 else False

            # Clasificación de causa
            cause = "OK"
            if deployable_at_center and mask_at_center == 0:
                cause = "A_MODEL_MISS"
            elif isolated_blob:
                cause = "B_CONTOUR_BLIND"
            elif veto_by_dil[0]:
                cause = "DROP_IN_BBOX_DIL0"
            elif drop_in_mask:
                cause = "DROP_IN_MASK"

            row = {
                "image": fname,
                "class": det.cls_name,
                "confidence": f"{det.confidence:.2f}",
                "cx": cx,
                "cy": cy,
                "drop_x": drop_x,
                "drop_y": drop_y,
                "mask_at_center": mask_at_center,
                "deployable_at_center": deployable_at_center,
                "frac_bbox_forbidden": f"{frac_forbidden:.2f}",
                "isolated_blob": isolated_blob,
                "drop_in_mask": drop_in_mask,
                "drop_in_bbox_dil0": veto_by_dil[0],
                "drop_in_bbox_dil15": veto_by_dil[15],
                "drop_in_bbox_dil25": veto_by_dil[25],
                "drop_in_bbox_dil35": veto_by_dil[35],
                "drop_in_bbox_dil50": veto_by_dil[50],
                "drop_inside_red_main": drop_inside_red,
                "cause": cause,
            }
            results.append(row)

            # Dibujar en overlay
            box_color = (0, 255, 0) if cause == "OK" else (0, 0, 255)
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), box_color, 2)
            cv2.putText(overlay, f"{det.cls_name} {det.confidence:.2f}", (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

            drop_color = (0, 255, 0) if cause == "OK" else (0, 0, 255)
            cv2.circle(overlay, (drop_x, drop_y), 6, drop_color, -1)
            cv2.arrowedLine(overlay, (drop_x, drop_y), (cx, cy), (255, 255, 0), 1, tipLength=0.15)

        # Guardar imagen procesada
        diag_out_path = os.path.join(args.out, f"diag_{fname}")
        cv2.imwrite(diag_out_path, overlay)

    # Guardar CSV
    csv_path = os.path.join(args.out, "results.csv")
    fieldnames = list(results[0].keys()) if results else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    log.info("Resultados guardados en %s (%d detecciones)", csv_path, len(results))


if __name__ == "__main__":
    main()
