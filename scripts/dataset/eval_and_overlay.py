"""
scripts/dataset/eval_and_overlay.py
Evaluación comparativa de modelos U-Net de muros (previo vs nuevo) y generación
de imágenes de overlay de optimize_jumps() en logs/eval_walls/.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.logger import get_logger
from vision.wall_seg_detector import WallSegDetector
from vision.yolo_detector import YoloDetector
import vision.wall_compartments as wc
from attack.jump_planner import compute_canonical_diamond_drop

log = get_logger("eval_walls")


def compute_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Tuple[float, float, int, int]:
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    dice = float((2.0 * inter + 1e-6) / (total + 1e-6))
    iou = float((inter + 1e-6) / (union + 1e-6))
    return dice, iou, int(pred_bin.sum()), int(gt_bin.sum())


def main():
    parser = argparse.ArgumentParser(description="Evalúa modelos de segmentación de muros y genera overlays de saltos.")
    parser.add_argument("--filter", type=str, default="", help="Filtro opcional por subcadena de nombre de archivo (ej: '0020,0014')")
    args = parser.parse_args()

    val_images_dir = os.path.join(PROJECT_ROOT, "datasets", "coc_walls_unet", "val", "images")
    val_masks_dir = os.path.join(PROJECT_ROOT, "datasets", "coc_walls_unet", "val", "masks")
    out_dir = os.path.join(PROJECT_ROOT, "logs", "eval_walls")
    os.makedirs(out_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(val_images_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(val_masks_dir, "*.png")))

    if not img_paths or len(img_paths) != len(mask_paths):
        log.error("No se encontraron pares válidos en %s y %s", val_images_dir, val_masks_dir)
        sys.exit(1)

    if args.filter:
        filters = [f.strip() for f in args.filter.split(",") if f.strip()]
        filtered = []
        for ip, mp in zip(img_paths, mask_paths):
            if any(f in os.path.basename(ip) for f in filters):
                filtered.append((ip, mp))
        if filtered:
            img_paths = [x[0] for x in filtered]
            mask_paths = [x[1] for x in filtered]

    model_prev_path = os.path.join(PROJECT_ROOT, "models", "wall_seg_prev.onnx")
    model_new_path = os.path.join(PROJECT_ROOT, "models", "wall_seg.onnx")

    log.info("Cargando detector baseline (prev): %s", model_prev_path)
    det_prev = WallSegDetector(model_path=model_prev_path, conf_thresh=0.5)

    log.info("Cargando detector nuevo: %s", model_new_path)
    det_new = WallSegDetector(model_path=model_new_path, conf_thresh=0.5)

    log.info("Cargando YoloDetector para edificios de botín...")
    yolo = YoloDetector()

    results: List[Dict] = []

    print("\n" + "=" * 90)
    print("EVALUACIÓN COMPARATIVA: models/wall_seg_prev.onnx vs models/wall_seg.onnx")
    print("=" * 90)

    for img_p, mask_p in zip(img_paths, mask_paths):
        stem = Path(img_p).name
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        gt_bin = (mask > 127).astype(np.uint8)

        # 1. Inferencia Baseline
        pred_prev = det_prev.predict(img)
        pred_prev_bin = (pred_prev > 127).astype(np.uint8)
        dice_prev, iou_prev, px_prev, gt_px = compute_metrics(pred_prev_bin, gt_bin)

        # 2. Inferencia Nuevo Modelo
        pred_new = det_new.predict(img)
        pred_new_bin = (pred_new > 127).astype(np.uint8)
        dice_new, iou_new, px_new, _ = compute_metrics(pred_new_bin, gt_bin)

        delta_dice = dice_new - dice_prev
        delta_iou = iou_new - iou_prev

        is_highlight = "0020" in stem
        results.append({
            "stem": stem,
            "dice_prev": dice_prev,
            "iou_prev": iou_prev,
            "dice_new": dice_new,
            "iou_new": iou_new,
            "delta_dice": delta_dice,
            "delta_iou": delta_iou,
            "highlight": is_highlight,
            "gt_px": gt_px,
            "px_prev": px_prev,
            "px_new": px_new,
        })

        # 3. Detección YOLO y Puntos de Salto
        detections = yolo.detect(img)
        drop_points = []
        for det in detections:
            dp = compute_canonical_diamond_drop(det.center)
            drop_points.append(dp)

        # 4. Generación de overlay con optimize_jumps usando pred_new
        h, w = img.shape[:2]
        opt_res = wc.optimize_jumps(
            img_shape=(h, w),
            base_img=img,
            wall_mask=pred_new,
            detections=detections,
            drop_points=drop_points,
            generate_debug_overlay=True,
        )

        overlay_out_path = os.path.join(out_dir, f"overlay_{stem}")
        if opt_res.debug_overlay is not None:
            cv2.imwrite(overlay_out_path, opt_res.debug_overlay)
            log.info("Overlay guardado: %s (saltos: %d, ganancia: %.1f)", overlay_out_path, len(opt_res.jumps), opt_res.total_gain)
        else:
            log.warning("No se generó debug_overlay para %s", stem)

    # Impresión de Tabla
    print(f"{'Imagen Val':<32} | {'Dice Prev':<9} | {'Dice Nuevo':<10} | {'Δ Dice':<8} | {'IoU Prev':<8} | {'IoU Nuevo':<9} | {'Δ IoU':<8}")
    print("-" * 96)
    for r in results:
        hl_tag = " ⭐ [TARGET]" if r["highlight"] else ""
        print(f"{r['stem']:<32} | {r['dice_prev']:<9.4f} | {r['dice_new']:<10.4f} | {r['delta_dice']:<+8.4f} | {r['iou_prev']:<8.4f} | {r['iou_new']:<9.4f} | {r['delta_iou']:<+8.4f}{hl_tag}")

    avg_dice_prev = np.mean([r["dice_prev"] for r in results])
    avg_dice_new = np.mean([r["dice_new"] for r in results])
    avg_iou_prev = np.mean([r["iou_prev"] for r in results])
    avg_iou_new = np.mean([r["iou_new"] for r in results])
    print("-" * 96)
    print(f"{'PROMEDIO GLOBAL':<32} | {avg_dice_prev:<9.4f} | {avg_dice_new:<10.4f} | {avg_dice_new - avg_dice_prev:<+8.4f} | {avg_iou_prev:<8.4f} | {avg_iou_new:<9.4f} | {avg_iou_new - avg_iou_prev:<+8.4f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
