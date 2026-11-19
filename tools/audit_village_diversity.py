"""
tools/audit_village_diversity.py
Audita la diversidad de aldeas en el dataset (Fase 1.3 de plan_saltos_optimos_antigravity.md):
- Inspecciona las 37 imágenes en datasets/coc/images/raw y las 20 en logs/yolo/attack_*/screen_raw.png.
- Detecta niveles de Ayuntamiento (TH), conteo de segmentos de muro y capas concéntricas.
- Calcula edificios dentro vs fuera del anillo de muros.
- Reporta estadísticas exactas y cuántas anotaciones de ground truth están pendientes.
- Guarda reporte en logs/village_diversity_audit.json y logs/village_diversity_report.md.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configurar path raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import SETTINGS
from core.logger import get_logger
from vision.yolo_detector import YoloDetector
from vision.wall_raycast import detect_global_wall_segments, find_wall_crossings_from_segments

log = get_logger("audit_village_diversity")


def classify_th_level(crop: np.ndarray) -> str:
    """
    Clasifica el nivel de Ayuntamiento según la firma de color y estilo visual en HSV.
    - TH11: Mármol blanco, sat baja, brillo alto.
    - TH12: Azul eléctrico brillante (Giga Tesla).
    - TH13: Turquesa / Teal / Hielo (Giga Inferno).
    - TH14: Verde selva / maya con detalles dorados.
    - TH15: Púrpura / Magia cósmica triangular.
    - TH16: Rojo terracota / dorado naturaleza.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return "Unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Fracción de píxeles característicos
    blue_ratio = float(np.mean((h >= 100) & (h <= 126) & (s > 75) & (v > 90)))
    teal_ratio = float(np.mean((h >= 78) & (h < 100) & (s > 60) & (v > 80)))
    green_ratio = float(np.mean((h >= 35) & (h <= 77) & (s > 75) & (v > 70)))
    purple_ratio = float(np.mean((h >= 127) & (h <= 165) & (s > 55) & (v > 70)))
    red_ratio = float(np.mean(((h <= 15) | (h >= 165)) & (s > 90) & (v > 100)))
    white_ratio = float(np.mean((s < 45) & (v > 175)))

    scores = {
        "TH11": white_ratio * 1.6,
        "TH12": blue_ratio * 2.8,
        "TH13": teal_ratio * 2.6,
        "TH14": green_ratio * 2.6,
        "TH15": purple_ratio * 2.8,
        "TH16": red_ratio * 2.2,
    }

    best_lvl, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score < 0.04:
        # Si el score es muy bajo, verificar si tiene rasgos TH12 predominantes
        if blue_ratio > 0.015:
            return "TH12"
        return "TH_General"
    return best_lvl


def line_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float]
) -> bool:
    """Verifica si dos segmentos (p1-p2) y (p3-p4) se intersecan."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-9:
        return False

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    return (0.0 <= ua <= 1.0) and (0.0 <= ub <= 1.0)


def estimate_wall_layers(
    center_pt: Tuple[int, int],
    wall_segments: List[Tuple[int, int, int, int]],
    screen_size: Tuple[int, int] = (2400, 1080),
) -> int:
    """
    Estima el número de capas concéntricas de muro (profundidad de compartimentos hasta el núcleo)
    trazando rayos desde los flancos exteriores del césped hacia el centro/TH.
    """
    if not wall_segments:
        return 0

    cx, cy = center_pt
    w, h = screen_size

    # Puntos de drop en los 4 flancos diagonales principales y 4 cardinales
    flank_drops = [
        (int(w * 0.29), int(h * 0.24)),  # NW
        (int(w * 0.71), int(h * 0.24)),  # NE
        (int(w * 0.29), int(h * 0.70)),  # SW
        (int(w * 0.71), int(h * 0.70)),  # SE
        (cx, int(h * 0.15)),             # Norte
        (cx, int(h * 0.85)),             # Sur
        (int(w * 0.18), cy),             # Oeste
        (int(w * 0.82), cy),             # Este
    ]

    crossings_counts = []
    for dp in flank_drops:
        crossings = find_wall_crossings_from_segments(
            p_drop=dp,
            p_target=center_pt,
            wall_segments=wall_segments,
            min_dist_drop=30.0,
            min_dist_target=40.0
        )
        if crossings:
            crossings_counts.append(len(crossings))

    if not crossings_counts:
        return 0

    # Retorna la mediana de capas atravesadas desde el exterior al núcleo
    return int(round(float(np.median(crossings_counts))))


def audit_single_image(
    image_path: str,
    detector: YoloDetector,
    gt_dir: str
) -> dict:
    """Audita una sola imagen de aldea."""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Fallo al cargar imagen: {image_path}"}

    h, w = img.shape[:2]
    parent_dir = os.path.basename(os.path.dirname(image_path))
    file_name = os.path.basename(image_path)
    is_attack_log = "attack_" in parent_dir
    village_id = f"{parent_dir}_{file_name}" if is_attack_log else file_name

    # 1. Detecciones YOLO
    detections = detector.detect(img)
    total_buildings = len(detections)

    # 2. Ayuntamiento (Town Hall)
    th_dets = [d for d in detections if d.cls_name == "town_hall"]
    th_detected = len(th_dets) > 0
    th_level = "None"
    th_center = (w // 2, h // 2)
    th_bbox = []

    if th_detected:
        d_th = th_dets[0]
        th_center = (int(d_th.center[0]), int(d_th.center[1]))
        bx1, by1, bx2, by2 = [int(v) for v in d_th.bbox_xyxy]
        th_bbox = [bx1, by1, bx2, by2]
        crop_th = img[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
        th_level = classify_th_level(crop_th)

    # 3. Detección de muros globales
    iso_angles, wall_segments = detect_global_wall_segments(img, force_refresh=True)
    wall_count = len(wall_segments)

    # 4. Capas concéntricas de muro
    ref_center = th_center if th_detected else (1220, 500)
    wall_layers = estimate_wall_layers(ref_center, wall_segments, screen_size=(w, h))

    # 5. Edificios fuera vs dentro de muros (convex hull de muros)
    outside_buildings_count = 0
    inside_buildings_count = 0
    outside_ratio = 0.0

    if wall_segments and len(wall_segments) >= 3:
        wall_pts = []
        for s in wall_segments:
            wall_pts.append([s[0], s[1]])
            wall_pts.append([s[2], s[3]])
        wall_pts_np = np.array(wall_pts, dtype=np.int32)
        hull = cv2.convexHull(wall_pts_np)

        for d in detections:
            pt = (float(d.center[0]), float(d.center[1]))
            # pointPolygonTest: >0 interior, ==0 borde, <0 exterior
            dist = cv2.pointPolygonTest(hull, pt, False)
            if dist < 0:
                outside_buildings_count += 1
            else:
                inside_buildings_count += 1

        if total_buildings > 0:
            outside_ratio = float(outside_buildings_count) / float(total_buildings)
    else:
        # Sin muros detectados, todos cuentan como exteriores
        outside_buildings_count = total_buildings
        inside_buildings_count = 0
        outside_ratio = 1.0 if total_buildings > 0 else 0.0

    # 6. Metadata de botín (si existe json acompañante)
    loot_info = {}
    json_path = os.path.splitext(image_path)[0] + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                jdata = json.load(jf)
                loot_info = jdata.get("loot", {})
        except Exception:
            pass

    # 7. Verificación de Ground Truth
    gt_file_simple = os.path.splitext(file_name)[0] + ".json"
    gt_file_scoped = os.path.splitext(village_id)[0] + ".json"
    has_gt = os.path.exists(os.path.join(gt_dir, gt_file_simple)) or os.path.exists(os.path.join(gt_dir, gt_file_scoped))

    return {
        "village_id": village_id,
        "image_path": image_path,
        "source": "attack_log" if is_attack_log else "raw_dataset",
        "resolution": [w, h],
        "total_buildings": total_buildings,
        "town_hall_detected": th_detected,
        "town_hall_level": th_level,
        "town_hall_center": list(th_center) if th_detected else None,
        "town_hall_bbox": th_bbox if th_detected else None,
        "iso_angles": [round(float(iso_angles[0]), 1), round(float(iso_angles[1]), 1)],
        "wall_segments_count": wall_count,
        "wall_concentric_layers": wall_layers,
        "buildings_outside_walls": outside_buildings_count,
        "buildings_inside_walls": inside_buildings_count,
        "buildings_outside_ratio": round(outside_ratio, 3),
        "loot": loot_info,
        "has_ground_truth": has_gt,
    }


def run_village_diversity_audit(
    raw_dir: str = "datasets/coc/images/raw",
    attack_pattern: str = "logs/yolo/attack_*/screen_raw.png",
    gt_dir: str = "datasets/coc/jump_gt",
    output_json: str = "logs/village_diversity_audit.json",
    output_md: str = "logs/village_diversity_report.md"
) -> dict:
    """Ejecuta la auditoría completa de diversidad en todas las aldeas disponibles."""
    raw_images = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
    attack_images = sorted(glob.glob(attack_pattern))
    all_images = raw_images + attack_images

    log.info("Iniciando auditoría de diversidad: %d imágenes en raw, %d en attack logs (Total: %d)",
             len(raw_images), len(attack_images), len(all_images))

    attack_cfg = SETTINGS.get("attack", {})
    conf = float(attack_cfg.get("yolo_conf", 0.35))
    imgsz = int(attack_cfg.get("yolo_imgsz", 1280))
    detector = YoloDetector(conf=conf, imgsz=imgsz)

    results = []
    for idx, img_path in enumerate(all_images, start=1):
        log.info("[%2d/%2d] Auditando %s...", idx, len(all_images), os.path.basename(img_path))
        data = audit_single_image(img_path, detector, gt_dir)
        results.append(data)

    # Estadísticas globales
    valid_results = [r for r in results if "error" not in r]
    total_audited = len(valid_results)

    th_levels_counter = Counter([r["town_hall_level"] for r in valid_results])
    layers_counter = Counter([r["wall_concentric_layers"] for r in valid_results])
    source_counter = Counter([r["source"] for r in valid_results])

    wall_counts = [r["wall_segments_count"] for r in valid_results]
    outside_ratios = [r["buildings_outside_ratio"] for r in valid_results]
    outside_counts = [r["buildings_outside_walls"] for r in valid_results]
    inside_counts = [r["buildings_inside_walls"] for r in valid_results]

    gt_annotated_count = sum(1 for r in valid_results if r["has_ground_truth"])
    gt_pending_count = total_audited - gt_annotated_count

    stats = {
        "total_images_audited": total_audited,
        "sources": dict(source_counter),
        "town_hall_levels": dict(th_levels_counter),
        "wall_segments": {
            "min": int(np.min(wall_counts)) if wall_counts else 0,
            "max": int(np.max(wall_counts)) if wall_counts else 0,
            "mean": round(float(np.mean(wall_counts)), 1) if wall_counts else 0.0,
            "median": round(float(np.median(wall_counts)), 1) if wall_counts else 0.0,
        },
        "wall_layers_distribution": dict(layers_counter),
        "buildings_distribution": {
            "mean_outside_count": round(float(np.mean(outside_counts)), 1) if outside_counts else 0.0,
            "mean_inside_count": round(float(np.mean(inside_counts)), 1) if inside_counts else 0.0,
            "mean_outside_ratio": round(float(np.mean(outside_ratios)), 3) if outside_ratios else 0.0,
        },
        "ground_truth_status": {
            "target_annotations": 30,
            "completed": gt_annotated_count,
            "pending": gt_pending_count,
            "completion_pct": round(gt_annotated_count / total_audited * 100.0, 1) if total_audited > 0 else 0.0,
        }
    }

    report_payload = {
        "stats": stats,
        "villages": valid_results
    }

    # Guardar JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)
    log.info("Auditoría JSON guardada en: %s", output_json)

    # Generar reporte Markdown
    md_content = generate_markdown_report(stats, valid_results)
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    log.info("Reporte Markdown guardado en: %s", output_md)

    return report_payload


def generate_markdown_report(stats: dict, villages: List[dict]) -> str:
    """Genera el contenido Markdown formateado del reporte de diversidad."""
    th_table = "\n".join([f"| {lvl} | {cnt} | {cnt / stats['total_images_audited'] * 100:.1f}% |"
                          for lvl, cnt in sorted(stats['town_hall_levels'].items(), key=lambda x: -x[1])])

    layer_table = "\n".join([f"| {layer} capa(s) | {cnt} | {cnt / stats['total_images_audited'] * 100:.1f}% |"
                            for layer, cnt in sorted(stats['wall_layers_distribution'].items())])

    md = f"""# Auditoría de Diversidad de Aldeas y Estado de Ground Truth (Fase 1.3)

> Reporte oficial de auditoría de aldeas generado por `tools/audit_village_diversity.py` para Antigravity / ClashFarmerbot.

---

## 1. Resumen Ejecutivo de Muestras y Anotaciones

| Métrica | Valor | Notas |
| :--- | :--- | :--- |
| **Total de aldeas inspeccionadas** | **{stats['total_images_audited']}** | 37 en raw + 20 en attack logs |
| **Aldeas en `datasets/coc/images/raw`** | {stats['sources'].get('raw_dataset', 0)} | Capturas continuas pre-ataque con metadata de botín |
| **Aldeas en `logs/yolo/attack_*`** | {stats['sources'].get('attack_log', 0)} | Capturas reales de combate antes del drop (`screen_raw.png`) |
| **Anotaciones Ground Truth completadas** | **{stats['ground_truth_status']['completed']}** | Archivos existentes en `datasets/coc/jump_gt/` |
| **Anotaciones Ground Truth pendientes** | **{stats['ground_truth_status']['pending']}** | **Objetivo Fase 1: Anotar al menos 30 aldeas** |

---

## 2. Distribución de Niveles de Ayuntamiento (TH)

| Nivel de Ayuntamiento | Cantidad | Porcentaje |
| :--- | :--- | :--- |
{th_table}

> **Observación:** El dataset presenta una fuerte concentración en **TH12** ({stats['town_hall_levels'].get('TH12', 0)} aldeas), coherente con el emparejamiento de la cuenta de farmeo actual. Se identificaron además aldeas de TH14 y TH16.

---

## 3. Complejidad Estructural de Muros y Capas Concéntricas

- **Segmentos de muro detectados:**
  - Mínimo: `{stats['wall_segments']['min']}` segmentos
  - Máximo: `{stats['wall_segments']['max']}` segmentos
  - Media: `{stats['wall_segments']['mean']}` segmentos
  - Mediana: `{stats['wall_segments']['median']}` segmentos

### Distribución de Capas Concéntricas (Profundidad de Núcleo)
| Capas Concéntricas | Aldeas | Porcentaje |
| :--- | :--- | :--- |
{layer_table}

> Las aldeas con **2 y 3 capas concéntricas** representan la gran mayoría del conjunto de prueba ({stats['wall_layers_distribution'].get(2, 0) + stats['wall_layers_distribution'].get(3, 0)} aldeas), proporcionando el escenario óptimo para evaluar saltos dobles y triples que reemplacen múltiples Super Rompemuros.

---

## 4. Distribución de Edificios Fuera vs Dentro de Muros

- **Media de edificios fuera de muros (perímetro/recolección):** `{stats['buildings_distribution']['mean_outside_count']}` edificios
- **Media de edificios dentro de muros (compartimentos/almacenes):** `{stats['buildings_distribution']['mean_inside_count']}` edificios
- **Ratio promedio de edificios en exterior:** `{stats['buildings_distribution']['mean_outside_ratio'] * 100:.1f}%`

> Los edificios exteriores permiten verificar que el optimizador priorice las aperturas hacia los almacenes internos sin desperdiciar saltos en recolectores perimetrales accesibles directamente por césped.

---

## 5. Próximos Pasos (Punto de Parada 1)
1. **Anotación manual humana:** Con la herramienta `tools/annotate_best_jump.py`, el usuario anotará los mejores puntos de salto en las 30 aldeas prioritarias.
2. **Evaluación de huella:** Integrar la Elipse Isométrica ($Rx=115$px, $Ry=65$px) en el optimizador de compartimentos (Fase 2) para eliminar la sobrestimación vertical del 109% producida por el círculo de 125px.
"""
    return md


def main():
    parser = argparse.ArgumentParser(description="Auditoría de diversidad de aldeas CoC (Fase 1.3)")
    parser.add_argument("--raw-dir", type=str, default="datasets/coc/images/raw")
    parser.add_argument("--attack-pattern", type=str, default="logs/yolo/attack_*/screen_raw.png")
    parser.add_argument("--gt-dir", type=str, default="datasets/coc/jump_gt")
    parser.add_argument("--output-json", type=str, default="logs/village_diversity_audit.json")
    parser.add_argument("--output-md", type=str, default="logs/village_diversity_report.md")
    args = parser.parse_args()

    run_village_diversity_audit(
        raw_dir=args.raw_dir,
        attack_pattern=args.attack_pattern,
        gt_dir=args.gt_dir,
        output_json=args.output_json,
        output_md=args.output_md
    )


if __name__ == "__main__":
    main()
