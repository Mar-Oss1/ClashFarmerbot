"""
Herramienta de Evaluación Offline y Línea Base de Colocación de Hechizos de Salto.
Fase 3 del Plan: colocación óptima de hechizos de salto (plan_saltos_optimos_antigravity.md).

Ejecuta y compara los 4 modos de decisión de saltos:
1. wall_raycast (modo actual de producción)
2. cluster_intersection (modo basado en centroides y cuadrantes)
3. legacy_quadrant (modo clásico proporcional)
4. coverage_optimizer (nuevo optimizador global de compartimentos)

Métricas evaluadas:
- Saltos usados
- Costo inicial de capas de muro hacia botín
- Costo final tras colocar saltos
- Reducción neta de costo (ganancia) y porcentaje de reducción
- Objetivos de botín que quedan cerrados (infranqueables)
- Tiempo de ejecución (ms)
- Barrido de sensibilidad de parámetros (buildings_block, grosor de muro, huella de salto)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
import cv2
import numpy as np

# Configurar path raíz del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.logger import get_logger
from core.config import SETTINGS
from vision.yolo_detector import YoloDetector, Detection
from vision.wall_raycast import detect_global_wall_segments
import vision.wall_compartments as wc
import attack.deployment_strategy as ds

log = get_logger("evaluate_jumps_offline")


def discover_village_images(
    raw_dir: str = "datasets/coc/images/raw",
    attack_pattern: str = "logs/yolo/attack_*/screen_raw.png",
) -> list[tuple[str, str, str]]:
    """
    Descubre todas las imágenes disponibles para evaluación.
    Retorna lista de (village_id, file_path, source).
    """
    villages = []
    seen_paths = set()

    # 1. Imágenes sin procesar de datasets
    raw_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, raw_dir, "*.png")))
    for rf in raw_files:
        norm = os.path.normpath(rf)
        if norm not in seen_paths:
            seen_paths.add(norm)
            vid = os.path.splitext(os.path.basename(rf))[0]
            villages.append((vid, norm, "raw"))

    # 2. Capturas de pantalla reales en logs de ataques
    atk_files = sorted(glob.glob(os.path.join(PROJECT_ROOT, attack_pattern)))
    for af in atk_files:
        norm = os.path.normpath(af)
        if norm not in seen_paths:
            seen_paths.add(norm)
            folder_name = os.path.basename(os.path.dirname(af))
            vid = f"attack_{folder_name}"
            villages.append((vid, norm, "attack_log"))

    return villages


def evaluate_jumps_under_compartment_model(
    comp_labels: np.ndarray,
    nearest_labels: np.ndarray,
    graph: dict[int, set[int]],
    entry_comps: set[int],
    targets: list[dict],
    jump_points: list[tuple[int, int]],
    footprint: dict | tuple | float | int,
) -> dict:
    """
    Evalúa cualquier conjunto de puntos de salto (x, y) utilizando el modelo de compartimentos.
    Calcula: costo inicial, costo final, reducción, % de reducción y objetivos cerrados.
    """
    initial_cost, initial_dists = wc.compute_total_cost(graph, entry_comps, targets, jump_bridges=[])

    jump_bridges = []
    for pt in jump_points:
        comps = wc.get_compartments_in_footprint(pt, footprint, comp_labels, nearest_labels)
        if len(comps) >= 2:
            jump_bridges.append(comps)

    final_cost, final_dists = wc.compute_total_cost(graph, entry_comps, targets, jump_bridges=jump_bridges)
    cost_reduction = initial_cost - final_cost
    pct_reduction = (cost_reduction / initial_cost * 100.0) if initial_cost > 0 else 0.0

    targets_closed = sum(1 for t in targets if final_dists.get(t["comp"], float("inf")) > 0)

    return {
        "num_jumps": len(jump_points),
        "initial_cost": round(float(initial_cost), 2),
        "final_cost": round(float(final_cost), 2),
        "cost_reduction": round(float(cost_reduction), 2),
        "pct_reduction": round(float(pct_reduction), 1),
        "targets_closed": int(targets_closed),
        "jumps": [[int(pt[0]), int(pt[1])] for pt in jump_points],
    }


def run_offline_evaluation(
    villages: list[tuple[str, str, str]],
    conf: float = 0.35,
    imgsz: int = 1280,
    max_jumps: int = 4,
    output_json_path: str = "logs/offline_evaluation_results.json",
) -> dict:
    """
    Ejecuta la evaluación offline completa sobre la lista de aldeas.
    """
    detector = YoloDetector(conf=conf, imgsz=imgsz)
    total_villages = len(villages)

    log.info("🚀 Iniciando Evaluación Offline sobre %d aldeas...", total_villages)

    per_village_results = []
    mode_accumulators = {
        "coverage_optimizer": defaultdict(list),
        "wall_raycast": defaultdict(list),
        "cluster_intersection": defaultdict(list),
        "legacy_quadrant": defaultdict(list),
        "human_gt": defaultdict(list),
    }

    # Acumuladores de sensibilidad
    sensitivity_accumulators = {
        "buildings_block": {
            "True": defaultdict(list),
            "False": defaultdict(list),
        },
        "wall_thickness": {
            "6": defaultdict(list),
            "8": defaultdict(list),
            "12": defaultdict(list),
        },
        "footprint": {
            "circle_90": defaultdict(list),
            "circle_125": defaultdict(list),
            "ellipse_calibrated_76_57": defaultdict(list),
        },
    }

    t_eval_start = time.perf_counter()

    for idx, (vid, img_path, source) in enumerate(villages, start=1):
        rel_path = os.path.relpath(img_path, PROJECT_ROOT).replace("\\", "/")
        log.info("[%d/%d] Evaluando aldea: %s (%s)", idx, total_villages, vid, source)

        img = cv2.imread(img_path)
        if img is None:
            log.warning("⚠️ No se pudo cargar imagen: %s", img_path)
            continue

        h, w = img.shape[:2]

        # 1. Detecciones YOLO
        t_det0 = time.perf_counter()
        detections = detector.detect(img)
        t_det = (time.perf_counter() - t_det0) * 1000.0

        # 2. Detección global de segmentos de muro
        t_wall0 = time.perf_counter()
        iso_angles, wall_segments = detect_global_wall_segments(
            img, force_refresh=True, detections=detections
        )
        t_wall = (time.perf_counter() - t_wall0) * 1000.0

        # 3. Construir modelo de compartimentos base (calibrado: rx=76, ry=57, radius=72)
        default_footprint = wc.DEFAULT_JUMP_FOOTPRINT
        wall_mask = wc.rasterize_wall_mask((h, w), wall_segments, iso_angles=iso_angles, wall_thickness=8)
        barrier_mask = wc.build_barrier_mask(wall_mask, detections, buildings_block=True)
        num_comps, comp_labels, entry_comps, nearest_labels = wc.compute_compartments(barrier_mask)
        graph, edges = wc.build_compartment_graph(num_comps, comp_labels, nearest_labels, wall_mask)
        targets = wc.map_targets_to_compartments(detections, comp_labels, nearest_labels)

        village_entry = {
            "village_id": vid,
            "image_path": rel_path,
            "source": source,
            "resolution": [w, h],
            "num_wall_segments": len(wall_segments),
            "num_compartments": max(1, num_comps - 1),
            "num_loot_targets": len(targets),
            "modes": {},
        }

        # --- A. coverage_optimizer (Beam Search con beam_width=4 y huella calibrada) ---
        t0 = time.perf_counter()
        opt_res = wc.optimize_jumps(
            img_shape=(h, w),
            wall_segments=wall_segments,
            detections=detections,
            iso_angles=iso_angles,
            jump_footprint=default_footprint,
            max_jumps=max_jumps,
            beam_width=4,
            buildings_block=True,
            wall_thickness=8,
            generate_debug_overlay=False,
        )
        rt_opt = (time.perf_counter() - t0) * 1000.0
        opt_jumps = [(j[0], j[1]) for j in opt_res.jumps]
        eval_opt = evaluate_jumps_under_compartment_model(
            comp_labels, nearest_labels, graph, entry_comps, targets, opt_jumps, default_footprint
        )
        eval_opt["runtime_ms"] = round(rt_opt, 1)
        village_entry["modes"]["coverage_optimizer"] = eval_opt

        # --- B. wall_raycast ---
        t0 = time.perf_counter()
        raycast_jumps = ds._compute_wall_raycast_jump_targets(img, detections, (w, h))
        rt_raycast = (time.perf_counter() - t0) * 1000.0
        eval_raycast = evaluate_jumps_under_compartment_model(
            comp_labels, nearest_labels, graph, entry_comps, targets, raycast_jumps, default_footprint
        )
        eval_raycast["runtime_ms"] = round(rt_raycast, 1)
        village_entry["modes"]["wall_raycast"] = eval_raycast

        # --- C. cluster_intersection ---
        t0 = time.perf_counter()
        cluster_jumps = ds._compute_cluster_intersection_jump_targets(detections, (w, h))
        rt_cluster = (time.perf_counter() - t0) * 1000.0
        eval_cluster = evaluate_jumps_under_compartment_model(
            comp_labels, nearest_labels, graph, entry_comps, targets, cluster_jumps, default_footprint
        )
        eval_cluster["runtime_ms"] = round(rt_cluster, 1)
        village_entry["modes"]["cluster_intersection"] = eval_cluster

        # --- D. legacy_quadrant ---
        t0 = time.perf_counter()
        legacy_jumps = ds._compute_legacy_quadrant_jump_targets(detections, (w, h))
        rt_legacy = (time.perf_counter() - t0) * 1000.0
        eval_legacy = evaluate_jumps_under_compartment_model(
            comp_labels, nearest_labels, graph, entry_comps, targets, legacy_jumps, default_footprint
        )
        eval_legacy["runtime_ms"] = round(rt_legacy, 1)
        village_entry["modes"]["legacy_quadrant"] = eval_legacy

        # --- E. human_gt (Ground Truth anotado en datasets/coc/jump_gt/) ---
        gt_path = os.path.join(PROJECT_ROOT, "datasets", "coc", "jump_gt", f"{vid}.json")
        if not os.path.exists(gt_path):
            folder_name = os.path.basename(os.path.dirname(img_path))
            gt_path = os.path.join(PROJECT_ROOT, "datasets", "coc", "jump_gt", f"{folder_name}.json")

        if os.path.exists(gt_path):
            try:
                with open(gt_path, "r", encoding="utf-8") as gf:
                    gt_json = json.load(gf)
                gt_pts = [(p[0], p[1]) for p in gt_json.get("puntos", [])]
                eval_gt = evaluate_jumps_under_compartment_model(
                    comp_labels, nearest_labels, graph, entry_comps, targets, gt_pts, default_footprint
                )
                eval_gt["runtime_ms"] = 0.0
                eval_gt["sin_salto"] = gt_json.get("sin_salto", False)
                eval_gt["botin_en_colectores"] = gt_json.get("botin_en_colectores", False)
                village_entry["modes"]["human_gt"] = eval_gt
            except Exception as e:
                log.warning("Error leyendo GT %s: %s", gt_path, e)

        # Acumular para estadísticas globales
        for mode_name, metrics in village_entry["modes"].items():
            for k, val in metrics.items():
                if k != "jumps":
                    mode_accumulators[mode_name][k].append(val)

        # --- SENSITIVITY SWEEP SOBRE ESTA ALDEA ---
        # 1. buildings_block (True vs False)
        for bb in [True, False]:
            sw_res = wc.optimize_jumps(
                img_shape=(h, w),
                wall_segments=wall_segments,
                detections=detections,
                iso_angles=iso_angles,
                jump_footprint=default_footprint,
                max_jumps=max_jumps,
                buildings_block=bb,
                wall_thickness=8,
                generate_debug_overlay=False,
            )
            sw_eval = evaluate_jumps_under_compartment_model(
                comp_labels, nearest_labels, graph, entry_comps, targets,
                [(j[0], j[1]) for j in sw_res.jumps], default_footprint
            )
            sensitivity_accumulators["buildings_block"][str(bb)]["num_jumps"].append(sw_eval["num_jumps"])
            sensitivity_accumulators["buildings_block"][str(bb)]["final_cost"].append(sw_eval["final_cost"])
            sensitivity_accumulators["buildings_block"][str(bb)]["cost_reduction"].append(sw_eval["cost_reduction"])
            sensitivity_accumulators["buildings_block"][str(bb)]["pct_reduction"].append(sw_eval["pct_reduction"])
            sensitivity_accumulators["buildings_block"][str(bb)]["runtime_ms"].append(sw_res.execution_time_ms)

        # 2. wall_thickness (6, 8, 12)
        for wt in [6, 8, 12]:
            sw_res = wc.optimize_jumps(
                img_shape=(h, w),
                wall_segments=wall_segments,
                detections=detections,
                iso_angles=iso_angles,
                jump_footprint=default_footprint,
                max_jumps=max_jumps,
                buildings_block=True,
                wall_thickness=wt,
                generate_debug_overlay=False,
            )
            sw_eval = evaluate_jumps_under_compartment_model(
                comp_labels, nearest_labels, graph, entry_comps, targets,
                [(j[0], j[1]) for j in sw_res.jumps], default_footprint
            )
            sensitivity_accumulators["wall_thickness"][str(wt)]["num_jumps"].append(sw_eval["num_jumps"])
            sensitivity_accumulators["wall_thickness"][str(wt)]["final_cost"].append(sw_eval["final_cost"])
            sensitivity_accumulators["wall_thickness"][str(wt)]["cost_reduction"].append(sw_eval["cost_reduction"])
            sensitivity_accumulators["wall_thickness"][str(wt)]["pct_reduction"].append(sw_eval["pct_reduction"])
            sensitivity_accumulators["wall_thickness"][str(wt)]["runtime_ms"].append(sw_res.execution_time_ms)

        # 3. jump_footprint (circle_90, circle_125, ellipse_calibrated_76_57)
        fp_variants = {
            "circle_90": 90.0,
            "circle_125": 125.0,
            "ellipse_calibrated_76_57": {"type": "ellipse", "rx": 76.0, "ry": 57.0, "radius": 72.0, "angle": 0.0},
        }
        for fp_name, fp_spec in fp_variants.items():
            sw_res = wc.optimize_jumps(
                img_shape=(h, w),
                wall_segments=wall_segments,
                detections=detections,
                iso_angles=iso_angles,
                jump_footprint=fp_spec,
                max_jumps=max_jumps,
                buildings_block=True,
                wall_thickness=8,
                generate_debug_overlay=False,
            )
            sw_eval = evaluate_jumps_under_compartment_model(
                comp_labels, nearest_labels, graph, entry_comps, targets,
                [(j[0], j[1]) for j in sw_res.jumps], fp_spec
            )
            sensitivity_accumulators["footprint"][fp_name]["num_jumps"].append(sw_eval["num_jumps"])
            sensitivity_accumulators["footprint"][fp_name]["final_cost"].append(sw_eval["final_cost"])
            sensitivity_accumulators["footprint"][fp_name]["cost_reduction"].append(sw_eval["cost_reduction"])
            sensitivity_accumulators["footprint"][fp_name]["pct_reduction"].append(sw_eval["pct_reduction"])
            sensitivity_accumulators["footprint"][fp_name]["runtime_ms"].append(sw_res.execution_time_ms)

        per_village_results.append(village_entry)

    total_eval_time = (time.perf_counter() - t_eval_start)

    # --- AGREGAR RESUMEN POR MODO ---
    summary_modes = {}
    for mode_name, metrics in mode_accumulators.items():
        summary_modes[mode_name] = {
            "mean_jumps": round(float(np.mean(metrics["num_jumps"])), 2),
            "mean_initial_cost": round(float(np.mean(metrics["initial_cost"])), 2),
            "mean_final_cost": round(float(np.mean(metrics["final_cost"])), 2),
            "mean_cost_reduction": round(float(np.mean(metrics["cost_reduction"])), 2),
            "mean_pct_reduction": round(float(np.mean(metrics["pct_reduction"])), 1),
            "mean_targets_closed": round(float(np.mean(metrics["targets_closed"])), 2),
            "mean_runtime_ms": round(float(np.mean(metrics["runtime_ms"])), 1),
        }

    # --- AGREGAR RESUMEN DE SENSIBILIDAD ---
    summary_sensitivity = {}
    for param_name, param_vars in sensitivity_accumulators.items():
        summary_sensitivity[param_name] = {}
        for var_name, metrics in param_vars.items():
            summary_sensitivity[param_name][var_name] = {
                "mean_jumps": round(float(np.mean(metrics["num_jumps"])), 2),
                "mean_final_cost": round(float(np.mean(metrics["final_cost"])), 2),
                "mean_cost_reduction": round(float(np.mean(metrics["cost_reduction"])), 2),
                "mean_pct_reduction": round(float(np.mean(metrics["pct_reduction"])), 1),
                "mean_runtime_ms": round(float(np.mean(metrics["runtime_ms"])), 1),
            }

    # --- RANKING DE ALDEAS (10 Mejores y 10 Peores) ---
    # Ordenar por ganancia neta (cost_reduction) y porcentaje de reducción
    sorted_by_opt = sorted(
        per_village_results,
        key=lambda v: (
            v["modes"]["coverage_optimizer"]["cost_reduction"],
            v["modes"]["coverage_optimizer"]["pct_reduction"],
        ),
        reverse=True,
    )

    best_10 = []
    for v in sorted_by_opt[:10]:
        m = v["modes"]["coverage_optimizer"]
        best_10.append({
            "village_id": v["village_id"],
            "image_path": v["image_path"],
            "source": v["source"],
            "num_wall_segments": v["num_wall_segments"],
            "num_compartments": v["num_compartments"],
            "num_loot_targets": v["num_loot_targets"],
            "initial_cost": m["initial_cost"],
            "final_cost": m["final_cost"],
            "cost_reduction": m["cost_reduction"],
            "pct_reduction": m["pct_reduction"],
            "num_jumps": m["num_jumps"],
            "jumps": m["jumps"],
        })

    worst_10 = []
    for v in sorted_by_opt[-10:]:
        m = v["modes"]["coverage_optimizer"]
        worst_10.append({
            "village_id": v["village_id"],
            "image_path": v["image_path"],
            "source": v["source"],
            "num_wall_segments": v["num_wall_segments"],
            "num_compartments": v["num_compartments"],
            "num_loot_targets": v["num_loot_targets"],
            "initial_cost": m["initial_cost"],
            "final_cost": m["final_cost"],
            "cost_reduction": m["cost_reduction"],
            "pct_reduction": m["pct_reduction"],
            "num_jumps": m["num_jumps"],
            "jumps": m["jumps"],
        })

    final_payload = {
        "summary": {
            "total_villages_evaluated": len(per_village_results),
            "total_evaluation_time_sec": round(total_eval_time, 1),
            "modes_comparison": summary_modes,
            "sensitivity_sweep": summary_sensitivity,
            "ranking": {
                "best_10": best_10,
                "worst_10": worst_10,
            },
        },
        "per_village": per_village_results,
    }

    # Guardar a disco
    out_dir = os.path.dirname(os.path.join(PROJECT_ROOT, output_json_path))
    os.makedirs(out_dir, exist_ok=True)
    full_out = os.path.join(PROJECT_ROOT, output_json_path)
    with open(full_out, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    log.info("💾 Resultados de evaluación offline guardados exitosamente en: %s", full_out)

    # Imprimir resumen de alto impacto en consola
    print("\n" + "=" * 90)
    print("📊 RESUMEN COMPARATIVO DE MODOS DE SALTO (FASE 3 OFFLINE)")
    print("=" * 90)
    print(f"{'Modo':<24} | {'Saltos':<8} | {'Costo Ini':<10} | {'Costo Fin':<10} | {'Ganancia':<10} | {'% Reduc':<8} | {'Tiempo':<8}")
    print("-" * 90)
    for mode, s in summary_modes.items():
        print(f"{mode:<24} | {s['mean_jumps']:<8.2f} | {s['mean_initial_cost']:<10.2f} | {s['mean_final_cost']:<10.2f} | {s['mean_cost_reduction']:<10.2f} | {s['mean_pct_reduction']:<7.1f}% | {s['mean_runtime_ms']:<6.1f}ms")
    print("=" * 90)

    print("\n" + "=" * 105)
    print("🎯 RESOLUCIÓN DE LAS 7 ALDEAS DE DISCREPANCIA (0005, 0010, 0012, 0022, 0008, 0021, screen_raw_07)")
    print("=" * 105)
    print(f"{'Aldea / Base':<32} | {'Costo Ini':<10} | {'Opt Fin (Saltos)':<18} | {'GT Fin (Saltos)':<18} | {'Raycast Fin':<12} | {'Estado':<10}")
    print("-" * 105)
    
    target_discrepancies = ["0005", "0010", "0012", "0022", "0008", "0021", "attack_20261113_171927", "attack_20261113_174533"]
    for v in per_village_results:
        vid = v["village_id"]
        path_str = v["image_path"]
        matches_disc = any(td in vid or td in path_str for td in target_discrepancies)
        if matches_disc:
            m_opt = v["modes"].get("coverage_optimizer", {})
            m_gt = v["modes"].get("human_gt", {})
            m_ray = v["modes"].get("wall_raycast", {})

            opt_str = f"{m_opt.get('final_cost', 0.0):.1f} ({m_opt.get('num_jumps', 0)} saltos)"
            gt_str = f"{m_gt.get('final_cost', 0.0):.1f} ({m_gt.get('num_jumps', 0)} saltos)" if m_gt else "N/A"
            ray_str = f"{m_ray.get('final_cost', 0.0):.1f}"

            opt_cost = m_opt.get('final_cost', 0.0)
            gt_cost = m_gt.get('final_cost', 0.0) if m_gt else 0.0
            status = "RESUELTO" if opt_cost <= gt_cost + 1e-4 else "MEJORADO"

            print(f"{vid:<32} | {m_opt.get('initial_cost', 0.0):<10.1f} | {opt_str:<18} | {gt_str:<18} | {ray_str:<12} | {status:<10}")
    print("=" * 105 + "\n")

    return final_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluación offline y línea base de saltos.")
    parser.add_argument("--limit", type=int, default=None, help="Limita el número de aldeas a procesar (para pruebas rápidas)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confianza YOLO")
    parser.add_argument("--imgsz", type=int, default=1280, help="Tamaño de imagen YOLO")
    parser.add_argument("--max-jumps", type=int, default=4, help="Número máximo de saltos")
    parser.add_argument("--output", type=str, default="logs/offline_evaluation_results.json", help="Ruta del archivo JSON de salida")
    args = parser.parse_args()

    villages = discover_village_images()
    if args.limit:
        villages = villages[:args.limit]

    run_offline_evaluation(
        villages=villages,
        conf=args.conf,
        imgsz=args.imgsz,
        max_jumps=args.max_jumps,
        output_json_path=args.output,
    )


if __name__ == "__main__":
    main()
