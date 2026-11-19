"""
attack/jump_planner.py
Planificador modular de Hechizos de Salto (Fase 4 - Mejora 05).
Despachador unificado que soporta:
  1. 'coverage_optimizer' / 'compartments': U-Net + Beam Search por compartimentos topológicos.
  2. 'wall_raycast': Raycasting físico real de aristas e intersecciones de muros (OpenCV).
  3. 'cluster_intersections': Agrupamiento voraz e intersecciones de edificios de botín.
  4. 'legacy_quadrant': Heurística heredada por cuadrantes.
"""
from __future__ import annotations
import math
import sys
import json
import os
import datetime
import cv2
import numpy as np

from core.logger import get_logger
from core.config import SETTINGS as DEFAULT_SETTINGS
from core.vision import save_debug_image_async
from vision.yolo_detector import Detection
from vision.wall_raycast import (
    estimate_iso_angles,
    detect_global_wall_segments,
    find_wall_crossings_from_segments,
    draw_wall_raycast_debug,
    find_wall_junctions,
    calculate_adaptive_jump_and_breach_placements,
)

log = get_logger("jump_planner")

# Constantes canónicas de referencia 2400x1080
CANONICAL_CENTER = (1220.0, 500.0)
CANONICAL_RX_BUILDABLE = 885.0
CANONICAL_RY_BUILDABLE = 664.0
CANONICAL_RX_GRASS = 955.0
CANONICAL_RY_GRASS = 715.0


def _get_settings() -> dict:
    """Obtiene SETTINGS dinámicamente, respetando mocks aplicados sobre deployment_strategy en tests."""
    if "attack.deployment_strategy" in sys.modules:
        ds = sys.modules["attack.deployment_strategy"]
        if hasattr(ds, "SETTINGS"):
            return ds.SETTINGS
    return DEFAULT_SETTINGS


def compute_canonical_diamond_drop(
    building_center: tuple[float, float],
    center_pt: tuple[float, float] = CANONICAL_CENTER,
    rx_grass: float = CANONICAL_RX_GRASS,
    ry_grass: float = CANONICAL_RY_GRASS,
    safe_y_min: int = 138,
    safe_y_max: int = 845,
    safe_x_min: int = 120,
    safe_x_max: int = 2280,
) -> tuple[int, int]:
    """
    Calcula el punto de drop exacto sobre el anillo de césped exterior legal (2-2.5 tiles fuera de la aldea):
    1. Traza el rayo director u desde el centro de la aldea C pasando por el centro del edificio B.
    2. Calcula la intersección analítica exacta con el rombo canónico exterior:
       |x - Cx| / Rx + |y - Cy| / Ry = 1
       t = 1.0 / (|ux| / Rx + |uy| / Ry)
    3. Si la coordenada Y toca el HUD superior o las cartas inferiores,
       desliza el punto sobre el perímetro del rombo hacia los flancos seguros.
    """
    cx, cy = center_pt
    bx, by = building_center
    dx = float(bx - cx)
    dy = float(by - cy)
    dist = float(np.hypot(dx, dy))

    if dist < 1e-3:
        ux, uy = 0.0, 1.0
    else:
        ux, uy = dx / dist, dy / dist

    denom = (abs(ux) / rx_grass) + (abs(uy) / ry_grass)
    t_intersec = 1.0 / max(denom, 1e-5)

    margin = 32.0
    drop_x = cx + (t_intersec + margin) * ux
    drop_y = cy + (t_intersec + margin) * uy

    if drop_y < safe_y_min:
        target_y = float(safe_y_min)
        rem_y = abs(target_y - cy) / ry_grass
        span_x = rx_grass * max(0.0, 1.0 - rem_y)
        sign_x = 1.0 if dx >= 0 else -1.0
        drop_x = cx + sign_x * (span_x + margin)
        drop_y = target_y
    elif drop_y > safe_y_max:
        target_y = float(safe_y_max)
        rem_y = abs(target_y - cy) / ry_grass
        span_x = rx_grass * max(0.0, 1.0 - rem_y)
        sign_x = 1.0 if dx >= 0 else -1.0
        drop_x = cx + sign_x * (span_x + margin)
        drop_y = target_y

    final_x = int(np.clip(drop_x, safe_x_min, safe_x_max))
    final_y = int(np.clip(drop_y, safe_y_min, safe_y_max))
    return final_x, final_y


def _compute_legacy_quadrant_jump_targets(
    detections: list[Detection], screen_size: tuple[int, int]
) -> list[tuple[int, int]]:
    """Heurística heredada de proyección por cuadrantes (jump_placement='legacy_quadrant')."""
    w, h = screen_size
    valid_classes = {"storage_gold", "storage_elixir", "storage_dark", "collector_dark"}
    valid_storages = [d for d in detections if d.cls_name in valid_classes and d.confidence >= 0.50]

    if not valid_storages:
        log.warning("No se encontraron almacenes válidos (conf >= 0.50) para cálculo de Saltos.")
        return []

    all_points = np.array([d.center for d in detections], dtype=np.float32)
    if len(all_points) >= 3:
        hull = cv2.convexHull(all_points)
        M = cv2.moments(hull)
        if M["m00"] > 0:
            center_x = float(M["m10"] / M["m00"])
            center_y = float(M["m01"] / M["m00"])
        else:
            center_x = float(np.mean(all_points[:, 0]))
            center_y = float(np.mean(all_points[:, 1]))
    elif len(all_points) > 0:
        center_x = float(np.mean(all_points[:, 0]))
        center_y = float(np.mean(all_points[:, 1]))
    else:
        center_x = float(w * 0.50)
        center_y = float(h * 0.50)

    log.info("Centroide de base YOLO (legacy): (%.1f, %.1f) a partir de %d detecciones", center_x, center_y, len(all_points))

    core_radius = 160
    quadrants_storages = {"SW": [], "SE": [], "N": [], "CORE": []}
    quadrants_projected = {"SW": [], "SE": [], "N": [], "CORE": []}

    for s in valid_storages:
        sx, sy = s.center
        dist = float(np.hypot(sx - center_x, sy - center_y))
        dx = sx - center_x
        dy = sy - center_y
        proj_x = int(round(sx + 0.40 * dx))
        proj_y = int(round(sy + 0.40 * dy))

        proj_x = int(np.clip(proj_x, int(0.12 * w), int(0.88 * w)))
        proj_y = int(np.clip(proj_y, int(0.12 * h), int(0.88 * h)))

        if dist < core_radius:
            quad = "CORE"
        elif sy >= center_y:
            quad = "SW" if sx < center_x else "SE"
        else:
            quad = "N"

        quadrants_storages[quad].append(s)
        quadrants_projected[quad].append((proj_x, proj_y))

    active_quads = [q for q in ("SW", "SE", "N", "CORE") if len(quadrants_storages[q]) > 0]
    active_quads.sort(key=lambda q: len(quadrants_storages[q]), reverse=True)
    top_quads = active_quads[:2]

    targets: list[tuple[int, int]] = []
    for q in top_quads:
        pts = quadrants_projected[q]
        avg_x = int(round(np.mean([p[0] for p in pts])))
        avg_y = int(round(np.mean([p[1] for p in pts])))
        targets.append((avg_x, avg_y))

    log.info("🎯 Jump targets calculados con YOLO legacy (máx 2): %s (cuadrantes: %s)", targets, top_quads)
    return targets


def _compute_cluster_intersection_jump_targets(
    detections: list[Detection], screen_size: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Algoritmo de Intersecciones de Muros y Máxima Cobertura para Hechizos de Salto:
    1. Genera candidatos de intersección entre parejas y tríos de almacenes cercanos.
    2. Genera candidatos de muro de entrada para almacenes aislados.
    3. Evalúa el coverage scoring ponderado por loot.
    4. Selecciona secuencialmente hasta max_jumps garantizando separación mínima R.
    """
    w, h = screen_size
    attack_cfg = _get_settings().get("attack", {})

    radius = float(attack_cfg.get("jump_spell_radius_px", 90))
    pair_min_dist = float(attack_cfg.get("pair_min_dist_px", 65))
    pair_max_dist = float(attack_cfg.get("pair_max_dist_px", 210))
    max_jumps = int(attack_cfg.get("max_jumps", 3))
    min_jump_score = float(attack_cfg.get("min_jump_score", 2.0))
    min_sep = float(attack_cfg.get("min_separation_between_jumps_px", 110))
    entry_offset_ratio = float(attack_cfg.get("jump_entry_wall_offset_ratio", 0.28))

    loot_weights = attack_cfg.get("jump_loot_weights", {
        "storage_dark": 3,
        "storage_gold": 2,
        "storage_elixir": 2,
        "collector_dark": 1,
        "collector_gold": 1,
        "collector_elixir": 1,
        "town_hall": 0,
    })

    yolo_conf = float(attack_cfg.get("yolo_conf", 0.40))
    valid_loot = [
        d for d in detections
        if loot_weights.get(d.cls_name, 0) > 0 and d.confidence >= yolo_conf
    ]
    storages = [d for d in valid_loot if "storage" in d.cls_name]

    if not valid_loot:
        log.warning("No hay almacenes ni recolectores detectados con suficiente confianza para saltos.")
        return []

    cx_base = float(attack_cfg.get("canonical_center_x", 1220))
    cy_base = float(attack_cfg.get("canonical_center_y", 500))
    center = (cx_base, cy_base)

    candidates: list[dict] = []

    # A) Candidatos por pareja de almacenes
    for i in range(len(storages)):
        for j in range(i + 1, len(storages)):
            s1, s2 = storages[i], storages[j]
            d_pair = float(np.hypot(s1.center[0] - s2.center[0], s1.center[1] - s2.center[1]))
            if pair_min_dist <= d_pair <= pair_max_dist:
                mid_x = (s1.center[0] + s2.center[0]) / 2.0
                mid_y = (s1.center[1] + s2.center[1]) / 2.0
                candidates.append({"point": (int(round(mid_x)), int(round(mid_y))), "type": "pair"})

    # B) Candidatos por trío de almacenes
    for i in range(len(storages)):
        for j in range(i + 1, len(storages)):
            for k in range(j + 1, len(storages)):
                s1, s2, s3 = storages[i], storages[j], storages[k]
                d12 = np.hypot(s1.center[0] - s2.center[0], s1.center[1] - s2.center[1])
                d23 = np.hypot(s2.center[0] - s3.center[0], s2.center[1] - s3.center[1])
                d31 = np.hypot(s3.center[0] - s1.center[0], s3.center[1] - s1.center[1])
                if max(d12, d23, d31) <= pair_max_dist:
                    t_x = (s1.center[0] + s2.center[0] + s3.center[0]) / 3.0
                    t_y = (s1.center[1] + s2.center[1] + s3.center[1]) / 3.0
                    candidates.append({"point": (int(round(t_x)), int(round(t_y))), "type": "trio"})

    # C) Candidatos de muro de entrada para almacenes aislados
    for s in storages:
        drop_x, drop_y = compute_canonical_diamond_drop(s.center, center_pt=center)
        v_dx = float(drop_x - s.center[0])
        v_dy = float(drop_y - s.center[1])
        v_len = max(float(np.hypot(v_dx, v_dy)), 1e-4)
        c_entry_x = s.center[0] + entry_offset_ratio * radius * (v_dx / v_len)
        c_entry_y = s.center[1] + entry_offset_ratio * radius * (v_dy / v_len)
        candidates.append({"point": (int(round(c_entry_x)), int(round(c_entry_y))), "type": "isolated"})

    min_x, max_x = int(0.12 * w), int(0.88 * w)
    min_y, max_y = int(0.14 * h), int(0.84 * h)
    for c in candidates:
        cx_c = int(np.clip(c["point"][0], min_x, max_x))
        cy_c = int(np.clip(c["point"][1], min_y, max_y))
        c["point"] = (cx_c, cy_c)

    scored_candidates = []
    for c in candidates:
        px, py = c["point"]
        covered = [
            b for b in valid_loot
            if np.hypot(b.center[0] - px, b.center[1] - py) <= radius
        ]
        score = sum(loot_weights.get(b.cls_name, 0) for b in covered)
        scored_candidates.append({
            "point": (px, py),
            "score": float(score),
            "type": c["type"],
            "covered_classes": [b.cls_name for b in covered],
        })

    scored_candidates.sort(key=lambda item: item["score"], reverse=True)

    selected_jumps = []
    for cand in scored_candidates:
        if len(selected_jumps) >= max_jumps:
            break
        if cand["score"] < min_jump_score:
            continue
        pt = cand["point"]
        if any(np.hypot(pt[0] - sj["point"][0], pt[1] - sj["point"][1]) < min_sep for sj in selected_jumps):
            continue
        selected_jumps.append(cand)

    targets = [j["point"] for j in selected_jumps]

    try:
        out_dir = os.path.join("logs", "yolo")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        decision_file = os.path.join(out_dir, f"jump_decision_{ts}.json")
        decision_log = {
            "timestamp": ts,
            "algorithm": "cluster_intersections",
            "screen_size": list(screen_size),
            "radius": radius,
            "max_jumps": max_jumps,
            "min_jump_score": min_jump_score,
            "candidates_evaluated": len(scored_candidates),
            "selected_jumps": [
                {
                    "point": j["point"],
                    "type": j["type"],
                    "score": j["score"],
                    "covered_classes": j["covered_classes"],
                }
                for j in selected_jumps
            ],
        }
        with open(decision_file, "w", encoding="utf-8") as f:
            json.dump(decision_log, f, indent=2, ensure_ascii=False)
        log.info("📝 Registro de decisión de saltos guardado en: %s", decision_file)
    except Exception as e_json:
        log.warning("No se pudo guardar registro JSON de decisión de saltos: %s", e_json)

    log.info("🎯 Jump targets calculados con cluster_intersections (total: %d): %s", len(targets), targets)
    return targets


def _compute_wall_raycast_jump_targets(
    screen: np.ndarray,
    detections: list[Detection],
    screen_size: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Calcula objetivos de Hechizos de Salto detectando la ubicación física real
    de los muros mediante raycasting de aristas OpenCV (vision/wall_raycast.py).
    """
    if screen is None or getattr(screen, "size", 0) == 0 or not detections:
        log.warning("⚠️ Screen o detecciones no disponibles para Wall Raycast. Usando cluster_intersections.")
        return _compute_cluster_intersection_jump_targets(detections, screen_size)

    w, h = screen_size
    attack_cfg = _get_settings().get("attack", {})
    max_jumps = int(attack_cfg.get("max_jumps", 4))
    min_sep = float(attack_cfg.get("min_separation_between_jumps_px", 110))
    wall_inset = float(attack_cfg.get("wall_inset_px", 20.0))
    tol_deg = float(attack_cfg.get("wall_tol_deg", 10.0))
    canny_low = int(attack_cfg.get("canny_low", 60))
    canny_high = int(attack_cfg.get("canny_high", 160))
    hough_thresh = int(attack_cfg.get("hough_threshold", 30))
    hough_min_len = int(attack_cfg.get("hough_min_len", 70))
    hough_max_gap = int(attack_cfg.get("hough_max_gap", 10))
    min_crossing_angle = float(attack_cfg.get("min_crossing_angle_deg", 25.0))
    min_dist_drop = float(attack_cfg.get("min_dist_from_drop_px", 40.0))
    min_dist_target = float(attack_cfg.get("min_dist_from_target_px", 50.0))
    endpoint_tol = float(attack_cfg.get("wall_endpoint_tol_px", 30.0))

    loot_weights = attack_cfg.get("jump_loot_weights", {
        "storage_dark": 3,
        "storage_gold": 2,
        "storage_elixir": 2,
        "town_hall": 0,
        "collector_dark": 1,
    })

    iso_angles, wall_segments = detect_global_wall_segments(
        screen,
        min_line_len=hough_min_len,
        max_line_gap=hough_max_gap,
        tol_deg=tol_deg,
        canny_low=canny_low,
        canny_high=canny_high,
        hough_thresh=hough_thresh,
        detections=detections,
    )

    # Filtrar y ordenar almacenes por peso de botín
    primary_targets = [
        d for d in detections
        if d.cls_name in {"storage_dark", "storage_gold", "storage_elixir", "town_hall"}
    ]
    if not primary_targets:
        primary_targets = [
            d for d in detections
            if "storage" in d.cls_name or "collector" in d.cls_name
        ]
    if not primary_targets:
        log.warning("No se encontraron objetivos de botín para Wall Raycast.")
        return _compute_cluster_intersection_jump_targets(detections, screen_size)

    primary_targets.sort(
        key=lambda d: loot_weights.get(d.cls_name, 0),
        reverse=True
    )

    # Centro de la aldea
    all_pts = np.array([d.center for d in detections], dtype=np.float32)
    center = (float(np.mean(all_pts[:, 0])), float(np.mean(all_pts[:, 1])))

    # 1. Intentar cálculo por Uniones Isométricas y Demanda Adaptativa
    if attack_cfg.get("adaptive_spells_by_demand", True):
        targets_with_drops = []
        try:
            from vision.deploy_zone_detector import get_deploy_zone_detector
            dz = get_deploy_zone_detector()
        except Exception:
            dz = None

        for target in primary_targets:
            if dz and dz.is_ready():
                drop_x, drop_y = dz.find_closest_deployable_point(
                    int(round(target.center[0])), int(round(target.center[1])), margin_px=25,
                    safe_bounds=(int(0.12 * w), int(0.14 * h), int(0.88 * w), int(0.84 * h))
                )
            else:
                p_target = (int(round(target.center[0])), int(round(target.center[1])))
                vx = p_target[0] - center[0]
                vy = p_target[1] - center[1]
                dist_c = max(float(np.hypot(vx, vy)), 1e-3)
                drop_x = int(np.clip(p_target[0] + 250.0 * (vx / dist_c), int(0.12 * w), int(0.88 * w)))
                drop_y = int(np.clip(p_target[1] + 250.0 * (vy / dist_c), int(0.14 * h), int(0.84 * h)))
            targets_with_drops.append((target, (drop_x, drop_y)))

        junctions = find_wall_junctions(wall_segments, iso_angles=iso_angles)
        selected_jumps, selected_swb, debug_rays = calculate_adaptive_jump_and_breach_placements(
            targets_with_drops=targets_with_drops,
            wall_segments=wall_segments,
            junctions=junctions,
            max_jumps=max_jumps,
            min_dist_between_jumps=min_sep,
            safe_y_min=int(0.14 * h),
            safe_y_max=int(0.84 * h),
            safe_x_min=int(0.12 * w),
            safe_x_max=int(0.88 * w),
        )

        try:
            draw_wall_raycast_debug(
                screen,
                debug_rays,
                selected_jumps,
                junctions=junctions,
                out_path="logs/yolo/wall_raycast_debug.png"
            )
        except Exception as e_dbg:
            log.warning("No se pudo renderizar wall_raycast_debug.png: %s", e_dbg)

        log.info("🎯 Jump targets calculados por Demanda Adaptativa e Intersecciones (total: %d): %s", len(selected_jumps), selected_jumps)
        if selected_jumps:
            return selected_jumps

    def _get_sector(pt: tuple[int, int], c_pt: tuple[float, float]) -> str:
        dx_s, dy_s = pt[0] - c_pt[0], pt[1] - c_pt[1]
        return ("N" if dy_s < 0 else "S") + ("W" if dx_s < 0 else "E")

    candidates = []
    debug_rays = []

    for target in primary_targets:
        p_target = (int(target.center[0]), int(target.center[1]))
        vx = p_target[0] - center[0]
        vy = p_target[1] - center[1]
        dist_c = max(float(np.hypot(vx, vy)), 1e-3)
        ux, uy = vx / dist_c, vy / dist_c

        p_drop_x = int(np.clip(p_target[0] + 250.0 * ux, int(0.10 * w), int(0.90 * w)))
        p_drop_y = int(np.clip(p_target[1] + 250.0 * uy, int(0.14 * h), int(0.82 * h)))
        p_drop = (p_drop_x, p_drop_y)

        dist_ray = float(np.hypot(p_target[0] - p_drop[0], p_target[1] - p_drop[1]))
        if dist_ray < (min_dist_drop + min_dist_target):
            continue

        debug_entry = {}
        crossings = find_wall_crossings_from_segments(
            p_drop=p_drop,
            p_target=p_target,
            wall_segments=wall_segments,
            min_crossing_angle_deg=min_crossing_angle,
            min_dist_drop=min_dist_drop,
            min_dist_target=min_dist_target,
            endpoint_tol=endpoint_tol,
            debug_collector=debug_entry,
        )
        debug_rays.append(debug_entry)

        if crossings:
            log.info(
                "🧱 [RAYCAST] %s en %s: %d cruce(s) de muro físico detectado(s): %s",
                target.cls_name, p_target, len(crossings), crossings
            )
            for idx_cr, (cx_w, cy_w) in enumerate(crossings):
                vec_tx = p_target[0] - cx_w
                vec_ty = p_target[1] - cy_w
                dist_t = max(float(np.hypot(vec_tx, vec_ty)), 1e-3)
                t_ux, t_uy = vec_tx / dist_t, vec_ty / dist_t

                jx = int(round(cx_w + wall_inset * t_ux))
                jy = int(round(cy_w + wall_inset * t_uy))

                bonus = 2.0 if (idx_cr == 0 or len(crossings) == 1) else 1.5
                if target.cls_name in {"storage_dark", "town_hall"} and idx_cr > 0:
                    bonus += 1.0

                cand_pt = (jx, jy)
                candidates.append({
                    "pt": cand_pt,
                    "target": target.cls_name,
                    "sector": _get_sector(cand_pt, center),
                    "priority": float(loot_weights.get(target.cls_name, 2)) + bonus,
                    "is_fallback": False,
                })
        else:
            log.info(
                "⚠️ [RAYCAST] %s en %s: 0 cruces detectados. Usando respaldo por ratio (0.75)...",
                target.cls_name, p_target
            )
            jx = int(round(p_drop[0] + 0.75 * (p_target[0] - p_drop[0])))
            jy = int(round(p_drop[1] + 0.75 * (p_target[1] - p_drop[1])))
            cand_pt = (jx, jy)
            candidates.append({
                "pt": cand_pt,
                "target": target.cls_name,
                "sector": _get_sector(cand_pt, center),
                "priority": float(loot_weights.get(target.cls_name, 2)),
                "is_fallback": True,
            })

    min_x, max_x = int(0.12 * w), int(0.88 * w)
    min_y, max_y = int(0.14 * h), int(0.84 * h)
    for c in candidates:
        cx_c = int(np.clip(c["pt"][0], min_x, max_x))
        cy_c = int(np.clip(c["pt"][1], min_y, max_y))
        c["pt"] = (cx_c, cy_c)

    candidates.sort(key=lambda c: c["priority"], reverse=True)

    available_sectors = set(c["sector"] for c in candidates)
    used_sectors = set()
    selected_jumps: list[tuple[int, int]] = []

    for step in range(max_jumps):
        best_cand = None
        best_score = -1.0
        rem_sectors = available_sectors - used_sectors

        for cand in candidates:
            pt = cand["pt"]
            if any(np.hypot(pt[0] - sj[0], pt[1] - sj[1]) < min_sep for sj in selected_jumps):
                continue
            if rem_sectors and cand["sector"] in used_sectors:
                continue
            if cand["priority"] > best_score:
                best_score = cand["priority"]
                best_cand = cand

        if best_cand:
            selected_jumps.append(best_cand["pt"])
            used_sectors.add(best_cand["sector"])

    if len(selected_jumps) < max_jumps:
        for cand in candidates:
            pt = cand["pt"]
            if all(np.hypot(pt[0] - sj[0], pt[1] - sj[1]) >= min_sep for sj in selected_jumps):
                selected_jumps.append(pt)
                if len(selected_jumps) >= max_jumps:
                    break

    if len(selected_jumps) < max_jumps:
        fallback_targets = _compute_cluster_intersection_jump_targets(detections, screen_size)
        for fb in fallback_targets:
            if all(np.hypot(fb[0] - sj[0], fb[1] - sj[1]) >= min_sep for sj in selected_jumps):
                selected_jumps.append(fb)
                if len(selected_jumps) >= max_jumps:
                    break

    try:
        draw_wall_raycast_debug(
            screen,
            debug_rays,
            selected_jumps,
            out_path="logs/yolo/wall_raycast_debug.png"
        )
    except Exception as e_dbg:
        log.warning("No se pudo renderizar wall_raycast_debug.png: %s", e_dbg)

    log.info("🎯 Jump targets calculados con Wall Raycast (total: %d): %s", len(selected_jumps), selected_jumps)
    return selected_jumps


def _compute_compartment_jump_targets(
    screen: np.ndarray | None,
    detections: list[Detection],
    screen_size: tuple[int, int],
    drop_points: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """
    Calcula saltos óptimos mediante segmentación de muros U-Net y optimización de
    cobertura por compartimentos con Beam Search (vision/wall_compartments.py).
    """
    if screen is None or getattr(screen, "size", 0) == 0:
        log.warning("⚠️ Modo 'compartments' / 'coverage_optimizer' activo pero screen es None; delegando a respaldo.")
        return []

    from vision.wall_compartments import optimize_jumps

    w, h = screen_size
    cfg = _get_settings()
    save_debug = cfg.get("debug", {}).get("save_debug_images", False)
    loot_weights = cfg.get("attack", {}).get("jump_loot_weights")

    dets_payload = [
        {
            "cls_name": getattr(d, "cls_name", ""),
            "center": getattr(d, "center", (0, 0)),
            "bbox_xyxy": getattr(d, "bbox_xyxy", (0, 0, 0, 0)),
            "confidence": getattr(d, "confidence", 1.0),
        }
        for d in detections
    ]

    res = optimize_jumps(
        img_shape=(h, w),
        base_img=screen,
        detections=dets_payload,
        drop_points=drop_points,
        loot_weights=loot_weights,
        generate_debug_overlay=save_debug,
    )

    if res.debug_overlay is not None and save_debug:
        save_debug_image_async("logs/yolo/compartments_debug.png", res.debug_overlay)

    targets = [(int(round(x)), int(round(y))) for x, y, _ in res.jumps]
    log.info("🎯 Jump targets calculados con U-Net + Beam Search (total: %d): %s", len(targets), targets)
    return targets


def compute_adaptive_jump_targets_from_detections(
    detections: list[Detection],
    screen_size: tuple[int, int],
    screen: np.ndarray | None = None,
    drop_points: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """
    Punto de entrada unificado para el cálculo de objetivos de Hechizos de Salto.
    Selecciona el algoritmo según config/settings.json -> attack.jump_placement:
    - 'coverage_optimizer' / 'compartments': U-Net + Beam Search por compartimentos.
    - 'wall_raycast' (por defecto): Detección física real de muros mediante OpenCV raycasting de aristas.
    - 'cluster_intersections': Intersección de muros compartidos y máxima cobertura.
    - 'legacy_quadrant': Heurística heredada de proyección por cuadrantes.
    """
    cfg = _get_settings()
    mode = cfg.get("attack", {}).get("jump_placement", "wall_raycast")

    if mode in {"coverage_optimizer", "compartments"}:
        if screen is not None:
            targets = _compute_compartment_jump_targets(screen, detections, screen_size, drop_points=drop_points)
            if targets:
                return targets
            log.warning("⚠️ Modo %s no devolvió saltos; ejecutando respaldo wall_raycast.", mode)
        else:
            log.warning("⚠️ Modo %s activo pero screen es None; usando respaldo wall_raycast.", mode)

        if screen is not None:
            targets = _compute_wall_raycast_jump_targets(screen, detections, screen_size)
            if targets:
                return targets
        return _compute_cluster_intersection_jump_targets(detections, screen_size)

    if mode == "wall_raycast":
        if screen is not None:
            targets = _compute_wall_raycast_jump_targets(screen, detections, screen_size)
            if targets:
                return targets
            log.warning("⚠️ Wall raycast no devolvió saltos; ejecutando respaldo cluster_intersections.")
        else:
            log.warning("⚠️ Modo wall_raycast activo pero screen es None; usando respaldo cluster_intersections.")
        return _compute_cluster_intersection_jump_targets(detections, screen_size)

    elif mode == "legacy_quadrant":
        return _compute_legacy_quadrant_jump_targets(detections, screen_size)

    return _compute_cluster_intersection_jump_targets(detections, screen_size)
