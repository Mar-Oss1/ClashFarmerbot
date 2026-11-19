"""
Módulo de Detección Real de Muros en Clash of Clans mediante Raycasting OpenCV.
Optimizado con detección global de segmentos Hough en una sola pasada por batalla
y filtrado geométrico robusto de cruces (ángulo de cruce, distancia de seguridad
y tolerancia colineal de extremos).
"""
from __future__ import annotations
import math
import os
import zlib
import cv2
import numpy as np
from core.logger import get_logger
from core.vision import save_debug_image_async

log = get_logger("wall_raycast")

# Caché por batalla para ángulos isométricos y segmentos de muro globales
_ISO_ANGLES_CACHE: tuple[float, float] | None = None
_WALL_SEGMENTS_CACHE: list[tuple[int, int, int, int]] | None = None
_LAST_SCREEN_HASH: int | None = None


def _compute_frame_fingerprint(img: np.ndarray) -> int:
    """
    Calcula una huella digital determinista, rápida (<0.1 ms) y sensible a cambios de aldea:
    1. Recorta la región central de la aldea (y: 15%..85%, x: 15%..85%).
    2. Reduce con cv2.resize a (32, 16) con INTER_AREA.
    3. Retorna zlib.adler32 de los bytes del thumbnail.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return 0
    h, w = img.shape[:2]
    y1, y2 = int(0.15 * h), int(0.85 * h)
    x1, x2 = int(0.15 * w), int(0.85 * w)
    center_crop = img[y1:y2, x1:x2]
    if center_crop.size == 0:
        return 0
    thumb = cv2.resize(center_crop, (32, 16), interpolation=cv2.INTER_AREA)
    return zlib.adler32(thumb.tobytes())


def reset_iso_angles_cache() -> None:
    """Limpia la caché de ángulos isométricos y segmentos al iniciar una nueva batalla."""
    global _ISO_ANGLES_CACHE, _WALL_SEGMENTS_CACHE, _LAST_SCREEN_HASH
    _ISO_ANGLES_CACHE = None
    _WALL_SEGMENTS_CACHE = None
    _LAST_SCREEN_HASH = None


def _angle_diff_180(ang1: float, ang2: float) -> float:
    """Calcula la menor diferencia angular entre dos ángulos en el rango [0, 180)."""
    diff = abs((ang1 - ang2) % 180.0)
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def detect_global_wall_segments(
    img: np.ndarray,
    force_refresh: bool = False,
    min_line_len: int = 70,
    max_line_gap: int = 10,
    tol_deg: float = 10.0,
    canny_low: int = 60,
    canny_high: int = 160,
    hough_thresh: int = 30,
    detections: list | None = None,
) -> tuple[tuple[float, float], list[tuple[int, int, int, int]]]:
    """
    Ejecuta HoughLinesP una sola vez sobre toda el área jugable de la aldea:
    1. Extrae los ángulos isométricos dominantes mediante histograma en bins de 5°.
    2. Filtra y almacena únicamente los segmentos largos (>= min_line_len) alineados
       con los ángulos isométricos dominantes (+- tol_deg).
    3. Almacena en caché por batalla para eliminar recálculos por cada almacén.
    
    Retorna: ((iso1, iso2), [segmentos_muro (x1, y1, x2, y2)])
    """
    global _ISO_ANGLES_CACHE, _WALL_SEGMENTS_CACHE, _LAST_SCREEN_HASH
    if img is None or img.size == 0:
        return (36.0, 144.0), []

    screen_hash = _compute_frame_fingerprint(img)
    if (
        not force_refresh
        and _ISO_ANGLES_CACHE is not None
        and _WALL_SEGMENTS_CACHE is not None
        and _LAST_SCREEN_HASH == screen_hash
    ):
        return _ISO_ANGLES_CACHE, _WALL_SEGMENTS_CACHE

    h, w = img.shape[:2]
    # Recorte del área de juego central (evita el HUD superior y la barra de tropas inferior)
    y1_roi, y2_roi = int(0.14 * h), int(0.82 * h)
    x1_roi, x2_roi = int(0.10 * w), int(0.90 * w)
    roi = img[y1_roi:y2_roi, x1_roi:x2_roi]

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    if detections:
        for det in detections:
            if hasattr(det, "bbox_xyxy"):
                bx1, by1, bx2, by2 = det.bbox_xyxy
                # Convertir a coordenadas locales del ROI y dejar margen de 6px en los bordes
                lx1 = max(0, (bx1 + 6) - x1_roi)
                ly1 = max(0, (by1 + 6) - y1_roi)
                lx2 = min(roi.shape[1], (bx2 - 6) - x1_roi)
                ly2 = min(roi.shape[0], (by2 - 6) - y1_roi)
                if lx2 > lx1 and ly2 > ly1:
                    edges[ly1:ly2, lx1:lx2] = 0

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(hough_thresh),
        minLineLength=int(min_line_len),
        maxLineGap=int(max_line_gap),
    )

    default_angles = (36.0, 144.0)
    if lines is None or len(lines) == 0:
        _ISO_ANGLES_CACHE = default_angles
        _WALL_SEGMENTS_CACHE = []
        _LAST_SCREEN_HASH = screen_hash
        return default_angles, []

    angles = []
    converted_lines = []
    for line in lines:
        pts = line[0] if (hasattr(line, "__len__") and len(line) == 1 and hasattr(line[0], "__len__")) else line
        lx1, ly1, lx2, ly2 = pts[0], pts[1], pts[2], pts[3]
        # Reconvertir a coordenadas absolutas de la pantalla
        gx1, gy1 = lx1 + x1_roi, ly1 + y1_roi
        gx2, gy2 = lx2 + x1_roi, ly2 + y1_roi
        dx = gx2 - gx1
        dy = gy2 - gy1
        deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
        angles.append(deg)
        converted_lines.append((gx1, gy1, gx2, gy2, deg))

    # Histograma de 0 a 180 grados en bins de 5 grados (36 bins)
    hist, bin_edges = np.histogram(angles, bins=36, range=(0.0, 180.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Pico 1: Cuadrante 1 (15° a 75°, típico ~30°-45°)
    mask_q1 = (bin_centers >= 15.0) & (bin_centers <= 75.0)
    if np.any(mask_q1) and np.max(hist[mask_q1]) > 0:
        best_idx_q1 = np.argmax(hist[mask_q1])
        peak_q1 = float(bin_centers[mask_q1][best_idx_q1])
    else:
        peak_q1 = 36.0

    # Pico 2: Cuadrante 2 (105° a 165°, típico ~135°-150°)
    mask_q2 = (bin_centers >= 105.0) & (bin_centers <= 165.0)
    if np.any(mask_q2) and np.max(hist[mask_q2]) > 0:
        best_idx_q2 = np.argmax(hist[mask_q2])
        peak_q2 = float(bin_centers[mask_q2][best_idx_q2])
    else:
        peak_q2 = 144.0

    iso_angles = (peak_q1, peak_q2)

    # Filtrar solo líneas alineadas con los ángulos isométricos dominantes
    valid_wall_segments: list[tuple[int, int, int, int]] = []
    for gx1, gy1, gx2, gy2, deg in converted_lines:
        diff1 = _angle_diff_180(deg, iso_angles[0])
        diff2 = _angle_diff_180(deg, iso_angles[1])
        if (diff1 <= tol_deg) or (diff2 <= tol_deg):
            valid_wall_segments.append((gx1, gy1, gx2, gy2))

    log.debug(
        "🧱 [GLOBAL] Segmentos de muro detectados: %d (Ángulos: %.1f°, %.1f°)",
        len(valid_wall_segments), peak_q1, peak_q2
    )

    _ISO_ANGLES_CACHE = iso_angles
    _WALL_SEGMENTS_CACHE = valid_wall_segments
    _LAST_SCREEN_HASH = screen_hash
    return _ISO_ANGLES_CACHE, _WALL_SEGMENTS_CACHE


def estimate_iso_angles(img: np.ndarray, force_refresh: bool = False) -> tuple[float, float]:
    """Wrapper de conveniencia que devuelve los ángulos isométricos dominantes."""
    angles, _ = detect_global_wall_segments(img, force_refresh=force_refresh)
    return angles


def find_wall_crossings_from_segments(
    p_drop: tuple[int, int],
    p_target: tuple[int, int],
    wall_segments: list[tuple[int, int, int, int]],
    min_crossing_angle_deg: float = 25.0,
    min_dist_drop: float = 40.0,
    min_dist_target: float = 50.0,
    endpoint_tol: float = 30.0,
    debug_collector: dict | None = None,
) -> list[tuple[int, int]]:
    """
    Calcula la intersección matemática del rayo p_drop -> p_target contra los
    segmentos globales de muro previamente detectados con los filtros:
    
    1. Ángulo de cruce: descarta líneas con diferencia angular < min_crossing_angle_deg
       (evita intersecciones espurias si el rayo corre casi paralelo al muro).
    2. Posición razonable: descarta cruces a < min_dist_drop del drop o < min_dist_target del almacén.
    3. Extremos de línea: acepta intersecciones que caigan hasta endpoint_tol px fuera del
       extremo del segmento (tramos tapados por edificios).
    4. Fusión de cruces a < 25 px promediando el centro del muro.
    """
    p_drop_f = (float(p_drop[0]), float(p_drop[1]))
    p_target_f = (float(p_target[0]), float(p_target[1]))

    rx = p_target_f[0] - p_drop_f[0]
    ry = p_target_f[1] - p_drop_f[1]
    ray_dist = float(np.hypot(rx, ry))
    if ray_dist < (min_dist_drop + min_dist_target):
        return []

    ray_angle = float(np.degrees(np.arctan2(ry, rx)) % 180.0)

    raw_crossings = []
    accepted_lines = []
    rejected_lines = []

    for seg in wall_segments:
        x1, y1, x2, y2 = seg
        sx = float(x2 - x1)
        sy = float(y2 - y1)
        seg_len_sq = sx * sx + sy * sy
        if seg_len_sq < 1.0:
            continue
        seg_len = math.sqrt(seg_len_sq)
        seg_angle = float(np.degrees(np.arctan2(sy, sx)) % 180.0)

        # Filtro 1: Ángulo de cruce
        cross_angle_diff = _angle_diff_180(ray_angle, seg_angle)
        if cross_angle_diff < min_crossing_angle_deg:
            rejected_lines.append(seg)
            continue

        # Intersección de línea rayo con segmento de muro
        x1_f, y1_f = p_drop_f
        x2_f, y2_f = p_target_f
        x3_f, y3_f = float(x1), float(y1)
        x4_f, y4_f = float(x2), float(y2)

        denom = (x1_f - x2_f) * (y3_f - y4_f) - (y1_f - y2_f) * (x3_f - x4_f)
        if abs(denom) < 1e-6:
            rejected_lines.append(seg)
            continue

        t = ((x1_f - x3_f) * (y3_f - y4_f) - (y1_f - y3_f) * (x3_f - x4_f)) / denom
        if not (0.0 < t < 1.0):
            rejected_lines.append(seg)
            continue

        # Distancias a lo largo del rayo
        dist_to_drop = t * ray_dist
        dist_to_target = (1.0 - t) * ray_dist

        # Filtro 2: Posición razonable en el trayecto
        if dist_to_drop < min_dist_drop or dist_to_target < min_dist_target:
            rejected_lines.append(seg)
            continue

        ix = x1_f + t * rx
        iy = y1_f + t * ry

        # Filtro 3: Tolerancia colineal en los extremos del segmento
        u = ((ix - x3_f) * sx + (iy - y3_f) * sy) / seg_len_sq
        margin_u = endpoint_tol / seg_len
        if not (-margin_u <= u <= 1.0 + margin_u):
            rejected_lines.append(seg)
            continue

        accepted_lines.append(seg)
        raw_crossings.append((ix, iy, dist_to_drop))

    # Ordenar cruces de afuera hacia adentro (menor a mayor distancia a p_drop)
    raw_crossings.sort(key=lambda c: c[2])

    # Fusión de cruces a < 25 px promediando el cluster
    clusters: list[list[tuple[float, float]]] = []
    for ix, iy, dist in raw_crossings:
        pt = (ix, iy)
        if not clusters:
            clusters.append([pt])
        else:
            last_cluster = clusters[-1]
            last_avg_x = sum(p[0] for p in last_cluster) / len(last_cluster)
            last_avg_y = sum(p[1] for p in last_cluster) / len(last_cluster)
            if np.hypot(pt[0] - last_avg_x, pt[1] - last_avg_y) < 25.0:
                last_cluster.append(pt)
            else:
                clusters.append([pt])

    merged_crossings: list[tuple[int, int]] = []
    for cluster in clusters:
        avg_x = int(round(sum(p[0] for p in cluster) / len(cluster)))
        avg_y = int(round(sum(p[1] for p in cluster) / len(cluster)))
        merged_crossings.append((avg_x, avg_y))

    if debug_collector is not None:
        debug_collector["p_drop"] = p_drop
        debug_collector["p_target"] = p_target
        debug_collector["accepted_lines"] = accepted_lines
        debug_collector["rejected_lines"] = rejected_lines
        debug_collector["crossings"] = merged_crossings

    return merged_crossings


def find_wall_crossings(
    img: np.ndarray,
    p_drop: tuple[int, int],
    p_target: tuple[int, int],
    iso_angles: tuple[float, float] | None = None,
    strip: int = 45,
    tol_deg: float = 10.0,
    canny_low: int = 60,
    canny_high: int = 160,
    hough_thresh: int = 30,
    hough_min_len: int = 70,
    hough_max_gap: int = 10,
    min_crossing_angle_deg: float = 25.0,
    min_dist_drop: float = 40.0,
    min_dist_target: float = 50.0,
    endpoint_tol: float = 30.0,
    debug_collector: dict | None = None,
) -> list[tuple[int, int]]:
    """
    Función principal de raycasting de muros:
    1. Si aún no se han extraído los segmentos globales de la imagen, los calcula una sola vez.
    2. Calcula los cruces del rayo con los segmentos aplicando los filtros geométricos.
    """
    if img is None or img.size == 0:
        return []

    _, wall_segments = detect_global_wall_segments(
        img,
        min_line_len=hough_min_len,
        max_line_gap=hough_max_gap,
        tol_deg=tol_deg,
        canny_low=canny_low,
        canny_high=canny_high,
        hough_thresh=hough_thresh,
    )

    return find_wall_crossings_from_segments(
        p_drop=p_drop,
        p_target=p_target,
        wall_segments=wall_segments,
        min_crossing_angle_deg=min_crossing_angle_deg,
        min_dist_drop=min_dist_drop,
        min_dist_target=min_dist_target,
        endpoint_tol=endpoint_tol,
        debug_collector=debug_collector,
    )


def find_wall_junctions(
    wall_segments: list[tuple[int, int, int, int]],
    tol_px: float = 18.0,
    min_intersect_angle_deg: float = 35.0,
    iso_angles: tuple[float, float] | None = None,
) -> list[dict]:
    """
    Encuentra las intersecciones geométricas entre segmentos de muro (cruces en T y +).
    Retorna lista de uniones: [{'pt': (x, y), 'type': 'cross'|'tee'|'corner', 'degree': 4|3|2, 'score': float}]
    """
    if not wall_segments or len(wall_segments) < 2:
        return []

    # Precalcular ángulos y longitudes de cada segmento
    seg_info = []
    for s in wall_segments:
        x1, y1, x2, y2 = s
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length >= 1.0:
            ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
            seg_info.append((s, length, ang))

    raw_junctions = []
    n = len(seg_info)
    for i in range(n):
        s1, len1, ang1 = seg_info[i]
        x1, y1, x2, y2 = s1
        for j in range(i + 1, n):
            s2, len2, ang2 = seg_info[j]
            # Verificar que no sean casi paralelos (deben cruzarse con ángulo >= min_intersect_angle_deg)
            if _angle_diff_180(ang1, ang2) < min_intersect_angle_deg:
                continue

            x3, y3, x4, y4 = s2
            denom = float((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            if abs(denom) < 1e-4:
                continue

            t = float(((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom)
            u = float(-((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom)

            t_tol = tol_px / len1
            u_tol = tol_px / len2

            if (-t_tol <= t <= 1.0 + t_tol) and (-u_tol <= u <= 1.0 + u_tol):
                ix = int(round(x1 + t * (x2 - x1)))
                iy = int(round(y1 + t * (y2 - y1)))

                is_end_t = (t < 0.15 or t > 0.85)
                is_end_u = (u < 0.15 or u > 0.85)
                if not is_end_t and not is_end_u:
                    jtype = "cross"  # Cruce 4 ramas (+)
                    degree = 4
                    score = 4.0
                elif (is_end_t and not is_end_u) or (not is_end_t and is_end_u):
                    jtype = "tee"    # Cruce 3 ramas (T)
                    degree = 3
                    score = 3.0
                else:
                    jtype = "corner" # Esquina 2 ramas (L)
                    degree = 2
                    score = 1.5

                raw_junctions.append(((ix, iy), jtype, degree, score))

    if not raw_junctions:
        return []

    # Agrupar uniones cercanas (< 25px) promediando sus coordenadas
    clusters: list[dict] = []
    for pt, jtype, deg, score in raw_junctions:
        matched = False
        for c in clusters:
            if np.hypot(pt[0] - c["pt"][0], pt[1] - c["pt"][1]) < 25.0:
                c["pts"].append(pt)
                if score > c["score"]:
                    c["score"] = score
                    c["type"] = jtype
                    c["degree"] = deg
                matched = True
                break
        if not matched:
            clusters.append({
                "pt": pt,
                "pts": [pt],
                "type": jtype,
                "degree": deg,
                "score": score
            })

    for c in clusters:
        avg_x = int(round(sum(p[0] for p in c["pts"]) / len(c["pts"])))
        avg_y = int(round(sum(p[1] for p in c["pts"]) / len(c["pts"])))
        c["pt"] = (avg_x, avg_y)

    return clusters


def calculate_adaptive_jump_and_breach_placements(
    targets_with_drops: list[tuple[any, tuple[int, int]]],
    wall_segments: list[tuple[int, int, int, int]],
    junctions: list[dict] | None = None,
    max_jumps: int = 4,
    min_dist_between_jumps: float = 110.0,
    swb_breach_max_dist_grass: float = 75.0,
    safe_y_min: int = 140,
    safe_y_max: int = 845,
    safe_x_min: int = 120,
    safe_x_max: int = 2280,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, any]], list[dict]]:
    """
    Decisión adaptativa por demanda real:
    - Analiza cada objetivo de botín prioritario.
    - Si tiene cruces de muro:
      * El cruce más exterior (Capa 1) asigna brecha de SWB.
      * Si hay 2+ capas, busca la unión geométrica (T o +) de mayor puntuación en el corredor
        para colocar 1 Hechizo de Salto que abra la mayor cantidad de compartimentos.
    - No duplica saltos si el objetivo ya cuenta con cobertura (< 140px).
    - Retorna: (jump_targets, swb_breach_targets, rays_debug_data)
    """
    if junctions is None:
        junctions = find_wall_junctions(wall_segments)

    selected_jumps: list[tuple[int, int]] = []
    selected_swb: list[tuple[int, int, any]] = []
    seen_swb_drops = set()
    rays_debug_data: list[dict] = []

    for d, (drop_x, drop_y) in targets_with_drops:
        target_center = (int(round(d.center[0])), int(round(d.center[1])))
        ray_debug: dict = {}
        crossings = find_wall_crossings_from_segments(
            p_drop=(drop_x, drop_y),
            p_target=target_center,
            wall_segments=wall_segments,
            debug_collector=ray_debug
        )
        rays_debug_data.append(ray_debug)

        # Verificar si este objetivo ya cuenta con cobertura de salto existente
        has_jump_coverage = any(
            np.hypot(target_center[0] - jx, target_center[1] - jy) < 140.0
            for jx, jy in selected_jumps
        )

        if crossings:
            c0_x, c0_y = crossings[0]
            dist_drop_to_c0 = np.hypot(c0_x - drop_x, c0_y - drop_y)

            # Capa 1: Si el muro exterior está cerca del césped, programar brecha SWB
            if dist_drop_to_c0 <= swb_breach_max_dist_grass:
                grid_k = (drop_x // 80, drop_y // 80)
                if grid_k not in seen_swb_drops:
                    seen_swb_drops.add(grid_k)
                    selected_swb.append((drop_x, drop_y, d))

            # Capas interiores (2 o más muros): colocar Hechizo de Salto en unión óptima
            if len(crossings) >= 2 and not has_jump_coverage and len(selected_jumps) < max_jumps:
                best_jump_pt = None
                best_score = -999.0

                # Vector del rayo
                dx_ray = target_center[0] - drop_x
                dy_ray = target_center[1] - drop_y
                ray_len = float(np.hypot(dx_ray, dy_ray))

                if ray_len > 1.0 and junctions:
                    ux_ray, uy_ray = dx_ray / ray_len, dy_ray / ray_len

                    for junc in junctions:
                        jx, jy = junc["pt"]
                        # Coordenada proyectada a lo largo del rayo
                        proj_t = (jx - drop_x) * ux_ray + (jy - drop_y) * uy_ray
                        # Debe estar entre la primera capa de muro y el objetivo
                        if proj_t < dist_drop_to_c0 - 15.0 or proj_t > ray_len - 15.0:
                            continue

                        # Distancia perpendicular al rayo
                        perp_dist = abs((jx - drop_x) * (-uy_ray) + (jy - drop_y) * ux_ray)
                        if perp_dist > 85.0:
                            continue

                        # Distancia a cruces interiores de muro
                        min_dist_to_interior = min(
                            np.hypot(jx - cx, jy - cy) for cx, cy in crossings[1:]
                        )

                        # Puntuación: mayor grado de unión (+ > T > L), menor distancia al corredor y a cruces
                        cand_score = junc["score"] - (0.015 * perp_dist) - (0.012 * min_dist_to_interior)

                        # Evitar solapamiento con saltos ya seleccionados
                        if any(np.hypot(jx - ox, jy - oy) < min_dist_between_jumps for ox, oy in selected_jumps):
                            continue

                        if cand_score > best_score:
                            best_score = cand_score
                            best_jump_pt = (jx, jy)

                # Si no encontramos unión de grafo cercana, usar el segundo cruce de muro o punto medio
                if best_jump_pt is None:
                    if len(crossings) >= 2:
                        fallback_pt = (
                            int(round((crossings[0][0] + crossings[1][0]) / 2.0)),
                            int(round((crossings[0][1] + crossings[1][1]) / 2.0)),
                        )
                    else:
                        fallback_pt = crossings[0]

                    if not any(np.hypot(fallback_pt[0] - ox, fallback_pt[1] - oy) < min_dist_between_jumps for ox, oy in selected_jumps):
                        best_jump_pt = fallback_pt

                if best_jump_pt is not None:
                    # Clampar a márgenes seguros de pantalla
                    cx_clamped = int(np.clip(best_jump_pt[0], safe_x_min, safe_x_max))
                    cy_clamped = int(np.clip(best_jump_pt[1], safe_y_min, safe_y_max))
                    selected_jumps.append((cx_clamped, cy_clamped))
                    log.info(
                        "🎯 Salto adaptativo asignado en (%d, %d) para %s (score=%.2f)",
                        cx_clamped, cy_clamped, d.cls_name, best_score if best_score > -900 else 1.0
                    )

    return selected_jumps, selected_swb, rays_debug_data


def draw_wall_raycast_debug(
    screen: np.ndarray,
    rays_data: list[dict],
    jump_targets: list[tuple[int, int]],
    junctions: list[dict] | None = None,
    out_path: str = "logs/yolo/wall_raycast_debug.png",
) -> None:
    """
    Genera y guarda la imagen anotada de depuración visual de raycasting de muros:
    - Segmentos de muro detectados (verde los que cruzan el rayo, azul/cian los globales)
    - Uniones de muros detectadas (magenta pequeños cruces T/+)
    - Rayos p_drop -> almacén (blanco)
    - Cruces de muro numerados (amarillo)
    - Círculos de Hechizo de Salto proyectados (verde translúcido)
    """
    if screen is None or screen.size == 0:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas = screen.copy()

    # Dibujar segmentos globales de fondo en cian suave
    if _WALL_SEGMENTS_CACHE is not None:
        for x1, y1, x2, y2 in _WALL_SEGMENTS_CACHE:
            cv2.line(canvas, (x1, y1), (x2, y2), (220, 180, 50), thickness=1)

    # Dibujar uniones de muros detectadas (magenta)
    if junctions:
        for junc in junctions:
            jx, jy = junc["pt"]
            jdeg = junc.get("degree", 2)
            cv2.circle(canvas, (jx, jy), 4 if jdeg <= 2 else 6, (255, 0, 255), -1)

    for data in rays_data:
        p_drop = data.get("p_drop")
        p_target = data.get("p_target")
        accepted = data.get("accepted_lines", [])
        rejected = data.get("rejected_lines", [])
        crossings = data.get("crossings", [])

        if p_drop and p_target:
            # Rayo principal blanco
            cv2.line(canvas, p_drop, p_target, (255, 255, 255), thickness=2)

        # Líneas que intersecan (verde brillante)
        for x1, y1, x2, y2 in accepted:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

        # Cruces numerados
        for idx, (cx, cy) in enumerate(crossings):
            cv2.circle(canvas, (cx, cy), radius=6, color=(0, 255, 255), thickness=-1)
            cv2.putText(
                canvas,
                f"W{idx+1}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

    # Dibujar los saltos finales
    for idx, (jx, jy) in enumerate(jump_targets):
        cv2.circle(canvas, (jx, jy), radius=85, color=(0, 255, 120), thickness=2)
        cv2.circle(canvas, (jx, jy), radius=5, color=(0, 255, 120), thickness=-1)
        cv2.putText(
            canvas,
            f"JUMP #{idx+1}",
            (jx - 35, jy - 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 120),
            2
        )

    save_debug_image_async(out_path, canvas)
    log.info("📸 Debug visual de Wall Raycast guardado en: %s", out_path)

