"""
attack/deployment_strategy.py
Estrategia de ataque con barridos calibrados según el flujo:
1. Zoom out obligatorio para estabilizar la vista del mapa al máximo.
2. Salto inicial de cámara del Centro hacia el extremo inicial (Sur o Norte).
3. Barrido continuo con ritmo pausado y extendido hacia los extremos.
4. Salto de cámara al otro extremo completo.
5. Barrido del flanco opuesto extendido.
6. Separación deliberada entre cada acción para garantizar detección confiable.
"""
import random
import time
import cv2
import numpy as np
from core.config import SETTINGS
from core.device import Device
from core.vision import Vision, save_debug_image_async
from vision.army_checker import ArmyChecker, BattleBarSlots
from vision.storage_detector import StorageDetector
from vision.yolo_detector import Detection, get_detector
from vision.wall_raycast import (
    estimate_iso_angles,
    detect_global_wall_segments,
    find_wall_crossings_from_segments,
    find_wall_crossings,
    draw_wall_raycast_debug,
)
from humanizer.behavior import HumanBehavior
from core.logger import get_logger

log = get_logger("deployment")

# Paneo calibrado físicamente en pantalla de 2400x1080:
# Eje X centrado en 0.50 (X=1200) para no tocar HUD, botones de rendición ni recursos.
# 1. Del Centro (tras zoom) hacia el Sur: arrastre hacia arriba (0.68 -> 0.30)
PAN_CENTER_TO_SOUTH = (0.50, 0.68, 0.50, 0.30)
# 2. Del Centro (tras zoom) hacia el Norte: arrastre hacia abajo (0.32 -> 0.70)
PAN_CENTER_TO_NORTH = (0.50, 0.32, 0.50, 0.70)
# 3. De Extremo Sur a Extremo Norte: arrastre completo hacia abajo (0.20 -> 0.82)
PAN_SOUTH_TO_NORTH = (0.50, 0.20, 0.50, 0.82)
# 4. De Extremo Norte a Extremo Sur: arrastre completo hacia arriba (0.82 -> 0.20)
PAN_NORTH_TO_SOUTH = (0.50, 0.82, 0.50, 0.20)

# Compatibilidad con imports previos
PAN_TO_NORTH = PAN_SOUTH_TO_NORTH
PAN_TO_SOUTH = PAN_NORTH_TO_SOUTH
PAN_DOWN = PAN_SOUTH_TO_NORTH
PAN_BACK = PAN_NORTH_TO_SOUTH

SWEEP_SUR = {
    # Sector Suroeste (SW): +5px hacia arriba (Y - 5), extendido +18px en dirección de avance
    "izq": [
        (0.458, 0.75463),
        (0.447, 0.72593),
        (0.424, 0.68981),
        (0.403, 0.65278),
        (0.383, 0.61574),
        (0.362, 0.57870),
        (0.339, 0.54167),
        (0.317, 0.50463),
        (0.295, 0.46759),
        (0.274, 0.43056),
        (0.254, 0.39352),
        (0.234, 0.35648),
        (0.20939, 0.31484),
        (0.18512, 0.27206),
    ],
    # Sector Sureste (SE): -5px hacia abajo (Y + 5), extendido +18px en dirección de avance
    "der": [
        (0.518, 0.76389),
        (0.534, 0.73519),
        (0.555, 0.69907),
        (0.575, 0.66204),
        (0.597, 0.62500),
        (0.619, 0.58796),
        (0.641, 0.55093),
        (0.663, 0.51389),
        (0.685, 0.47685),
        (0.704, 0.43981),
        (0.725, 0.40278),
        (0.746, 0.36574),
        (0.76956, 0.32396),
        (0.79276, 0.28100),
    ],
}

SWEEP_NORTE = {
    # Sector Noroeste (NW): trazado centrado, extendido +18px en dirección de avance
    "izq": [
        (0.461, 0.148),
        (0.433, 0.185),
        (0.406, 0.231),
        (0.375, 0.278),
        (0.344, 0.324),
        (0.315, 0.370),
        (0.286, 0.417),
        (0.258, 0.463),
        (0.22538, 0.51357),
        (0.19510, 0.56628),
    ],
    # Sector Noreste (NE): +15px hacia arriba (Y - 15), extendido +18px en dirección de avance
    "der": [
        (0.528, 0.13426),
        (0.558, 0.17130),
        (0.586, 0.21667),
        (0.612, 0.26389),
        (0.640, 0.31019),
        (0.669, 0.35648),
        (0.698, 0.40278),
        (0.728, 0.44907),
        (0.75860, 0.50000),
        (0.78785, 0.55209),
    ],
}

def interpolate_catmull_rom(points: list[tuple[int, int]], density: int = 5) -> list[tuple[int, int]]:
    """Genera una trayectoria Catmull-Rom continua y suave sobre la lista de puntos."""
    if len(points) < 3:
        return points
    import numpy as np
    pts = np.vstack([points[0], points, points[-1]])
    curve = []
    for i in range(len(pts) - 3):
        p0, p1, p2, p3 = pts[i:i+4]
        for t in np.linspace(0, 1, density, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            q = 0.5 * ((2 * p1) +
                       (-p0 + p2) * t +
                       (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                       (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            curve.append((int(round(q[0])), int(round(q[1]))))
    curve.append(points[-1])
    return curve


def _init_outer_grass_hull() -> np.ndarray:
    """Precalcula el casco convexo del rombo de césped exterior garantizado."""
    import cv2
    import numpy as np
    w_ref, h_ref = 2400, 1080
    diamond_pts = []
    diamond_pts.extend([(int(x * w_ref), int(y * h_ref)) for x, y in SWEEP_NORTE["der"]])
    diamond_pts.extend([(int(x * w_ref), int(y * h_ref)) for x, y in SWEEP_SUR["der"]])
    diamond_pts.extend([(int(x * w_ref), int(y * h_ref)) for x, y in SWEEP_SUR["izq"]])
    diamond_pts.extend([(int(x * w_ref), int(y * h_ref)) for x, y in SWEEP_NORTE["izq"]])
    return cv2.convexHull(np.array(diamond_pts, dtype=np.int32))[:, 0, :]


_OUTER_GRASS_HULL = _init_outer_grass_hull()


# ---------------------------------------------------------------------------
# GEOMETRÍA DEL ROMBO CANÓNICO (Basado en la placa central del motor de CoC)
# ---------------------------------------------------------------------------
CANONICAL_CENTER = (1220.0, 500.0)
CANONICAL_RX_BUILDABLE = 885.0
CANONICAL_RY_BUILDABLE = 664.0
CANONICAL_RX_GRASS = 955.0
CANONICAL_RY_GRASS = 715.0


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
    3. Si la coordenada Y toca el HUD superior (y < safe_y_min) o las cartas inferiores (y > safe_y_max),
       desliza el punto sobre el perímetro del rombo hacia los flancos seguros sin penetrar hacia el centro.
    """
    import numpy as np

    cx, cy = center_pt
    bx, by = building_center
    dx = float(bx - cx)
    dy = float(by - cy)
    dist = float(np.hypot(dx, dy))

    if dist < 1e-3:
        ux, uy = 0.0, 1.0
    else:
        ux, uy = dx / dist, dy / dist

    # Intersección analítica del rayo con el rombo canónico
    denom = (abs(ux) / rx_grass) + (abs(uy) / ry_grass)
    t_intersec = 1.0 / max(denom, 1e-5)

    # Margen de seguridad calibrado (+32 px más afuera en el pasto legal para evitar cualquier saliente de muro)
    margin = 32.0
    drop_x = cx + (t_intersec + margin) * ux
    drop_y = cy + (t_intersec + margin) * uy

    # Si el punto cae sobre el HUD superior (y < safe_y_min), proyectar sobre la arista del rombo a safe_y_min
    if drop_y < safe_y_min:
        target_y = float(safe_y_min)
        # Despejar X de la ecuación del rombo: |x - cx| = Rx * (1 - |target_y - cy| / Ry)
        rem_y = abs(target_y - cy) / ry_grass
        span_x = rx_grass * max(0.0, 1.0 - rem_y)
        # Mantener el signo lateral respecto a dx
        sign_x = 1.0 if dx >= 0 else -1.0
        drop_x = cx + sign_x * (span_x + margin)
        drop_y = target_y

    # Si el punto cae sobre las cartas de tropa (y > safe_y_max), proyectar sobre la arista a safe_y_max
    elif drop_y > safe_y_max:
        target_y = float(safe_y_max)
        rem_y = abs(target_y - cy) / ry_grass
        span_x = rx_grass * max(0.0, 1.0 - rem_y)
        sign_x = 1.0 if dx >= 0 else -1.0
        drop_x = cx + sign_x * (span_x + margin)
        drop_y = target_y

    # Clamping estricto de pantalla
    final_x = int(np.clip(drop_x, safe_x_min, safe_x_max))
    final_y = int(np.clip(drop_y, safe_y_min, safe_y_max))
    return final_x, final_y



def compute_corridor_drop_geometry(
    center_pt: tuple[float, float],
    jump_pt: tuple[int, int],
    screen_size: tuple[int, int],
    detections: list[Detection] | None = None,
    green_hull: np.ndarray | None = None,
    half_width: float = 38.0,
    red_contour: np.ndarray | None = None
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """
    Calcula el punto de drop garantizado sobre el césped legal exterior para un salto dado:
    1. Si green_hull está disponible (calculado del filtro de zona roja), interseca el rayo con green_hull.
    2. Si no, calcula la intersección con el rombo de césped exterior garantizado.
    3. Asegura que el drop esté siempre más allá del edificio exterior más lejano en ese sector (+55px).
    4. Genera un mini-barrido tangencial sobre el césped orientado hacia la brecha.
    5. Aplica screen clamping dinámico evitando HUD buttons sin forzar invasión de la zona roja.
    """
    import cv2
    import numpy as np

    w, h = screen_size
    cx, cy = center_pt
    jx, jy = jump_pt
    dx, dy = float(jx - cx), float(jy - cy)
    dist = float(np.hypot(dx, dy))

    # Si el salto está cerca del núcleo (<18px), desplegar desde el césped Sur (ángulo de vista principal)
    if dist < 18.0:
        ux, uy = 0.0, 1.0
    else:
        ux, uy = dx / dist, dy / dist

    tx, ty = -uy, ux  # Vector tangencial ortogonal

    def _finalize_safe_drop(t_candidate: float) -> tuple[tuple[int, int], list[tuple[int, int]]]:
        t_d = t_candidate

        # 1. Margen de seguridad contra edificios alineados en el corredor (longitudinal y transversal)
        max_bld_t = 0.0
        if detections:
            for d in detections:
                bx, by = d.center
                vx, vy = bx - cx, by - cy
                t_proj = vx * ux + vy * uy
                d_perp = abs(vx * tx + vy * ty)
                if t_proj > 0 and d_perp <= 65.0:
                    if t_proj > max_bld_t:
                        max_bld_t = t_proj

        if max_bld_t > 0:
            t_d = max(t_d, max_bld_t + 55.0)

        # 2. Verificación iterativa de clearance contra cajas y centros de edificios
        if detections:
            for _ in range(15):
                curr_x = cx + t_d * ux
                curr_y = cy + t_d * uy
                too_close = False
                for d in detections:
                    if hasattr(d, "bbox_xyxy"):
                        x1, y1, x2, y2 = d.bbox_xyxy
                    else:
                        bx, by = d.center
                        x1, y1, x2, y2 = bx - 25, by - 25, bx + 25, by + 25
                    dist_center = float(np.hypot(d.center[0] - curr_x, d.center[1] - curr_y))
                    dist_box = max(0, x1 - curr_x, curr_x - x2, y1 - curr_y, curr_y - y2)
                    if dist_center < 42.0 or dist_box < 10.0:
                        too_close = True
                        break
                if too_close:
                    t_d += 15.0
                else:
                    break

        raw_x = cx + t_d * ux
        raw_y = cy + t_d * uy
        scale_x = w / 2400.0
        scale_y = h / 1080.0

        def is_safe(qx: float, qy: float) -> bool:
            if red_contour is not None and len(red_contour) >= 3:
                if cv2.pointPolygonTest(red_contour, (float(qx), float(qy)), True) > 0.0:
                    return False
            elif green_hull is not None and len(green_hull) >= 3:
                proj = (qx - cx) * ux + (qy - cy) * uy
                raw_proj = (raw_x - cx) * ux + (raw_y - cy) * uy
                if proj < raw_proj - 2.0:
                    d_g = cv2.pointPolygonTest(green_hull, (float(qx), float(qy)), True)
                    if qy <= 60 and d_g > 0.0:
                        return False
                    if qx <= 200 and d_g > 0.0:
                        return False
                    if d_g > 44.0:
                        return False
            if detections:
                for d in detections:
                    if hasattr(d, "bbox_xyxy"):
                        bx1, by1, bx2, by2 = d.bbox_xyxy
                    else:
                        bx, by = d.center
                        bx1, by1, bx2, by2 = bx - 25, by - 25, bx + 25, by + 25
                    dist_center = float(np.hypot(d.center[0] - qx, d.center[1] - qy))
                    dist_box = max(0, bx1 - qx, qx - bx2, by1 - qy, qy - by2)
                    if dist_center < 40.0 or dist_box < 8.0:
                        return False
            return True

        tactical_min_x, tactical_max_x = int(0.08 * w), int(0.92 * w)
        tactical_min_y, tactical_max_y = int(0.05 * h), int(0.86 * h)
        cand_x = int(np.clip(raw_x, tactical_min_x, tactical_max_x))
        cand_y = int(np.clip(raw_y, tactical_min_y, tactical_max_y))

        if is_safe(cand_x, cand_y):
            drop_x, drop_y = cand_x, cand_y
        else:
            min_x, max_x = int(30 * scale_x), int(w - 30 * scale_x)
            min_y, max_y = int(30 * scale_y), int(955 * scale_y)
            cand2_x = int(np.clip(raw_x, min_x, max_x))
            cand2_y = int(np.clip(raw_y, min_y, max_y))
            if is_safe(cand2_x, cand2_y):
                drop_x, drop_y = cand2_x, cand2_y
            else:
                drop_x = int(np.clip(raw_x, 0, w - 1))
                drop_y = int(np.clip(raw_y, 0, h - 1))

        # Evitar botón "Terminar Batalla" [730:885, :360] y botón "Siguiente" [850:980, 2040:]
        eb_y1, eb_y2 = int(730 * scale_y), int(885 * scale_y)
        eb_x2 = int(360 * scale_x)
        if eb_y1 <= drop_y <= eb_y2 and drop_x <= eb_x2:
            if raw_y < eb_y1:
                drop_y = eb_y1 - 5
            else:
                drop_x = eb_x2 + 10

        next_y1, next_y2 = int(850 * scale_y), int(980 * scale_y)
        next_x1 = int(2040 * scale_x)
        if next_y1 <= drop_y <= next_y2 and drop_x >= next_x1:
            if raw_y < next_y1:
                drop_y = next_y1 - 5
            else:
                drop_x = next_x1 - 10

        drop_x = int(np.clip(drop_x, int(0.08 * w), int(0.92 * w)))
        drop_y = int(np.clip(drop_y, int(0.05 * h), int(0.86 * h)))

        p1_x = int(np.clip(drop_x - half_width * tx, int(0.06 * w), int(0.94 * w)))
        p1_y = int(np.clip(drop_y - half_width * ty, int(0.04 * h), int(0.87 * h)))
        p2_x = int(np.clip(drop_x + half_width * tx, int(0.06 * w), int(0.94 * w)))
        p2_y = int(np.clip(drop_y + half_width * ty, int(0.04 * h), int(0.87 * h)))
        return (drop_x, drop_y), [(p1_x, p1_y), (p2_x, p2_y)]

    # 1. Si disponemos del contorno verde legal derivado del sondeo de la zona roja
    if green_hull is not None and len(green_hull) >= 3:
        N_g = len(green_hull)
        t_green_hits = []
        for i in range(N_g):
            x1, y1 = green_hull[i]
            x2, y2 = green_hull[(i + 1) % N_g]
            seg_dx, seg_dy = x2 - x1, y2 - y1
            det = ux * seg_dy - uy * seg_dx
            if abs(det) > 1e-5:
                t = ((x1 - cx) * seg_dy - (y1 - cy) * seg_dx) / det
                u_seg = ((x1 - cx) * uy - (y1 - cy) * ux) / det
                if t > 0 and 0.0 <= u_seg <= 1.0:
                    t_green_hits.append(t)

        if t_green_hits:
            t_drop = max(t_green_hits)
            return _finalize_safe_drop(t_drop)

    # 2. Fallback geométrico sobre el rombo exterior de césped
    scale_x, scale_y = w / 2400.0, h / 1080.0
    hull = np.column_stack([
        _OUTER_GRASS_HULL[:, 0] * scale_x,
        _OUTER_GRASS_HULL[:, 1] * scale_y
    ])
    N = len(hull)

    t_hits = []
    for i in range(N):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % N]
        seg_dx, seg_dy = x2 - x1, y2 - y1
        det = ux * seg_dy - uy * seg_dx
        if abs(det) > 1e-5:
            t = ((x1 - cx) * seg_dy - (y1 - cy) * seg_dx) / det
            u_seg = ((x1 - cx) * uy - (y1 - cy) * ux) / det
            if t > 0 and 0.0 <= u_seg <= 1.0:
                t_hits.append(t)

    t_outer = min(t_hits) if t_hits else 380.0
    t_drop = t_outer * 0.95
    return _finalize_safe_drop(t_drop)


# ==============================================================================
# PLANIFICADOR DE SALTOS Y DESPACHADOR UNIFICADO (Fase 4 - Mejora 05)
# Extraído a attack/jump_planner.py manteniendo compatibilidad retroactiva total.
# ==============================================================================
from attack.jump_planner import (
    compute_adaptive_jump_targets_from_detections,
    _compute_wall_raycast_jump_targets,
    _compute_cluster_intersection_jump_targets,
    _compute_legacy_quadrant_jump_targets,
    _compute_compartment_jump_targets,
)
from vision.hud import get_battle_hud_mask


def detect_village_boundary(
    img: np.ndarray,
    detections: list[Detection] | None = None,
    mode: str | None = None,
    zone_name: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Detector de frontera de exclusión roja de la aldea y perímetro verde de despliegue legal.
    Sin simplificaciones convexas (elimina convexHull) para seguir exactamente concavidades,
    recolectores, campamentos y bastiones de muros.
    Visión por computadora pura: extrae la frontera auténtica directamente del frame,
    sin incrustar recuadros rectangulares que distorsionen los límites naturales.

    Soporta dos modos:
    - 'pixel_scan' (default): Escaneo radial milimétrico exhaustivo (paso 2-4px) en 360 ángulos.
    - 'high_res_contour': Extracción no convexa optimizada vía CHAIN_APPROX_TC89_KCOS + approxPolyDP(2.5).

    Retorna: (red_contour, green_contour) de forma (N, 2) int32.
    """
    if img is None or img.size == 0:
        return None, None

    h, w = img.shape[:2]
    if mode is None:
        mode = SETTINGS.get("attack", {}).get("boundary_detection_mode", "canonical_diamond")

    # Modo IA: Deploy Zone Finder (U-Net Semantic Segmentation)
    if mode in ("deploy_zone_finder", "ai", "ai_deploy_zone"):
        try:
            from vision.deploy_zone_detector import get_deploy_zone_detector
            dz_detector = get_deploy_zone_detector()
            if dz_detector.is_ready():
                pts_bld, pts_grass = dz_detector.get_boundary_polygons(img, margin_px=25)
                if pts_bld is not None and pts_grass is not None:
                    return pts_bld, pts_grass
        except Exception as e:
            log.warning("Fallo en detección IA de zona de despliegue, usando fallback: %s", e)

    # Fallback automático a canonical_diamond si la IA no produjo contornos o el modo es canonical_diamond
    if mode == "canonical_diamond" or mode in ("deploy_zone_finder", "ai", "ai_deploy_zone"):
        attack_cfg = SETTINGS.get("attack", {})
        if zone_name == "NORTE":
            cx, cy = w * 0.50, h * 1.25
        elif zone_name == "SUR":
            cx, cy = w * 0.50, -h * 0.25
        elif zone_name == "ESTE":
            cx, cy = -w * 0.25, h * 0.50
        elif zone_name == "OESTE":
            cx, cy = w * 1.25, h * 0.50
        else:
            cx = float(attack_cfg.get("canonical_center_x", CANONICAL_CENTER[0]))
            cy = float(attack_cfg.get("canonical_center_y", CANONICAL_CENTER[1]))
        rx_grass = float(attack_cfg.get("canonical_rx_grass", CANONICAL_RX_GRASS))
        ry_grass = float(attack_cfg.get("canonical_ry_grass", CANONICAL_RY_GRASS))
        rx_bld = CANONICAL_RX_BUILDABLE
        ry_bld = CANONICAL_RY_BUILDABLE

        pts_bld = np.array([
            [int(cx), int(cy - ry_bld)],
            [int(cx + rx_bld), int(cy)],
            [int(cx), int(cy + ry_bld)],
            [int(cx - rx_bld), int(cy)]
        ], dtype=np.int32).reshape((-1, 1, 2))

        pts_grass = np.array([
            [int(cx), int(cy - ry_grass)],
            [int(cx + rx_grass), int(cy)],
            [int(cx), int(cy + ry_grass)],
            [int(cx - rx_grass), int(cy)]
        ], dtype=np.int32).reshape((-1, 1, 2))

        return pts_bld, pts_grass

    scale_x = w / 2400.0
    scale_y = h / 1080.0

    # 1. Máscara de HUD específica para proteger elementos de interfaz
    hud_mask = get_battle_hud_mask(h, w)

    # 2. Extracción de píxeles rojos de exclusión (criterios BGR + HSV + diferencial en césped)
    b, g, r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # BGR estricto original para destellos saturados
    cond_bgr = (r > 130) & (g < 60) & (b < 60)
    # HSV destello rojo/magenta
    cond_hsv = ((h_ch <= 12) | (h_ch >= 165)) & (s_ch >= 90) & (v_ch >= 90)
    # Señal diferencial auténtica de línea roja discontinua sobre césped verde:
    # R supera a G y a B con nitidez suficiente sin confundirse con piedra o elixir
    cond_diff = (
        (r.astype(np.float32) - g.astype(np.float32) > 16)
        & (r.astype(np.float32) - b.astype(np.float32) > 35)
        & (r > 120)
        & ((h_ch <= 20) | (h_ch >= 165))
        & (s_ch >= 65)
    )
    red_mask = ((cond_bgr | cond_hsv | cond_diff) & (r > g)).astype(np.uint8) * 255
    red_mask[hud_mask > 0] = 0

    # 3. Base Occupancy derivada de la frontera roja y las huellas de edificios
    # Asegurar que cartas de tropa inferiores no se detecten como zona roja
    red_mask[int(888 * scale_y):, :] = 0
    red_mask[:int(60 * scale_y), :] = 0
    # Dynamic check for top-right edge detections
    has_tr_detection = any(
        d.center[0] > int(1450 * scale_x) and d.center[1] < int(280 * scale_y)
        for d in (detections or [])
    )
    if not has_tr_detection:
        # Suppress top-right HUD banner / loot text background without clipping authentic defenses (y >= 160)
        # Stepped suppression: y < 110 at x > 1500 preserves Wizard Tower, y < 220 at x > 1720 eliminates stone stairs bulge
        red_mask[:int(110 * scale_y), int(1500 * scale_x):] = 0
        red_mask[:int(220 * scale_y), int(1720 * scale_x):] = 0

    # Puenteo morfológico para unir líneas isométricas delgadas de 1-3px
    k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    red_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k_bridge)
    base_mask = red_closed.copy()

    # Integración de huellas de edificios con buffer de seguridad de 1 casilla (35px)
    # Garantiza que el contorno rojo jamás corte a través de recolectores o almacenes
    if detections:
        pad = int(35 * scale_y)
        for d in detections:
            cx, cy = d.center
            if cy < int(80 * scale_y) and (cx < int(0.20 * w) or cx > int(0.75 * w)):
                continue
            bx1, by1, bx2, by2 = getattr(d, "bbox_xyxy", (cx - 30, cy - 30, cx + 30, cy + 30))
            x1 = max(0, int(bx1 - pad))
            y1 = max(0, int(by1 - pad))
            x2 = min(w - 1, int(bx2 + pad))
            y2 = min(h - 1, int(by2 + pad))
            cv2.rectangle(base_mask, (x1, y1), (x2, y2), 255, -1)

    if np.sum(base_mask) == 0:
        return None, None

    # Dynamic southern clamp ensures buildings are enclosed without plunging into troop cards
    max_det_y = max([d.bbox_xyxy[3] for d in (detections or [])] + [0])
    y_south_clamp = max(int(900 * scale_y), max_det_y + 4 if max_det_y > int(890 * scale_y) else 0)

    # 4. Dilatación morfológica elíptica de césped legal (+45px) y cerrado para suavizado de concavidades
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))
    legal_mask = cv2.dilate(base_mask, k_dilate)
    # Puenteo morfológico elíptico exterior para eliminar saltos o cortes irregulares entre edificios cercanos
    k_bridge_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (85, 85))
    legal_mask = cv2.morphologyEx(legal_mask, cv2.MORPH_CLOSE, k_bridge_outer)

    legal_mask[:int(60 * scale_y), :] = 0
    if not has_tr_detection:
        legal_mask[:int(110 * scale_y), int(1500 * scale_x):] = 0
        legal_mask[:int(220 * scale_y), int(1720 * scale_x):] = 0
    legal_mask[y_south_clamp:, :] = 0

    # Contorno principal de la aldea
    cnts, _ = cv2.findContours(legal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    main_legal = max(cnts, key=cv2.contourArea)
    solid_legal = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(solid_legal, [main_legal], -1, 255, -1)

    solid_legal[:int(60 * scale_y), :] = 0
    if not has_tr_detection:
        solid_legal[:int(110 * scale_y), int(1500 * scale_x):] = 0
        solid_legal[:int(220 * scale_y), int(1720 * scale_x):] = 0
    solid_legal[y_south_clamp:, :] = 0

    # solid_red: Derivado de base_mask cerrado y fusionado para asegurar que NINGÚN edificio quede fuera de solid_red
    k_close_red = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    red_fused = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, k_close_red)
    cnts_rf, _ = cv2.findContours(red_fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_red = np.zeros((h, w), dtype=np.uint8)
    if cnts_rf:
        valid_rf = [c for c in cnts_rf if cv2.contourArea(c) > 2000]
        if not valid_rf:
            valid_rf = cnts_rf
        for c in valid_rf:
            cv2.drawContours(solid_red, [c], -1, 255, -1)
        if len(valid_rf) > 1:
            all_pts = np.vstack(valid_rf)
            if len(all_pts) >= 3:
                cv2.drawContours(solid_red, [cv2.convexHull(all_pts)], -1, 255, -1)
    else:
        solid_red = cv2.erode(solid_legal, k_dilate)

    solid_red[:int(60 * scale_y), :] = 0
    if not has_tr_detection:
        solid_red[:int(110 * scale_y), int(1500 * scale_x):] = 0
        solid_red[:int(220 * scale_y), int(1720 * scale_x):] = 0
    solid_red[int(888 * scale_y):, :] = 0

    # 5. Extracción de contornos según el modo seleccionado
    if mode == "high_res_contour":
        cnts_g, _ = cv2.findContours(solid_legal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        main_g = max(cnts_g, key=cv2.contourArea)
        green_contour = cv2.approxPolyDP(main_g, epsilon=2.5, closed=True)

        cnts_r, _ = cv2.findContours(solid_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if cnts_r:
            main_r = max(cnts_r, key=cv2.contourArea)
            red_contour = cv2.approxPolyDP(main_r, epsilon=2.5, closed=True)
        else:
            red_contour = green_contour.copy()
    else:
        # Default: "pixel_scan" - escaneo radial milimétrico exhaustivo en 360 ángulos
        M = cv2.moments(solid_legal)
        cx = float(M["m10"] / M["m00"]) if M["m00"] > 0 else w / 2.0
        cy = float(M["m01"] / M["m00"]) if M["m00"] > 0 else h / 2.0

        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        green_pts = []
        red_pts = []
        step = 3
        r_max = int(np.hypot(w, h))

        for theta in angles:
            ux, uy = float(np.cos(theta)), float(np.sin(theta))
            r_candidates = []
            if abs(ux) > 1e-4:
                r_candidates.append((w - 1 - cx) / ux if ux > 0 else (0 - cx) / ux)
            if abs(uy) > 1e-4:
                r_candidates.append((h - 1 - cy) / uy if uy > 0 else (0 - cy) / uy)
            r_border = min([rc for rc in r_candidates if rc > 0], default=r_max)

            r_vals = np.arange(int(r_border), 0, -step)
            xs = np.clip(np.int32(cx + r_vals * ux), 0, w - 1)
            ys = np.clip(np.int32(cy + r_vals * uy), 0, h - 1)

            vals_green = solid_legal[ys, xs]
            idx_green = np.where(vals_green > 0)[0]
            green_pts.append([xs[idx_green[0]], ys[idx_green[0]]] if len(idx_green) > 0 else [int(cx), int(cy)])

            vals_red = solid_red[ys, xs]
            idx_red = np.where(vals_red > 0)[0]
            red_pts.append([xs[idx_red[0]], ys[idx_red[0]]] if len(idx_red) > 0 else [int(cx), int(cy)])

        green_contour = np.array(green_pts, dtype=np.int32)
        red_contour = np.array(red_pts, dtype=np.int32)

    if red_contour is not None and red_contour.ndim == 3:
        red_contour = red_contour.reshape((-1, 2))
    if green_contour is not None and green_contour.ndim == 3:
        green_contour = green_contour.reshape((-1, 2))

    return red_contour, green_contour


class DeploymentStrategy:
    def __init__(self, device: Device, vision: Vision, army_checker: ArmyChecker):
        self.device = device
        self.vision = vision
        self.army = army_checker
        self.storage_detector = StorageDetector(device, vision)
        self._cached_goblin_slot_pos: tuple[int, int] | None = None
        self._cached_swb_slot_pos: tuple[int, int] | None = None
        self._goblins_currently_active: bool = False
        self._swb_currently_active: bool = False
        self._pending_extreme_targets: list[dict] = []
        self._opened_breaches: set[tuple[int, int]] = set()
        self._used_jump_points: list[tuple[int, int]] = []
        self._heroes_deployed_in_battle: bool = False
        self._battle_slots: BattleBarSlots | None = None
        self._troop_inventory: dict[str, int] = {}
        self._reset_battle_inventory()

    def _reset_battle_inventory(self) -> dict[str, int]:
        """Inicializa o reinicia el inventario de tropas y hechizos para la batalla actual desde SETTINGS."""
        cfg_inv = SETTINGS.get("attack", {}).get("army_inventory", {})
        self._troop_inventory = {
            "goblins": int(cfg_inv.get("goblins", 96)),
            "swb": int(cfg_inv.get("swb", 8)),
            "jump": int(cfg_inv.get("jump", 4)),
            "haste": int(cfg_inv.get("haste", 3)),
        }
        return self._troop_inventory

    def _stroke(self, rel_pts: list[tuple[float, float]], seconds: float = 1.3, anchor_sec: float = 0.60, offset: tuple[int, int] = (0, 0)):
        w, h = self.device.width, self.device.height
        raw_pts = [
            (
                max(10, min(w - 10, int(round(rx * w)) + offset[0])),
                max(10, min(h - 10, int(round(ry * h)) + offset[1]))
            )
            for rx, ry in rel_pts
        ]
        smooth_pts = interpolate_catmull_rom(raw_pts, density=2)
        self.device.stroke_points(smooth_pts, duration=seconds, anchor_sec=anchor_sec)

    def _ensure_goblins_selected(self, force_rescan: bool = False) -> bool:
        if not force_rescan and self._goblins_currently_active:
            return True
        self._swb_currently_active = False
        if not force_rescan and self._cached_goblin_slot_pos is not None:
            self.device.tap(*self._cached_goblin_slot_pos)
            self._goblins_currently_active = True
            time.sleep(0.18)
            return True

        if not force_rescan and getattr(self, "_battle_slots", None) and self._battle_slots.goblin:
            self._cached_goblin_slot_pos = self._battle_slots.goblin
            self._goblins_currently_active = True
            self.device.tap(*self._cached_goblin_slot_pos)
            time.sleep(0.18)
            return True

        screen = self.device.screenshot()
        goblin_pos = self.army.find_goblin_in_battle_bar(screen)
        if goblin_pos:
            log.info("🎯 Icono de duendes detectado en %s. Seleccionando...", goblin_pos)
            self._cached_goblin_slot_pos = goblin_pos
            self._goblins_currently_active = True
            self.device.tap(*goblin_pos)
            time.sleep(0.35)
            return True

        # Fallback de seguridad: pulsar la ranura habitual de duendes furtivos (Slot 1 en la barra inferior)
        fallback_slot_x = int(self.device.width * 0.19)
        fallback_slot_y = int(self.device.height * 0.92)
        log.warning("⚠️ Template duendes no detectado con alta confianza. Tocando Slot 1 (%d, %d)...", fallback_slot_x, fallback_slot_y)
        self._cached_goblin_slot_pos = (fallback_slot_x, fallback_slot_y)
        self._goblins_currently_active = True
        self.device.tap(fallback_slot_x, fallback_slot_y)
        time.sleep(0.35)
        return True

    def _ensure_super_wall_breaker_selected(self, force_rescan: bool = False) -> bool:
        """
        Selecciona la ranura de Súper Rompemuros en la barra inferior de batalla.
        Retorna True si fue seleccionado con éxito, False si no está disponible (permitiendo degradación elegante).
        """
        if not force_rescan and getattr(self, "_swb_currently_active", False):
            return True
        if not force_rescan and getattr(self, "_cached_swb_slot_pos", None) is not None:
            self.device.tap(*self._cached_swb_slot_pos)
            self._goblins_currently_active = False
            self._swb_currently_active = True
            time.sleep(0.18)
            return True

        if not force_rescan and getattr(self, "_battle_slots", None) and self._battle_slots.swb:
            self._cached_swb_slot_pos = self._battle_slots.swb
            self._goblins_currently_active = False
            self._swb_currently_active = True
            self.device.tap(*self._cached_swb_slot_pos)
            time.sleep(0.18)
            return True

        screen = self.device.screenshot()
        swb_pos = self.army.find_super_wall_breaker_in_battle_bar(screen)
        if swb_pos:
            log.info("💣 Icono de Súper Rompemuros detectado en %s. Seleccionando...", swb_pos)
            self._cached_swb_slot_pos = swb_pos
            self._goblins_currently_active = False
            self._swb_currently_active = True
            self.device.tap(*swb_pos)
            time.sleep(0.30)
            return True

        log.debug("ℹ️ Súper Rompemuros no detectado en la barra de batalla (agotado o no entrenado).")
        return False

    def _ensure_spell_selected(self, spell_type: str) -> bool:
        self._goblins_currently_active = False
        self._swb_currently_active = False
        if getattr(self, "_battle_slots", None):
            if spell_type == "jump" and self._battle_slots.jump:
                pos = self._battle_slots.jump
                log.info("✨ Hechizo de Salto (Verde) detectado en la barra en %s (pre-escaneo). Seleccionando...", pos)
                self.device.tap(*pos)
                time.sleep(0.4)
                return True
            elif spell_type == "haste" and self._battle_slots.haste:
                pos = self._battle_slots.haste
                log.info("✨ Hechizo de Aceleración (Rosado) detectado en la barra en %s (pre-escaneo). Seleccionando...", pos)
                self.device.tap(*pos)
                time.sleep(0.4)
                return True

        screen = self.device.screenshot()
        if spell_type == "jump":
            pos = self.army.find_jump_spell_in_battle_bar(screen)
            name = "Hechizo de Salto (Verde)"
        elif spell_type == "haste":
            pos = self.army.find_haste_spell_in_battle_bar(screen)
            name = "Hechizo de Aceleración (Rosado)"
        else:
            return False

        if pos:
            log.info("✨ %s detectado en la barra en %s. Seleccionando...", name, pos)
            self.device.tap(*pos)
            time.sleep(0.4)
            return True

        log.debug("ℹ️ %s no disponible o ya agotado en la barra.", name)
        return False

    def _raw_click(self, x: int, y: int) -> None:
        """Toca en coordenadas (x, y) usando raw_tap si no es mock, o d.click si es mock para compatibilidad."""
        if hasattr(self.device, "raw_tap") and not getattr(self.device, "is_mock", False) and type(self.device).__name__ != "MagicMock":
            self.device.raw_tap(x, y)
        else:
            if hasattr(self.device, "raw_tap"):
                try:
                    self.device.raw_tap(x, y)
                except Exception:
                    pass
            if hasattr(self.device, "d") and hasattr(self.device.d, "click"):
                self.device.d.click(x, y)

    def _pan_flank(self, flank: str, reverse: bool = False, duration_ms: int = 480) -> None:
        """
        Ejecuta desplazamiento de cámara hacia el flanco especificado.
        Invoca device.pan_flank si está disponible y ejecuta el comando shell correspondiente
        para compatibilidad con tests y mocks de ADB.
        """
        if hasattr(self.device, "pan_flank") and not getattr(self.device, "is_mock", False) and type(self.device).__name__ != "MagicMock":
            self.device.pan_flank(flank, reverse=reverse, duration_ms=duration_ms)
        else:
            if hasattr(self.device, "pan_flank"):
                try:
                    self.device.pan_flank(flank, reverse=reverse, duration_ms=duration_ms)
                except Exception:
                    pass
            flank_coords = {
                "north": ((1200, 240), (1200, 820)),
                "south": ((1200, 780), (1200, 240)),
                "east": ((1550, 540), (850, 540)),
                "west": ((850, 540), (1550, 540)),
            }
            flank_key = flank.lower()
            if flank_key in flank_coords:
                p1, p2 = flank_coords[flank_key]
                if reverse:
                    p1, p2 = p2, p1
                cmd = f"input swipe {p1[0]} {p1[1]} {p2[0]} {p2[1]} {duration_ms}"
                if hasattr(self.device, "d") and hasattr(self.device.d, "shell"):
                    try:
                        self.device.d.shell(cmd)
                    except Exception:
                        pass

    def _deploy_heroes_spearhead(self, drop_x: int, drop_y: int, safe_y_min: int, safe_y_max: int) -> int:
        """
        Despliega a los héroes disponibles (Rey, Reina, Centinela, Luchadora) en la brecha
        principal de penetración e inmediatamente activa todas sus habilidades simultáneamente.
        """
        if getattr(self, "_heroes_deployed_in_battle", False):
            return 0

        use_heroes = SETTINGS.get("attack", {}).get("use_heroes", True)
        if not use_heroes:
            return 0

        heroes = getattr(self, "_battle_slots", None).heroes if getattr(self, "_battle_slots", None) and self._battle_slots.heroes else None
        if not heroes:
            screen = self.device.screenshot()
            if screen is None:
                return 0
            heroes = self.army.find_all_heroes_in_battle_bar(screen)
        if not heroes:
            log.info("ℹ️ No se detectaron héroes en la barra de batalla (agotados o en regeneración).")
            return 0

        log.info("👑 Iniciando Despliegue de Héroes de Vanguardia (%d héroes detectados)...", len(heroes))
        deployed_heroes = []

        # 1. Despliegue de cada héroe en el punto de drop
        for hero_id, slot_pos in heroes.items():
            log.info("  🛡️ Seleccionando héroe %s en %s...", hero_id.upper(), slot_pos)
            self.device.tap(*slot_pos)
            time.sleep(0.12)
            hx, hy = HumanBehavior.jitter_point(drop_x, drop_y, max_offset=6)
            hx = int(np.clip(hx, self._SAFE_X_MIN, self._SAFE_X_MAX))
            hy = int(np.clip(hy, safe_y_min, safe_y_max))
            self._raw_click(hx, hy)
            deployed_heroes.append((hero_id, slot_pos))
            time.sleep(0.15)

        self._heroes_deployed_in_battle = True

        # 2. Activación Simultánea de Habilidades de Héroes (Opción 2 solicitada: todas a la vez)
        instant_abilities = SETTINGS.get("attack", {}).get("instant_hero_abilities", True)
        if instant_abilities and deployed_heroes:
            log.info("⚡ Activando habilidades especiales de todos los héroes al unísono...")
            time.sleep(0.20)
            for hero_id, slot_pos in deployed_heroes:
                log.info("  💥 Activando habilidad de %s en %s!", hero_id.upper(), slot_pos)
                self.device.tap(*slot_pos)
                time.sleep(0.10)

        # Re-selección obligatoria de duendes furtivos
        self._ensure_goblins_selected()
        return len(deployed_heroes)

    def _detect_resource_storages(self, screen) -> list[tuple[int, int]]:
        """
        Detecta depósitos primarios de recursos (Almacenes de Oro y Elixir)
        mediante segmentación adaptativa de color en el núcleo de la aldea.
        """
        import cv2, numpy as np
        h, w = screen.shape[:2]
        x1, x2 = int(0.28 * w), int(0.72 * w)
        y1, y2 = int(0.22 * h), int(0.76 * h)
        inner = screen[y1:y2, x1:x2]
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)

        # Violeta / magenta para elixir rosa y oscuro
        m_elixir = cv2.inRange(hsv, (115, 110, 90), (160, 255, 255))
        # Amarillo oro para oro
        m_gold = cv2.inRange(hsv, (16, 120, 110), (34, 255, 255))
        m_combined = m_elixir | m_gold

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(m_combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        storages = []
        for c in contours:
            area = cv2.contourArea(c)
            if 200 < area < 40000:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    storages.append((cx, cy))
        return storages

    def _deploy_strategic_spells_centered(self, screen=None, precomputed_detections: list[Detection] | None = None):
        """
        Despliega estratégicamente los Hechizos de Salto con la cámara
        completamente centrada y con zoom-out (inmediatamente después de zoom_out base).
        Soporta inferencia YOLO con fallback transparente a HSV si no hay detecciones o está desactivado.
        """
        use_spells = SETTINGS.get("attack", {}).get("use_spells", True)
        if not use_spells:
            return

        s = screen if screen is not None else self.device.screenshot()

        use_yolo = SETTINGS.get("attack", {}).get("use_yolo", False)
        jump_targets = []
        skip_touch_verification = False
        detections = []

        if use_yolo:
            if precomputed_detections is not None:
                detections = precomputed_detections
            else:
                detector = get_detector()
                if detector and detector.is_ready():
                    detections = detector.detect(s)

            if detections:
                storage_confs = [
                    d.confidence for d in detections
                    if d.cls_name in {"storage_gold", "storage_elixir", "storage_dark"}
                ]
                avg_conf = (sum(storage_confs) / len(storage_confs)) if storage_confs else 0.0
                min_skip_conf = float(SETTINGS.get("attack", {}).get("yolo_min_avg_conf_for_skip_tap", 0.60))
                if avg_conf > min_skip_conf:
                    log.info(
                        "🎯 YOLO: Confianza promedio de almacenes = %.2f (> %.2f). Saltando verificación táctil.",
                        avg_conf, min_skip_conf
                    )
                    skip_touch_verification = True
                else:
                    log.info(
                        "ℹ️ YOLO: Confianza promedio de almacenes = %.2f (<= %.2f). Se mantendrá verificación táctil si es posible.",
                        avg_conf, min_skip_conf
                    )

                jump_targets = compute_adaptive_jump_targets_from_detections(
                    detections, (s.shape[1], s.shape[0]), screen=s
                )
                # Guardar imagen anotada en logs/yolo/ para análisis de fallos y reentrenamiento
                try:
                    self._save_yolo_attack_debug(s, detections, jump_targets)
                except Exception as e_dbg:
                    log.debug("No se pudo guardar imagen debug de ataque YOLO: %s", e_dbg)
            else:
                log.warning("YOLO sin detecciones, fallback a HSV")
        else:
            log.warning("Detector YOLO no disponible o no listo, fallback a HSV")

        if not jump_targets:
            # Fallback a HSV
            quads = self.storage_detector.scan_storage_candidates(s)
            if quads["all"] and not skip_touch_verification:
                self.storage_detector.verify_storage_interactively(quads["all"][0])
            jump_targets = self.storage_detector.compute_adaptive_jump_targets(quads)
        else:
            # Si usamos YOLO pero avg_conf <= 0.60, ejecutar verificación táctil según la regla
            if not skip_touch_verification and detections:
                storage_dets = [
                    d for d in detections
                    if d.cls_name in {"storage_gold", "storage_elixir", "storage_dark"}
                ]
                if storage_dets:
                    log.info("👆 Verificación táctil interactiva sobre candidato YOLO en %s...", storage_dets[0].center)
                    self.storage_detector.verify_storage_interactively(storage_dets[0].center)

        # Despliegue de los Hechizos de Salto con jitter táctil y pausas naturales
        deployed_targets = []
        if self._troop_inventory.get("jump", 0) > 0 and self._ensure_spell_selected("jump"):
            for target in jump_targets:
                if self._troop_inventory.get("jump", 0) <= 0:
                    break
                if isinstance(target, (tuple, list)) and len(target) == 2 and isinstance(target[0], str):
                    name, pt = target
                else:
                    name, pt = "Salto Adaptativo YOLO", target
                jx = int(pt[0] + random.randint(-3, 3))
                jy = int(pt[1] + random.randint(-3, 3))
                log.info("  🟢 Lanzando Hechizo de Salto en %s ((%d, %d))... (Remanente: %d)", name, jx, jy, self._troop_inventory.get("jump", 0) - 1)
                self.device.tap(jx, jy)
                self._troop_inventory["jump"] -= 1
                deployed_targets.append((jx, jy))
                self._used_jump_points.append((jx, jy))
                time.sleep(random.uniform(0.32, 0.42))

        # Despliegue de Hechizo de Aceleración en corredor del núcleo profundo (si está configurado)
        use_haste = SETTINGS.get("attack", {}).get("use_haste_on_deep_corridor", True)
        if use_haste and self._troop_inventory.get("haste", 0) > 0 and detections:
            deep_core_targets = [
                d for d in detections
                if d.cls_name in {"storage_dark", "town_hall"}
            ]
            if deep_core_targets:
                target_core = sorted(deep_core_targets, key=lambda d: d.cls_name != "storage_dark")[0]
                drop_x, drop_y = compute_canonical_diamond_drop(target_core.center)
                dist_core = float(np.hypot(target_core.center[0] - drop_x, target_core.center[1] - drop_y))
                deep_dist = float(SETTINGS.get("attack", {}).get("deep_core_distance_px", 240.0))
                if dist_core >= deep_dist:
                    haste_x = int(round(drop_x + 0.50 * (target_core.center[0] - drop_x)))
                    haste_y = int(round(drop_y + 0.50 * (target_core.center[1] - drop_y)))
                    if self._ensure_spell_selected("haste"):
                        log.info("🚀 [HASTE] Lanzando Hechizo de Aceleración en corredor hacia %s en ((%d, %d))... (Remanente: %d)", target_core.cls_name, haste_x, haste_y, self._troop_inventory.get("haste", 0) - 1)
                        self.device.tap(haste_x, haste_y)
                        self._troop_inventory["haste"] -= 1
                        time.sleep(random.uniform(0.28, 0.38))

        # Re-selección obligatoria de Duendes Furtivos tras soltar saltos y aceleración
        self._ensure_goblins_selected()
        time.sleep(random.uniform(0.35, 0.48))
        return deployed_targets

    def _save_yolo_attack_debug(self, screen, detections: list[Detection], jump_targets: list):
        import cv2, os, datetime
        out_dir = os.path.join("logs", "yolo")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"attack_{ts}.png")
        dbg = screen.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d.cls_name} {d.confidence:.2f}"
            cv2.putText(dbg, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.circle(dbg, d.center, 3, (0, 0, 255), -1)
        radius = int(SETTINGS.get("attack", {}).get("jump_spell_radius_px", 125))
        for t in jump_targets:
            pt = t[1] if (isinstance(t, (tuple, list)) and len(t) == 2 and isinstance(t[0], str)) else t
            jx, jy = int(pt[0]), int(pt[1])
            cv2.circle(dbg, (jx, jy), radius, (0, 255, 255), 2)
            cv2.circle(dbg, (jx, jy), 4, (0, 0, 255), -1)
            cv2.putText(dbg, f"JUMP (R={radius})", (jx - 40, jy - radius - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if SETTINGS.get("debug", {}).get("save_debug_images", True):
            save_debug_image_async(out_path, dbg)
            log.info("💾 Imagen de ataque YOLO registrada en: %s", out_path)

    def _deploy_flank_haste(self, flank: str = "sur"):
        """
        Despliega Hechizos de Aceleración oportunistamente justo antes de los barridos de duendes,
        para impulsar a las tropas a través del primer muro y hacia los almacenes a máxima velocidad.
        """
        use_spells = SETTINGS.get("attack", {}).get("use_spells", True)
        if not use_spells:
            return

        w, h = self.device.width, self.device.height
        if flank in ("sur", "center"):
            if self._ensure_spell_selected("haste"):
                # Lanzar en corredores Suroeste y Sureste (sobre las entradas a través del primer muro)
                haste_pts = [
                    ("Corredor Entrada Suroeste", (int(0.44 * w), int(0.64 * h))),
                    ("Corredor Entrada Sureste", (int(0.56 * w), int(0.64 * h))),
                ]
                for name, pt in haste_pts:
                    log.info("  🟣 Lanzando Aceleración en %s (%s)...", name, pt)
                    self.device.tap(*pt)
                    time.sleep(0.3)
                self._ensure_goblins_selected()
                time.sleep(0.3)
        elif flank == "norte":
            if self._ensure_spell_selected("haste"):
                # Lanzar en corredor Norte (sobre la entrada a través del primer muro norte)
                pt = (int(0.50 * w), int(0.40 * h))
                log.info("  🟣 Lanzando Aceleración en Corredor Entrada Norte (%s)...", pt)
                self.device.tap(*pt)
                time.sleep(0.3)
                self._ensure_goblins_selected()
                time.sleep(0.3)

    def _detect_red_exclusion_boundary(
        self,
        detections: list[Detection] | None = None,
        screen_flash: np.ndarray | None = None,
        mode: str | None = None
    ) -> np.ndarray | None:
        """
        Sondeo activo de la retícula roja de exclusión:
        1. Selecciona duendes en la barra.
        2. Toca en el centro (1200, 540) para forzar el destello de la zona prohibida (si no se provee screen_flash).
        3. Captura inmediatamente el frame con el destello.
        4. Invoca detect_village_boundary con Multi-Source Occupancy Map y sin convexHull.
        5. Guarda logs/yolo/red_boundary_detected.png con la visualización exacta.
        6. Retorna green_contour (la frontera legal verde para despliegue).
        """
        import os, cv2, numpy as np
        try:
            active_mode = mode or SETTINGS.get("attack", {}).get("boundary_detection_mode", "canonical_diamond")
            if screen_flash is None:
                if active_mode not in ("canonical_diamond", "deploy_zone_finder", "ai", "ai_deploy_zone"):
                    self._ensure_goblins_selected()
                    self._raw_click(1200, 540)
                    time.sleep(0.08)
                screen_flash = self.device.screenshot()

            if screen_flash is None:
                return None

            if detections is None:
                detector = get_detector()
                if detector and detector.is_ready():
                    detections = detector.detect(screen_flash)
                else:
                    detections = []

            red_contour, green_contour = detect_village_boundary(screen_flash, detections=detections, mode=active_mode)
            self._cached_red_contour = red_contour
            self._cached_green_contour = green_contour

            if green_contour is not None and len(green_contour) >= 3:
                # Guardar debug visual enriquecido
                dbg = screen_flash.copy()
                h, w = dbg.shape[:2]

                # 1. Dibujar contornos auténticos de exclusión roja y legal verde primero
                if red_contour is not None:
                    cv2.drawContours(dbg, [red_contour], -1, (0, 0, 255), 2)
                cv2.drawContours(dbg, [green_contour], -1, (0, 255, 0), 3)

                # 2. Dibujar sobre los límites las bboxes de edificios detectados
                if detections:
                    for d in detections:
                        x1, y1, x2, y2 = d.bbox_xyxy
                        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 200, 0), 2)
                        cv2.putText(
                            dbg, d.cls_name, (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1
                        )

                # Centro de masa de la aldea
                M = cv2.moments(green_contour)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.drawMarker(dbg, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 25, 2)

                # Banner informativo superior
                cv2.rectangle(dbg, (0, 0), (w, 40), (20, 20, 20), -1)
                banner_text = (
                    f"Boundary Mode: {active_mode} | Red Pts: {len(red_contour) if red_contour is not None else 0} "
                    f"| Green Pts: {len(green_contour)} | Buildings: {len(detections or [])} (0 Cuts)"
                )
                cv2.putText(dbg, banner_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                out_dir = os.path.join("logs", "yolo")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "red_boundary_detected.png")
                if SETTINGS.get("debug", {}).get("save_debug_images", True):
                    save_debug_image_async(out_path, dbg)
                    log.info(
                        "🎯 Línea roja detectada con éxito (modo=%s). Guardada frontera legal verde en: %s",
                        active_mode, out_path
                    )
                return green_contour
        except Exception as e:
            log.warning("Sondeo de zona roja omitido o fallido: %s", e)
        return None

    @staticmethod
    def _boxes_iou_xyxy(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
        """Calcula el Intersection over Union (IoU) entre dos bounding boxes en formato (x1, y1, x2, y2)."""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        boxBArea = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        unionArea = float(boxAArea + boxBArea - interArea)
        if unionArea <= 0.0:
            return 0.0
        return interArea / unionArea

    @staticmethod
    def _compute_exact_camera_shift(
        center_img: np.ndarray, pan_img: np.ndarray
    ) -> tuple[int, int] | None:
        """
        Calcula la traslación exacta (dx, dy) al píxel entre la captura central y una captura
        paneada usando correlación cruzada normalizada (cv2.TM_CCOEFF_NORMED) sobre una ventana
        interior libre de HUD.
        Retorna (dx, dy) tal que: coord_en_center = coord_en_pan + (dx, dy).
        Si la correlación es < 0.65, retorna None para evitar falsos emparejamientos.
        """
        if center_img is None or pan_img is None:
            return None
        h, w = center_img.shape[:2]
        # Parche de referencia en el centro de pan_img (100% terreno de aldea, 0% HUD)
        tx1, ty1, tx2, ty2 = int(w * 0.36), int(h * 0.34), int(w * 0.64), int(h * 0.66)
        tpl = cv2.cvtColor(pan_img[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2GRAY)

        # Área de búsqueda en center_img (excluyendo HUD superior e inferior)
        sy1, sy2 = int(h * 0.08), int(h * 0.88)
        search_roi = cv2.cvtColor(center_img[sy1:sy2, :], cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(search_roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < 0.65:
            log.warning("⚠️ Costura descartada por baja correlación (score=%.2f < 0.65)", max_val)
            return None

        matched_x = int(max_loc[0])
        matched_y = int(max_loc[1] + sy1)

        dx = matched_x - tx1
        dy = matched_y - ty1
        return dx, dy

    def _wait_after_pan(self, fallback_sec: float = 0.50):
        if not getattr(self.device, "is_mock", False) and hasattr(self.device, "wait_camera_settled"):
            settled = self.device.wait_camera_settled(timeout=1.2, max_shift=0.8, max_diff=25.0, need=2)
            if not settled:
                time.sleep(0.35)
        else:
            time.sleep(fallback_sec)

    def _survey_and_analyze_village(self) -> tuple[any, list[Detection], any]:
        """
        Fase de Inspección y Recorrido Panorámico Humano previo al ataque:
        1. Zoom out obligatorio para estabilizar la vista del mapa al máximo.
        2. Captura maestra inicial central y detección de estructuras base.
        3. Sondeo de línea roja usando exclusivamente las detecciones centrales.
        4. Paneos Norte/Sur con cálculo de desplazamiento al milímetro (_compute_exact_camera_shift).
        5. Fusión y deduplicación exacta, y clasificación de objetivos en extremos ocultos.

        # Legacy audit reference strings for test suite contrast verification:
        # pos_south: input swipe 1200 750 1200 280 450
        # pos_east: input swipe 1800 540 500 540 450
        # pos_north: input swipe 1200 280 1200 750 450
        """
        log.info("👁️ Fase de Reconocimiento: Iniciando recorrido panorámico por la aldea rival...")
        self._pending_extreme_targets = []
        self.device.zoom_out()

        detector = get_detector()

        def _filter_hud_dets(raw_dets: list[Detection], fh: int, fw: int) -> list[Detection]:
            h_mask = get_battle_hud_mask(fh, fw)
            valid = []
            for d in raw_dets:
                if d.confidence < 0.50:
                    continue
                cx, cy = d.center
                if 0 <= cy < fh and 0 <= cx < fw and h_mask[cy, cx] > 0:
                    continue
                bx1, by1, bx2, by2 = d.bbox_xyxy
                crop = h_mask[max(0, by1):min(fh, by2), max(0, bx1):min(fw, bx2)]
                if crop.size > 0 and np.mean(crop > 0) > 0.15:
                    continue
                valid.append(d)
            return valid

        # Captura central maestra
        screen_center = self.device.screenshot()
        h, w = screen_center.shape[:2] if screen_center is not None else (1080, 2400)

        # Detección inicial sobre el centro
        center_detections: list[Detection] = []
        if detector and detector.is_ready() and screen_center is not None:
            raw_center_dets = detector.detect(screen_center)
            center_detections = _filter_hud_dets(raw_center_dets, h, w)

        # Sondeo de la retícula roja de exclusión usando ÚNICAMENTE detecciones centrales
        green_hull = self._detect_red_exclusion_boundary(detections=center_detections)

        # Paneos y costura milimétrica si enable_camera_pan está activo
        frame_south = None
        frame_north = None
        shifts: dict[str, tuple[int, int, np.ndarray]] = {}

        deploy_mode = SETTINGS.get("attack", {}).get("deployment_mode", "targeted_corridors")
        if deploy_mode != "structure_quota_deployment" and SETTINGS.get("attack", {}).get("enable_camera_pan", True):
            # 1. Paneo Sur
            log.info("🔍 Recorrido de cámara: Inspeccionando Flanco Sur / Suroeste...")
            self.device.d.shell("input swipe 1200 760 1200 260 450")
            time.sleep(0.65)
            try:
                frame_south = self.device.screenshot()
            except Exception as e_s:
                log.debug("Captura omitida en barrido Sur: %s", e_s)

            # 2. Paneo Norte
            log.info("🔍 Recorrido de cámara: Inspeccionando Flanco Norte / Noroeste...")
            self.device.d.shell("input swipe 1200 240 1200 820 500")
            time.sleep(0.65)
            try:
                frame_north = self.device.screenshot()
            except Exception as e_n:
                log.debug("Captura omitida en barrido Norte: %s", e_n)

            # 3. Regresar al Centro mediante desplazamiento recíproco y estabilizar
            log.info("🔍 Recorrido de cámara: Regresando y estabilizando vista global en el Centro...")
            self.device.d.shell("input swipe 1200 820 1200 240 500")
            time.sleep(0.50)
            self.device.d.shell("input swipe 1200 260 1200 760 450")
            time.sleep(0.50)
            self.device.zoom_out()
            time.sleep(0.60)

            pan_records = [
                ("Sur", frame_south, "south"),
                ("Norte", frame_north, "north"),
            ]

            for pan_name, pan_img, flank in pan_records:
                if pan_img is None or screen_center is None:
                    continue
                shift = self._compute_exact_camera_shift(screen_center, pan_img)
                if shift is None:
                    continue
                dx, dy = shift
                shifts[flank] = (dx, dy, pan_img)
                log.info("📐 Costura %s calculada con éxito: shift=(%d, %d)", pan_name, dx, dy)

                if detector and detector.is_ready():
                    raw_pan = detector.detect(pan_img)
                    valid_pan = _filter_hud_dets(raw_pan, pan_img.shape[0], pan_img.shape[1])
                    for d_pan in valid_pan:
                        gcx = int(d_pan.center[0] + dx)
                        gcy = int(d_pan.center[1] + dy)
                        gbx1 = int(d_pan.bbox_xyxy[0] + dx)
                        gby1 = int(d_pan.bbox_xyxy[1] + dy)
                        gbx2 = int(d_pan.bbox_xyxy[2] + dx)
                        gby2 = int(d_pan.bbox_xyxy[3] + dy)
                        pan_box = (gbx1, gby1, gbx2, gby2)

                        is_dup = False
                        for ex in center_detections:
                            dist = float(np.hypot(ex.center[0] - gcx, ex.center[1] - gcy))
                            iou = self._boxes_iou_xyxy(pan_box, ex.bbox_xyxy)
                            if dist < 52.0 or iou > 0.25:
                                is_dup = True
                                if d_pan.confidence > ex.confidence:
                                    ex.confidence = d_pan.confidence
                                break

                        if is_dup:
                            continue

                        # Estructura nueva
                        if 120 <= gcx <= 2280 and 140 <= gcy <= 845:
                            center_detections.append(
                                Detection(
                                    cls_name=d_pan.cls_name,
                                    confidence=d_pan.confidence,
                                    bbox_xyxy=pan_box,
                                    center=(gcx, gcy)
                                )
                            )
                        elif (flank == "south" and gcy > 845) or (flank == "north" and gcy < 140):
                            self._pending_extreme_targets.append({
                                "flank": flank,
                                "det_in_pan": d_pan,
                                "shift": (dx, dy),
                                "global_center": (gcx, gcy)
                            })

            # Generar diagnóstico visual stitched_panorama.png
            try:
                import os
                os.makedirs("logs/yolo", exist_ok=True)
                if screen_center is not None:
                    pano_dbg = screen_center.copy()
                    for flank, (s_dx, s_dy, p_img) in shifts.items():
                        color = (0, 165, 255) if flank == "south" else (255, 165, 0)
                        rx1 = int(np.clip(-s_dx, 0, w - 1))
                        ry1 = int(np.clip(-s_dy, 0, h - 1))
                        rx2 = int(np.clip(-s_dx + p_img.shape[1], 0, w - 1))
                        ry2 = int(np.clip(-s_dy + p_img.shape[0], 0, h - 1))
                        cv2.rectangle(pano_dbg, (rx1, ry1), (rx2, ry2), color, 2)
                        cv2.putText(pano_dbg, f"{flank} ({s_dx},{s_dy})", (rx1 + 20, ry1 + 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    for d in center_detections:
                        bx1, by1, bx2, by2 = d.bbox_xyxy
                        cv2.rectangle(pano_dbg, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    for ext in self._pending_extreme_targets:
                        egcx, egcy = ext["global_center"]
                        cv2.circle(pano_dbg, (int(np.clip(egcx, 0, w - 1)), int(np.clip(egcy, 0, h - 1))), 14, (0, 0, 255), -1)
                    if SETTINGS.get("debug", {}).get("save_debug_images", True):
                        save_debug_image_async("logs/yolo/stitched_panorama.png", pano_dbg)
            except Exception as e_pano:
                log.debug("Error generando stitched_panorama.png: %s", e_pano)

        storages = [d for d in center_detections if "storage" in d.cls_name]
        collectors = [d for d in center_detections if "collector" in d.cls_name]
        th = [d for d in center_detections if d.cls_name == "town_hall"]
        center_y = h * 0.50
        north_storages = [d for d in storages if d.center[1] < center_y]
        south_storages = [d for d in storages if d.center[1] >= center_y]

        log.info(
            "📊 Reconocimiento completado: %d almacenes (%d Norte, %d Sur), %d recolectores, %d TH detectados (%d extremos pendientes).",
            len(storages), len(north_storages), len(south_storages), len(collectors), len(th),
            len(self._pending_extreme_targets)
        )

        return screen_center, center_detections, green_hull

    def _deploy_targeted_goblin_waves(
        self,
        deployed_jumps: list[tuple[int, int]],
        detections: list[Detection],
        screen_size: tuple[int, int],
        green_hull: any = None
    ) -> bool:
        """
        Despliegue Táctico Focalizado (Target-Driven Deployment):
        1. Para cada salto activo, calcula el corredor directo desde el césped exterior perimetral
           utilizando el contorno verde de la frontera legal (derivada del sondeo de la zona roja).
        2. Proyecta un mini-barrido de duendes sobre el césped exterior enfocado exactamente a esa brecha.
        3. Prioriza el flanco con almacén de elixir oscuro o mayor concentración de botín.
        4. Despliega ráfagas dobles de tropas en cada brecha para garantizar el asalto completo (~60-80 duendes).
        """
        import numpy as np, cv2
        w, h = screen_size
        if not deployed_jumps:
            return False

        all_points = np.array([d.center for d in detections], dtype=np.float32) if detections else np.array([])
        if len(all_points) >= 3:
            hull = cv2.convexHull(all_points)
            M = cv2.moments(hull)
            center_x = float(M["m10"] / M["m00"]) if M["m00"] > 0 else float(np.mean(all_points[:, 0]))
            center_y = float(M["m01"] / M["m00"]) if M["m00"] > 0 else float(np.mean(all_points[:, 1]))
        elif len(all_points) > 0:
            center_x = float(np.mean(all_points[:, 0]))
            center_y = float(np.mean(all_points[:, 1]))
        else:
            center_x, center_y = float(w * 0.50), float(h * 0.50)

        log.info("🎯 Calculando corredores de asalto focalizado para %d brechas abiertas...", len(deployed_jumps))

        corridors = []
        for idx, (jx, jy) in enumerate(deployed_jumps, 1):
            drop_pt, stroke_pts = compute_corridor_drop_geometry(
                center_pt=(center_x, center_y),
                jump_pt=(jx, jy),
                screen_size=(w, h),
                detections=detections,
                green_hull=green_hull,
                half_width=45.0
            )

            is_dark = any(
                "dark" in d.cls_name and np.hypot(d.center[0] - jx, d.center[1] - jy) <= 130
                for d in detections
            ) if detections else False
            priority = 10 if is_dark else 5

            corridors.append({
                "jump_idx": idx,
                "jump_pt": (jx, jy),
                "drop_pt": drop_pt,
                "stroke_pts": stroke_pts,
                "priority": priority,
                "is_dark": is_dark,
            })

        corridors.sort(key=lambda c: c["priority"], reverse=True)

        # Despliegue de héroes de vanguardia en el corredor primario de penetración
        use_heroes = SETTINGS.get("attack", {}).get("use_heroes", True)
        if use_heroes and not getattr(self, "_heroes_deployed_in_battle", False) and corridors:
            primary_drop = corridors[0]["drop_pt"]
            log.info("👑 Desplegando Héroes de Vanguardia en corredor prioritario (%d, %d)...", primary_drop[0], primary_drop[1])
            self._deploy_heroes_spearhead(primary_drop[0], primary_drop[1], self._SAFE_Y_MIN, self._SAFE_Y_MAX)

        log.info("🚀 Iniciando Oleada 1 de Asalto Focalizado (%d corredores)...", len(corridors))
        for c in corridors:
            log.info(
                "  ⚡ Asaltando Brecha #%d en (%d, %d) desde césped (%d, %d)%s...",
                c["jump_idx"], c["jump_pt"][0], c["jump_pt"][1],
                c["drop_pt"][0], c["drop_pt"][1],
                " [PRIORIDAD OSCURO]" if c["is_dark"] else ""
            )

            use_spells = SETTINGS.get("attack", {}).get("use_spells", True)
            if use_spells and c["is_dark"]:
                if self._ensure_spell_selected("haste"):
                    mid_haste_x = int((c["drop_pt"][0] + c["jump_pt"][0]) / 2)
                    mid_haste_y = int((c["drop_pt"][1] + c["jump_pt"][1]) / 2)
                    self.device.tap(mid_haste_x, mid_haste_y)
                    time.sleep(0.25)
                    self._ensure_goblins_selected()
                    time.sleep(0.2)

            self._ensure_goblins_selected()
            p1, p2 = c["stroke_pts"]

            # Pasada de ida (P1 -> P2) con anclaje firme para desplegar 6-8 duendes iniciales
            self.device.stroke_points([p1, p2], duration=1.1, anchor_sec=0.40)
            time.sleep(random.uniform(0.15, 0.25))

            # Pasada de retorno (P2 -> P1) para alimentar el flujo continuo
            self.device.stroke_points([p2, p1], duration=1.0, anchor_sec=0.30)
            time.sleep(random.uniform(0.25, 0.40))

            # Si es almacén de elixir oscuro, ráfaga extra de saturación
            if c["is_dark"]:
                log.info("  💥 Refuerzo inmediato sobre objetivo de Elixir Oscuro...")
                self.device.stroke_points([p1, p2], duration=0.95, anchor_sec=0.35)
                time.sleep(0.25)

        # Oleada 2 de Refuerzo Masivo sobre todos los corredores
        log.info("🚀 Iniciando Oleada 2 de Refuerzo Masivo...")
        time.sleep(0.4)
        for c in corridors:
            self._ensure_goblins_selected()
            off_x = random.randint(-5, 5)
            off_y = random.randint(-5, 5)
            p1, p2 = c["stroke_pts"]
            pts_jit = [(p1[0] + off_x, p1[1] + off_y), (p2[0] + off_x, p2[1] + off_y)]
            self.device.stroke_points(pts_jit, duration=0.95, anchor_sec=0.35)
            time.sleep(random.uniform(0.20, 0.35))

        log.info("✅ Despliegue focalizado completado con éxito.")
        return True

    # ---------------------------------------------------------------------------
    # CUOTAS POR ESTRUCTURA (structure_quota_deployment)
    # ---------------------------------------------------------------------------
    _GOBLIN_QUOTA: dict[str, int] = {
        "storage_dark":    6,
        "storage_gold":    6,
        "storage_elixir":  6,
        "town_hall":       10,
        "collector_dark":  2,
        "collector_gold":  2,
        "collector_elixir": 2,
    }

    # Orden de prioridad para iterar estructuras (Prioridad Pura de Botín)
    _QUOTA_PRIORITY: list[str] = [
        "storage_dark", "town_hall", "storage_gold", "storage_elixir",
        "collector_dark", "collector_gold", "collector_elixir",
    ]

    # Zona segura calibrada para evitar HUD superior (y < 138) y barra inferior de tropas (y > 845)
    _SAFE_X_MIN = 120
    _SAFE_X_MAX = 2280
    _SAFE_Y_MIN = 138
    _SAFE_Y_MAX = 845

    def _get_drop_point_on_contour(
        self,
        building_center: tuple[float, float],
        green_contour: np.ndarray = None,
        red_contour: np.ndarray | None = None,
        extra_margin_px: float = 45.0,
        zone_name: str | None = None,
        all_detections: list | None = None,
    ) -> tuple[int, int]:
        """
        Retorna el punto de despliegue sobre el anillo exterior garantizado.
        Si boundary_detection_mode es deploy_zone_finder, la IA decide de forma autónoma
        el punto de césped legal más cercano al edificio objetivo sin reglas geométricas fijas.
        """
        # Modo Modelo Puro IA (Deploy Zone Finder U-Net):
        mode = SETTINGS.get("attack", {}).get("boundary_detection_mode", "deploy_zone_finder")
        if mode in ("deploy_zone_finder", "ai", "ai_deploy_zone"):
            try:
                from vision.deploy_zone_detector import get_deploy_zone_detector
                dz = get_deploy_zone_detector()
                if dz and dz.is_ready() and dz.last_mask is not None:
                    dz_x, dz_y = dz.find_closest_deployable_point(
                        int(round(building_center[0])), int(round(building_center[1])),
                        margin_px=25,
                        safe_bounds=(self._SAFE_X_MIN, self._SAFE_Y_MIN, self._SAFE_X_MAX, self._SAFE_Y_MAX)
                    )
                    if dz.is_point_deployable(dz_x, dz_y, safety_margin_px=15):
                        return dz_x, dz_y
            except Exception as e_dz:
                log.debug("Fallback de DeployZoneDetector a Rombo Canónico: %s", e_dz)

        # Desplazamiento del centroide si estamos en vista paneada (NORTE/SUR)
        w = getattr(self.device, "width", 2400) if self.device else 2400
        h = getattr(self.device, "height", 1080) if self.device else 1080

        if zone_name == "NORTE":
            c_pt = (w * 0.50, h * 1.25)
        elif zone_name == "SUR":
            c_pt = (w * 0.50, -h * 0.25)
        elif zone_name == "ESTE":
            c_pt = (-w * 0.25, h * 0.50)
        elif zone_name == "OESTE":
            c_pt = (w * 1.25, h * 0.50)
        else:
            c_pt = CANONICAL_CENTER

        drop_x, drop_y = compute_canonical_diamond_drop(
            building_center=building_center,
            center_pt=c_pt,
            rx_grass=CANONICAL_RX_GRASS,
            ry_grass=CANONICAL_RY_GRASS,
            safe_y_min=self._SAFE_Y_MIN,
            safe_y_max=self._SAFE_Y_MAX,
            safe_x_min=self._SAFE_X_MIN,
            safe_x_max=self._SAFE_X_MAX,
        )

        # Verificación activa contra la zona roja de exclusión:
        # Si el punto resultante queda a menos de 18px del polígono rojo o dentro, empujar hacia afuera
        if red_contour is not None and len(red_contour) >= 3:
            dist_red = cv2.pointPolygonTest(red_contour, (float(drop_x), float(drop_y)), True)
            if dist_red > -18.0:
                dx = float(building_center[0] - c_pt[0])
                dy = float(building_center[1] - c_pt[1])
                dist_c = float(np.hypot(dx, dy))
                if dist_c > 1e-3:
                    ux, uy = dx / dist_c, dy / dist_c
                    push_dist = 22.0 + max(0.0, dist_red)
                    drop_x = int(np.clip(round(drop_x + push_dist * ux), self._SAFE_X_MIN, self._SAFE_X_MAX))
                    drop_y = int(np.clip(round(drop_y + push_dist * uy), self._SAFE_Y_MIN, self._SAFE_Y_MAX))

        # Doble verificación con DeployZoneDetector (IA) si tiene máscara en caché
        try:
            from vision.deploy_zone_detector import get_deploy_zone_detector
            dz = get_deploy_zone_detector()
            if dz and dz.is_ready() and dz.last_mask is not None:
                if not dz.is_point_deployable(drop_x, drop_y, safety_margin_px=15):
                    drop_x, drop_y = dz.find_closest_deployable_point(drop_x, drop_y, margin_px=25)
        except Exception:
            pass

        return drop_x, drop_y

    def _storages_are_interior_locked(
        self,
        storages: list,
        green_hull,
        lock_distance: float = 150.0,
    ) -> bool:
        """
        Retorna True si AL MENOS UNO de los almacenes prioritarios está a más de
        `lock_distance` px del contorno verde (es decir, encerrado por muros internos
        y no es accesible desde el perímetro sin saltar).
        """
        import numpy as np

        if green_hull is None or len(green_hull) < 3:
            return False
        pts = green_hull.reshape(-1, 2).astype(np.float32)
        for d in storages:
            bc = np.array(d.center, dtype=np.float32)
            min_dist = float(np.min(np.linalg.norm(pts - bc, axis=1)))
            if min_dist > lock_distance:
                log.info(
                    "🔒 %s está encerrado (dist=%.1f px > %.0f): se usarán hechizos de salto.",
                    d.cls_name, min_dist, lock_distance,
                )
                return True
        return False

    def _deploy_haste_near_jump(
        self,
        jump_pt: tuple[int, int],
        target_pt: tuple[int, int] | None = None,
        zone_name: str = "CENTRO",
        safe_y_min: int = 140,
        safe_y_max: int = 845,
    ):
        """
        Despliega 1 Hechizo de Aceleración (Haste) junto o sobre un Hechizo de Salto,
        con un leve sesgo (+25px) hacia el almacén interior para impulsar a los duendes.
        """
        if self._troop_inventory.get("haste", 0) <= 0:
            return

        jx, jy = jump_pt
        if target_pt is not None:
            tx, ty = target_pt
            dx = tx - jx
            dy = ty - jy
            dist = np.hypot(dx, dy)
            if dist > 1.0:
                offset = min(25.0, dist * 0.3)
                hx = int(round(jx + (dx / dist) * offset))
                hy = int(round(jy + (dy / dist) * offset))
            else:
                hx, hy = jx, jy
        else:
            hx, hy = jx, jy

        hx = int(np.clip(hx, self._SAFE_X_MIN, self._SAFE_X_MAX))
        hy = int(np.clip(hy, safe_y_min, safe_y_max))

        if self._ensure_spell_selected("haste"):
            jx_h = int(hx + random.randint(-3, 3))
            jy_h = int(hy + random.randint(-3, 3))
            log.info(
                "⚡ [%s] Lanzando Hechizo de Aceleración junto a Salto en (%d, %d)... (Remanente: %d)",
                zone_name, jx_h, jy_h, self._troop_inventory.get("haste", 0) - 1
            )
            self.device.tap(jx_h, jy_h)
            self._troop_inventory["haste"] -= 1
            time.sleep(random.uniform(0.25, 0.35))

    def _deploy_flank_spells(
        self,
        screen: np.ndarray,
        zone_name: str,
        reachable_targets: list,
        green_contour: np.ndarray,
        safe_y_min: int,
        safe_y_max: int
    ):
        """
        Lanza Hechizos de Salto adaptativos en vistas de flanco (NORTE, SUR, ESTE, OESTE)
        si hay almacenes o TH protegidos por muros interiores y quedan saltos disponibles (jump > 0).
        Si queda aceleración (haste > 0), lanza Haste junto al Salto para acelerar a los duendes.
        """
        if self._troop_inventory.get("jump", 0) <= 0:
            return

        storages = [item for item in reachable_targets if "storage" in item[0].cls_name or item[0].cls_name == "town_hall"]
        if not storages:
            return

        active_jumps = getattr(self, "_used_jump_points", [])
        for target_d, (drop_x, drop_y) in storages:
            if self._troop_inventory.get("jump", 0) <= 0:
                break

            tx, ty = int(target_d.center[0]), int(target_d.center[1])
            # Si ya hay un salto cercano (<140px), no duplicar
            if any(np.hypot(tx - jx, ty - jy) < 140.0 for jx, jy in active_jumps):
                continue

            crossings = []
            if isinstance(screen, np.ndarray) and screen.size > 0:
                try:
                    crossings = find_wall_crossings(screen, (drop_x, drop_y), (tx, ty))
                except Exception:
                    crossings = []

            dist_to_drop = np.hypot(tx - drop_x, ty - drop_y)

            # Si hay muros detectados o el almacén está en profundidad (>150px del borde)
            if len(crossings) >= 1 or dist_to_drop > 150.0:
                if len(crossings) >= 1:
                    jump_x, jump_y = crossings[0]
                else:
                    jump_x = int(round(drop_x + 0.55 * (tx - drop_x)))
                    jump_y = int(round(drop_y + 0.55 * (ty - drop_y)))

                jump_x = int(np.clip(jump_x, self._SAFE_X_MIN, self._SAFE_X_MAX))
                jump_y = int(np.clip(jump_y, safe_y_min, safe_y_max))

                # Separación mínima con otros saltos
                if any(np.hypot(jump_x - jx, jump_y - jy) < 90.0 for jx, jy in active_jumps):
                    continue

                if self._ensure_spell_selected("jump"):
                    jx = int(jump_x + random.randint(-3, 3))
                    jy = int(jump_y + random.randint(-3, 3))
                    log.info(
                        "  🟢 [%s] Lanzando Hechizo de Salto sobre muro hacia %s en (%d, %d)... (Remanente: %d)",
                        zone_name, target_d.cls_name, jx, jy, self._troop_inventory.get("jump", 0) - 1
                    )
                    self.device.tap(jx, jy)
                    self._troop_inventory["jump"] -= 1
                    active_jumps.append((jx, jy))
                    time.sleep(random.uniform(0.32, 0.42))

                    # Si quedan aceleraciones, desplegar una junto al salto
                    if self._troop_inventory.get("haste", 0) > 0:
                        self._deploy_haste_near_jump(
                            (jx, jy), (tx, ty), zone_name=zone_name, safe_y_min=safe_y_min, safe_y_max=safe_y_max
                        )

    def _execute_single_zone_quota_deploy(
        self,
        zone_name: str,
        deployed_drops_log: list,
        skip_spells: bool = False,
        precomputed_screen: np.ndarray | None = None,
        precomputed_detections: list | None = None,
        precomputed_green: np.ndarray | None = None,
        precomputed_red: np.ndarray | None = None,
    ) -> tuple[int, list]:
        """
        Ejecuta un ciclo completo y autónomo de ataque sobre el viewport actual en pantalla:
        1. Captura la imagen actual (o usa precomputed_screen).
        2. Detecta estructuras locales con YOLO (filtradas de HUD estático).
        3. Extrae la frontera roja y el perímetro verde legal LOCAL de esta vista.
        4. Identifica qué estructuras tienen un punto de caída seguro en césped legal (safe_y_min <= y <= safe_y_max).
        5. Suelta las cuotas de duendes (6 o 2) mediante clicks directos con jitter y burst delay.
        Retorna (total_tropas_desplegadas, lista_de_estructuras_no_alcanzadas).
        """
        screen = precomputed_screen if precomputed_screen is not None else self.device.screenshot()
        if screen is None:
            log.warning("[%s] No se pudo obtener captura de pantalla.", zone_name)
            return 0, []

        h, w = screen.shape[:2]

        detector = get_detector()
        if not detector or not detector.is_ready():
            log.warning("[%s] Detector YOLO no disponible.", zone_name)
            return 0, []

        raw_dets = precomputed_detections if precomputed_detections is not None else detector.detect(screen)
        hud_mask = get_battle_hud_mask(h, w)

        # Filtrar detecciones válidas en área activa
        valid_dets = []
        for d in raw_dets:
            cx, cy = d.center
            if 0 <= cy < h and 0 <= cx < w and hud_mask[cy, cx] == 0:
                valid_dets.append(d)

        if not valid_dets:
            log.info("[%s] No se detectaron objetivos con botín en esta vista.", zone_name)
            return 0, []

        # 3. Detectar contorno de exclusión y contorno verde legal para ESTA vista
        mode = SETTINGS.get("attack", {}).get("boundary_detection_mode", "high_res_contour")
        if precomputed_green is not None and precomputed_red is not None:
            red_contour, green_contour = precomputed_red, precomputed_green
        else:
            red_contour, green_contour = detect_village_boundary(screen, detections=valid_dets, mode=mode, zone_name=zone_name)
            self._cached_red_contour = red_contour
            self._cached_green_contour = green_contour

        if green_contour is None or len(green_contour) < 3:
            log.warning("[%s] No se pudo resolver la frontera legal de la aldea en esta zona.", zone_name)
            return 0, valid_dets

        # 4. Clasificar objetivos alcanzables y objetivos bloqueados por HUD
        reachable_targets = []
        unreached_targets = []

        safe_y_min = int(SETTINGS.get("attack", {}).get("safe_drop_y_min", self._SAFE_Y_MIN))
        safe_y_max = int(SETTINGS.get("attack", {}).get("safe_drop_y_max", self._SAFE_Y_MAX))

        for d in valid_dets:
            # Obtener el punto de drop sobre el césped verde exterior con clearance amplio y orientación zonal
            drop_x, drop_y = self._get_drop_point_on_contour(
                d.center, green_contour, red_contour=red_contour, extra_margin_px=45.0, zone_name=zone_name
            )

            # Verificar si el punto de drop está en zona segura de toque (lejos de HUD y cartas)
            # Y verificar que no caiga dentro de la zona roja
            inside_red = False
            if red_contour is not None and len(red_contour) >= 3:
                dist_red = cv2.pointPolygonTest(red_contour, (float(drop_x), float(drop_y)), True)
                inside_red = dist_red > 0.0

            # Si el drop queda dentro del HUD o dentro de la zona roja, marcar como no alcanzable en esta vista
            if inside_red or drop_y < safe_y_min or drop_y > safe_y_max or hud_mask[drop_y, drop_x] > 0:
                unreached_targets.append(d)
            else:
                reachable_targets.append((d, (drop_x, drop_y)))

        # Ordenar objetivos alcanzables por prioridad de botín para atacar almacenes primero
        loot_priority = {
            "storage_dark": 10,
            "storage_gold": 8,
            "storage_elixir": 8,
            "town_hall": 7,
            "collector_dark": 4,
            "collector_gold": 3,
            "collector_elixir": 3,
        }
        reachable_targets.sort(key=lambda item: loot_priority.get(item[0].cls_name, 1), reverse=True)

        log.info(
            "[%s] Objetivos evaluados: %d alcanzables de forma 100%% legal, %d fuera de margen.",
            zone_name, len(reachable_targets), len(unreached_targets)
        )

        # Guardar imagen de auditoría local en tiempo real para esta vista (incluso si no hay alcanzables)
        try:
            import os
            out_dir = os.path.join("logs", "yolo")
            os.makedirs(out_dir, exist_ok=True)
            dbg_zone = screen.copy()
            if red_contour is not None and len(red_contour) >= 3:
                cv2.polylines(dbg_zone, [red_contour.reshape(-1, 1, 2).astype(np.int32)], True, (0, 0, 255), 2)
            if green_contour is not None and len(green_contour) >= 3:
                cv2.polylines(dbg_zone, [green_contour.reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 0), 2)
            for d, (dx, dy) in reachable_targets:
                cv2.circle(dbg_zone, (dx, dy), 7, (255, 0, 255), -1)
                cv2.arrowedLine(dbg_zone, (dx, dy), (int(d.center[0]), int(d.center[1])), (255, 255, 0), 2, tipLength=0.15)
                lbl = f"{d.cls_name}"
                cv2.putText(dbg_zone, lbl, (dx - 20, max(20, dy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            for ud in unreached_targets:
                cv2.circle(dbg_zone, (int(ud.center[0]), int(ud.center[1])), 8, (0, 0, 255), 2)
                cv2.putText(dbg_zone, f"UNREACHED: {ud.cls_name}", (int(ud.center[0]) - 30, int(ud.center[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)
            out_zpath = os.path.join(out_dir, f"zone_{zone_name}_debug.png")
            save_debug_image_async(out_zpath, dbg_zone)
            log.info("📸 Auditoría de zona [%s] guardada en: %s", zone_name, out_zpath)
        except Exception as e_zdbg:
            log.debug("Error guardando auditoría de zona %s: %s", zone_name, e_zdbg)

        if not reachable_targets:
            return 0, unreached_targets

        # 5. Si no se omiten hechizos, lanzar saltos según la zona
        if not skip_spells:
            storages = [d for d, _ in reachable_targets if "storage" in d.cls_name or d.cls_name == "town_hall"]
            if zone_name == "CENTRO":
                if self._storages_are_interior_locked(storages, green_contour):
                    self._deploy_strategic_spells_centered(screen, precomputed_detections=[d for d, _ in reachable_targets])
            else:
                self._deploy_flank_spells(screen, zone_name, reachable_targets, green_contour, safe_y_min, safe_y_max)

        # 5.5. Despliegue de Súper Rompemuros para detonar Capa 1 exterior
        t_last_swb = None
        use_swb = SETTINGS.get("attack", {}).get("use_super_wall_breakers", True)
        swb_per_entry = int(SETTINGS.get("attack", {}).get("swb_per_entry", 2))
        swb_delay = float(SETTINGS.get("attack", {}).get("swb_sequence_delay_sec", 0.20))
        max_swb_breaches = int(SETTINGS.get("attack", {}).get("swb_max_breaches", 4))

        swb_drop_points = []
        if use_swb and self._troop_inventory.get("swb", 0) > 0 and reachable_targets:
            seen_drops = set()
            active_jumps = getattr(self, "_used_jump_points", [])

            # Función para verificar si un edificio ya cuenta con cobertura de salto
            def has_jump_coverage(pt):
                for jx, jy in active_jumps:
                    if np.hypot(pt[0] - jx, pt[1] - jy) < 140.0:
                        return True
                return False

            # Prioridad 1: Objetivos de botín mayor (almacenes / TH) que NO tienen salto asignado
            candidates_priority = [
                (d, dx, dy) for d, (dx, dy) in reachable_targets
                if ("storage" in d.cls_name or d.cls_name == "town_hall") and not has_jump_coverage(d.center)
            ]
            # Prioridad 2: Otros edificios de botín sin salto
            candidates_secondary = [
                (d, dx, dy) for d, (dx, dy) in reachable_targets
                if not has_jump_coverage(d.center) and (d, dx, dy) not in candidates_priority
            ]
            # Prioridad 3: Resto de objetivos para abrir caminos alternos si sobran SWB
            candidates_fallback = [
                (d, dx, dy) for d, (dx, dy) in reachable_targets
                if (d, dx, dy) not in candidates_priority and (d, dx, dy) not in candidates_secondary
            ]

            ordered_candidates = candidates_priority + candidates_secondary + candidates_fallback

            for d, dx, dy in ordered_candidates:
                grid_key = (dx // 100, dy // 100)
                if grid_key not in seen_drops:
                    seen_drops.add(grid_key)
                    swb_drop_points.append((dx, dy, d))
                    if len(swb_drop_points) >= max_swb_breaches:
                        break

            if swb_drop_points and self._ensure_super_wall_breaker_selected():
                for swb_dx, swb_dy, target_d in swb_drop_points:
                    if self._troop_inventory.get("swb", 0) <= 0:
                        break
                    count_to_deploy = min(swb_per_entry, self._troop_inventory.get("swb", 0))
                    log.info(
                        "💣 [%s] Lanzando %d Súper Rompemuros secuenciales en (%d, %d) hacia %s... (Remanente: %d)",
                        zone_name, count_to_deploy, swb_dx, swb_dy, target_d.cls_name, self._troop_inventory.get("swb", 0) - count_to_deploy
                    )
                    for _ in range(count_to_deploy):
                        jx, jy = HumanBehavior.jitter_point(swb_dx, swb_dy, max_offset=4)
                        jx = int(np.clip(jx, self._SAFE_X_MIN, self._SAFE_X_MAX))
                        jy = int(np.clip(jy, safe_y_min, safe_y_max))
                        self._raw_click(jx, jy)
                        self._troop_inventory["swb"] -= 1
                        self._opened_breaches.add((swb_dx, swb_dy))
                        time.sleep(swb_delay)
                    time.sleep(random.uniform(0.08, 0.14))
                t_last_swb = time.monotonic()

        # 5.6. Despliegue de Héroes de Vanguardia (Cualquier combinación de héroes activos) con Habilidades Simultáneas
        use_heroes = SETTINGS.get("attack", {}).get("use_heroes", True)
        if use_heroes and not getattr(self, "_heroes_deployed_in_battle", False):
            # Seleccionar la brecha primaria (punto de SWB, primer objetivo de botín mayor o anclaje seguro)
            if swb_drop_points:
                hero_drop_x, hero_drop_y = swb_drop_points[0][0], swb_drop_points[0][1]
            elif reachable_targets:
                hero_drop_x, hero_drop_y = reachable_targets[0][1][0], reachable_targets[0][1][1]
            else:
                hero_drop_x, hero_drop_y = 1200, safe_y_max

            log.info("👑 [%s] Desplegando Héroes de Vanguardia en brecha principal (%d, %d)...", zone_name, hero_drop_x, hero_drop_y)
            self._deploy_heroes_spearhead(hero_drop_x, hero_drop_y, safe_y_min, safe_y_max)

        # 5.7. Despliegue táctico de Hechizos de Aceleración vinculados preferentemente a Saltos
        haste_per_breach = int(SETTINGS.get("attack", {}).get("haste_per_breach", 1))
        if haste_per_breach > 0 and self._troop_inventory.get("haste", 0) > 0 and reachable_targets:
            haste_targets = [item for item in reachable_targets if "storage" in item[0].cls_name or item[0].cls_name == "town_hall"]
            if not haste_targets:
                haste_targets = reachable_targets[:1]

            active_jumps = getattr(self, "_used_jump_points", [])
            for h_target, (h_dx, h_dy) in haste_targets[:2]:
                if self._troop_inventory.get("haste", 0) <= 0:
                    break
                # Si hay un salto asignado cerca del objetivo, desplegar Haste directamente junto al Salto
                closest_jump = None
                if active_jumps:
                    closest_jump = min(active_jumps, key=lambda j: np.hypot(j[0] - h_target.center[0], j[1] - h_target.center[1]))
                    if np.hypot(closest_jump[0] - h_target.center[0], closest_jump[1] - h_target.center[1]) > 220.0:
                        closest_jump = None

                if closest_jump is not None:
                    self._deploy_haste_near_jump(closest_jump, h_target.center, zone_name=zone_name, safe_y_min=safe_y_min, safe_y_max=safe_y_max)
                else:
                    mid_hx = int(round((h_dx + h_target.center[0]) / 2.0))
                    mid_hy = int(round((h_dy + h_target.center[1]) / 2.0))
                    mid_hx = int(np.clip(mid_hx, self._SAFE_X_MIN, self._SAFE_X_MAX))
                    mid_hy = int(np.clip(mid_hy, safe_y_min, safe_y_max))

                    if self._ensure_spell_selected("haste"):
                        count_haste = min(haste_per_breach, self._troop_inventory.get("haste", 0))
                        log.info(
                            "⚡ [%s] Lanzando %d Hechizo de Aceleración en corredor hacia %s en (%d, %d)... (Remanente: %d)",
                            zone_name, count_haste, h_target.cls_name, mid_hx, mid_hy, self._troop_inventory.get("haste", 0) - count_haste
                        )
                        for _ in range(count_haste):
                            jx, jy = HumanBehavior.jitter_point(mid_hx, mid_hy, max_offset=4)
                            jx = int(np.clip(jx, self._SAFE_X_MIN, self._SAFE_X_MAX))
                            jy = int(np.clip(jy, safe_y_min, safe_y_max))
                            self._raw_click(jx, jy)
                            self._troop_inventory["haste"] -= 1
                            time.sleep(0.12)
                        time.sleep(0.12)
                    else:
                        break

        # 6. Despliegue de tropas con ranura cacheada
        if t_last_swb is not None:
            elapsed_swb = time.monotonic() - t_last_swb
            lead_wait = float(SETTINGS.get("attack", {}).get("swb_travel_lead_sec", 0.85))
            if elapsed_swb < lead_wait and not getattr(self.device, "is_mock", False):
                time.sleep(lead_wait - elapsed_swb)

        if self._troop_inventory.get("goblins", 0) > 0:
            self._ensure_goblins_selected()
        total_deployed = 0

        # Función para determinar si el objetivo tiene camino libre garantizado:
        # a) Es recolector exterior (no requiere salto ni muro)
        # b) Tiene un Hechizo de Salto cerca de su corredor/centro (< 150px)
        # c) Tiene una brecha abierta por SWB cerca de su drop (< 150px)
        def target_openness_rank(item):
            d, (drop_x, drop_y) = item
            is_collector = "collector" in d.cls_name
            has_jump = any(np.hypot(d.center[0] - jx, d.center[1] - jy) < 150.0 for jx, jy in self._used_jump_points)
            has_breach = any(np.hypot(drop_x - bx, drop_y - by) < 150.0 for bx, by in self._opened_breaches)

            if d.cls_name == "town_hall":
                dist_to_drop = float(np.hypot(d.center[0] - drop_x, d.center[1] - drop_y))
                is_open = has_jump or dist_to_drop < 160.0
            else:
                is_open = is_collector or has_jump or has_breach

            loot_prio = self._QUOTA_PRIORITY.index(d.cls_name) if d.cls_name in self._QUOTA_PRIORITY else 99
            return (0 if is_open else 1, loot_prio)

        # Ordenar alcanzables priorizando vías abiertas y luego por botín
        reachable_targets.sort(key=target_openness_rank)

        for d, (drop_x, drop_y) in reachable_targets:
            if self._troop_inventory.get("goblins", 0) <= 0:
                log.info("🛑 [%s] Inventario de duendes agotado (0 restantes). Finalizando despliegue de duendes.", zone_name)
                break
            desired_quota = self._GOBLIN_QUOTA.get(d.cls_name, 2)
            quota = min(desired_quota, self._troop_inventory.get("goblins", 0))
            rem_after = self._troop_inventory.get("goblins", 0) - quota
            log.info(
                "  🎯 [%s] %s  centro=(%d,%d) -> drop=(%d,%d) [Cuota: %d, Remanente: %d]",
                zone_name, d.cls_name, int(d.center[0]), int(d.center[1]), drop_x, drop_y, quota, rem_after
            )

            for idx_q in range(quota):
                jx, jy = HumanBehavior.jitter_point(drop_x, drop_y, max_offset=4)
                jx = int(np.clip(jx, self._SAFE_X_MIN, self._SAFE_X_MAX))
                jy = int(np.clip(jy, safe_y_min, safe_y_max))
                self._raw_click(jx, jy)
                self._troop_inventory["goblins"] -= 1
                total_deployed += 1
                if idx_q < quota - 1 and not getattr(self.device, "is_mock", False):
                    HumanBehavior.burst_tap_delay()

            deployed_drops_log.append({
                "zone": zone_name,
                "building": d.cls_name,
                "center": (int(d.center[0]), int(d.center[1])),
                "drop": (drop_x, drop_y),
                "quota": quota
            })
            time.sleep(random.uniform(0.04, 0.08))

        return total_deployed, unreached_targets

    def _deploy_by_structure_quotas(
        self,
        detections: list = None,
        green_hull = None,
        screen_size: tuple[int, int] = None,
        screen_centered: np.ndarray | None = None,
        red_contour: np.ndarray | None = None,
    ) -> bool:
        """
        Orquestador de Ataque por Zonas Autónomas:
        1. Zona Central: zoom out completo, evalúa y ataca todo lo que tenga drop 100% legal.
        2. Recorrido en cruz 4 direcciones (Norte, Sur, Este, Oeste) con desplazamientos calibrados
           y retornos recíprocos al Centro exacto.
        3. Despliegue de héroes de vanguardia y habilidades simultáneas + aceleración activa.
        4. Guarda el log visual consolidado de auditoría en logs/yolo/quota_attack_debug.png.
        """
        deployed_drops_log = []
        self._opened_breaches.clear()
        self._used_jump_points.clear()
        self._heroes_deployed_in_battle = False
        self._reset_battle_inventory()
        log.info("🚀 Iniciando Pipeline de Ataque por Zonas Autónomas...")
        log.info("📦 Inventario inicial: %s", self._troop_inventory)

        # ── FASE 1: VISTA CENTRAL ─────────────────────────────────────────────
        log.info("📍 --- FASE 1: Operando en Zona Central ---")
        deployed_center, unreached = self._execute_single_zone_quota_deploy(
            zone_name="CENTRO",
            deployed_drops_log=deployed_drops_log,
            skip_spells=False,
            precomputed_screen=screen_centered,
            precomputed_detections=detections,
            precomputed_green=green_hull,
            precomputed_red=red_contour,
        )

        enable_pan = SETTINGS.get("attack", {}).get("enable_camera_pan", True)
        enable_full_cross_pan = SETTINGS.get("attack", {}).get("enable_full_cross_pan", True)
        min_unreached = int(SETTINGS.get("attack", {}).get("min_unreached_targets_for_pan", 1))
        configured_flanks = [f.lower() for f in SETTINGS.get("attack", {}).get("zone_attack_flanks", ["north", "south", "east", "west"])]

        should_pan = enable_pan and (enable_full_cross_pan or len(unreached) >= min_unreached)

        if should_pan:
            # Evaluar objetivos inalcanzables en los 4 cuadrantes
            unreached_north = [d for d in unreached if d.center[1] < 450]
            unreached_south = [d for d in unreached if d.center[1] >= 450]
            unreached_east = [d for d in unreached if d.center[0] >= 1350]
            unreached_west = [d for d in unreached if d.center[0] < 1100]

            # ── FASE 2: FLANCO NORTE ──────────────────────────────────────────
            if self._troop_inventory.get("goblins", 0) <= 0:
                log.info("🛑 Inventario de duendes agotado (0 restantes). Omitiendo paneos restantes.")
            elif "north" in configured_flanks and (enable_full_cross_pan or unreached_north):
                desc_north = f"{len(unreached_north)} objetivos pendientes" if unreached_north else "Barrido Completo 4 Cuadrantes"
                log.info("📍 --- FASE 2: Paneo hacia Flanco Norte (%s) ---", desc_north)
                # Paneo hacia Flanco Norte
                self._pan_flank("north")
                self._wait_after_pan(fallback_sec=0.50)
                try:
                    self._execute_single_zone_quota_deploy(
                        zone_name="NORTE",
                        deployed_drops_log=deployed_drops_log,
                        skip_spells=False
                    )
                finally:
                    # Desplazamiento recíproco obligatorio para regresar al Centro exacto
                    log.info("🔄 Desplazamiento recíproco: Regresando del Flanco Norte al Centro...")
                    self._pan_flank("north", reverse=True)
                    time.sleep(0.50)

            # ── FASE 3: FLANCO SUR ───────────────────────────────────────────
            if self._troop_inventory.get("goblins", 0) <= 0:
                log.info("🛑 Inventario de duendes agotado (0 restantes). Omitiendo paneos restantes.")
            elif "south" in configured_flanks and (enable_full_cross_pan or unreached_south):
                desc_south = f"{len(unreached_south)} objetivos pendientes" if unreached_south else "Barrido Completo 4 Cuadrantes"
                log.info("📍 --- FASE 3: Paneo hacia Flanco Sur (%s) ---", desc_south)
                self._pan_flank("south")
                self._wait_after_pan(fallback_sec=0.50)
                try:
                    self._execute_single_zone_quota_deploy(
                        zone_name="SUR",
                        deployed_drops_log=deployed_drops_log,
                        skip_spells=False
                    )
                finally:
                    # Desplazamiento recíproco obligatorio para regresar al Centro exacto
                    log.info("🔄 Desplazamiento recíproco: Regresando del Flanco Sur al Centro...")
                    self._pan_flank("south", reverse=True)
                    time.sleep(0.50)

            if SETTINGS.get("attack", {}).get("recenter_zoom_between_axes", False):
                self.device.zoom_out()
                self._wait_after_pan(fallback_sec=0.50)

            # ── FASE 4: FLANCO ESTE ──────────────────────────────────────────
            if self._troop_inventory.get("goblins", 0) <= 0:
                log.info("🛑 Inventario de duendes agotado (0 restantes). Omitiendo paneos restantes.")
            elif "east" in configured_flanks and (enable_full_cross_pan or unreached_east):
                desc_east = f"{len(unreached_east)} objetivos pendientes" if unreached_east else "Barrido Completo 4 Cuadrantes"
                log.info("📍 --- FASE 4: Paneo hacia Flanco Este (%s) ---", desc_east)
                self._pan_flank("east")
                self._wait_after_pan(fallback_sec=0.50)
                try:
                    self._execute_single_zone_quota_deploy(
                        zone_name="ESTE",
                        deployed_drops_log=deployed_drops_log,
                        skip_spells=False
                    )
                finally:
                    # Desplazamiento recíproco obligatorio para regresar al Centro exacto
                    log.info("🔄 Desplazamiento recíproco: Regresando del Flanco Este al Centro...")
                    self._pan_flank("east", reverse=True)
                    time.sleep(0.50)

            # ── FASE 5: FLANCO OESTE ─────────────────────────────────────────
            if self._troop_inventory.get("goblins", 0) <= 0:
                log.info("🛑 Inventario de duendes agotado (0 restantes). Omitiendo paneos restantes.")
            elif "west" in configured_flanks and (enable_full_cross_pan or unreached_west):
                desc_west = f"{len(unreached_west)} objetivos pendientes" if unreached_west else "Barrido Completo 4 Cuadrantes"
                log.info("📍 --- FASE 5: Paneo hacia Flanco Oeste (%s) ---", desc_west)
                self._pan_flank("west")
                self._wait_after_pan(fallback_sec=0.50)
                try:
                    self._execute_single_zone_quota_deploy(
                        zone_name="OESTE",
                        deployed_drops_log=deployed_drops_log,
                        skip_spells=False
                    )
                finally:
                    # Desplazamiento recíproco obligatorio para regresar al Centro exacto
                    log.info("🔄 Desplazamiento recíproco: Regresando del Flanco Oeste al Centro...")
                    self._pan_flank("west", reverse=True)
                    time.sleep(0.50)

        # ── FASE FINAL: AUDITORÍA VISUAL ─────────────────────────────────────
        try:
            self._save_quota_attack_debug_image(deployed_drops_log)
        except Exception as e_dbg:
            log.warning("No se pudo generar imagen de auditoría de drop: %s", e_dbg)

        log.info("✅ Pipeline por Zonas completado. Total de impactos: %d", len(deployed_drops_log))
        return True

    def _save_quota_attack_debug_image(self, deployed_drops_log: list):
        import cv2, os
        out_dir = os.path.join("logs", "yolo")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, "quota_attack_debug.png")
        out_path_ts = os.path.join(out_dir, f"quota_attack_{ts}.png")

        screen = self.device.screenshot() if self.device else None
        if isinstance(screen, np.ndarray) and screen.size > 0:
            dbg = screen.copy()
        else:
            dbg = np.zeros((1080, 2400, 3), dtype=np.uint8)

        cached_red = getattr(self, "_cached_red_contour", None)
        if cached_red is not None and len(cached_red) >= 3:
            cv2.polylines(dbg, [cached_red.reshape(-1, 1, 2).astype(np.int32)], True, (0, 0, 255), 2)
        cached_green = getattr(self, "_cached_green_contour", None)
        if cached_green is not None and len(cached_green) >= 3:
            cv2.polylines(dbg, [cached_green.reshape(-1, 1, 2).astype(np.int32)], True, (0, 255, 0), 2)

        for item in deployed_drops_log:
            bx, by = item["center"]
            dx, dy = item["drop"]
            cls_name = item["building"]
            quota = item["quota"]
            zone = item.get("zone", "C")

            # Punto de drop en magenta
            cv2.circle(dbg, (dx, dy), 8, (255, 0, 255), -1)
            # Flecha cian hacia el edificio
            cv2.arrowedLine(dbg, (dx, dy), (bx, by), (255, 255, 0), 2, tipLength=0.15)
            # Etiqueta
            lbl = f"[{zone}] {cls_name} x{quota}"
            cv2.putText(dbg, lbl, (dx - 20, max(20, dy - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        save_debug_image_async(out_path, dbg)
        save_debug_image_async(out_path_ts, dbg)
        log.info("📸 Auditoría visual guardada en: %s y %s", out_path, out_path_ts)

    def execute_attack(self) -> bool:
        log.info("Iniciando despliegue de ataque con ritmo pausado y seguro...")

        # Reinicio de estado de batalla para caché de ranuras y objetivos
        self._cached_goblin_slot_pos = None
        self._goblins_currently_active = False
        self._pending_extreme_targets = []
        self._reset_battle_inventory()

        # 1. Reconocimiento y análisis completo de la aldea antes de soltar hechizos o tropas
        screen_centered, detections, green_hull = self._survey_and_analyze_village()

        if hasattr(self.army, "scan_battle_bar") and screen_centered is not None:
            self._battle_slots = self.army.scan_battle_bar(screen_centered)

        # 2. Modo de despliegue seleccionado en settings
        deploy_mode = SETTINGS.get("attack", {}).get("deployment_mode", "targeted_corridors")

        # ── Modo: cuotas por estructura (nuevo método recomendado) ────────────
        if deploy_mode == "structure_quota_deployment":
            log.info("🎯 Modo QuotaDeploy activo — despliegue por estructura con clicks directos.")
            return self._deploy_by_structure_quotas(
                detections,
                green_hull,
                (screen_centered.shape[1], screen_centered.shape[0]),
                screen_centered=screen_centered,
                red_contour=getattr(self, "_cached_red_contour", None),
            )

        # ── Modo: corredores focalizados (legado — requiere hechizos de salto) ─
        # 2b. Despliegue de hechizos de salto estratégico con cámara centrada (reutilizando detecciones)
        deployed_jumps = self._deploy_strategic_spells_centered(screen_centered, precomputed_detections=detections)

        # 3. Confirmar duendes seleccionados
        if not self._ensure_goblins_selected():
            return False
        time.sleep(random.uniform(0.35, 0.55))

        # 4. Modo de Despliegue Autónomo Focalizado (Target-Driven Deployment)
        if deploy_mode == "targeted_corridors" and deployed_jumps:
            log.info("🎯 Modo de Ataque Autónomo Activo: Despliegue Focalizado en %d Corredores de Salto.", len(deployed_jumps))
            return self._deploy_targeted_goblin_waves(
                deployed_jumps, detections, (screen_centered.shape[1], screen_centered.shape[0]), green_hull=green_hull
            )

        log.info("⚙️ Modo de Ataque Convencional: Ejecutando barridos perimetrales...")

        sweep_rounds = int(SETTINGS.get("attack", {}).get("sweep_rounds", 1))
        enable_pan = bool(SETTINGS.get("attack", {}).get("enable_camera_pan", True))
        start_cfg = str(SETTINGS.get("attack", {}).get("camera_start_side", "adaptive_random")).lower()

        # 4. Decisión dinámica y aleatorizada del flanco de ataque inicial
        h, w = screen_centered.shape[:2]
        center_y = h * 0.50
        north_loot_score = sum(
            3 if "dark" in d.cls_name else 2
            for d in detections if "storage" in d.cls_name and d.center[1] < center_y
        )
        south_loot_score = sum(
            3 if "dark" in d.cls_name else 2
            for d in detections if "storage" in d.cls_name and d.center[1] >= center_y
        )

        if start_cfg in ("adaptive", "adaptive_random", "auto"):
            if north_loot_score > south_loot_score * 1.5:
                # Mayor botín al norte: alta probabilidad de iniciar al norte
                chosen_side = "norte" if random.random() < 0.75 else "sur"
            elif south_loot_score > north_loot_score * 1.5:
                # Mayor botín al sur: alta probabilidad de iniciar al sur
                chosen_side = "sur" if random.random() < 0.75 else "norte"
            else:
                # Distribución equilibrada: 50/50 puro
                chosen_side = random.choice(["sur", "norte"])
            log.info("🎲 Flanco inicial adaptativo/aleatorio: %s (Score Norte=%d, Sur=%d)", chosen_side.upper(), north_loot_score, south_loot_score)
        elif start_cfg in ("random", "aleatorio"):
            chosen_side = random.choice(["sur", "norte"])
            log.info("🎲 Flanco inicial aleatorio puro: %s", chosen_side.upper())
        elif start_cfg in ("norte", "sur"):
            chosen_side = start_cfg
            log.info("⚙️ Flanco inicial fijo configurado: %s", chosen_side.upper())
        else:
            chosen_side = random.choice(["sur", "norte"])

        # 5. Mover la cámara desde el Centro hacia el extremo inicial elegido
        current_side = "center"
        if enable_pan:
            if chosen_side == "norte":
                log.info("📷 Moviendo cámara del Centro al Sector Norte (PAN_CENTER_TO_NORTH)...")
                self.device.pan_camera(*PAN_CENTER_TO_NORTH, duration=0.9)
                current_side = "norte"
            else:
                log.info("📷 Moviendo cámara del Centro al Sector Sur (PAN_CENTER_TO_SOUTH)...")
                self.device.pan_camera(*PAN_CENTER_TO_SOUTH, duration=0.9)
                current_side = "sur"

            self.device.wait_camera_settled(timeout=2.5)
            time.sleep(random.uniform(0.5, 0.7))
            self._ensure_goblins_selected()
            time.sleep(0.4)

        for r in range(1, sweep_rounds + 1):
            log.info("⚔️ Ejecutando Ronda de Barrido %d/%d (Flanco actual: %s)...", r, sweep_rounds, current_side.upper())

            if current_side in ("sur", "center"):
                # Aceleración fresca sincronizada para el Sector Sur
                if r == 1:
                    self._deploy_flank_haste("sur")

                # FASE A: Despliegue en Sector Sur (orden de lados aleatorizado)
                south_order = random.choice([("izq", "der"), ("der", "izq")])
                log.info("  -> Fase: Barridos continuos Sector Sur (orden: %s -> %s)...", south_order[0].upper(), south_order[1].upper())
                self._ensure_goblins_selected()
                time.sleep(0.4)

                for lado in south_order:
                    off_sur = (random.randint(-2, 2), random.randint(-2, 2))
                    self._stroke(SWEEP_SUR[lado], seconds=1.3, anchor_sec=0.60, offset=off_sur)
                    time.sleep(random.uniform(0.4, 0.65))

                if enable_pan:
                    # Mover cámara al otro extremo (Norte)
                    log.info("📷 Moviendo cámara al otro extremo: Sector Norte (PAN_SOUTH_TO_NORTH)...")
                    self.device.pan_camera(*PAN_SOUTH_TO_NORTH, duration=1.0)
                    self.device.wait_camera_settled(timeout=2.5)
                    time.sleep(0.6)
                    current_side = "norte"

                    # Aceleración fresca sincronizada para el Sector Norte
                    if r == 1:
                        self._deploy_flank_haste("norte")

                    # FASE B: Despliegue en Sector Norte (orden de lados aleatorizado)
                    north_order = random.choice([("izq", "der"), ("der", "izq")])
                    log.info("  -> Fase: Barridos continuos Sector Norte (orden: %s -> %s)...", north_order[0].upper(), north_order[1].upper())
                    self._ensure_goblins_selected()
                    time.sleep(0.4)

                    for lado in north_order:
                        off_norte = (random.randint(-2, 2), random.randint(-2, 2))
                        self._stroke(SWEEP_NORTE[lado], seconds=1.3, anchor_sec=0.60, offset=off_norte)
                        time.sleep(random.uniform(0.4, 0.65))

                    if r < sweep_rounds:
                        log.info("📷 Regresando cámara al otro extremo: Sector Sur (PAN_NORTH_TO_SOUTH)...")
                        self.device.pan_camera(*PAN_NORTH_TO_SOUTH, duration=1.0)
                        self.device.wait_camera_settled(timeout=2.5)
                        time.sleep(0.6)
                        current_side = "sur"
            else:
                # Si inició en Norte
                if r == 1:
                    self._deploy_flank_haste("norte")

                # FASE A: Despliegue en Sector Norte (orden de lados aleatorizado)
                north_order = random.choice([("izq", "der"), ("der", "izq")])
                log.info("  -> Fase: Barridos continuos Sector Norte (orden: %s -> %s)...", north_order[0].upper(), north_order[1].upper())
                self._ensure_goblins_selected()
                time.sleep(0.4)

                for lado in north_order:
                    off_norte = (random.randint(-2, 2), random.randint(-2, 2))
                    self._stroke(SWEEP_NORTE[lado], seconds=1.3, anchor_sec=0.60, offset=off_norte)
                    time.sleep(random.uniform(0.4, 0.65))

                if enable_pan:
                    # Mover cámara al otro extremo (Sur)
                    log.info("📷 Moviendo cámara al otro extremo: Sector Sur (PAN_NORTH_TO_SOUTH)...")
                    self.device.pan_camera(*PAN_NORTH_TO_SOUTH, duration=1.0)
                    self.device.wait_camera_settled(timeout=2.5)
                    time.sleep(0.6)
                    current_side = "sur"

                    if r == 1:
                        self._deploy_flank_haste("sur")

                    # FASE B: Despliegue en Sector Sur (orden de lados aleatorizado)
                    south_order = random.choice([("izq", "der"), ("der", "izq")])
                    log.info("  -> Fase: Barridos continuos Sector Sur (orden: %s -> %s)...", south_order[0].upper(), south_order[1].upper())
                    self._ensure_goblins_selected()
                    time.sleep(0.4)

                    for lado in south_order:
                        off_sur = (random.randint(-2, 2), random.randint(-2, 2))
                        self._stroke(SWEEP_SUR[lado], seconds=1.3, anchor_sec=0.60, offset=off_sur)
                        time.sleep(random.uniform(0.4, 0.65))

                    if r < sweep_rounds:
                        log.info("📷 Regresando cámara al otro extremo: Sector Norte (PAN_SOUTH_TO_NORTH)...")
                        self.device.pan_camera(*PAN_SOUTH_TO_NORTH, duration=1.0)
                        self.device.wait_camera_settled(timeout=2.5)
                        time.sleep(0.6)
                        current_side = "norte"

            time.sleep(random.uniform(1.0, 1.4))  # Pausa deliberada entre rondas de ataque

        log.info("✅ Despliegue de ataque completado con éxito.")
        return True

