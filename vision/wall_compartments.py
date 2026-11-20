"""
Módulo de Optimización de Hechizos de Salto por Modelo de Compartimentos en Clash of Clans.
Modela la aldea como regiones cerradas por muros (compartimentos), calcula las capas
de muro mínimas desde las zonas de drop hacia cada objetivo de botín y selecciona
mediante Beam Search los puntos de salto que maximizan la reducción del costo global.

Fase 2 del Plan: colocación óptima de hechizos de salto (plan_saltos_optimos_antigravity.md).
Paso 3: Integración modular con WallSegDetector (ONNX + fallback Lab), calibración física
real (rx=76, ry=57, radius=72), filtro de densidad de muro y Beam Search.
"""
from __future__ import annotations
import math
import os
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
import cv2
import numpy as np

from core.logger import get_logger
from core.config import PROJECT_ROOT, SETTINGS

log = get_logger("wall_compartments")

# Pesos por defecto para objetivos de botín según settings.json
DEFAULT_LOOT_WEIGHTS: dict[str, float] = {
    "storage_dark": 3.0,
    "storage_gold": 2.0,
    "storage_elixir": 2.0,
    "collector_gold": 1.0,
    "collector_elixir": 1.0,
    "collector_dark": 1.0,
    "town_hall": 0.0,
}

# Calibración Física Real de la Huella del Salto (resolución base 2400x1080)
# rx = 76.0, ry = 57.0 (semi-ejes isométricos), radio efectivo de fusión = 72.0 px (~2.8 casillas)
DEFAULT_JUMP_RX: float = 76.0
DEFAULT_JUMP_RY: float = 57.0
DEFAULT_FUSION_RADIUS: float = 72.0
DEFAULT_MIN_WALL_PIXELS: int = 30
DEFAULT_CORE_RADIUS: float = 42.0
DEFAULT_BEAM_WIDTH: int = 4
DEFAULT_MAX_JUMPS: int = 4
DEFAULT_MIN_SEP_PX: float = 95.0

DEFAULT_JUMP_FOOTPRINT: dict = {
    "type": "ellipse",
    "rx": DEFAULT_JUMP_RX,
    "ry": DEFAULT_JUMP_RY,
    "radius": DEFAULT_FUSION_RADIUS,
    "angle": 0.0,
}


@dataclass
class JumpCandidate:
    """Representa un salto evaluado o seleccionado."""
    x: int
    y: int
    gain: float
    cost_before: float
    cost_after: float
    bridged_compartments: list[int] = field(default_factory=list)
    drop_dist: float = 0.0
    travel_dist: float = 0.0


@dataclass
class CompartmentAnalysisResult:
    """Resultado estructurado del análisis de compartimentos y optimización de saltos."""
    jumps: list[tuple[int, int, float]]  # [(x, y, gain), ...]
    jump_details: list[JumpCandidate]
    initial_cost: float
    final_cost: float
    total_gain: float
    num_compartments: int
    comp_labels: np.ndarray
    wall_mask: np.ndarray
    edges: list[tuple[int, int]]
    target_assignments: list[dict]
    execution_time_ms: float
    debug_overlay: np.ndarray | None = None
    graph: dict[int, set[int]] = field(default_factory=dict)
    entry_comps: set[int] = field(default_factory=set)
    nearest_labels: np.ndarray | None = None
    grid_scale: int = 1


def _angle_diff_180(ang1: float, ang2: float) -> float:
    """Diferencia angular mínima en el rango [0, 180)."""
    diff = abs((ang1 - ang2) % 180.0)
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def _get_line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Genera un kernel morfológico lineal a lo largo del ángulo angle_deg."""
    rad = math.radians(angle_deg)
    half = length // 2
    k = np.zeros((length, length), dtype=np.uint8)
    for i in range(-half, half + 1):
        x = int(round(half + i * math.cos(rad)))
        y = int(round(half + i * math.sin(rad)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1
    return k


from vision.hud import get_battle_hud_mask


def build_clean_wall_canvas(
    img: np.ndarray,
    detections: list | None = None,
    deploy_zone_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Preprocesador limpio para extracción de muros como fallback obligatorio:
    1. Enmascara el HUD de batalla (y >= 888 y bordes superiores/laterales).
    2. Suprime el césped exterior mediante máscara U-Net de deploy zone finder (si disponible).
    3. Suprime el interior de cajas de edificios YOLO (erosionadas 5px) para eliminar
       falsos bordes de techos.
    4. Extrae gradiente en espacio de color Lab (canales L y b, resistente a aldeas
       nevadas y muros oscuros).
    """
    if img is None or img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    h, w = img.shape[:2]
    hud_mask = get_battle_hud_mask(h, w)

    # 2. Convertir a Lab y calcular gradiente en canales L y b
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        L = lab[:, :, 0]
        b = lab[:, :, 2]
    else:
        L = img
        b = img

    grad_L_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    grad_L_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    mag_L = np.hypot(grad_L_x, grad_L_y)

    grad_b_x = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    grad_b_y = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    mag_b = np.hypot(grad_b_x, grad_b_y)

    grad = cv2.addWeighted(mag_L, 0.6, mag_b, 0.4, 0.0)
    canvas = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canvas[hud_mask > 0] = 0

    # 3. Supresión de césped exterior vía máscara U-Net
    if deploy_zone_mask is None:
        try:
            from vision.deploy_zone_detector import get_deploy_zone_detector
            dzd = get_deploy_zone_detector()
            if dzd and dzd.is_ready():
                deploy_zone_mask = dzd.predict_mask(img)
        except Exception as e:
            log.debug("No se pudo obtener deploy_zone_mask para build_clean_wall_canvas: %s", e)

    if deploy_zone_mask is not None:
        if deploy_zone_mask.shape[:2] != (h, w):
            deploy_zone_mask = cv2.resize(deploy_zone_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # En deploy_zone_detector: 1 = no-despliegue (interior aldea), 0 = césped exterior
        canvas[deploy_zone_mask == 0] = 0

    # 4. Supresión del interior de cajas de edificios YOLO (erosionadas 5 px)
    if detections is None:
        try:
            from vision.yolo_detector import get_detector
            det = get_detector()
            if det and det.is_ready():
                detections = det.detect(img)
        except Exception as e:
            log.debug("No se pudo obtener detector YOLO para supresión de cajas: %s", e)

    if detections:
        for det in detections:
            bbox = getattr(det, "bbox_xyxy", None)
            if bbox is None and isinstance(det, dict):
                bbox = det.get("bbox_xyxy") or det.get("bbox")
            if bbox is not None:
                bx1, by1, bx2, by2 = [int(v) for v in bbox[:4]]
                bx1_e = max(0, min(w - 1, bx1 + 5))
                by1_e = max(0, min(h - 1, by1 + 5))
                bx2_e = max(0, min(w, bx2 - 5))
                by2_e = max(0, min(h, by2 - 5))
                if bx2_e > bx1_e and by2_e > by1_e:
                    canvas[by1_e:by2_e, bx1_e:bx2_e] = 0

    return canvas


# Re-exportación oficial de WallSegDetector (U-Net MobileNetV2) desde vision.wall_seg_detector
from vision.wall_seg_detector import (
    WallSegDetector,
    get_wall_seg_detector,
    reset_wall_seg_detector,
    build_cyan_overlay,
    IMG_W,
    IMG_H,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


_CORE_MASKS_CACHE: dict[float, tuple[np.ndarray, int]] = {}


def _count_wall_pixels_in_core(
    pt: tuple[int, int],
    wall_mask: np.ndarray,
    core_radius: float = DEFAULT_CORE_RADIUS,
) -> int:
    """
    Cuenta la cantidad de píxeles de muro presentes dentro del radio core_radius
    alrededor de pt=(x, y) de forma eficiente utilizando máscaras circulares en caché.
    """
    if wall_mask is None or getattr(wall_mask, "size", 0) == 0 or core_radius <= 0:
        return 0

    cx, cy = int(round(pt[0])), int(round(pt[1]))
    h, w = wall_mask.shape[:2]
    r = int(math.ceil(core_radius))

    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)

    if x1 <= x0 or y1 <= y0:
        return 0

    sub_mask = wall_mask[y0:y1, x0:x1]
    if core_radius not in _CORE_MASKS_CACHE:
        ys, xs = np.ogrid[-r : r + 1, -r : r + 1]
        _CORE_MASKS_CACHE[core_radius] = ((xs ** 2 + ys ** 2) <= (core_radius ** 2), r)

    full_circle, cr = _CORE_MASKS_CACHE[core_radius]
    mask_slice = full_circle[y0 - cy + cr : y1 - cy + cr, x0 - cx + cr : x1 - cx + cr]
    return int(np.count_nonzero(sub_mask[mask_slice] > 0))


def is_valid_wall_centered_jump(
    pt: tuple[int, int],
    wall_mask: np.ndarray,
    core_radius: float = DEFAULT_CORE_RADIUS,
    min_wall_pixels: int = DEFAULT_MIN_WALL_PIXELS,
) -> bool:
    """
    Filtro de Densidad de Muro:
    Verifica que el punto candidato de salto (pt = (x, y)) posea al menos
    min_wall_pixels (por defecto 30) de muro dentro de su núcleo (core_radius = 42 px),
    evitando que el bot coloque saltos en el centro vacío de un compartimento.
    """
    if min_wall_pixels <= 0 or core_radius <= 0:
        return True

    return _count_wall_pixels_in_core(pt, wall_mask, core_radius=core_radius) >= min_wall_pixels


def rasterize_wall_mask(
    shape: tuple[int, int],
    wall_segments: list[tuple[int, int, int, int]],
    iso_angles: tuple[float, float] = (36.0, 144.0),
    wall_thickness: int = 8,
    endpoint_tol: float = 25.0,
    morph_kernel_size: int = 5,
) -> np.ndarray:
    """
    Rasteriza los segmentos de muro en una máscara binaria (uint8: 0 o 255):
    1. Prolonga los extremos a lo largo de la dirección isométrica dominante más cercana
       por endpoint_tol píxeles para cerrar huecos causados por edificios sobre el muro.
    2. Dibuja los segmentos con grosor wall_thickness.
    3. Aplica cierre morfológico para sellar micro-fisuras.
    """
    h, w = shape[:2]
    wall_mask = np.zeros((h, w), dtype=np.uint8)

    if not wall_segments:
        return wall_mask

    iso1, iso2 = iso_angles

    for seg in wall_segments:
        x1, y1, x2, y2 = seg[:4]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)

        if length < 1e-3:
            continue

        if endpoint_tol > 0:
            deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
            diff1 = _angle_diff_180(deg, iso1)
            diff2 = _angle_diff_180(deg, iso2)
            chosen_iso = iso1 if diff1 <= diff2 else iso2

            iso_rad = math.radians(chosen_iso)
            ux, uy = math.cos(iso_rad), math.sin(iso_rad)

            if (dx * ux + dy * uy) < 0:
                ux, uy = -ux, -uy

            px1 = int(round(x1 - ux * endpoint_tol))
            py1 = int(round(y1 - uy * endpoint_tol))
            px2 = int(round(x2 + ux * endpoint_tol))
            py2 = int(round(y2 + uy * endpoint_tol))
        else:
            px1, py1 = int(round(x1)), int(round(y1))
            px2, py2 = int(round(x2)), int(round(y2))

        cv2.line(wall_mask, (px1, py1), (px2, py2), 255, thickness=int(wall_thickness))

    if morph_kernel_size > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(morph_kernel_size), int(morph_kernel_size)))
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k)

    return wall_mask


def build_barrier_mask(
    wall_mask: np.ndarray,
    detections: list | None = None,
    buildings_block: bool = True,
    building_margin: int = 0,
) -> np.ndarray:
    """
    Combina la máscara de muros con los bounding boxes de edificios YOLO.
    Si buildings_block=True, los edificios actúan como barreras infranqueables
    para evitar fugas de muros incompletos.
    """
    barrier_mask = wall_mask.copy()
    if not buildings_block or not detections:
        return barrier_mask

    h, w = barrier_mask.shape[:2]

    for det in detections:
        bbox = getattr(det, "bbox_xyxy", None)
        if bbox is None and isinstance(det, dict):
            bbox = det.get("bbox_xyxy") or det.get("bbox")
        if bbox is None and isinstance(det, (list, tuple)) and len(det) >= 4:
            bbox = det[:4]

        if bbox is not None:
            bx1, by1, bx2, by2 = [int(v) for v in bbox[:4]]
            if building_margin != 0:
                bx1 += building_margin
                by1 += building_margin
                bx2 -= building_margin
                by2 -= building_margin
            bx1 = max(0, min(w - 1, bx1))
            by1 = max(0, min(h - 1, by1))
            bx2 = max(0, min(w, bx2))
            by2 = max(0, min(h, by2))
            if bx2 > bx1 and by2 > by1:
                cv2.rectangle(barrier_mask, (bx1, by1), (bx2, by2), 255, -1)

    return barrier_mask


def compute_compartments(
    obstacle_mask: np.ndarray,
    drop_points: list[tuple[int, int]] | None = None,
) -> tuple[int, np.ndarray, set[int], np.ndarray]:
    """
    Calcula los componentes conexos del espacio libre (donde obstacle_mask == 0).
    Cada componente conexo > 0 representa un compartimento.
    Mapea además todos los píxeles de obstáculos a su compartimento libre más cercano
    utilizando cv2.distanceTransformWithLabels.

    Retorna:
        num_compartments (total de componentes incluyendo fondo 0)
        comp_labels (matriz 2D con etiquetas de compartimento, 0 = obstáculo)
        entry_comps (conjunto de IDs de compartimentos de entrada)
        nearest_labels (matriz 2D donde cada obstáculo tiene la etiqueta de su compartimento más cercano)
    """
    h, w = obstacle_mask.shape[:2]
    free_space = (obstacle_mask == 0).astype(np.uint8)

    num_labels, comp_labels = cv2.connectedComponents(free_space, connectivity=8)

    src_dist = (255 - free_space * 255).astype(np.uint8)
    dist_map, nearest_labels = cv2.distanceTransformWithLabels(
        src_dist, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP
    )

    entry_comps: set[int] = set()

    if drop_points:
        for pt in drop_points:
            px, py = int(round(pt[0])), int(round(pt[1]))
            if 0 <= px < w and 0 <= py < h:
                lbl = comp_labels[py, px]
                if lbl == 0:
                    lbl = nearest_labels[py, px]
                if lbl > 0:
                    entry_comps.add(int(lbl))

    if not entry_comps and num_labels > 1:
        border_pixels = np.concatenate([
            comp_labels[0, :],
            comp_labels[h - 1, :],
            comp_labels[:, 0],
            comp_labels[:, w - 1],
        ])
        border_nonzero = border_pixels[border_pixels > 0]
        if len(border_nonzero) > 0:
            vals, counts = np.unique(border_nonzero, return_counts=True)
            exterior_comp = int(vals[np.argmax(counts)])
            entry_comps.add(exterior_comp)
        else:
            entry_comps.add(1)

    return num_labels, comp_labels, entry_comps, nearest_labels


def build_compartment_graph(
    num_compartments: int,
    comp_labels: np.ndarray,
    nearest_labels: np.ndarray,
    wall_mask: np.ndarray,
    max_wall_width: float = 35.0,
) -> tuple[dict[int, set[int]], list[tuple[int, int]]]:
    """
    Construye el grafo de adyacencia de compartimentos:
    Nodos = compartimentos libres (1 .. num_compartments - 1).
    Arista entre (u, v) si comparten una banda de muro en wall_mask.
    Costo base de cada arista = 1 capa de muro.

    Retorna: (grafo_adyacencia, lista_de_aristas)
    """
    graph: dict[int, set[int]] = {c: set() for c in range(1, num_compartments)}
    edges_set: set[tuple[int, int]] = set()

    if num_compartments <= 2:
        return graph, []

    # Transición horizontal a través de píxeles de muro
    mask_h = (nearest_labels[:, 1:] != nearest_labels[:, :-1])
    on_wall_h = (wall_mask[:, 1:] > 0) | (wall_mask[:, :-1] > 0)
    valid_h = mask_h & on_wall_h
    u_h = nearest_labels[:, 1:][valid_h]
    v_h = nearest_labels[:, :-1][valid_h]

    # Transición vertical a través de píxeles de muro
    mask_v = (nearest_labels[1:, :] != nearest_labels[:-1, :])
    on_wall_v = (wall_mask[1:, :] > 0) | (wall_mask[:-1, :] > 0)
    valid_v = mask_v & on_wall_v
    u_v = nearest_labels[1:, :][valid_v]
    v_v = nearest_labels[:-1, :][valid_v]

    all_u = np.concatenate([u_h, u_v])
    all_v = np.concatenate([v_h, v_v])

    valid = (all_u > 0) & (all_v > 0) & (all_u != all_v)
    if np.any(valid):
        u_sel = all_u[valid]
        v_sel = all_v[valid]
        min_uv = np.minimum(u_sel, v_sel).astype(np.int64)
        max_uv = np.maximum(u_sel, v_sel).astype(np.int64)
        encoded = min_uv * 1000000 + max_uv
        unique_pairs = np.unique(encoded)

        for code in unique_pairs:
            u = int(code // 1000000)
            v = int(code % 1000000)
            if u in graph and v in graph:
                graph[u].add(v)
                graph[v].add(u)
                edges_set.add((u, v))

    return graph, sorted(edges_set)


def map_targets_to_compartments(
    detections: list,
    comp_labels: np.ndarray,
    nearest_labels: np.ndarray,
    loot_weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Asigna cada detección de botín relevante a su compartimento correspondiente.
    Si el centro cae en un obstáculo (edificio/muro), utiliza el compartimento libre más cercano.
    """
    weights = loot_weights if loot_weights is not None else DEFAULT_LOOT_WEIGHTS
    h, w = comp_labels.shape[:2]
    assigned_targets: list[dict] = []

    for det in detections:
        cls_name = getattr(det, "cls_name", None)
        if cls_name is None and isinstance(det, dict):
            cls_name = det.get("cls_name", "")

        weight = float(weights.get(cls_name, 0.0))
        if weight <= 0.0:
            continue

        center = getattr(det, "center", None)
        if center is None and isinstance(det, dict):
            center = det.get("center")
        if center is None:
            bbox = getattr(det, "bbox_xyxy", None)
            if bbox is None and isinstance(det, dict):
                bbox = det.get("bbox_xyxy") or det.get("bbox")
            if bbox is not None:
                center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

        if center is None:
            continue

        cx = max(0, min(w - 1, int(round(center[0]))))
        cy = max(0, min(h - 1, int(round(center[1]))))

        comp = comp_labels[cy, cx]
        if comp == 0:
            comp = nearest_labels[cy, cx]

        if comp > 0:
            assigned_targets.append({
                "det": det,
                "cls_name": cls_name,
                "weight": weight,
                "center": (cx, cy),
                "comp": int(comp),
            })

    return assigned_targets


def compute_total_cost(
    graph: dict[int, set[int]],
    entry_comps: set[int],
    targets: list[dict],
    jump_bridges: list[set[int]] | None = None,
) -> tuple[float, dict[int, float]]:
    """
    Calcula las capas mínimas de muro desde entry_comps hacia todos los compartimentos
    utilizando 0-1 BFS / Dijkstra:
    - Aristas estándar del grafo cuestan 1 capa.
    - Compartimentos dentro de la huella de un mismo salto quedan unidos con costo 0.

    Costo total = sum(peso * min_capas(entrada, compartimento_objetivo)).
    Retorna: (costo_total, diccionario_distancias)
    """
    if not targets or not entry_comps:
        return 0.0, {}

    all_nodes = set(graph.keys())
    for bridge in (jump_bridges or []):
        all_nodes.update(bridge)

    dist: dict[int, float] = {u: float("inf") for u in all_nodes}
    deque_bfs = deque()

    for s in entry_comps:
        if s in dist:
            dist[s] = 0.0
            deque_bfs.append((s, 0.0))

    bridges = [b for b in (jump_bridges or []) if len(b) > 1]
    zero_adj: dict[int, list[int]] = {}
    for bridge in bridges:
        b_list = list(bridge)
        for u in b_list:
            if u not in zero_adj:
                zero_adj[u] = []
            for v in b_list:
                if u != v:
                    zero_adj[u].append(v)

    while deque_bfs:
        u, d = deque_bfs.popleft()
        if d > dist[u]:
            continue

        # 1. Movimiento gratuito (costo 0) a través de huellas de salto compartidas
        if u in zero_adj:
            for v in zero_adj[u]:
                if d < dist[v]:
                    dist[v] = d
                    deque_bfs.appendleft((v, d))

        # 2. Movimiento normal a través de muros (costo 1)
        d_next = d + 1.0
        for v in graph.get(u, ()):
            if d_next < dist[v]:
                dist[v] = d_next
                deque_bfs.append((v, d_next))

    total_cost = 0.0
    for t in targets:
        comp_id = t["comp"]
        d = dist.get(comp_id, float("inf"))
        cost_layers = d if d < float("inf") else 10.0
        total_cost += t["weight"] * cost_layers

    return total_cost, dist


def get_compartments_in_footprint(
    center: tuple[int, int],
    footprint: dict | tuple | float | int,
    comp_labels: np.ndarray,
    nearest_labels: np.ndarray | None = None,
) -> set[int]:
    """
    Determina qué compartimentos libres son intersectados o cubiertos por la huella
    del hechizo de salto situada en center.
    Soporta:
    - Huella circular: radio r (int, float o {"type": "circle", "radius": r})
    - Huella elíptica isométrica: {"type": "ellipse", "rx": rx, "ry": ry, "angle": ang} o tuple (rx, ry)
    Calibración física por defecto: rx = 76.0, ry = 57.0, radius = 72.0 px.
    """
    h, w = comp_labels.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))

    # Interpretar parámetros de huella
    is_ellipse = False
    rx, ry = DEFAULT_JUMP_RX, DEFAULT_JUMP_RY
    angle_deg = 0.0

    if isinstance(footprint, dict):
        f_type = footprint.get("type", footprint.get("shape", "ellipse"))
        if f_type == "ellipse":
            is_ellipse = True
            rx = float(footprint.get("rx", DEFAULT_JUMP_RX))
            ry = float(footprint.get("ry", DEFAULT_JUMP_RY))
            angle_deg = float(footprint.get("angle", 0.0))
        else:
            rx = ry = float(footprint.get("radius", DEFAULT_FUSION_RADIUS))
    elif isinstance(footprint, (tuple, list)):
        if len(footprint) >= 2:
            is_ellipse = True
            rx, ry = float(footprint[0]), float(footprint[1])
            angle_deg = float(footprint[2]) if len(footprint) > 2 else 0.0
        else:
            rx = ry = float(footprint[0])
    else:
        rx = ry = float(footprint)

    max_r = max(rx, ry)
    x0 = max(0, int(cx - max_r - 1))
    x1 = min(w, int(cx + max_r + 2))
    y0 = max(0, int(cy - max_r - 1))
    y1 = min(h, int(cy + max_r + 2))

    if x1 <= x0 or y1 <= y0:
        return set()

    ys, xs = np.ogrid[y0 - cy : y1 - cy, x0 - cx : x1 - cx]

    if is_ellipse:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        x_rot = xs * cos_a + ys * sin_a
        y_rot = -xs * sin_a + ys * cos_a
        mask = ((x_rot / rx) ** 2 + (y_rot / ry) ** 2) <= 1.0
    else:
        mask = (xs ** 2 + ys ** 2) <= (rx ** 2)

    sub_comps = comp_labels[y0:y1, x0:x1]
    touched = set(np.unique(sub_comps[mask])) - {0}

    if len(touched) < 2 and nearest_labels is not None:
        sub_nearest = nearest_labels[y0:y1, x0:x1]
        nearest_touched = set(np.unique(sub_nearest[mask])) - {0}
        if len(nearest_touched) > len(touched):
            touched = nearest_touched

    return touched


def sample_jump_candidates(
    wall_mask: np.ndarray,
    wall_segments: list[tuple[int, int, int, int]] | None = None,
    jump_radius: float = DEFAULT_FUSION_RADIUS,
    step: int = 20,
    bbox_roi: tuple[int, int, int, int] | None = None,
    hud_mask: np.ndarray | None = None,
    hud_margin_px: int = 15,
) -> list[tuple[int, int]]:
    """
    Genera candidatos a punto de salto:
    1. Puntos uniformemente espaciados a lo largo de los segmentos de muro detectados.
    2. Puntos medios entre pares de segmentos de muro cercanos (cruces entre capas concéntricas).
    3. Muestreo en rejilla sobre las celdas con muro de wall_mask.
    Excluye candidatos que caigan dentro del HUD de batalla.
    """
    candidates_set: set[tuple[int, int]] = set()
    h, w = wall_mask.shape[:2]

    dilated_hud = None
    if hud_mask is not None:
        if hud_margin_px > 0:
            k = max(3, int(hud_margin_px * 2 + 1))
            dilated_hud = cv2.dilate(hud_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        else:
            dilated_hud = hud_mask

    pts_seg: list[tuple[int, int]] = []
    pts_mids: list[tuple[int, int]] = []

    if wall_segments:
        s_arr = np.array(wall_segments, dtype=np.float32)[:, :4]
        centers = np.column_stack([(s_arr[:, 0] + s_arr[:, 2]) / 2.0, (s_arr[:, 1] + s_arr[:, 3]) / 2.0])
        n_segs = len(centers)

        if n_segs > 1:
            diff = centers[:, None, :] - centers[None, :, :]
            dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
            triu = np.triu(np.ones((n_segs, n_segs), dtype=bool), k=1)
            i, j = np.where((dist >= 20.0) & (dist <= (2.0 * jump_radius - 10.0)) & triu)
            if len(i) > 0:
                pair_mids = (centers[i] + centers[j]) / 2.0
                step_snap = max(15.0, float(step))
                uniq = np.unique(np.round(pair_mids / step_snap).astype(np.int32), axis=0) * int(step_snap)
                for p in uniq:
                    px, py = int(p[0]), int(p[1])
                    if 0 <= px < w and 0 <= py < h:
                        pts_mids.append((px, py))

        for seg in wall_segments:
            x1, y1, x2, y2 = seg[:4]
            mx, my = int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
            if 0 <= mx < w and 0 <= my < h:
                pts_seg.append((mx, my))

    pts_grid: list[tuple[int, int]] = []
    step_grid = max(25, step)
    sub = wall_mask[::step_grid, ::step_grid]
    gy, gx = np.where(sub > 0)
    for py, px in zip(gy * step_grid, gx * step_grid):
        pts_grid.append((int(px), int(py)))

    ordered_candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pt in pts_mids + pts_seg + pts_grid:
        if bbox_roi:
            rx1, ry1, rx2, ry2 = bbox_roi
            if not (rx1 <= pt[0] <= rx2 and ry1 <= pt[1] <= ry2):
                continue
        if dilated_hud is not None:
            if 0 <= pt[1] < dilated_hud.shape[0] and 0 <= pt[0] < dilated_hud.shape[1]:
                if dilated_hud[pt[1], pt[0]] > 0:
                    continue
        if pt not in seen:
            seen.add(pt)
            ordered_candidates.append(pt)

    return ordered_candidates


def optimize_jumps(
    img_shape: tuple[int, int],
    wall_segments: list[tuple[int, int, int, int]] | None = None,
    detections: list | None = None,
    drop_points: list[tuple[int, int]] | None = None,
    iso_angles: tuple[float, float] = (36.0, 144.0),
    jump_footprint: dict | tuple | float | int | None = None,
    max_jumps: int = DEFAULT_MAX_JUMPS,
    min_marginal_gain: float = 0.5,
    loot_weights: dict[str, float] | None = None,
    buildings_block: bool | None = None,
    wall_thickness: int = 8,
    endpoint_tol: float = 25.0,
    morph_kernel_size: int = 5,
    candidate_step: int = 20,
    time_budget_sec: float = 0.300,
    grid_scale: int = 1,
    generate_debug_overlay: bool = True,
    base_img: np.ndarray | None = None,
    hud_mask: np.ndarray | None = None,
    hud_margin_px: int = 15,
    spell_config: dict | None = None,
    force_max_jumps: bool = False,
    wall_mask: np.ndarray | None = None,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    min_sep_px: float = DEFAULT_MIN_SEP_PX,
    min_wall_pixels: int | None = None,
    core_radius: float = DEFAULT_CORE_RADIUS,
    wall_detection_mode: str | None = None,
    deploy_zone_mask: np.ndarray | None = None,
    distance_penalty_factor: float | None = None,
    target_dist_weight: float | None = None,
) -> CompartmentAnalysisResult:
    """
    Función maestra de optimización de saltos por modelo de compartimentos.
    Implementa:
    1. WallSegDetector (U-Net con fallback transparente a raycast si el modelo no existe).
    2. Calibración física real: rx=76.0, ry=57.0, effective_radius=72.0 px.
    3. Filtro de densidad de muro (is_valid_wall_centered_jump con core_radius=42 px, min_wall_pixels=30).
    4. Beam Search con anticipación (beam_width=4, min_sep_px=95.0 px) para encadenar
       saltos intermedios de ganancia 0.0 hacia el núcleo.
    5. Penalización suave por distancia al despliegue (drop_points o borde de deploy_zone_mask)
       para preferir siempre muros exteriores ante candidatos con igual ganancia.
    Presupuesto de ejecución: < 300 ms por aldea.
    """
    t_start = time.perf_counter()
    h, w = img_shape[:2]

    # Cargar defaults de settings.json para jump_optimizer si no se suministran
    cfg_opt = SETTINGS.get("attack", {}).get("jump_optimizer", {})
    if jump_footprint is None:
        rx_val = float(cfg_opt.get("rx", DEFAULT_JUMP_RX))
        ry_val = float(cfg_opt.get("ry", DEFAULT_JUMP_RY))
        r_val = float(cfg_opt.get("effective_radius", DEFAULT_FUSION_RADIUS))
        jump_footprint = {"type": "ellipse", "rx": rx_val, "ry": ry_val, "radius": r_val, "angle": 0.0}

    beam_width = int(cfg_opt.get("beam_width", beam_width))
    min_sep_px = float(cfg_opt.get("min_sep_px", min_sep_px))
    core_radius = float(cfg_opt.get("core_radius", core_radius))
    if buildings_block is None:
        if "buildings_block" in cfg_opt:
            buildings_block = bool(cfg_opt["buildings_block"])
        else:
            buildings_block = True
    if distance_penalty_factor is None:
        distance_penalty_factor = float(cfg_opt.get("distance_penalty_factor", 0.0005))
    if target_dist_weight is None:
        target_dist_weight = float(cfg_opt.get("target_dist_weight", 1.1))

    # Precomputar mapa de distancia euclidiana a la zona de despliegue si está disponible
    if not drop_points and deploy_zone_mask is None and base_img is not None:
        try:
            from vision.deploy_zone_detector import get_deploy_zone_detector
            dzd = get_deploy_zone_detector()
            if dzd is not None and dzd.is_ready():
                deploy_zone_mask = dzd.predict_mask(base_img)
        except Exception as e:
            log.debug("Auto deploy_zone_mask retrieval skipped en optimize_jumps: %s", e)

    deploy_dist_map = None
    if deploy_zone_mask is not None and isinstance(deploy_zone_mask, np.ndarray):
        if deploy_zone_mask.shape[:2] != (h, w):
            dz_native = cv2.resize(deploy_zone_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            dz_native = deploy_zone_mask
        # En deploy_zone_mask: 0 = exterior legal (césped), 1 = interior aldea
        deploy_dist_map = cv2.distanceTransform((dz_native > 0).astype(np.uint8), cv2.DIST_L2, 3)

    def _get_cand_drop_dist(cx_orig: int, cy_orig: int) -> float:
        if drop_points:
            return float(min(math.hypot(cx_orig - dp[0], cy_orig - dp[1]) for dp in drop_points))
        if deploy_dist_map is not None:
            c_y = min(h - 1, max(0, cy_orig))
            c_x = min(w - 1, max(0, cx_orig))
            return float(deploy_dist_map[c_y, c_x])
        return 0.0

    # Escalamiento dinámico de min_wall_pixels si no se pasó explícitamente
    if min_wall_pixels is not None:
        min_wall_pixels = int(min_wall_pixels)
    else:
        base_min_wall = int(cfg_opt.get("min_wall_pixels", DEFAULT_MIN_WALL_PIXELS))
        if w < 2000:
            min_wall_pixels = max(10, int(round(base_min_wall * (w / 2400.0))))
        else:
            min_wall_pixels = base_min_wall

    # Determinar modo de detección de muros ("unet" vs "raycast")
    if wall_detection_mode is None:
        if spell_config and "wall_detection_mode" in spell_config:
            wall_detection_mode = spell_config["wall_detection_mode"]
        else:
            wall_detection_mode = cfg_opt.get(
                "wall_detection_mode",
                SETTINGS.get("attack", {}).get("wall_detection_mode", "unet"),
            )
    wall_detection_mode = str(wall_detection_mode).lower().strip()

    # Sobrescritura desde spell_config si existe
    if spell_config:
        if "max_spells" in spell_config or "max_jumps" in spell_config:
            max_jumps = int(spell_config.get("max_spells") or spell_config.get("max_jumps", max_jumps))
        if "shape" in spell_config:
            sh_type = spell_config.get("shape", "ellipse")
            if sh_type == "ellipse":
                rx_s = float(spell_config.get("rx", DEFAULT_JUMP_RX))
                ry_s = float(spell_config.get("ry", DEFAULT_JUMP_RY))
                jump_footprint = {"shape": "ellipse", "rx": rx_s, "ry": ry_s}
            elif sh_type == "circle":
                r_s = float(spell_config.get("radius", DEFAULT_FUSION_RADIUS))
                jump_footprint = r_s
        if "min_marginal_gain" in spell_config:
            min_marginal_gain = float(spell_config["min_marginal_gain"])
        if "beam_width" in spell_config:
            beam_width = int(spell_config["beam_width"])
        if "min_sep_px" in spell_config:
            min_sep_px = float(spell_config["min_sep_px"])
        if "min_wall_pixels" in spell_config:
            min_wall_pixels = int(spell_config["min_wall_pixels"])
        if "core_radius" in spell_config:
            core_radius = float(spell_config["core_radius"])

    # Escalamiento adaptativo para cumplir con el presupuesto estricto (<300 ms) en imágenes HD (1080p)
    scale = int(grid_scale)
    if scale <= 1 and (w >= 1200 or h >= 700):
        scale = 2

    # 1. Obtener o construir la máscara de muros sin escalar (resolución nativa)
    unscaled_wall_mask: np.ndarray | None = None
    if wall_mask is not None:
        unscaled_wall_mask = wall_mask.copy()
    elif wall_detection_mode == "unet" and base_img is not None:
        try:
            seg_det = get_wall_seg_detector()
            unscaled_wall_mask = seg_det.predict(base_img)
            hud_native = get_battle_hud_mask(h, w)
            unscaled_wall_mask[hud_native > 0] = 0
            log.info("🧱 Muros detectados vía WallSegDetector (U-Net): %d px", int(np.count_nonzero(unscaled_wall_mask)))
        except (FileNotFoundError, Exception) as e:
            log.warning("⚠️ Fallback automático: WallSegDetector U-Net no disponible (%s). Usando raycast.", e)
            wall_detection_mode = "raycast"

    # Si estamos en modo raycast (o tras fallback de unet)
    if unscaled_wall_mask is None:
        if wall_segments is None and base_img is not None:
            try:
                from vision.wall_raycast import detect_global_wall_segments
                iso_angles, wall_segments = detect_global_wall_segments(base_img, detections=detections)
            except Exception as e:
                log.warning("Fallo al detectar segmentos de muro por raycast: %s", e)
                wall_segments = []

        if wall_segments is not None:
            unscaled_wall_mask = rasterize_wall_mask(
                shape=(h, w),
                wall_segments=wall_segments,
                iso_angles=iso_angles,
                wall_thickness=wall_thickness,
                endpoint_tol=endpoint_tol,
                morph_kernel_size=morph_kernel_size,
            )
        else:
            unscaled_wall_mask = np.zeros((h, w), dtype=np.uint8)

    # Preparar representaciones a escala de trabajo
    if scale > 1:
        sh, sw = int(h // scale), int(w // scale)
        scaled_shape = (sh, sw)
        scaled_walls = [
            (int(round(s[0] / scale)), int(round(s[1] / scale)), int(round(s[2] / scale)), int(round(s[3] / scale)))
            for s in (wall_segments or [])
        ]
        scaled_dets = []
        for det in (detections or []):
            bbox = getattr(det, "bbox_xyxy", None)
            if bbox is None and isinstance(det, dict):
                bbox = det.get("bbox_xyxy") or det.get("bbox")
            center = getattr(det, "center", None)
            if center is None and isinstance(det, dict):
                center = det.get("center")
            cls_name = getattr(det, "cls_name", "") if hasattr(det, "cls_name") else det.get("cls_name", "")

            s_bbox = tuple(int(round(v / scale)) for v in bbox) if bbox is not None else None
            s_center = (int(round(center[0] / scale)), int(round(center[1] / scale))) if center is not None else None
            scaled_dets.append({"cls_name": cls_name, "bbox_xyxy": s_bbox, "center": s_center})

        scaled_drop = [(int(round(p[0] / scale)), int(round(p[1] / scale))) for p in (drop_points or [])]

        if isinstance(jump_footprint, dict):
            scaled_fp = dict(jump_footprint)
            if "radius" in scaled_fp:
                scaled_fp["radius"] = float(scaled_fp["radius"]) / scale
            if "rx" in scaled_fp:
                scaled_fp["rx"] = float(scaled_fp["rx"]) / scale
            if "ry" in scaled_fp:
                scaled_fp["ry"] = float(scaled_fp["ry"]) / scale
        elif isinstance(jump_footprint, (list, tuple)):
            scaled_fp = tuple(float(v) / scale if idx < 2 else v for idx, v in enumerate(jump_footprint))
        else:
            scaled_fp = float(jump_footprint) / scale

        scaled_step = max(15, int(round(candidate_step / scale)))
        scaled_hud = cv2.resize(hud_mask, (sw, sh), interpolation=cv2.INTER_NEAREST) if hud_mask is not None else None
        scaled_wall_mask = cv2.resize(unscaled_wall_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    else:
        scale = 1
        scaled_shape = (h, w)
        scaled_walls = wall_segments or []
        scaled_dets = detections or []
        scaled_drop = drop_points
        scaled_fp = jump_footprint
        scaled_step = candidate_step
        scaled_hud = hud_mask
        scaled_wall_mask = unscaled_wall_mask

    if scaled_hud is None:
        scaled_hud = get_battle_hud_mask(scaled_shape[0], scaled_shape[1])

    # 2. Barreras de edificios sobre la máscara de muros
    barrier_mask = build_barrier_mask(
        wall_mask=scaled_wall_mask,
        detections=scaled_dets,
        buildings_block=buildings_block,
    )

    # 3. Componentes conexos (compartimentos)
    num_comps, comp_labels, entry_comps, nearest_labels = compute_compartments(
        obstacle_mask=barrier_mask,
        drop_points=scaled_drop,
    )

    # 4. Grafo de adyacencia de compartimentos
    graph, edges = build_compartment_graph(
        num_compartments=num_comps,
        comp_labels=comp_labels,
        nearest_labels=nearest_labels,
        wall_mask=scaled_wall_mask,
    )

    # 5. Mapear objetivos a compartimentos
    targets = map_targets_to_compartments(
        detections=scaled_dets,
        comp_labels=comp_labels,
        nearest_labels=nearest_labels,
        loot_weights=loot_weights,
    )

    # Si no hay muros detectados, o 1 solo compartimento, o sin objetivos: terminar con 0 saltos
    wall_pixel_count = int(np.count_nonzero(unscaled_wall_mask))
    if num_comps <= 1 or not targets or (not wall_segments and wall_pixel_count == 0):
        initial_cost, _ = compute_total_cost(graph, entry_comps, targets, jump_bridges=[])
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        debug_overlay = None
        if generate_debug_overlay:
            debug_overlay = render_debug_overlay(
                base_img=base_img if base_img is not None else np.zeros((h, w, 3), dtype=np.uint8),
                comp_labels=comp_labels,
                wall_mask=scaled_wall_mask,
                selected_jumps=[],
                footprint=jump_footprint,
                targets=targets,
                edges=edges,
                grid_scale=scale,
            )
        return CompartmentAnalysisResult(
            jumps=[],
            jump_details=[],
            initial_cost=initial_cost,
            final_cost=initial_cost,
            total_gain=0.0,
            num_compartments=max(1, num_comps - 1),
            comp_labels=comp_labels,
            wall_mask=scaled_wall_mask,
            edges=edges,
            target_assignments=targets,
            execution_time_ms=t_elapsed,
            debug_overlay=debug_overlay,
            graph=graph,
            entry_comps=entry_comps,
            nearest_labels=nearest_labels,
            grid_scale=scale,
        )

    # 6. Costo inicial
    initial_cost, initial_dist = compute_total_cost(graph, entry_comps, targets, jump_bridges=[])

    if (initial_cost <= 0.0 and not force_max_jumps) or max_jumps <= 0:
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        debug_overlay = None
        if generate_debug_overlay:
            debug_overlay = render_debug_overlay(
                base_img=base_img if base_img is not None else np.zeros((h, w, 3), dtype=np.uint8),
                comp_labels=comp_labels,
                wall_mask=scaled_wall_mask,
                selected_jumps=[],
                footprint=jump_footprint,
                targets=targets,
                edges=edges,
                grid_scale=scale,
            )
        return CompartmentAnalysisResult(
            jumps=[],
            jump_details=[],
            initial_cost=initial_cost,
            final_cost=initial_cost,
            total_gain=0.0,
            num_compartments=max(1, num_comps - 1),
            comp_labels=comp_labels,
            wall_mask=scaled_wall_mask,
            edges=edges,
            target_assignments=targets,
            execution_time_ms=t_elapsed,
            debug_overlay=debug_overlay,
            graph=graph,
            entry_comps=entry_comps,
            nearest_labels=nearest_labels,
            grid_scale=scale,
        )

    # 7. Muestrear candidatos a salto
    r_approx = DEFAULT_FUSION_RADIUS / scale
    if isinstance(scaled_fp, (int, float)):
        r_approx = float(scaled_fp)
    elif isinstance(scaled_fp, dict):
        r_approx = float(scaled_fp.get("rx", scaled_fp.get("radius", DEFAULT_FUSION_RADIUS / scale)))
    elif isinstance(scaled_fp, (list, tuple)) and len(scaled_fp) > 0:
        r_approx = float(scaled_fp[0])

    candidates = sample_jump_candidates(
        wall_mask=scaled_wall_mask,
        wall_segments=scaled_walls,
        jump_radius=r_approx,
        step=scaled_step,
        hud_mask=scaled_hud,
        hud_margin_px=max(3, int(round(hud_margin_px / scale))),
    )

    max_candidates = 200
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    t_search_start = time.perf_counter()
    loop_deadline = time_budget_sec - 0.030

    # Filtrar candidatos válidos:
    # A) Filtro de densidad de muro (is_valid_wall_centered_jump con core_radius=42, min_wall_pixels=30)
    # B) Cubre al menos 2 compartimentos libres en comp_labels/nearest_labels
    raw_valid_candidates: list[tuple[tuple[int, int], set[int], int, int, int, float]] = []
    for cand in candidates:
        if (time.perf_counter() - t_search_start) >= loop_deadline:
            break

        cand_orig_x = int(round(cand[0] * scale))
        cand_orig_y = int(round(cand[1] * scale))
        cand_orig = (cand_orig_x, cand_orig_y)

        wall_px = 0
        if unscaled_wall_mask is not None:
            wall_px = _count_wall_pixels_in_core(
                cand_orig,
                unscaled_wall_mask,
                core_radius=core_radius,
            )
            if min_wall_pixels > 0 and wall_px < min_wall_pixels:
                continue

        touched_comps = get_compartments_in_footprint(
            center=cand,
            footprint=scaled_fp,
            comp_labels=comp_labels,
            nearest_labels=nearest_labels,
        )
        if len(touched_comps) >= 2:
            cand_drop_dist = _get_cand_drop_dist(cand_orig_x, cand_orig_y)
            cand_targets = [t for t in targets if t["comp"] in (touched_comps - entry_comps)]
            if not cand_targets:
                cand_targets = [t for t in targets if t["comp"] not in entry_comps]
            if not cand_targets:
                cand_targets = targets
            if cand_targets:
                cand_target_dist = min(
                    math.hypot(cand_orig_x - t["center"][0] * scale, cand_orig_y - t["center"][1] * scale)
                    for t in cand_targets
                )
            else:
                cand_target_dist = 0.0
            cand_travel_dist = cand_drop_dist + target_dist_weight * cand_target_dist
            raw_valid_candidates.append(
                (cand, touched_comps, wall_px, cand_orig_x, cand_orig_y, cand_travel_dist, cand_drop_dist)
            )

    # Poda por equivalencia topológica:
    # Agrupar por comp_key = frozenset(touched_comps)
    candidates_by_comp: dict[frozenset[int], list] = defaultdict(list)
    for cand, touched_comps, wall_px, cx_orig, cy_orig, cand_travel_dist, cand_drop_dist in raw_valid_candidates:
        comp_key = frozenset(touched_comps)
        candidates_by_comp[comp_key].append(
            (cand, touched_comps, wall_px, cx_orig, cy_orig, cand_travel_dist, cand_drop_dist)
        )

    valid_candidates: list[tuple[tuple[int, int], set[int], float, float]] = []
    for comp_key, c_list in candidates_by_comp.items():
        # Prioridad: 1) densidad de muro suficiente, 2) menor recorrido total, 3) mayor densidad de muro
        c_list.sort(key=lambda item: (item[2] >= min_wall_pixels, -item[5], item[2]), reverse=True)
        best = c_list[0]
        valid_candidates.append((best[0], best[1], best[5], best[6]))

        # Conservar hasta 2 representantes solo si están separados más de min_sep_px del primero
        if len(c_list) > 1:
            best_x_orig, best_y_orig = best[3], best[4]
            for second in c_list[1:]:
                sec_x_orig, sec_y_orig = second[3], second[4]
                if math.hypot(sec_x_orig - best_x_orig, sec_y_orig - best_y_orig) >= min_sep_px:
                    valid_candidates.append((second[0], second[1], second[5], second[6]))
                    break

    log.debug(
        "✂️ [BEAM] Candidatos podados por equivalencia topológica: %d -> %d",
        len(raw_valid_candidates),
        len(valid_candidates),
    )

    # 8. Búsqueda con Anticipación (Beam Search con beam_width = 4)
    # Reemplaza la parada temprana voraz (min_marginal_gain) para habilitar encadenamiento
    # de 2 a 4 saltos hacia el núcleo aun cuando saltos intermedios tengan ganancia marginal 0.0.
    @dataclass
    class BeamState:
        jumps: list[JumpCandidate]
        active_bridges: list[set[int]]
        cost: float
        total_gain: float
        dists: dict[int, float]
        sum_target_dists: float
        total_travel_dist: float = 0.0

    initial_target_dists = sum(initial_dist.get(t["comp"], 10.0) * t["weight"] for t in targets)
    root_state = BeamState(
        jumps=[],
        active_bridges=[],
        cost=initial_cost,
        total_gain=0.0,
        dists=initial_dist,
        sum_target_dists=initial_target_dists,
        total_travel_dist=0.0,
    )

    current_beam: list[BeamState] = [root_state]
    all_evaluated_states: list[BeamState] = [root_state]
    cost_cache: dict[tuple, tuple[float, dict[int, float]]] = {}
    total_states_evaluated = 0

    for depth in range(max_jumps):
        if (time.perf_counter() - t_search_start) >= loop_deadline:
            log.warning("⏳ Presupuesto de tiempo excedido en Beam Search nivel %d (%.1f ms)", depth, (time.perf_counter() - t_search_start) * 1000.0)
            break

        next_expansions: list[BeamState] = []

        for state in current_beam:
            if (time.perf_counter() - t_search_start) >= loop_deadline:
                break

            # Si ya se alcanzó costo 0 y no se fuerzan saltos máximos, no es necesario expandir esta rama
            if not force_max_jumps and state.cost <= 0.0:
                continue

            for cand, touched_comps, cand_travel_dist, cand_drop_dist in valid_candidates:
                if (time.perf_counter() - t_search_start) >= loop_deadline:
                    break

                cand_orig_x = int(round(cand[0] * scale))
                cand_orig_y = int(round(cand[1] * scale))

                # Separación mínima requerida entre saltos (min_sep_px = 95.0 px)
                if any(math.hypot(cand_orig_x - j.x, cand_orig_y - j.y) < min_sep_px for j in state.jumps):
                    continue

                # Evitar agregar puente idéntico o subconjunto de uno ya activo
                if any(touched_comps.issubset(b) for b in state.active_bridges):
                    continue

                test_bridges = state.active_bridges + [touched_comps]
                bridge_key = tuple(sorted(tuple(sorted(b)) for b in test_bridges))

                if bridge_key in cost_cache:
                    cand_cost, cand_dists = cost_cache[bridge_key]
                else:
                    cand_cost, cand_dists = compute_total_cost(
                        graph=graph,
                        entry_comps=entry_comps,
                        targets=targets,
                        jump_bridges=test_bridges,
                    )
                    cost_cache[bridge_key] = (cand_cost, cand_dists)

                total_states_evaluated += 1

                marginal_gain = state.cost - cand_cost
                if marginal_gain < -1e-6:
                    continue

                total_gain = initial_cost - cand_cost
                cand_sum_td = sum(cand_dists.get(t["comp"], 10.0) * t["weight"] for t in targets)

                # Objetivos directamente desbloqueados o acercados por este salto
                unlocked = [
                    t for t in targets
                    if cand_dists.get(t["comp"], 10.0) < state.dists.get(t["comp"], 10.0)
                ]
                if not unlocked:
                    unlocked = [t for t in targets if cand_dists.get(t["comp"], 10.0) > 0.0]
                if not unlocked:
                    unlocked = [t for t in targets if t["comp"] in (touched_comps - entry_comps)]
                if not unlocked:
                    unlocked = targets

                d_drop = _get_cand_drop_dist(cand_orig_x, cand_orig_y)
                if unlocked:
                    d_target = min(
                        math.hypot(cand_orig_x - t["center"][0] * scale, cand_orig_y - t["center"][1] * scale)
                        for t in unlocked
                    )
                else:
                    d_target = 0.0
                cand_travel = d_drop + target_dist_weight * d_target

                cand_jump = JumpCandidate(
                    x=cand_orig_x,
                    y=cand_orig_y,
                    gain=marginal_gain,
                    cost_before=state.cost,
                    cost_after=cand_cost,
                    bridged_compartments=sorted(touched_comps),
                    drop_dist=d_drop,
                    travel_dist=cand_travel,
                )

                new_state = BeamState(
                    jumps=state.jumps + [cand_jump],
                    active_bridges=test_bridges,
                    cost=cand_cost,
                    total_gain=total_gain,
                    dists=cand_dists,
                    sum_target_dists=cand_sum_td,
                    total_travel_dist=state.total_travel_dist + cand_travel,
                )
                next_expansions.append(new_state)

        if not next_expansions:
            break

        # Ordenar expansiones por:
        # 1) Ganancia efectiva con penalización suave por recorrido total (despliegue -> salto -> objetivo)
        # 2) Mayor ganancia pura
        # 3) Menor recorrido acumulado (total_travel_dist)
        # 4) Menor distancia agregada a objetivos (sum_target_dists)
        # 5) Menor costo restante
        next_expansions.sort(
            key=lambda s: (
                s.total_gain - distance_penalty_factor * s.total_travel_dist,
                s.total_gain,
                -s.total_travel_dist,
                -s.sum_target_dists,
                -s.cost,
            ),
            reverse=True,
        )

        # Seleccionar top beam_width con diversidad de configuración
        new_beam: list[BeamState] = []
        seen_signatures: set[tuple] = set()

        for s in next_expansions:
            sig = (
                round(s.cost, 2),
                tuple(sorted(tuple(sorted(b)) for b in s.active_bridges)),
            )
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                new_beam.append(s)
                all_evaluated_states.append(s)
                if len(new_beam) >= beam_width:
                    break

        if not new_beam:
            new_beam = next_expansions[:beam_width]
            all_evaluated_states.extend(new_beam)

        current_beam = new_beam

    log.debug(
        "🧠 [BEAM] Búsqueda completada: %d estados evaluados en %.1f ms",
        total_states_evaluated,
        (time.perf_counter() - t_start) * 1000.0,
    )

    # Selección del estado óptimo global entre todos los estados evaluados
    if force_max_jumps:
        best_state = max(
            all_evaluated_states,
            key=lambda s: (
                s.total_gain - distance_penalty_factor * s.total_travel_dist,
                s.total_gain,
                -s.total_travel_dist,
                len(s.jumps),
                -s.cost,
            ),
        )
    else:
        states_with_gain = [s for s in all_evaluated_states if s.total_gain >= min_marginal_gain]
        if states_with_gain:
            max_gain = max(s.total_gain for s in states_with_gain)
            candidates_best = [s for s in states_with_gain if abs(s.total_gain - max_gain) < 1e-4]
            # De los que empatan en ganancia máxima, seleccionar el que use menos saltos,
            # y a igual número de saltos, el de menor recorrido total (despliegue -> salto -> objetivo)
            best_state = min(candidates_best, key=lambda s: (len(s.jumps), s.total_travel_dist))
        else:
            best_state = root_state

    selected_jumps = best_state.jumps
    final_cost = best_state.cost
    total_gain = best_state.total_gain
    t_elapsed = (time.perf_counter() - t_start) * 1000.0

    jumps_list = [(j.x, j.y, j.gain) for j in selected_jumps]

    debug_overlay = None
    if generate_debug_overlay:
        debug_overlay = render_debug_overlay(
            base_img=base_img if base_img is not None else np.zeros((h, w, 3), dtype=np.uint8),
            comp_labels=comp_labels,
            wall_mask=scaled_wall_mask,
            selected_jumps=selected_jumps,
            footprint=jump_footprint,
            targets=targets,
            edges=edges,
            grid_scale=scale,
        )

    return CompartmentAnalysisResult(
        jumps=jumps_list,
        jump_details=selected_jumps,
        initial_cost=initial_cost,
        final_cost=final_cost,
        total_gain=total_gain,
        num_compartments=max(1, num_comps - 1),
        comp_labels=comp_labels,
        wall_mask=scaled_wall_mask,
        edges=edges,
        target_assignments=targets,
        execution_time_ms=t_elapsed,
        debug_overlay=debug_overlay,
        graph=graph,
        entry_comps=entry_comps,
        nearest_labels=nearest_labels,
        grid_scale=scale,
    )


def render_debug_overlay(
    base_img: np.ndarray,
    comp_labels: np.ndarray,
    wall_mask: np.ndarray,
    selected_jumps: list[JumpCandidate],
    footprint: dict | tuple | float | int,
    targets: list[dict] | None = None,
    edges: list[tuple[int, int]] | None = None,
    grid_scale: int = 1,
) -> np.ndarray:
    """
    Genera una imagen de auditoría visual con:
    - Compartimentos coloreados en tonos pastel semitransparentes.
    - Segmentos y contornos de muros destacados.
    - Objetivos de botín asignados y marcados.
    - Huellas de saltos seleccionados numeradas con su ganancia.
    """
    h, w = comp_labels.shape[:2]
    canvas = base_img.copy() if base_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
    if canvas.shape[:2] != (h, w):
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)

    num_comps = int(np.max(comp_labels))
    if num_comps > 0:
        palette = np.zeros((num_comps + 1, 3), dtype=np.uint8)
        for c in range(1, num_comps + 1):
            hue = int((c * 179 / max(num_comps, 1)) % 180)
            color_hsv = np.uint8([[[hue, 140, 220]]])
            palette[c] = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]

        color_overlay = palette[comp_labels]
        cv2.addWeighted(color_overlay, 0.30, canvas, 0.70, 0, canvas)

    if wall_mask is not None:
        if wall_mask.shape[:2] == (h, w):
            canvas[wall_mask > 0] = [240, 240, 240]
        else:
            resized_wall = cv2.resize(wall_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            canvas[resized_wall > 0] = [240, 240, 240]

    for t in (targets or []):
        cx, cy = t["center"]
        cls_name = t["cls_name"]
        color = (0, 215, 255) if "gold" in cls_name else (255, 0, 255) if "dark" in cls_name else (255, 128, 0)
        cv2.circle(canvas, (cx, cy), 8, color, -1)
        cv2.circle(canvas, (cx, cy), 10, (255, 255, 255), 1)
        cv2.putText(canvas, f"C{t['comp']}", (cx - 10, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    is_ellipse = False
    rx, ry = DEFAULT_JUMP_RX, DEFAULT_JUMP_RY
    angle_deg = 0.0

    if isinstance(footprint, dict):
        if footprint.get("type") == "ellipse" or footprint.get("shape") == "ellipse":
            is_ellipse = True
            rx = float(footprint.get("rx", DEFAULT_JUMP_RX))
            ry = float(footprint.get("ry", DEFAULT_JUMP_RY))
            angle_deg = float(footprint.get("angle", 0.0))
        else:
            rx = ry = float(footprint.get("radius", DEFAULT_FUSION_RADIUS))
    elif isinstance(footprint, (tuple, list)):
        if len(footprint) >= 2:
            is_ellipse = True
            rx, ry = float(footprint[0]), float(footprint[1])
            angle_deg = float(footprint[2]) if len(footprint) > 2 else 0.0
        else:
            rx = ry = float(footprint[0])
    else:
        rx = ry = float(footprint)

    scale = max(1, int(grid_scale))
    scaled_rx = rx / scale
    scaled_ry = ry / scale

    for idx, jump in enumerate(selected_jumps, start=1):
        jx = int(round(jump.x / scale))
        jy = int(round(jump.y / scale))
        jump_color = (0, 255, 0) if idx == 1 else (0, 255, 255)

        if is_ellipse:
            cv2.ellipse(
                canvas,
                (jx, jy),
                (int(round(scaled_rx)), int(round(scaled_ry))),
                angle_deg,
                0,
                360,
                jump_color,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(canvas, (jx, jy), int(round(scaled_rx)), jump_color, 2, cv2.LINE_AA)

        cv2.circle(canvas, (jx, jy), 5, (0, 0, 255), -1)
        label = f"Salto #{idx} (+{jump.gain:.1f})"
        cv2.putText(canvas, label, (jx + 8, jy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (jx + 8, jy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas
