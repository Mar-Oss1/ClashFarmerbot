"""
vision/hud.py
Módulo centralizado para máscaras de HUD de batalla y elementos interactivos
de pantalla en Clash of Clans.
"""
import functools
import cv2
import numpy as np


@functools.lru_cache(maxsize=16)
def _build_battle_hud_mask_cached(h: int, w: int, margin_px: int = 0) -> np.ndarray:
    """
    Genera la máscara de exclusión de elementos interactivos del HUD de batalla
    (barra de despliegue inferior, botón finalizar batalla, botón siguiente, medidores superiores).
    Resolución de referencia: 2400x1080.
    
    Usa lru_cache para evitar recalcular la misma matriz en llamadas sucesivas.
    Retorna un array inmutable (read-only) internamente.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    scale_x = w / 2400.0
    scale_y = h / 1080.0

    # 1. Margen superior extremo
    mask[: int(60 * scale_y), :] = 255

    # 2. Barra inferior de despliegue de tropas (y >= 888)
    y1_bar = int(888 * scale_y)
    mask[y1_bar:, :] = 255

    # 3. Botón 'Terminar Batalla' (esquina inferior izquierda)
    y1_eb, y2_eb = int(730 * scale_y), int(885 * scale_y)
    x1_eb, x2_eb = 0, int(360 * scale_x)
    mask[y1_eb:y2_eb, x1_eb:x2_eb] = 255

    # 4. Botón 'Siguiente' / menú lateral derecho
    y1_next, y2_next = int(850 * scale_y), int(980 * scale_y)
    x1_next = int(2040 * scale_x)
    mask[y1_next:y2_next, x1_next:] = 255

    # 5. Medidores superiores (Loot superior izquierdo, recursos superior derecho)
    mask[: int(100 * scale_y), : int(380 * scale_x)] = 255
    mask[: int(160 * scale_y), int(1900 * scale_x) :] = 255

    if margin_px > 0:
        k_size = max(3, int(margin_px * 2 + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        mask = cv2.dilate(mask, kernel)

    mask.flags.writeable = False
    return mask


def get_battle_hud_mask(h: int, w: int, margin_px: int = 0) -> np.ndarray:
    """
    Retorna una copia mutable de la máscara de HUD de batalla para compatibilidad total
    con callers que puedan modificar la matriz in-place.
    """
    return _build_battle_hud_mask_cached(h, w, margin_px).copy()
