"""
core/ui_cache.py
Motor de Caché Inteligente de Coordenadas de Interfaz (UI) para ClashFarmerbot.
Permite clics de latencia cero y micro-verificaciones en ROIs reducidas (<1 ms),
eliminando búsquedas template matching de pantalla completa redundantes.
"""
from __future__ import annotations
import random
import cv2
import numpy as np
from core.logger import get_logger
from core.config import SETTINGS

log = get_logger("ui_cache")


class UICache:
    """
    Administrador centralizado de coordenadas UI pre-calibradas con auto-aprendizaje dinámico.
    """

    # Coordenadas de referencia normalizadas (x / 2400, y / 1080)
    DEFAULT_NORMALIZED_COORDS: dict[str, tuple[float, float]] = {
        "next_btn": (2185.0 / 2400.0, 892.0 / 1080.0),            # Botón 'Siguiente' en batalla
        "attack_btn": (195.0 / 2400.0, 945.0 / 1080.0),           # Botón 'Atacar' en aldea
        "find_match_btn": (1820.0 / 2400.0, 780.0 / 1080.0),      # Botón 'Buscar' multijugador
        "confirm_attack_btn": (1200.0 / 2400.0, 778.0 / 1080.0),  # Confirmar ataque / gastar oro
        "surrender_btn": (214.0 / 2400.0, 808.0 / 1080.0),        # Rendirse / Terminar batalla
        "confirm_surrender_btn": (1400.0 / 2400.0, 670.0 / 1080.0), # 'Vale' de confirmación
        "return_home_btn": (1200.0 / 2400.0, 955.0 / 1080.0),     # Volver a casa
        "claim_btn": (1200.0 / 2400.0, 847.0 / 1080.0),           # Reclamar Star Bonus
        "claim_reward_btn": (1202.0 / 2400.0, 927.0 / 1080.0),    # Botón Claim Reward de fin de batalla con cofre
        "chest_center": (1200.0 / 2400.0, 500.0 / 1080.0),        # Centro de toque para cofres
        "continue_btn": (1180.0 / 2400.0, 952.0 / 1080.0),        # Botón Continue de recompensas
    }

    def __init__(self):
        self._norm_coords: dict[str, tuple[float, float]] = dict(self.DEFAULT_NORMALIZED_COORDS)
        self._learned_hits: dict[str, int] = {k: 0 for k in self._norm_coords}
        self.enabled: bool = SETTINGS.get("ui_cache", {}).get("enabled", True)
        self.use_micro_verify: bool = SETTINGS.get("ui_cache", {}).get("use_micro_verify", True)
        self.learn_on_detect: bool = SETTINGS.get("ui_cache", {}).get("learn_on_detect", True)

    def get_pos(self, name: str, width: int = 2400, height: int = 1080, jitter: int = 0) -> tuple[int, int]:
        """
        Retorna la coordenada absoluta en pantalla escalada a (width, height),
        con variación humanizada opcional (jitter).
        """
        if name not in self._norm_coords:
            raise KeyError(f"Elemento UI '{name}' no existe en UICache.")

        norm_x, norm_y = self._norm_coords[name]
        cx = int(round(norm_x * width))
        cy = int(round(norm_y * height))

        if jitter > 0:
            cx += random.randint(-jitter, jitter)
            cy += random.randint(-jitter, jitter)

        # Garantizar que se mantenga dentro de los límites legales de pantalla
        cx = max(10, min(width - 10, cx))
        cy = max(10, min(height - 10, cy))
        return (cx, cy)

    def update_pos(self, name: str, x: int, y: int, width: int = 2400, height: int = 1080):
        """
        Auto-aprendizaje dinámico: actualiza la posición normalizada
        cuando un template es localizado con alta confianza.
        """
        if not self.learn_on_detect or width <= 0 or height <= 0:
            return

        norm_x = float(x) / float(width)
        norm_y = float(y) / float(height)

        # Suavizado exponencial ligero (EMA) si ya existía para amortiguar ruido
        if name in self._norm_coords:
            old_x, old_y = self._norm_coords[name]
            alpha = 0.35  # Ponderación de la nueva observación
            smooth_x = (1.0 - alpha) * old_x + alpha * norm_x
            smooth_y = (1.0 - alpha) * old_y + alpha * norm_y
            self._norm_coords[name] = (smooth_x, smooth_y)
        else:
            self._norm_coords[name] = (norm_x, norm_y)

        self._learned_hits[name] = self._learned_hits.get(name, 0) + 1
        log.debug("📍 [UI_CACHE] Aprendida pos para '%s': (%d, %d) en %dx%d (hits: %d)",
                  name, x, y, width, height, self._learned_hits[name])

    def micro_verify(
        self,
        screen: np.ndarray,
        name: str,
        template: np.ndarray,
        threshold: float = 0.68,
        margin_x: int = 45,
        margin_y: int = 45,
    ) -> tuple[int, int] | None:
        """
        Micro-verificación de template en una ventana reducida alrededor de la coordenada en caché.
        Tarda < 1 ms al procesar un área pequeña (~90x90 px) en vez de toda la pantalla.
        Retorna la coordenada central exacta (cx, cy) si hay coincidencia, o None si no.
        """
        if not self.enabled or screen is None or template is None:
            return None

        h, w = screen.shape[:2]
        cx, cy = self.get_pos(name, width=w, height=h, jitter=0)

        th, tw = template.shape[:2]
        # Calcular ROI con márgenes suficientes para que quepa el template holgadamente
        pad_x = max(margin_x, tw // 2 + 10)
        pad_y = max(margin_y, th // 2 + 10)

        x1 = max(0, cx - pad_x)
        y1 = max(0, cy - pad_y)
        x2 = min(w, cx + pad_x)
        y2 = min(h, cy + pad_y)

        # Si el recorte es menor que el template, no se puede buscar
        if (y2 - y1) < th or (x2 - x1) < tw:
            return None

        crop = screen[y1:y2, x1:x2]
        # Convertir a escala de grises si es necesario
        if crop.ndim == 3 and template.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif crop.ndim == 2 and template.ndim == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Evitar falsos positivos en recortes o templates sin contraste (std == 0)
        if np.std(crop) < 1.0 or np.std(template) < 1.0:
            return None

        res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if not np.isnan(max_val) and max_val >= threshold:
            # Coordenada del centro del template en el sistema global
            best_x = x1 + max_loc[0] + tw // 2
            best_y = y1 + max_loc[1] + th // 2
            # Auto-actualizar caché si la confianza es muy alta
            if max_val >= 0.80:
                self.update_pos(name, best_x, best_y, width=w, height=h)
            return (best_x, best_y)

        return None
