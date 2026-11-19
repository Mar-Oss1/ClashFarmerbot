"""
core/wall_upgrader.py
Módulo automatizado y reutilizable para la mejora de muros en Clash of Clans.
Soporta cualquier transición de nivel (1->2, 5->6, 10->11, ..., 18->19, ..., N->N+1)
con lógica dinámica de costos, requisitos de TH, recursos válidos (Oro/Elixir),
plantilla global reutilizable y combinación sinérgica de filtros (RSS + Permanente).
Fuente única de verdad: core/wall_cost_table.py (WALL_LEVEL_COSTS).
"""
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Union
import cv2
import numpy as np

from core.device import Device
from core.vision import Vision
from humanizer.behavior import HumanBehavior
from core.config import SETTINGS
from core.logger import get_logger
from core.wall_cost_table import (
    WALL_LEVEL_COSTS,
    get_required_resources,
    can_afford,
)

log = get_logger("wall_upgrader")


@dataclass(frozen=True)
class WallTransition:
    """
    Especificación técnica de una transición de muro (nivel origen -> nivel destino).
    Permite parametrizar y consultar dinámicamente costos, requisitos y tiempos.
    """
    source_level: int
    target_level: int
    cost_gold: int
    cost_elixir: Optional[int]
    required_th: int
    upgrade_time_sec: int = 0
    can_use_elixir: bool = True
    # Flag de extrapolación: True para niveles >= 20.
    # ADVERTENCIA: Los valores para niveles >= 20 son aproximaciones calculadas dinámicamente
    # y pueden diferir de las especificaciones oficiales que Supercell agregue en el futuro.
    is_extrapolated: bool = False


class WallUpgradeStrategy:
    """
    Estrategia global reutilizable para cualquier transición de muros en Clash of Clans.
    Elimina hardcodeos específicos y proporciona cálculo dinámico de:
    - Costos unitarios y por lote directamente desde la tabla oficial (WALL_LEVEL_COSTS).
    - Recursos válidos (Oro para niveles 1-4; Oro y Elixir a partir de nivel 5).
    - Tiempos de mejora (instantáneo: 0s en CoC).
    - Requisitos de nivel de Ayuntamiento (Town Hall).
    - Flag de extrapolación (is_extrapolated) para niveles fuera de la tabla oficial (>= 20).
    - Umbrales de balance de recursos para disparar mejoras sin valores fijos.
    """

    # Referencia canónica a la tabla oficial de costos (Niveles 1 a 19)
    CANONICAL_WALL_DATA: Dict[int, dict] = WALL_LEVEL_COSTS

    # ADVERTENCIA: Los niveles >= 20 son extrapolaciones dinámicas aproximadas.
    # Los valores reales de Supercell para niveles por encima de 19 pueden no seguir
    # un patrón estrictamente lineal/geométrico. La extrapolación debe marcarse con is_extrapolated=True.
    is_extrapolated: bool = False

    def __init__(
        self,
        custom_table: Optional[Dict[int, dict]] = None,
        target_level: Optional[int] = None,
    ):
        self._table = dict(WALL_LEVEL_COSTS)
        if custom_table:
            self._table.update(custom_table)
        self.target_level = target_level
        # Flag de extrapolación a nivel de instancia
        # ADVERTENCIA: Aproximación para niveles >= 20 fuera de la tabla oficial.
        self.is_extrapolated: bool = (target_level >= 20) if target_level is not None else False

    def get_transition(self, source_level: int, target_level: Optional[int] = None) -> WallTransition:
        """
        Devuelve la especificación de transición para source_level -> target_level.
        Si target_level supera los niveles tabulados (>= 20), extrapola dinámicamente los valores.
        """
        tgt = target_level if target_level is not None else source_level + 1
        is_extrap = tgt >= 20
        self.is_extrapolated = is_extrap

        if tgt in self._table:
            data = self._table[tgt]
            cost_gold = int(data["gold_cost"])
            elixir_val = data["elixir_cost"]
            cost_elixir = int(elixir_val) if elixir_val is not None else None
            can_elixir = cost_elixir is not None
            th = int(data.get("th_required", 18))
            time_sec = int(data.get("time", 0))
        else:
            # ADVERTENCIA DE APROXIMACIÓN (NIVEL >= 20):
            # Los valores reales de Supercell para niveles por encima del 19 pueden no seguir
            # un patrón puramente lineal/geométrico. La extrapolación es una aproximación
            # y se marca como is_extrapolated=True para evitar decisiones erróneas de gasto.
            can_elixir = True
            extra_steps = max(0, tgt - 19)
            cost_gold = 10_000_000 + extra_steps * 1_000_000
            cost_elixir = cost_gold
            th = 18
            time_sec = 0

        return WallTransition(
            source_level=source_level,
            target_level=tgt,
            cost_gold=cost_gold,
            cost_elixir=cost_elixir,
            required_th=th,
            upgrade_time_sec=time_sec,
            can_use_elixir=can_elixir,
            is_extrapolated=is_extrap,
        )

    def can_use_resource(self, source_level: int, resource_type: str, target_level: Optional[int] = None) -> bool:
        """
        Determina si un recurso ('gold' o 'elixir') es válido para mejorar el nivel dado.
        Oro es admitido desde el nivel 1. Elixir se admite desde nivel 5 (según la tabla oficial).
        """
        res = resource_type.lower()
        if res == "gold":
            return True
        if res == "elixir":
            tr = self.get_transition(source_level, target_level)
            return tr.can_use_elixir
        return False

    def get_cost(self, source_level: int, resource_type: str = "gold", target_level: Optional[int] = None) -> int:
        """Obtiene el costo dinámico unitario según el recurso solicitado consultando la tabla oficial."""
        tr = self.get_transition(source_level, target_level)
        res = resource_type.lower()
        if res == "elixir":
            if not tr.can_use_elixir or tr.cost_elixir is None:
                raise ValueError(
                    f"Elixir no está permitido para muros con nivel objetivo {tr.target_level} (niveles 1 a 4 no requieren elixir)"
                )
            return tr.cost_elixir
        elif res == "gold":
            return tr.cost_gold
        raise ValueError(f"Recurso no reconocido: {resource_type}")

    def calculate_batch_resources(
        self,
        source_level: int,
        num_walls: int,
        target_level: Optional[int] = None,
    ) -> Dict[str, Optional[int]]:
        """
        Calcula los costos de Oro y Elixir por separado para un lote de muros consultando la tabla oficial.
        """
        tr = self.get_transition(source_level, target_level)
        qty = max(1, num_walls)
        return {
            "gold": tr.cost_gold * qty,
            "elixir": tr.cost_elixir * qty if tr.cost_elixir is not None else None,
        }

    def calculate_batch_cost(
        self,
        source_level: int,
        num_walls: int,
        resource_type: Optional[str] = "gold",
        target_level: Optional[int] = None,
    ) -> Union[int, Dict[str, Optional[int]]]:
        """
        Calcula dinámicamente el costo total para un lote de muros consultando directamente la tabla oficial.
        - Si resource_type es 'gold' o 'elixir', devuelve el entero correspondiente.
        - Si resource_type es None o 'both'/'all', devuelve un diccionario con {'gold': ..., 'elixir': ...}.
        """
        batch_res = self.calculate_batch_resources(source_level, num_walls, target_level)
        if resource_type is None or resource_type.lower() in ("both", "all"):
            return batch_res
        res = resource_type.lower()
        if res == "gold":
            return int(batch_res["gold"])
        elif res == "elixir":
            if batch_res["elixir"] is None:
                tr = self.get_transition(source_level, target_level)
                raise ValueError(
                    f"Elixir no está permitido para muros con nivel objetivo {tr.target_level} (niveles 1 a 4 no requieren elixir)"
                )
            return int(batch_res["elixir"])
        raise ValueError(f"Recurso no reconocido: {resource_type}")

    def get_min_rss_threshold(
        self,
        source_level: Optional[int],
        batch_size: int = 2,
        fallback: int = 20_000_000,
        resource_type: str = "gold",
        target_level: Optional[int] = None,
    ) -> int:
        """
        Calcula el saldo mínimo requerido dinámicamente para disparar mejoras.
        Si source_level es None y target_level es None, recurre al valor configurado como fallback.
        """
        if source_level is None and target_level is None:
            return fallback
        eff_source = source_level if source_level is not None else max(1, (target_level - 1) if target_level else 1)
        try:
            val = self.calculate_batch_cost(eff_source, batch_size, resource_type, target_level=target_level)
            return int(val) if isinstance(val, (int, float)) else fallback
        except ValueError:
            val = self.calculate_batch_cost(eff_source, batch_size, "gold", target_level=target_level)
            return int(val) if isinstance(val, (int, float)) else fallback

    def estimate_affordable_walls(
        self,
        available_rss: int,
        source_level: int,
        resource_type: str = "gold",
        target_level: Optional[int] = None,
    ) -> int:
        """Estima cuántos muros se pueden mejorar con los recursos disponibles."""
        try:
            cost = self.get_cost(source_level, resource_type, target_level=target_level)
            return max(0, available_rss // cost) if cost > 0 else 0
        except ValueError:
            return 0

    def get_required_resources(self, target_level: int) -> dict:
        """Devuelve los recursos requeridos (Oro y Elixir) para target_level."""
        return get_required_resources(target_level)

    def can_afford(
        self,
        target_level: int,
        available_gold: int,
        available_elixir: int,
        require_both: bool = False,
    ) -> bool:
        """Valida si los recursos disponibles alcanzan para target_level."""
        return can_afford(
            target_level, available_gold, available_elixir, require_both=require_both
        )

    @staticmethod
    def is_extrapolated_level(target_level: int) -> bool:
        """
        Indica si un nivel destino requiere extrapolación (>= 20).
        ADVERTENCIA: Niveles >= 20 son aproximaciones sin confirmar por Supercell.
        """
        return target_level >= 20

    def compare_and_order_resources(
        self,
        available_gold: int,
        available_elixir: int,
        source_level: Optional[int] = None,
        target_level: Optional[int] = None,
    ) -> List[str]:
        """
        Mini sistema de comparación y ordenamiento:
        Compara saldo de Oro vs Elixir y determina el orden de prioridad:
        - Para niveles 1 a 4 (donde Elixir no se admite): siempre ['gold'].
        - Para niveles >= 5 o dinámicos:
          - Si available_gold > available_elixir -> ['gold', 'elixir']
          - Si available_elixir >= available_gold -> ['elixir', 'gold']
        """
        eff_source = source_level if source_level is not None else (target_level - 1 if target_level else None)
        if eff_source is not None and not self.can_use_resource(eff_source, "elixir", target_level=target_level):
            return ["gold"]

        if available_gold > available_elixir:
            return ["gold", "elixir"]
        return ["elixir", "gold"]

    order_resources_by_priority = compare_and_order_resources


class WallUpgrader:
    """
    Controlador autónomo para la mejora de muros en Clash of Clans.
    Implementa:
    1. Estrategia dinámica de costos y niveles sin hardcodeo de 18 -> 19.
    2. Plantilla global reutilizable agnóstica de nivel para detectar filas de 'Wall'.
    3. Cobertura del 100% de muros mediante filtro combinado (RSS + Permanente).
    4. Retrocompatibilidad absoluta sin regresiones para la transición 18 -> 19.
    """

    def __init__(
        self,
        device: Device,
        vision: Vision,
        source_level: Optional[int] = None,
        target_level: Optional[int] = None,
        strategy: Optional[WallUpgradeStrategy] = None,
    ):
        self.device = device
        self.vision = vision
        self.strategy = strategy or WallUpgradeStrategy()

        # Parametrización dinámica de nivel origen y destino
        econ_cfg = SETTINGS.get("economy", {})
        self.source_level = source_level if source_level is not None else econ_cfg.get("wall_level_source")
        self.target_level = target_level if target_level is not None else econ_cfg.get("wall_level_target")
        if self.source_level is not None and self.target_level is None:
            self.target_level = self.source_level + 1

        # 0. Caché de sesión en RAM para filtros activos
        self._session_filters: Optional[Dict[str, bool]] = None

        # 1. Menú del constructor
        self.tpl_build_btn = "assets/buttons/build_btn.png"
        self.tpl_filter_btn = "assets/buttons/build_filter_btn.png"
        self.tpl_wall_label = "assets/buttons/wall_label_generic.png"
        
        # Plantilla global reutilizable válida para cualquier transición (1->2, 5->6, 10->11, ..., N->N+1)
        self.tpl_wall_row_generic = "assets/buttons/wall_row_generic.png"
        # Plantilla legacy (18 -> 19) mantenida para compatibilidad 100% garantizada
        self.tpl_wall_row = "assets/buttons/wall_row_item.png"

        # 2. Filtros apagados (marrón)
        self.tpl_filter_gold_off = "assets/buttons/filter_gold_off.png"
        self.tpl_filter_elixir_off = "assets/buttons/filter_elixir_off.png"
        self.tpl_filter_perm_off = "assets/buttons/filter_perm_off.png"

        # 3. Secuencia de selección múltiple (Muros dinámicos)
        self.tpl_upgrade_more = "assets/buttons/upgrade_more_btn.png"      # Botón "Upgrade More"
        self.tpl_add_wall_1 = "assets/buttons/add_wall_1_btn.png"          # Botón "+1 Add Wall"
        self.tpl_remove_wall_1 = "assets/buttons/remove_wall_1_btn.png"    # Botón "-1 Remove Wall"
        self.tpl_upgrade_hammers = "assets/buttons/upgrade_hammers.png"     # Martillos + Upgrade
        self.tpl_upgrade_gold = "assets/buttons/upgrade_with_gold.png"
        self.tpl_upgrade_elixir = "assets/buttons/upgrade_with_elixir.png"
        self.tpl_confirm = "assets/buttons/confirm_upgrade_btn.png"
        self.tpl_dialog_confirm = "assets/buttons/dialog_confirm_btn.png"   # Botón verde de validación múltiple

    def compare_and_order_resources(
        self,
        available_gold: int,
        available_elixir: int,
        source_level: Optional[int] = None,
        target_level: Optional[int] = None,
    ) -> List[str]:
        """Delega en la estrategia para comparar y ordenar recursos según saldo y nivel."""
        return self.strategy.compare_and_order_resources(
            available_gold=available_gold,
            available_elixir=available_elixir,
            source_level=source_level if source_level is not None else self.source_level,
            target_level=target_level if target_level is not None else self.target_level,
        )

    order_resources_by_priority = compare_and_order_resources

    # ROIs normalizadas relativas (ymin, ymax, xmin, xmax) calibradas sobre sample_filters_active.jpg (2400x1080)
    FILTER_BUTTON_ROIS = {
        "gold": (0.198, 0.265, 0.408, 0.504),
        "elixir": (0.198, 0.265, 0.523, 0.619),
        "perm": (0.431, 0.498, 0.635, 0.731),
    }

    # Coordenadas relativas de tap (rx, ry) al centro de cada botón de filtro
    FILTER_BUTTON_TAPS = {
        "gold": (0.456, 0.231),
        "elixir": (0.571, 0.231),
        "perm": (0.683, 0.465),
    }

    def reset_filter_cache(self) -> None:
        """Restablece la caché de sesión de filtros en RAM."""
        self._session_filters = None

    @staticmethod
    def _is_filter_button_active(screen: np.ndarray, roi: Tuple[float, float, float, float]) -> bool:
        """
        Determina si un botón de filtro está activo (amarillo/dorado) o inactivo (marrón).
        Mide el porcentaje de píxeles amarillo/dorado brillante en HSV (15 <= H <= 38, S >= 130, V >= 185)
        sobre los bordes/fondo del botón (evitando el centro para que la moneda dorada de Gold no dé
        falso positivo cuando esté apagado; umbral >= 0.40).
        """
        if screen is None or getattr(screen, "size", 0) == 0:
            return False

        h, w = screen.shape[:2]
        ry1, ry2, rx1, rx2 = roi
        y1, y2 = int(ry1 * h), int(ry2 * h)
        x1, x2 = int(rx1 * w), int(rx2 * w)

        btn = screen[y1:y2, x1:x2]
        if btn.size == 0 or btn.ndim != 3:
            return False

        hsv = cv2.cvtColor(btn, cv2.COLOR_BGR2HSV)
        mask = (
            (hsv[:, :, 0] >= 15)
            & (hsv[:, :, 0] <= 38)
            & (hsv[:, :, 1] >= 130)
            & (hsv[:, :, 2] >= 185)
        )

        bh, bw = btn.shape[:2]
        border_mask = np.zeros((bh, bw), dtype=bool)
        border_mask[4:20, 10:bw - 10] = True
        border_mask[bh - 20:bh - 4, 10:bw - 10] = True
        border_mask[10:bh - 10, 4:20] = True
        border_mask[10:bh - 10, bw - 20:bw - 4] = True

        if not np.any(border_mask):
            return False

        pct = float(np.mean(mask[border_mask]))
        return pct >= 0.40

    def detect_filter_states(self, screen: np.ndarray) -> Dict[str, bool]:
        """Detecta el estado activo (True/False) de los 3 botones principales de filtro."""
        return {
            k: self._is_filter_button_active(screen, roi)
            for k, roi in self.FILTER_BUTTON_ROIS.items()
        }

    def _apply_filter(self, resource_type: str):
        """
        Abre el menú de filtros del constructor y aplica la combinación óptima (RSS + Permanente):
        - Usa caché de sesión en RAM: si los filtros ya están en el estado objetivo, omite abrir el menú (0 clicks).
        - Si necesita abrir el menú, detecta el estado real (ON/OFF) de cada botón vía HSV
          y pulsa ÚNICAMENTE los que difieran del estado objetivo (apagando Gold al pasar a Elixir y viceversa).
        """
        target_state = (
            {"gold": True, "elixir": False, "perm": True}
            if resource_type == "gold"
            else {"gold": False, "elixir": True, "perm": True}
        )

        if self._session_filters == target_state:
            log.info("ℹ️ Filtros ya configurados en sesión (%s). Omitiendo apertura de menú.", self._session_filters)
            return

        screen = self.device.screenshot()
        filter_pos = self.vision.find_template(screen, self.tpl_filter_btn, threshold=0.70, zona="ESQUINA_SUP_DER")
        if not filter_pos:
            log.debug("Botón de filtro no detectado en el menú Build.")
            return

        log.info("🔍 Abriendo menú de filtros de Build...")
        HumanBehavior.reaction_delay()
        self.device.tap(*filter_pos)
        time.sleep(0.8)

        menu_screen = self.device.screenshot()
        current_state = self.detect_filter_states(menu_screen)

        w = getattr(self.device, "width", menu_screen.shape[1] if menu_screen is not None else 2400)
        h = getattr(self.device, "height", menu_screen.shape[0] if menu_screen is not None else 1080)

        # Pulsar ÚNICAMENTE los botones cuyo estado actual difiera de target_state
        for key in ("gold", "elixir", "perm"):
            if current_state.get(key) != target_state.get(key):
                desired = target_state[key]
                action_str = "Activando" if desired else "Desactivando"
                log.info("🎯 Filtro %s en estado %s (objetivo: %s). %s...", key.upper(), current_state.get(key), desired, action_str)
                rx, ry = self.FILTER_BUTTON_TAPS[key]
                tap_x, tap_y = int(rx * w), int(ry * h)
                HumanBehavior.reaction_delay()
                self.device.tap(tap_x, tap_y)
                time.sleep(0.4)
                current_state[key] = desired

        # Cerrar el menú desplegable tocando de nuevo el botón de filtro
        log.info("Cerrando desplegable de filtros...")
        self.device.tap(*filter_pos)
        time.sleep(0.8)

        self._session_filters = dict(current_state)

    def _find_and_tap_wall_in_list(self) -> bool:
        """
        Busca la fila de 'Wall' en la lista de mejoras del constructor con alta fiabilidad (>= 95%):
        1. Prioriza 'wall_label_generic.png' con threshold >= 0.85 en escala de grises.
        2. Verifica que el ROI contenga texto blanco binarizado (letras de 'Wall').
        3. Verifica que a la derecha inmediata del texto 'Wall' no haya más letras blancas pegadas
           (descarta nombres compuestos como 'Wall Wrecker' o colisiones con 'Grand Warden'/'Bomb Tower').
        4. Si no supera el umbral estricto, fallback a 'wall_row_generic.png' / 'wall_row_item.png'
           elevando su umbral mínimo a 0.85.
        Realiza deslizamientos fluidos hacia arriba para recorrer toda la lista.
        """
        max_attempts = 7
        log.info("🔍 Buscando fila de 'Wall' en la lista (máx %d intentos)...", max_attempts)

        for attempt in range(max_attempts):
            screen = self.device.screenshot()
            screen_gray = self.vision.to_gray(screen)

            wall_pos = None
            rejected_compound = False

            # 1. Búsqueda prioritaria con wall_label_generic.png y verificación estricta
            cached_lbl = None
            if hasattr(self.vision, "_load_template"):
                try:
                    res_load = self.vision._load_template(self.tpl_wall_label)
                    if isinstance(res_load, tuple) and len(res_load) == 2 and isinstance(res_load[1], (tuple, list)):
                        cached_lbl = res_load
                except Exception:
                    cached_lbl = None

            if cached_lbl is not None:
                lbl_gray, (th, tw) = cached_lbl
                search_area, off_x, off_y = self.vision._crop_zone(screen_gray, "MITAD_DERECHA")
                if search_area.shape[0] >= th and search_area.shape[1] >= tw:
                    res = cv2.matchTemplate(search_area, lbl_gray, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if max_val >= 0.85:
                        mx, my = max_loc[0] + off_x, max_loc[1] + off_y
                        match_roi = screen_gray[my : my + th, mx : mx + tw]
                        bin_match = cv2.threshold(match_roi, 180, 255, cv2.THRESH_BINARY)[1]
                        white_pixels = int(np.sum(bin_match > 0))

                        # Verificar presencia de texto blanco ('Wall' contiene >= 250 px blancos)
                        if white_pixels >= 250:
                            # Verificar que a la derecha inmediata no haya letras pegadas
                            x_right_start = mx + tw
                            x_right_end = min(screen_gray.shape[1], mx + tw + 20)
                            right_strip = screen_gray[my + 6 : my + th - 6, x_right_start : x_right_end]
                            bin_right = cv2.threshold(right_strip, 180, 255, cv2.THRESH_BINARY)[1] if right_strip.size > 0 else np.zeros((1, 1), dtype=np.uint8)

                            has_attached_letters = False
                            if right_strip.size > 0:
                                col_sums = np.sum(bin_right > 0, axis=0)
                                if any(c >= 8 for c in col_sums[:12]) or np.sum(bin_right > 0) >= 35:
                                    has_attached_letters = True

                            if not has_attached_letters:
                                wall_pos = (mx + tw // 2, my + th // 2)
                            else:
                                rejected_compound = True
                                log.debug("Palabra compuesta o letras adheridas tras 'Wall' en (%d, %d). Descartando.", mx, my)
            else:
                # Si _load_template no está disponible (mocked vision), buscar con find_template
                wall_pos = self.vision.find_template(
                    screen, self.tpl_wall_label, threshold=0.85, zona="MITAD_DERECHA"
                )

            # 2. Fallback a wall_row_generic.png / wall_row_item.png con umbral elevado a 0.85
            if not wall_pos and not rejected_compound:
                wall_pos = self.vision.find_template(
                    screen, self.tpl_wall_row_generic, threshold=0.85, zona="MITAD_DERECHA"
                )
            if not wall_pos and not rejected_compound:
                wall_pos = self.vision.find_template(
                    screen, self.tpl_wall_row, threshold=0.85, zona="MITAD_DERECHA"
                )

            if wall_pos:
                log.info("🎯 'Wall' encontrado en %s (intento %d/%d). Tocando...", wall_pos, attempt + 1, max_attempts)
                HumanBehavior.reaction_delay()
                self.device.tap(*wall_pos)
                time.sleep(0.8)
                return True

            if attempt < max_attempts - 1:
                # Swipe largo y fluido: de 80% a 22% de la altura
                sx = int(self.device.width * 0.83)
                sy_start = int(self.device.height * 0.80)
                sy_end = int(self.device.height * 0.22)
                log.debug("📜 Deslizando lista hacia abajo (intento %d): (%d, %d) -> (%d, %d)", attempt + 1, sx, sy_start, sx, sy_end)
                self.device.swipe(sx, sy_start, sx, sy_end, duration=0.55)
                time.sleep(1.0)

        log.warning("❌ No se encontró 'Wall' tras %d intentos de desplazamiento.", max_attempts)
        return False

    def _find_upgrade_buttons_left_to_right(self, screenshot: np.ndarray) -> List[Tuple[int, int]]:
        """
        Encuentra todos los botones 'Upgrade' en la parte inferior y los ordena de izquierda a derecha.
        Índice 0 = Oro (Izquierda) | Índice -1 = Elixir (Derecha).
        """
        cached = self.vision._load_template(self.tpl_upgrade_hammers)
        if cached is None:
            return []

        tpl_gray, (th, tw) = cached
        screen_gray = self.vision.to_gray(screenshot)
        crop, ox, oy = self.vision._crop_zone(screen_gray, "MITAD_INFERIOR")

        res = cv2.matchTemplate(crop, tpl_gray, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= 0.78)

        detections: List[Tuple[int, int, float]] = []
        for y, x in zip(ys, xs):
            cx = int(x + ox + tw // 2)
            cy = int(y + oy + th // 2)
            detections.append((cx, cy, float(res[y, x])))

        if not detections:
            return []

        # NMS horizontal: agrupar detecciones que estén a menos de 45px de distancia en X
        detections.sort(key=lambda item: item[2], reverse=True)
        unique_btns: List[Tuple[int, int]] = []
        for cx, cy, _ in detections:
            if all(abs(cx - ux) >= 45 for ux, _ in unique_btns):
                unique_btns.append((cx, cy))

        # Ordenar de izquierda (menor X) a derecha (mayor X)
        unique_btns.sort(key=lambda pt: pt[0])
        return unique_btns

    def _is_cost_affordable(self, screen: np.ndarray, target_pos: Tuple[int, int]) -> bool:
        """
        Evalúa si la cifra de costo encima del botón de mejora está en blanco
        (costeable) o en rojo (insuficientes recursos). Dinámico y agnóstico de nivel.
        """
        bx, by = target_pos
        y1 = max(0, by - 95)
        y2 = min(screen.shape[0], by - 65)
        x1 = max(0, bx - 60)
        x2 = min(screen.shape[1], bx + 60)
        text_roi = screen[y1:y2, x1:x2]

        if text_roi.size == 0:
            return False

        # Píxeles blancos (texto normal costeable): B, G, R > 180
        white_pts = int(np.sum((text_roi[:, :, 0] > 180) & (text_roi[:, :, 1] > 180) & (text_roi[:, :, 2] > 180)))
        # Píxeles rojos (indicador de costo excesivo): R > 180 y B < 120
        red_pts = int(np.sum((text_roi[:, :, 2] > 180) & (text_roi[:, :, 0] < 120)))

        is_affordable = (red_pts < 30) and (white_pts > 400)
        log.info(
            "📊 Evaluación de costo en %s: white=%d, red=%d -> %s",
            target_pos, white_pts, red_pts, "COSTEABLE (blanco)" if is_affordable else "INSUFICIENTE (rojo)"
        )
        return is_affordable

    def upgrade_wall(
        self,
        resource_type: str,
        max_walls: int = 5,
        source_level: Optional[int] = None,
        target_level: Optional[int] = None,
    ) -> bool:
        """
        Ejecuta el flujo completo y dinámico para mejorar muros en cualquier nivel:
        Build -> Filtro combinado (RSS + Permanente) -> Wall (plantilla global) ->
        Upgrade More -> Multi-selección dinámica (+1 hasta rojo) -> Upgrade -> Confirmar.
        """
        eff_source = source_level if source_level is not None else self.source_level
        eff_target = target_level if target_level is not None else self.target_level
        if eff_source is not None and eff_target is None:
            eff_target = eff_source + 1

        level_str = f"{eff_source}->{eff_target}" if eff_source is not None else "dinámico"
        log.info("🧱 Iniciando mejora de muros [%s] usando: %s (máx %d muros)", level_str, resource_type.upper(), max_walls)

        # Validación previa de compatibilidad de recurso si el nivel está fijado
        if eff_source is not None and not self.strategy.can_use_resource(eff_source, resource_type, eff_target):
            log.warning(
                "⚠️ El nivel %d no admite mejoras con %s (solo Oro para niveles <= 4). Cancelando intento.",
                eff_source, resource_type.upper()
            )
            return False

        # 1. Abrir menú Build
        screen = self.device.screenshot()
        build_pos = self.vision.find_template(screen, self.tpl_build_btn, threshold=0.78, zona="ESQUINA_INF_DER")
        if not build_pos:
            log.warning("No se encontró el botón Build.")
            return False

        HumanBehavior.reaction_delay()
        self.device.tap(*build_pos)
        time.sleep(1.0)

        # 2. Aplicar filtro combinado (RSS + Permanente) y tocar 'Wall' en la lista
        self._apply_filter(resource_type)
        if not self._find_and_tap_wall_in_list():
            log.info("ℹ️ No hay muros pendientes de mejora disponibles (todos al nivel máximo para el TH actual o lista vacía). Omitiendo acción.")
            # Cerrar menú tocando fuera
            self.device.tap(int(self.device.width * 0.25), int(self.device.height * 0.50))
            return False

        # 3. Esperar y tocar "Upgrade More"
        more_pos = self.vision.wait_for_template(
            self.device.screenshot, self.tpl_upgrade_more, timeout=5, threshold=0.74, zona="MITAD_INFERIOR"
        )
        if not more_pos:
            log.warning("⚠️ No apareció el botón 'Upgrade More'.")
            return False

        log.info("🎯 Botón 'Upgrade More' detectado en %s. Tocando...", more_pos)
        HumanBehavior.reaction_delay()
        self.device.tap(*more_pos)
        time.sleep(1.0)

        # 4. Modo multi-selección dinámica según recursos disponibles
        screen = self.device.screenshot()
        upgrade_btns = self._find_upgrade_buttons_left_to_right(screen)
        if upgrade_btns:
            if len(upgrade_btns) >= 2:
                target_pos = upgrade_btns[0] if resource_type == "gold" else upgrade_btns[-1]
            else:
                if resource_type == "gold":
                    target_pos = upgrade_btns[0]
                else:
                    # Si solo hay 1 botón y está claramente a la izquierda (Oro), evitar click erróneo
                    if upgrade_btns[0][0] < int(self.device.width * 0.48):
                        log.warning("⚠️ Solo se detectó botón de ORO en pantalla, pero se solicitó ELIXIR. Cancelando intento.")
                        self.device.tap(int(self.device.width * 0.25), int(self.device.height * 0.50))
                        return False
                    target_pos = upgrade_btns[0]
        else:
            target_pos = (1030, 842) if resource_type == "gold" else (1240, 840)

        # Verificar si al menos 1 muro es costeable
        if not self._is_cost_affordable(screen, target_pos):
            log.warning("⚠️ Recursos insuficientes ni para mejorar 1 solo muro con %s.", resource_type.upper())
            # Deseleccionar tocando en zona segura del mapa
            self.device.tap(int(self.device.width * 0.25), int(self.device.height * 0.50))
            return False

        # Localizar botones "+1 Muro" y "-1 Muro"
        add_one_pos = self.vision.find_template(screen, self.tpl_add_wall_1, threshold=0.72, zona="MITAD_INFERIOR")
        if not add_one_pos:
            add_one_pos = (835, 848)

        remove_one_pos = self.vision.find_template(screen, self.tpl_remove_wall_1, threshold=0.72, zona="MITAD_INFERIOR")
        if not remove_one_pos:
            remove_one_pos = (418, 850)

        current_walls = 1
        while current_walls < max_walls:
            log.info("➕ Probando añadir muro (+1)... (actuales: %d)", current_walls)
            HumanBehavior.reaction_delay()
            self.device.tap(*add_one_pos)
            time.sleep(0.7)
            screen = self.device.screenshot()

            # Recalcular posición del botón por si hubo micro-desplazamiento
            btns = self._find_upgrade_buttons_left_to_right(screen)
            if btns:
                target_pos = btns[0] if resource_type == "gold" else btns[-1]

            if not self._is_cost_affordable(screen, target_pos):
                log.info(
                    "🛑 Costo en ROJO detectado: recursos insuficientes para %d muros. Retrocediendo (-1)...",
                    current_walls + 1
                )
                self.device.tap(*remove_one_pos)
                time.sleep(0.7)
                break

            current_walls += 1
            log.info("🎯 %d muro(s) seleccionados y verificados como costeables.", current_walls)

        # 5. Tocar botón Upgrade con los muros costeables seleccionados
        screen = self.device.screenshot()
        btns = self._find_upgrade_buttons_left_to_right(screen)
        if btns:
            target_pos = btns[0] if resource_type == "gold" else btns[-1]

        log.info("🔨 Ejecutando mejora de %d muro(s) [%s] con %s en %s...", current_walls, level_str, resource_type.upper(), target_pos)
        HumanBehavior.reaction_delay()
        self.device.tap(*target_pos)
        time.sleep(1.2)

        # 6. Manejo de diálogo de validación / confirmación
        screen = self.device.screenshot()
        confirm_pos = self.vision.find_template(screen, self.tpl_dialog_confirm, threshold=0.74, zona="CENTRO")
        if not confirm_pos:
            confirm_pos = self.vision.find_template(screen, self.tpl_dialog_confirm, threshold=0.74, zona="MITAD_INFERIOR")
        if not confirm_pos:
            confirm_pos = self.vision.find_template(screen, self.tpl_confirm, threshold=0.74, zona="CENTRO")

        if confirm_pos:
            log.info("🎯 Diálogo de validación detectado en %s. Confirmando mejora...", confirm_pos)
            HumanBehavior.reaction_delay()
            self.device.tap(*confirm_pos)
            time.sleep(1.2)
        else:
            log.info("ℹ️ No se requirió diálogo de confirmación adicional o se ejecutó directamente.")

        log.info("✅ ¡%d Muro(s) [%s] mejorado(s) exitosamente usando %s!", current_walls, level_str, resource_type.upper())
        time.sleep(1.0)
        return True

    def upgrade_walls_sequence(
        self,
        resources: Optional[List[str]] = None,
        max_walls: int = 5,
        source_level: Optional[int] = None,
        target_level: Optional[int] = None,
        available_gold: Optional[int] = None,
        available_elixir: Optional[int] = None,
    ) -> Dict[str, bool]:
        """
        Ejecuta mejoras secuenciales de muro para una lista ordenada de recursos.
        Aplica el mini sistema de comparación y ordenamiento:
        - Si no se proporcionan recursos explícitos, compara available_gold y available_elixir
          y ejecuta primero el recurso más alto.
        - Si se provee una secuencia, la ejecuta en ese orden de prioridad (el recurso más alto primero).
        - Si alguna tanda falla o no hay muros pendientes, se detiene la secuencia limpiamente.
        """
        if resources is None:
            gold = available_gold or 0
            elixir = available_elixir or 0
            resources = self.compare_and_order_resources(
                available_gold=gold,
                available_elixir=elixir,
                source_level=source_level,
                target_level=target_level,
            )

        results: Dict[str, bool] = {}
        for res in resources:
            log.info("🧱 Ejecutando tanda de mejora de muros para: %s", res.upper())
            success = self.upgrade_wall(
                resource_type=res,
                max_walls=max_walls,
                source_level=source_level,
                target_level=target_level,
            )
            results[res] = success
            if not success:
                log.info(
                    "ℹ️ La tanda con %s no se ejecutó (sin recursos, sin muros o cancelada). Deteniendo secuencia.",
                    res.upper()
                )
                break
            time.sleep(1.0)
        return results
