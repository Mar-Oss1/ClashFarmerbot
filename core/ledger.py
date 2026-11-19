"""
core/ledger.py
Contabilidad de sesión, cálculo de ganancias, seguimiento de capacidad real de almacenes
y disparadores de mejora de muros basados en porcentaje de almacenamiento (trigger del 90%).
"""
import time
from typing import Dict, List, Optional
from core.config import SETTINGS
from core.logger import get_logger
from core.storage_cost_table import (
    FALLBACK_TH18_CAPACITIES,
    get_fallback_storage_capacities,
)

log = get_logger("ledger")


class Ledger:
    def __init__(self):
        self.start_time = time.time()
        self.rss_start_session: dict[str, int] | None = None
        self.rss_current: dict[str, int] = {"gold": 0, "elixir": 0, "dark_elixir": 0}
        self.total_profit: dict[str, int] = {"gold": 0, "elixir": 0, "dark_elixir": 0}
        self.attacks_completed: int = 0
        self.villages_skipped: int = 0

        # Almacenamiento de capacidad real leída de tooltips (o fallback canónico TH18: 24M / 420k)
        self.max_capacities: Dict[str, int] = dict(FALLBACK_TH18_CAPACITIES)
        self.production_per_hour: Dict[str, int] = {"gold": 0, "elixir": 0, "dark_elixir": 0}
        # "In Treasury" es un dato puramente informativo (Clan Castle); no altera la capacidad de almacenes normales
        self.in_treasury: Dict[str, int] = {"gold": 0, "elixir": 0, "dark_elixir": 0}
        self.tooltip_cached: bool = False

    def record_tooltip_data(self, tooltip_data: Dict[str, Optional[Dict[str, Optional[int]]]]):
        """
        Almacena en caché de sesión los 3 valores leídos del tooltip por cada recurso:
        - Capacidad máxima real (Total Capacity)
        - Producción por hora (Production per hour)
        - Guardado en tesorería (In Treasury)

        Si la lectura de algún recurso falla o es incompleta, se aplica de forma transparente
        el fallback oficial de core/storage_cost_table.py (TH18 con almacenes TH17: 24M Oro/Elixir y 420k Oscuro).
        """
        if not tooltip_data:
            log.warning("⚠️ Datos de tooltip vacíos. Aplicando capacidades de fallback TH18...")
            self.set_fallback_capacities(th_level=18)
            return

        for res in ("gold", "elixir", "dark_elixir"):
            item = tooltip_data.get(res)
            fallback_cap = FALLBACK_TH18_CAPACITIES.get(res, 24_000_000 if res != "dark_elixir" else 420_000)

            cap_val = item.get("total_capacity") if item else None
            is_valid_cap = False
            if cap_val is not None:
                if res in ("gold", "elixir") and cap_val >= 5_000_000:
                    is_valid_cap = True
                elif res == "dark_elixir" and 10_000 <= cap_val <= 600_000:
                    is_valid_cap = True

            if is_valid_cap:
                self.max_capacities[res] = int(cap_val)
            else:
                if cap_val is not None:
                    log.warning(
                        "⚠️ Capacidad leída para %s (%s) rechazada por control de cordura (<5M en Oro/Elixir). Aplicando fallback canónico TH18: %s",
                        res, f"{cap_val:,}", f"{fallback_cap:,}"
                    )
                else:
                    log.info("ℹ️ Usando capacidad fallback para %s: %s", res, f"{fallback_cap:,}")
                self.max_capacities[res] = fallback_cap

            if item and item.get("production_per_hour") is not None:
                self.production_per_hour[res] = int(item["production_per_hour"])

            if item and item.get("in_treasury") is not None:
                self.in_treasury[res] = int(item["in_treasury"])

        self.tooltip_cached = True
        log.info(
            "📊 Caché de sesión de almacenes actualizada -> Oro: %s (prod: %s/h, treasury: %s) | "
            "Elixir: %s (prod: %s/h, treasury: %s) | Oscuro: %s (prod: %s/h, treasury: %s)",
            f"{self.max_capacities['gold']:,}",
            f"{self.production_per_hour['gold']:,}",
            f"{self.in_treasury['gold']:,}",
            f"{self.max_capacities['elixir']:,}",
            f"{self.production_per_hour['elixir']:,}",
            f"{self.in_treasury['elixir']:,}",
            f"{self.max_capacities['dark_elixir']:,}",
            f"{self.production_per_hour['dark_elixir']:,}",
            f"{self.in_treasury['dark_elixir']:,}",
        )

    def set_fallback_capacities(self, th_level: int = 18):
        """Aplica las capacidades de fallback canónicas para el TH indicado."""
        fallback = get_fallback_storage_capacities(th_level=th_level)
        self.max_capacities.update(fallback)
        self.tooltip_cached = True
        log.info("🛡️ Capacidades fijadas vía fallback (TH%d): %s", th_level, self.max_capacities)

    def calculate_resource_percentages(self) -> Dict[str, float]:
        """
        Calcula de forma INDEPENDIENTE el porcentaje de llenado de cada almacén:
        - gold_pct = gold_actual / max_gold
        - elixir_pct = elixir_actual / max_elixir
        - dark_pct = dark_actual / max_dark

        Nota técnica: 'In Treasury' se almacena de forma independiente como dato informativo
        y NO altera el cálculo de capacidad de almacenamiento normal.
        """
        pcts: Dict[str, float] = {}
        for res, key_pct in (("gold", "gold_pct"), ("elixir", "elixir_pct"), ("dark_elixir", "dark_pct")):
            cur = float(self.rss_current.get(res, 0))
            cap = float(self.max_capacities.get(res, FALLBACK_TH18_CAPACITIES.get(res, 1)))
            pcts[key_pct] = round(cur / cap, 4) if cap > 0 else 0.0
        return pcts

    @property
    def gold_pct(self) -> float:
        return self.calculate_resource_percentages()["gold_pct"]

    @property
    def elixir_pct(self) -> float:
        return self.calculate_resource_percentages()["elixir_pct"]

    @property
    def dark_pct(self) -> float:
        return self.calculate_resource_percentages()["dark_pct"]

    def compare_and_order_resources(
        self,
        source_level: Optional[int] = None,
        target_level: Optional[int] = None,
    ) -> List[str]:
        """
        Mini sistema de comparación y ordenamiento de recursos para muros:
        - Muros de nivel 1 a 4: Únicamente admiten Oro (retorna ['gold']).
        - Muros de nivel >= 5 o dinámicos:
          Compara Oro vs Elixir según:
          1. Porcentaje de llenado de almacén (saturation pct).
          2. Saldo absoluto de recursos disponibles.
          Retorna ['gold', 'elixir'] si Oro es más alto, o ['elixir', 'gold'] si Elixir es más alto.
        - Elixir Oscuro queda estrictamente excluido en todas las condiciones.
        """
        from core.wall_upgrader import WallUpgradeStrategy
        strategy = WallUpgradeStrategy(target_level=target_level)
        eff_source = source_level if source_level is not None else (target_level - 1 if target_level else None)

        if eff_source is not None and not strategy.can_use_resource(eff_source, "elixir", target_level=target_level):
            return ["gold"]

        pcts = self.calculate_resource_percentages()
        g_pct = pcts.get("gold_pct", 0.0)
        e_pct = pcts.get("elixir_pct", 0.0)

        gold_val = self.rss_current.get("gold", 0)
        elixir_val = self.rss_current.get("elixir", 0)

        # Comparación por saturación porcentual; desempata por saldo absoluto
        if g_pct > e_pct:
            return ["gold", "elixir"]
        elif e_pct > g_pct:
            return ["elixir", "gold"]
        else:
            return ["gold", "elixir"] if gold_val > elixir_val else ["elixir", "gold"]

    def get_storage_trigger_resources(
        self,
        source_level: Optional[int] = None,
        target_level: Optional[int] = None,
    ) -> List[str]:
        """
        Evalúa el disparador de mejora de muros basado en la capacidad de almacenes:
        - STORAGE_UPGRADE_TRIGGER_THRESHOLD = 0.90 en config/settings.json (configurable).
        - REGLA DE NEGOCIO: El Elixir Oscuro NUNCA dispara mejoras de muro (los muros solo usan
          Oro o Elixir; dark_pct se guarda únicamente para uso futuro en tropas/hechizos oscuros).
        - Si ambos (Oro y Elixir) >= 90%: aplica el mini sistema de comparación y ordenamiento
          para priorizar el recurso más saturado / con mayor saldo antes.
        - Si solo Elixir >= 90%: tanda con Elixir.
        - Si solo Oro >= 90%: tanda con Oro.
        - Si no se alcanza el umbral: lista vacía [].
        """
        econ = SETTINGS.get("economy", {})
        if not econ.get("enable_wall_upgrade", True) and not econ.get("upgrade_walls", True):
            return []

        threshold = float(econ.get("storage_upgrade_trigger_threshold", 0.90))
        pcts = self.calculate_resource_percentages()
        g_pct = pcts["gold_pct"]
        e_pct = pcts["elixir_pct"]
        d_pct = pcts["dark_pct"]

        # ADVERTENCIA / REGLA: El Elixir Oscuro jamás dispara mejoras de muro
        if d_pct >= threshold:
            log.info(
                "ℹ️ dark_pct (%.1f%%) supera el umbral (%.1f%%). Elixir Oscuro NUNCA dispara mejoras de muro (reservado para tropas/héroes).",
                d_pct * 100, threshold * 100
            )

        gold_triggered = g_pct >= threshold
        elixir_triggered = e_pct >= threshold

        eff_source = source_level if source_level is not None else econ.get("wall_level_source")
        eff_target = target_level if target_level is not None else econ.get("wall_level_target")

        # Validación de nivel: si el nivel origen <= 4, Elixir no puede usarse para muros
        if elixir_triggered and (eff_source is not None or eff_target is not None):
            from core.wall_upgrader import WallUpgradeStrategy
            strategy = WallUpgradeStrategy(target_level=eff_target)
            eff_s = eff_source if eff_source is not None else (eff_target - 1 if eff_target else 1)
            if not strategy.can_use_resource(eff_s, "elixir", target_level=eff_target):
                log.info("ℹ️ Elixir al >= 90%% pero el nivel %d solo admite Oro. Desactivando trigger de Elixir.", eff_s)
                elixir_triggered = False

        if gold_triggered and elixir_triggered:
            ordered = self.compare_and_order_resources(source_level=eff_source, target_level=eff_target)
            primary_name = "ORO" if ordered[0] == "gold" else "ELIXIR"
            primary_pct = g_pct if ordered[0] == "gold" else e_pct
            secondary_pct = e_pct if ordered[0] == "gold" else g_pct
            log.info(
                "🚨 TRIGGER DOBLE ALMACÉN (>= %.1f%%): Oro (%.1f%%) y Elixir (%.1f%%) saturados. "
                "Ordenando por recurso más alto: priorizando %s (%.1f%% vs %.1f%%). Secuencia: %s",
                threshold * 100, g_pct * 100, e_pct * 100, primary_name, primary_pct * 100, secondary_pct * 100, ordered
            )
            return ordered
        elif elixir_triggered:
            log.info(
                "🚨 TRIGGER ALMACÉN (>= %.1f%%): Elixir saturado (%.1f%%). Priorizando gasto en muros con ELIXIR.",
                threshold * 100, e_pct * 100
            )
            return ["elixir"]
        elif gold_triggered:
            log.info(
                "🚨 TRIGGER ALMACÉN (>= %.1f%%): Oro saturado (%.1f%%). Priorizando gasto en muros con ORO.",
                threshold * 100, g_pct * 100
            )
            return ["gold"]

        return []

    def get_wall_upgrade_resources_sequence(
        self,
        source_level: Optional[int] = None,
        target_level: Optional[int] = None,
    ) -> List[str]:
        """
        Retorna la secuencia de recursos a utilizar para mejorar muros:
        1. Prioridad: Disparador por saturación de almacén (>= 90% capacidad real),
           ordenando con el mini sistema de comparación (el recurso más alto primero).
        2. Fallback: Umbral convencional min_rss_to_spend de la economía si no hay saturación,
           ordenando también por el recurso disponible más alto.
        """
        storage_triggers = self.get_storage_trigger_resources(source_level=source_level, target_level=target_level)
        if storage_triggers:
            return storage_triggers

        legacy_res = self.get_wall_upgrade_resource(source_level=source_level, target_level=target_level)
        return [legacy_res] if legacy_res else []

    def record_initial_snapshot(self, rss: dict[str, int]):
        if not rss:
            return
        gold = rss.get("gold", 0)
        elixir = rss.get("elixir", 0)
        if gold <= 0 and elixir <= 0:
            return

        if self.rss_start_session is None:
            # Evitar registrar snapshot con Elixir u Oscuro en 0 si Oro es millonario (fallo de OCR inicial)
            if gold > 500_000 and (elixir == 0 or rss.get("dark_elixir", 0) == 0):
                log.warning("⚠️ Snapshot inicial incompleto (lectura en 0 con Oro alto: %s). Reintentando en siguiente captura.", rss)
                return
            self.rss_start_session = rss.copy()
            self.rss_current = rss.copy()
            log.info("📊 Snapshot inicial registrado con éxito: %s", self.rss_start_session)
        else:
            # Si el snapshot inicial tenía 0 en elixir u oscuro por error de OCR, corregir línea base
            for k in ("elixir", "dark_elixir"):
                if self.rss_start_session.get(k, 0) == 0 and rss.get(k, 0) > 10_000:
                    log.info("🔄 Corrigiendo línea base preexistente de %s en snapshot inicial: %s", k, f"{rss[k]:,}")
                    self.rss_start_session[k] = rss[k]
                    self.rss_current[k] = rss[k]

    def record_post_attack(self, new_rss: dict[str, int]):
        self.attacks_completed += 1
        if not any(new_rss.values()):
            log.warning("⚠️ No se pudieron leer recursos propios tras el ataque (valores en 0).")
            return

        gain = {"gold": 0, "elixir": 0, "dark_elixir": 0}
        if any(self.rss_current.values()):
            for k in ("gold", "elixir", "dark_elixir"):
                # Si el valor anterior era 0 pero el nuevo es alto, fue un fallo de OCR previo, no botín ganado
                if self.rss_current.get(k, 0) == 0 and new_rss.get(k, 0) > 50_000:
                    log.warning("⚠️ Detectada recuperación de recurso %s (de 0 a %s). Registrando saldo real sin sumar a botín.", k, f"{new_rss[k]:,}")
                    self.rss_current[k] = new_rss[k]
                    if self.rss_start_session:
                        self.rss_start_session[k] = new_rss[k]
                    continue

                diff = new_rss.get(k, 0) - self.rss_current.get(k, 0)
                if diff > 0:
                    gain[k] = diff
                    self.total_profit[k] += diff

        self.rss_current = new_rss.copy()
        log.info(
            "💰 Botín de este ataque -> Oro: +%s | Elixir: +%s | Oscuro: +%s",
            f"{gain['gold']:,}",
            f"{gain['elixir']:,}",
            f"{gain['dark_elixir']:,}",
        )
        log.info(
            "📈 Total Acumulado en Sesión (#%d) -> Oro: +%s | Elixir: +%s | Oscuro: +%s",
            self.attacks_completed,
            f"{self.total_profit['gold']:,}",
            f"{self.total_profit['elixir']:,}",
            f"{self.total_profit['dark_elixir']:,}",
        )

    def get_wall_upgrade_resource(
        self,
        source_level: Optional[int] = None,
        target_level: Optional[int] = None,
    ) -> Optional[str]:
        """
        Retorna el recurso a utilizar para mejora de muros.
        Prioriza la señal de saturación del 90% si está activa.
        Si no, aplica el umbral convencional min_rss_to_spend.
        """
        econ = SETTINGS.get("economy", {})
        if not econ.get("enable_wall_upgrade", False) and not econ.get("upgrade_walls", False):
            return None

        # Si el trigger del 90% está activo para algún recurso, consumirlo prioritariamente
        storage_triggers = self.get_storage_trigger_resources(source_level=source_level, target_level=target_level)
        if storage_triggers:
            return storage_triggers[0]

        eff_source = source_level if source_level is not None else econ.get("wall_level_source")
        eff_target = target_level if target_level is not None else econ.get("wall_level_target")
        batch_size = econ.get("wall_batch_size", 2)
        base_threshold = econ.get("min_rss_to_spend", 20_000_000)

        # Cálculo dinámico de umbral y validación de recurso según el nivel
        if eff_source is not None or eff_target is not None:
            from core.wall_upgrader import WallUpgradeStrategy
            strategy = WallUpgradeStrategy(target_level=eff_target)
            eff_s = eff_source if eff_source is not None else max(1, (eff_target - 1) if eff_target else 1)
            threshold = strategy.get_min_rss_threshold(eff_s, batch_size=batch_size, fallback=base_threshold, target_level=eff_target)
            can_elixir = strategy.can_use_resource(eff_s, "elixir", target_level=eff_target)
        else:
            threshold = base_threshold
            can_elixir = True

        gold = self.rss_current.get("gold", 0)
        elixir = self.rss_current.get("elixir", 0)

        gold_ready = gold >= threshold
        elixir_ready = can_elixir and (elixir >= threshold)

        # Mini sistema de comparación y ordenamiento:
        # Si ambos superan el umbral, seleccionar el recurso que sea más alto
        if gold_ready and elixir_ready:
            ordered = self.compare_and_order_resources(source_level=eff_source, target_level=eff_target)
            log.info(
                "📊 Comparación de recursos (ambos superan %s): Oro=%s, Elixir=%s -> Seleccionando el más alto (%s)",
                f"{threshold:,}", f"{gold:,}", f"{elixir:,}", ordered[0].upper()
            )
            return ordered[0]
        elif elixir_ready:
            return "elixir"
        elif gold_ready:
            return "gold"
        return None

    def print_summary(self):
        elapsed_min = max(1.0, (time.time() - self.start_time) / 60.0)
        pcts = self.calculate_resource_percentages()
        log.info("═" * 60)
        log.info("🏆 RESUMEN DE SESIÓN (%.1f minutos)", elapsed_min)
        log.info("   • Ataques realizados : %d", self.attacks_completed)
        log.info("   • Aldeas saltadas    : %d", self.villages_skipped)
        log.info("   • Oro farmeado       : +%s (Llenado: %.1f%%)", f"{self.total_profit['gold']:,}", pcts["gold_pct"] * 100)
        log.info("   • Elixir farmeado    : +%s (Llenado: %.1f%%)", f"{self.total_profit['elixir']:,}", pcts["elixir_pct"] * 100)
        log.info("   • Oscuro farmeado    : +%s (Llenado: %.1f%%)", f"{self.total_profit['dark_elixir']:,}", pcts["dark_pct"] * 100)
        log.info(
            "   • Capacidades Máximas: Oro: %s | Elixir: %s | Oscuro: %s",
            f"{self.max_capacities['gold']:,}",
            f"{self.max_capacities['elixir']:,}",
            f"{self.max_capacities['dark_elixir']:,}",
        )
        log.info(
            "   • In Treasury (CC)   : Oro: %s | Elixir: %s | Oscuro: %s",
            f"{self.in_treasury['gold']:,}",
            f"{self.in_treasury['elixir']:,}",
            f"{self.in_treasury['dark_elixir']:,}",
        )
        log.info("═" * 60)
