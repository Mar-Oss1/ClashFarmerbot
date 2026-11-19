"""
humanizer/behavior.py
Funciones de aleatorización conductual para mitigar detección por patrones.
"""
import random
import time
from core.config import SETTINGS

_HUM = SETTINGS["humanizer"]


class HumanBehavior:
    @staticmethod
    def sleep_range(min_sec: float, max_sec: float):
        """Pausa con distribución ligeramente sesgada al centro (más natural que uniforme pura)."""
        mean = (min_sec + max_sec) / 2.0
        std = (max_sec - min_sec) / 4.0
        delay = random.gauss(mean, std)
        delay = max(min_sec, min(max_sec, delay))
        time.sleep(delay)

    @classmethod
    def reaction_delay(cls):
        """Tiempo de reacción humano tras aparecer un botón o cargar una aldea."""
        cls.sleep_range(_HUM["reaction_min_sec"], _HUM["reaction_max_sec"])

    @classmethod
    def between_sides_delay(cls):
        """Pausa al cambiar de flanco durante el despliegue."""
        cls.sleep_range(_HUM["between_sides_min_sec"], _HUM["between_sides_max_sec"])

    @classmethod
    def burst_tap_delay(cls):
        """Cadencia humana rápida entre toques consecutivos al soltar múltiples tropas (85ms a 135ms)."""
        cls.sleep_range(
            _HUM.get("burst_tap_min_sec", 0.085),
            _HUM.get("burst_tap_max_sec", 0.135),
        )

    @staticmethod
    def jitter_point(x: int, y: int, max_offset: int | None = None) -> tuple[int, int]:
        """Dispersión gaussiana centrada en el objetivo y truncada a +-max_offset."""
        offset = max_offset if max_offset is not None else _HUM["tap_jitter_px"]
        std = max(1.0, offset / 2.2)
        dx = int(round(max(-offset, min(offset, random.gauss(0, std)))))
        dy = int(round(max(-offset, min(offset, random.gauss(0, std)))))
        return max(0, x + dx), max(0, y + dy)

    @staticmethod
    def hold_duration() -> float:
        return random.uniform(_HUM["hold_deploy_min_sec"], _HUM["hold_deploy_max_sec"])