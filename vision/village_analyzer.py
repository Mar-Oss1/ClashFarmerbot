"""
vision/village_analyzer.py
Fachada de percepción unificada de la aldea en combate (Fase 4 - Mejora 05).
Agrupa y cachea el análisis de escena (YOLO, HUD mask, contornos legales de despliegue y máscaras de muros)
en un único objeto VillageScene para evitar re-inferencias redundantes entre subsistemas.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from core.logger import get_logger
from vision.yolo_detector import Detection, get_detector
from vision.hud import get_battle_hud_mask

log = get_logger("village_analyzer")


@dataclass
class VillageScene:
    """Representa el estado perceptual completo de la aldea en una zona o encuadre determinado."""
    screen: np.ndarray
    zone_name: str = "CENTRO"
    detections: list[Detection] = field(default_factory=list)
    deploy_mask: np.ndarray | None = None
    red_contour: np.ndarray | None = None
    green_contour: np.ndarray | None = None
    wall_mask: np.ndarray | None = None


class VillageAnalyzer:
    """Analizador integral de la aldea que encapsula YOLO, filtros de HUD y límites legales de despliegue."""

    @staticmethod
    def analyze(
        screen: np.ndarray,
        zone_name: str = "CENTRO",
        boundary_mode: str | None = None,
        detections: list[Detection] | None = None,
    ) -> VillageScene:
        """
        Ejecuta el pipeline de percepción sobre screen reutilizando detecciones si se proporcionan:
        1. Detección YOLO filtrada por la máscara del HUD de batalla.
        2. Detección de fronteras legales (roja/verde) vía detect_village_boundary.
        Retorna VillageScene listo para ser consumido por planificadores de saltos y cuotas.
        """
        if screen is None or getattr(screen, "size", 0) == 0:
            return VillageScene(screen=screen, zone_name=zone_name)

        h, w = screen.shape[:2]

        # 1. Detecciones YOLO (si no fueron precalculadas)
        if detections is None:
            detector = get_detector()
            if detector and detector.is_ready():
                raw_dets = detector.detect(screen)
                hud_mask = get_battle_hud_mask(h, w)
                detections = [
                    d for d in raw_dets
                    if hud_mask[min(h - 1, max(0, int(d.center[1]))), min(w - 1, max(0, int(d.center[0])))] == 0
                ]
            else:
                detections = []

        # 2. Detección de frontera legal de aldea
        from attack.deployment_strategy import detect_village_boundary
        red_contour, green_contour = detect_village_boundary(screen, detections=detections, mode=boundary_mode)

        return VillageScene(
            screen=screen,
            zone_name=zone_name,
            detections=detections,
            red_contour=red_contour,
            green_contour=green_contour,
        )
