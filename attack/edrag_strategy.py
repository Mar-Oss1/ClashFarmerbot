"""
attack/edrag_strategy.py
Estrategia de ataque autónoma: Dragones Eléctricos + Furias (EDrag 50% Rush).
Enfocada en victorias rápidas del 50%+ para farmeo constante del Bonus de Liga.
Despliega en el Triángulo Norte de la aldea:
  1. Zoom out y Paneo decisivo hacia el Norte para encuadrar la zona norte.
  2. Furias de apertura cubriendo la entrada de ambos lados del triángulo.
  3. Despliegue de 11 Dragones Eléctricos en ambos lados a la vez en césped exterior alto y alejado.
  4. Los 4 Héroes juntos en césped exterior protegido (RC, Minion Prince, Warden, Duke Dragon).
  5. Activación rápida de habilidades de héroes tras 2 segundos.
  6. Furias de avance escalonadas hacia el núcleo / Ayuntamiento tras 10 segundos.
  (Sin máquina de asedio).
"""
import os
import time
import random
import datetime
import cv2
import numpy as np

from core.device import Device
from core.vision import Vision, save_debug_image_async
from humanizer.behavior import HumanBehavior
from core.logger import get_logger
from core.config import SETTINGS
from vision.army_checker import ArmyChecker

log = get_logger("edrag_strategy")


class EDragStrategy:
    """Implementa la secuencia de ataque por el Triángulo Norte con Dragones Eléctricos y Furias."""

    def __init__(
        self,
        device: Device,
        vision: Vision,
        human: HumanBehavior = None,
        army_checker: ArmyChecker | None = None,
    ):
        self.device = device
        self.vision = vision
        self.human = human or HumanBehavior()
        self.army = army_checker or ArmyChecker(self.vision)
        self.w = device.width
        self.h = device.height

        cfg = SETTINGS.get("attack", {}).get("edrag_rush", {})
        self.zoom_mode = str(cfg.get("zoom_mode", "half")).lower()
        self.rage_opening_count = int(cfg.get("rage_opening_count", 2))
        self.rage_core_delay_sec = float(cfg.get("rage_core_delay_sec", 10.0))
        self.hero_ability_delay_sec = float(cfg.get("hero_ability_delay_sec", 2.0))
        self._last_detections = []

    def compute_north_triangle_drops(
        self,
        screen=None,
        only_anchors: bool = False,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Calcula puntos de despliegue legal en el césped exterior norte,
        altos y alejados de la línea roja construible a ambos lados del vértice norte.
        Con zoom moderado (mitad disponible), la superficie de césped exterior es enorme.
        Si only_anchors=True, valida únicamente los 2 puntos ancla (nw_drops[1] y ne_drops[1])
        ahorrando búsquedas y cómputo de IA innecesarios para puntos que no se despliegan.
        """
        safe_top_y = 145
        if screen is not None:
            try:
                from vision.yolo_detector import get_detector
                detector = get_detector()
                if detector and detector.is_ready():
                    dets = detector.detect(screen)
                    if dets:
                        self._last_detections = dets
                        min_b_y = min(d.bbox_xyxy[1] for d in dets)
                        # Margen seguro por encima del edificio más alto, nunca menor a 140 (área activa de pantalla)
                        safe_top_y = max(140, min(175, min_b_y - 80))
            except Exception as e:
                log.debug("YOLO safe drop check omitido: %s", e)

        # Flanco Noroeste: césped exterior amplio y seguro a la izquierda
        nw_drops = [
            (1020, safe_top_y - 5),
            (945, safe_top_y + 10),
            (880, safe_top_y + 35),
            (810, safe_top_y + 65),
            (740, safe_top_y + 95),
        ]
        # Flanco Noreste: césped exterior amplio y seguro a la derecha
        ne_drops = [
            (1380, safe_top_y),
            (1455, safe_top_y + 15),
            (1520, safe_top_y + 45),
            (1590, safe_top_y + 75),
            (1660, safe_top_y + 105),
        ]

        # Validación y ajuste fino vía DeployZoneDetector (IA U-Net)
        if screen is not None:
            try:
                from vision.deploy_zone_detector import get_deploy_zone_detector
                dz = get_deploy_zone_detector()
                if dz and dz.is_ready():
                    mask = dz.predict_mask(screen)
                    if mask is not None:
                        if only_anchors:
                            # Validar únicamente los 2 puntos ancla de ataque
                            pt_nw = nw_drops[1]
                            if not dz.is_point_deployable(pt_nw[0], pt_nw[1], mask=mask, safety_margin_px=15):
                                nw_drops[1] = dz.find_closest_deployable_point(pt_nw[0], pt_nw[1], margin_px=25, mask=mask)
                            pt_ne = ne_drops[1]
                            if not dz.is_point_deployable(pt_ne[0], pt_ne[1], mask=mask, safety_margin_px=15):
                                ne_drops[1] = dz.find_closest_deployable_point(pt_ne[0], pt_ne[1], margin_px=25, mask=mask)
                        else:
                            nw_drops = [
                                dz.find_closest_deployable_point(pt[0], pt[1], margin_px=25, mask=mask)
                                if not dz.is_point_deployable(pt[0], pt[1], mask=mask, safety_margin_px=15)
                                else pt
                                for pt in nw_drops
                            ]
                            ne_drops = [
                                dz.find_closest_deployable_point(pt[0], pt[1], margin_px=25, mask=mask)
                                if not dz.is_point_deployable(pt[0], pt[1], mask=mask, safety_margin_px=15)
                                else pt
                                for pt in ne_drops
                            ]
                        log.info("🛡️ DeployZoneDetector validó puntos del Triángulo Norte (IA activa)")
            except Exception as e:
                log.debug("DeployZoneDetector en EDrag check omitido: %s", e)

        nw_drops = [(int(np.clip(pt[0], 120, 2280)), int(np.clip(pt[1], 140, 845))) for pt in nw_drops]
        ne_drops = [(int(np.clip(pt[0], 120, 2280)), int(np.clip(pt[1], 140, 845))) for pt in ne_drops]

        return nw_drops, ne_drops

    def get_opening_rage_points(self) -> list[tuple[int, int]]:
        """
        Puntos de impacto para los Hechizos de Furia de apertura en el cono norte.
        Cubren la entrada de los dragones desde ambos flancos sobre las primeras defensas.
        """
        return [
            (1100, 260),  # Flanco izquierdo del cono norte
            (1300, 260),  # Flanco derecho del cono norte
            (1200, 200),  # Vértice Norte central (si hay 3 furias)
        ]

    def get_core_rage_points(self, detections: list | None = None) -> list[tuple[int, int]]:
        """
        Puntos de impacto para las Furias escalonadas hacia el núcleo / Ayuntamiento
        tras el avance de los dragones y héroes.
        """
        target_center = None
        if detections:
            # 1. Buscar town_hall en la mitad norte/central (300 < y < 650)
            th_candidates = [
                d for d in detections
                if getattr(d, "cls_name", "") == "town_hall" and 300 < getattr(d, "center", (0, 0))[1] < 650
            ]
            if th_candidates:
                target_center = th_candidates[0].center
            else:
                # 2. Buscar centroide de almacenes en 300 < y < 650
                storage_candidates = [
                    d for d in detections
                    if "storage" in getattr(d, "cls_name", "") and 300 < getattr(d, "center", (0, 0))[1] < 650
                ]
                if storage_candidates:
                    avg_x = sum(d.center[0] for d in storage_candidates) / len(storage_candidates)
                    avg_y = sum(d.center[1] for d in storage_candidates) / len(storage_candidates)
                    target_center = (avg_x, avg_y)

        if target_center is not None:
            c_x = int(np.clip(target_center[0], 950, 1450))
            c_y = int(np.clip(target_center[1], 320, 620))
            left_x = int(np.clip(c_x - 110, 120, self.w - 120))
            left_y = int(np.clip(c_y - 50, 310, 640))
            right_x = int(np.clip(c_x + 110, 120, self.w - 120))
            right_y = int(np.clip(c_y - 50, 310, 640))
            return [
                (c_x, c_y),
                (left_x, left_y),
                (right_x, right_y),
            ]

        return [
            (1200, 420),  # Ayuntamiento / Núcleo central
            (1100, 360),  # Compartimento interior izquierdo
            (1300, 360),  # Compartimento interior derecho
        ]

    def _tap_jitter(self, x: int, y: int):
        jx, jy = HumanBehavior.jitter_point(x, y, max_offset=6)
        jx = max(10, min(self.w - 10, jx))
        jy = max(10, min(self.h - 10, jy))
        self.device.tap(jx, jy)

    def execute_attack(self) -> bool:
        """
        Ejecuta la secuencia completa de ataque EDrag Rush por el Triángulo Norte:
        1. Zoom out y Paneo decisivo hacia el Norte.
        2. Furias de apertura en el cono norte.
        3. 11 Dragones Eléctricos en ambos lados a la vez (césped exterior alto y alejado).
        4. 4 Héroes juntos en césped exterior seguro (RC, Prince, Warden, Duke Dragon).
        5. Activación de habilidades de héroes.
        6. Furias de avance al núcleo.
        """
        log.info("🐉 ════════════════════════════════════════════════════════════")
        log.info("🐉 INICIANDO ESTRATEGIA DE ATAQUE: DRAGONES ELÉCTRICOS + FURIAS")
        log.info("🐉 Objetivo: 50%+ de Destrucción y Captura del Bonus de Liga")
        log.info("🐉 ════════════════════════════════════════════════════════════")

        # 1. Asegurar encuadre de combate óptimo según zoom_mode
        if self.zoom_mode == "half":
            log.info("🔍 Aplicando Zoom Moderado ('mitad disponible') para mayor superficie de césped...")
            if hasattr(self.device, "zoom_half"):
                self.device.zoom_half()
            else:
                self.device.zoom_out()
            log.info("🔍 Paneo de cámara hacia el sector Norte para encuadrar césped exterior amplio...")
            self.device.pan_camera(0.50, 0.35, 0.50, 0.70, duration=0.65)
        elif self.zoom_mode == "none":
            log.info("🔍 Manteniendo zoom nativo de batalla...")
            self.device.wait_camera_settled(timeout=1.0)
            log.info("🔍 Paneo de cámara suave hacia el sector Norte...")
            self.device.pan_camera(0.50, 0.30, 0.50, 0.60, duration=0.55)
        else:
            log.info("🔍 Ejecutando Zoom Out completo...")
            self.device.zoom_out()
            log.info("🔍 Paneo de cámara hacia el sector Norte...")
            self.device.pan_camera(0.50, 0.20, 0.50, 0.82, duration=0.85)

        self.device.wait_camera_settled()

        screen = self.device.screenshot()
        if screen is None:
            log.error("❌ No se pudo capturar la pantalla para el ataque.")
            return False

        # Guardar captura inicial del ataque tras el paneo de forma asíncrona
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_debug_image_async(f"logs/yolo/edrag_attack_start_{ts}.png", screen)

        # 3. Localizar tropas, hechizos y héroes en la barra de batalla (escaneo único)
        slots = self.army.scan_battle_bar(screen)
        edrag_pos = slots.edrag
        if edrag_pos is None and hasattr(self.army, "find_edrag_in_battle_bar"):
            edrag_pos = self.army.find_edrag_in_battle_bar(screen)

        rage_pos = slots.rage
        if rage_pos is None and hasattr(self.army, "find_rage_spell_in_battle_bar"):
            rage_pos = self.army.find_rage_spell_in_battle_bar(screen)

        heroes = slots.heroes
        if not heroes and hasattr(self.army, "find_all_heroes_in_battle_bar"):
            heroes = self.army.find_all_heroes_in_battle_bar(screen)

        log.info("📦 Reconocimiento de barra de batalla:")
        log.info("   • Dragones Eléctricos : %s", edrag_pos)
        log.info("   • Hechizo de Furia    : %s", rage_pos)
        log.info("   • Héroes detectados   : %s", list(heroes.keys()))

        if not edrag_pos:
            log.warning("⚠️ No se detectó ranura dinámica de E-Drag. Usando ranura estándar en (407, 955)...")
            edrag_pos = (407, 955)

        if not rage_pos:
            log.warning("⚠️ No se detectó ranura dinámica de Furia. Usando ranura estándar en (1345, 960)...")
            rage_pos = (1345, 960)

        nw_drops, ne_drops = self.compute_north_triangle_drops(screen, only_anchors=True)

        # ─────────────────────────────────────────────────────────────
        # FASE 1: HECHIZOS DE FURIA DE APERTURA (Escalonado Táctico)
        # ─────────────────────────────────────────────────────────────
        if rage_pos:
            log.info("✨ Seleccionando Hechizo de Furia en %s...", rage_pos)
            self.device.tap(*rage_pos)
            HumanBehavior.reaction_delay()

            opening_rages = self.get_opening_rage_points()[:self.rage_opening_count]
            for idx_r, r_pt in enumerate(opening_rages, 1):
                log.info("⚡ [APERTURA] Lanzando Furia #%d en el Triángulo Norte: %s", idx_r, r_pt)
                self._tap_jitter(r_pt[0], r_pt[1])
                time.sleep(0.18)
        else:
            log.warning("⚠️ No se detectó Hechizo de Furia en la barra.")

        # ─────────────────────────────────────────────────────────────
        # FASE 2: DESPLIEGUE EN 2 TAPS SOSTENIDOS SEPARADOS (DRAGONES ELÉCTRICOS)
        # ─────────────────────────────────────────────────────────────
        log.info("🐉 Seleccionando Dragones Eléctricos en %s...", edrag_pos)
        self.device.tap(*edrag_pos)
        HumanBehavior.reaction_delay()

        # Puntos de despliegue sostenido: 1 en flanco Noroeste y 1 en flanco Noreste
        # Ambos en césped exterior amplio y seguro, altos y alejados de la línea roja
        pt_nw = nw_drops[1]
        pt_ne = ne_drops[1]

        log.info("🌊 [1/2] Despliegue sostenido de Dragones Eléctricos en Flanco Noroeste: %s (sosteniendo ~0.9s)...", pt_nw)
        if hasattr(self.device, "hold_tap"):
            self.device.hold_tap(pt_nw[0], pt_nw[1], duration=0.90)
        else:
            self._tap_jitter(pt_nw[0], pt_nw[1])
        time.sleep(0.12)

        log.info("🌊 [2/2] Despliegue sostenido de Dragones Eléctricos en Flanco Noreste: %s (sosteniendo ~0.9s)...", pt_ne)
        if hasattr(self.device, "hold_tap"):
            self.device.hold_tap(pt_ne[0], pt_ne[1], duration=0.90)
        else:
            self._tap_jitter(pt_ne[0], pt_ne[1])
        time.sleep(0.12)

        log.info("✅ Desplegados Dragones Eléctricos en 2 tiradas sostenidas continuas con éxito.")

        # ─────────────────────────────────────────────────────────────
        # FASE 3: DESPLIEGUE UNIVERSAL DE TODOS LOS HÉROES DETECTADOS EN EL PUNTO ANCLA
        # ─────────────────────────────────────────────────────────────
        hero_drop_anchor = pt_nw
        deployed_heroes = []

        if heroes:
            log.info("👑 Desplegando %d Héroes detectados secuencialmente en el punto ancla de dragones %s:", len(heroes), hero_drop_anchor)
            for idx_h, (h_id, h_slot) in enumerate(heroes.items(), 1):
                log.info("   • [%d/%d] Seleccionando héroe %s en %s...", idx_h, len(heroes), h_id.upper(), h_slot)
                self.device.tap(*h_slot, humanize=False)
                time.sleep(0.08)
                log.info("     -> Soltando %s en punto ancla de dragones: %s", h_id.upper(), hero_drop_anchor)
                self.device.tap(hero_drop_anchor[0], hero_drop_anchor[1], humanize=False)
                deployed_heroes.append((h_id, h_slot))
                time.sleep(0.10)
        else:
            log.info("ℹ️ No se detectaron héroes disponibles en la barra de batalla (agotados o no llevados).")

        # ─────────────────────────────────────────────────────────────
        # FASE 4: ACTIVACIÓN RÁPIDA DE HABILIDADES DE HÉROES
        # ─────────────────────────────────────────────────────────────
        if deployed_heroes:
            log.info("⏳ Esperando %.1fs para que los héroes tomen agro y penetren defensas...", self.hero_ability_delay_sec)
            time.sleep(self.hero_ability_delay_sec)

            log.info("⚡ Activando habilidades especiales de todos los héroes desplegados mediante toques en la barra...")
            for h_id, h_slot in deployed_heroes:
                log.info("   💥 Disparando habilidad especial de %s en %s!", h_id.upper(), h_slot)
                self.device.tap(*h_slot)
                time.sleep(0.10)

        # ─────────────────────────────────────────────────────────────
        # FASE 5: FURIAS DE AVANCE AL NÚCLEO / AYUNTAMIENTO (tras 10s)
        # ─────────────────────────────────────────────────────────────
        log.info("⏳ Esperando %.1fs de combate antes de soltar las Furias de avance al núcleo...", self.rage_core_delay_sec)
        time.sleep(self.rage_core_delay_sec)

        if rage_pos:
            log.info("✨ Re-seleccionando Hechizo de Furia para el avance central...")
            self.device.tap(*rage_pos)
            HumanBehavior.reaction_delay()

            core_rages = self.get_core_rage_points(getattr(self, "_last_detections", None))
            for idx_c, c_pt in enumerate(core_rages, 1):
                log.info("⚡ [NÚCLEO] Lanzando Furia de Avance #%d en: %s", idx_c, c_pt)
                self._tap_jitter(c_pt[0], c_pt[1])
                time.sleep(0.20)

        # Guardar captura final del despliegue completado de forma asíncrona
        s_deployed = self.device.screenshot()
        if s_deployed is not None:
            save_debug_image_async(f"logs/yolo/edrag_attack_deployed_{ts}.png", s_deployed)

        log.info("🚀 Despliegue de Dragones Eléctricos, Furias y Héroes completado con éxito.")
        return True
