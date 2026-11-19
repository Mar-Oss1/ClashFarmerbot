"""
vision/storage_detector.py
Módulo de detección y reconocimiento interactivo de almacenes:
1. Escaneo pasivo por densidad de color (Oro, Elixir rosa y Elixir oscuro).
2. Verificación interactiva mediante toque de selección y lectura del rótulo flotante ("Almacén").
3. Cálculo dinámico de vectores de asalto para orientar Hechizos de Salto sobre el primer muro
   directamente alineados con los almacenes reales de la aldea rival.
"""
import time
import cv2
import numpy as np

from core.logger import get_logger
from core.device import Device
from core.vision import Vision

log = get_logger("storage_detector")


class StorageDetector:
    def __init__(self, device: Device | None = None, vision: Vision | None = None):
        self.device = device
        self.vision = vision
        self.tpl_tag_almacen = "assets/buttons/tag_almacen.png"
        self.tpl_pill_cap = "assets/buttons/pill_tag_cap.png"

    @staticmethod
    def _extract_color_masks(screen: np.ndarray):
        h, w = screen.shape[:2]
        # Área jugable de la aldea (ignora HUD de botín en esquina sup-izq y barras inferiores)
        x1, x2 = int(0.26 * w), int(0.74 * w)
        y1, y2 = int(0.20 * h), int(0.78 * h)
        roi = screen[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Máscaras adaptativas para almacenes interiores
        masks = {
            "storage_elixir": cv2.inRange(hsv, (118, 90, 70), (165, 255, 255)),
            "storage_gold": cv2.inRange(hsv, (16, 65, 65), (62, 255, 255)),
            "storage_dark": cv2.inRange(hsv, (55, 60, 20), (165, 255, 255)),
        }
        m_combined = masks["storage_elixir"] | masks["storage_gold"] | masks["storage_dark"]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(m_combined, cv2.MORPH_CLOSE, kernel)
        return roi, (x1, y1, x2, y2), masks, closed

    def detect_boxes(self, screenshot: np.ndarray, min_conf: float = 0.30) -> list[dict]:
        """
        Detección visual pasiva de cajas delimitadoras (bounding boxes) de almacenes
        a partir de segmentación por color HSV y análisis morfológico.
        Devuelve lista de dicts:
            {"cls": "storage_gold"|"storage_elixir"|"storage_dark",
             "bbox_xyxy": (x1, y1, x2, y2),
             "confidence": float}
        """
        roi, (x1, y1, x2, y2), masks, closed = self._extract_color_masks(screenshot)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results: list[dict] = []
        expected_area = 2500.0  # ~50x50 px a 2400x1080 con zoom-out

        for c in cnts:
            a = cv2.contourArea(c)
            if a < 200 or a > 35000:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            # Descartar formas extremadamente alargadas (muros o barras de interfaz)
            aspect_ratio = float(bw) / max(1.0, float(bh))
            if aspect_ratio < 0.35 or aspect_ratio > 2.85:
                continue

            # Máscara local del contorno para evaluar pureza de color
            cnt_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.drawContours(cnt_mask, [c], -1, 255, -1)

            overlaps = {
                cls_name: cv2.countNonZero(cv2.bitwise_and(m, cnt_mask))
                for cls_name, m in masks.items()
            }
            best_cls = max(overlaps, key=overlaps.get)
            max_overlap = overlaps[best_cls]

            # Descartar contornos ambiguos con poca señal de color
            if max_overlap < 25:
                continue

            # Confianza heurística basada en ratio de área esperada y densidad del color
            area_ratio = min(a / expected_area, expected_area / a)
            color_purity = max_overlap / max(1.0, float(a))
            conf = float(np.clip(0.55 * area_ratio + 0.45 * color_purity, 0.15, 0.99))

            if conf >= min_conf:
                gx1 = int(bx + x1)
                gy1 = int(by + y1)
                gx2 = int(gx1 + bw)
                gy2 = int(gy1 + bh)
                results.append({
                    "cls": best_cls,
                    "bbox_xyxy": (gx1, gy1, gx2, gy2),
                    "confidence": round(conf, 3),
                })

        return results

    def scan_storage_candidates(self, screen: np.ndarray) -> dict[str, list[tuple[int, int]]]:
        """
        Escanea la pantalla centrada y extrae los candidatos de almacenes
        clasificados por cuadrantes: 'sw', 'se', 'north', 'core'.
        """
        h, w = screen.shape[:2]
        roi, (x1, y1, x2, y2), _, closed = self._extract_color_masks(screen)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        center_x, center_y = int(0.50 * w), int(0.50 * h)

        for c in cnts:
            a = cv2.contourArea(c)
            if 200 < a < 35000:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    # Filtrar elementos fuera del perímetro de la base (minas/extractores en césped exterior)
                    dist_center = np.hypot(cx - center_x, cy - center_y)
                    if dist_center < 490:
                        candidates.append(((cx, cy), a))

        # Clasificación por cuadrantes
        center_x, center_y = int(0.50 * w), int(0.50 * h)
        core_radius = 160

        quadrants = {"sw": [], "se": [], "north": [], "core": [], "all": []}
        for (cx, cy), a in sorted(candidates, key=lambda x: x[1], reverse=True):
            dist_center = np.hypot(cx - center_x, cy - center_y)
            quadrants["all"].append((cx, cy))
            if dist_center < core_radius:
                quadrants["core"].append((cx, cy))
            elif cy >= center_y:
                if cx < center_x:
                    quadrants["sw"].append((cx, cy))
                else:
                    quadrants["se"].append((cx, cy))
            else:
                quadrants["north"].append((cx, cy))

        log.info(
            "📍 Almacenes detectados: Total=%d (SW=%d, SE=%d, Norte=%d, Núcleo=%d)",
            len(quadrants["all"]),
            len(quadrants["sw"]),
            len(quadrants["se"]),
            len(quadrants["north"]),
            len(quadrants["core"]),
        )
        return quadrants

    def verify_storage_interactively(self, candidate_pt: tuple[int, int]) -> bool:
        """
        Toca un candidato y comprueba si aparece el rótulo flotante 'Almacén'
        o la etiqueta de selección masiva sobre el edificio.
        """
        log.info("👆 Tocando candidato en %s para verificar rótulo de almacén...", candidate_pt)
        self.device.tap(*candidate_pt)
        time.sleep(0.4)

        s_tapped = self.device.screenshot()
        # 1. Comprobar texto 'Almacén'
        pos = self.vision.find_template(s_tapped, self.tpl_tag_almacen, threshold=0.68, zona="COMPLETA")
        if pos:
            log.info("✨ Rótulo 'Almacén' confirmado con éxito en %s!", pos)
            return True

        # 2. Comprobar capuchón de píldora de selección flotante
        pos_cap = self.vision.find_template(s_tapped, self.tpl_pill_cap, threshold=0.70, zona="COMPLETA")
        if pos_cap:
            log.info("✨ Etiqueta de selección confirmada en %s!", pos_cap)
            return True

        log.debug("ℹ️ Candidato en %s no mostró etiqueta de almacén conocida.", candidate_pt)
        return False

    def compute_adaptive_jump_targets(
        self, quadrants: dict[str, list[tuple[int, int]]]
    ) -> list[tuple[str, tuple[int, int]]]:
        """
        Calcula las coordenadas óptimas de los 4 Hechizos de Salto adaptadas
        a la posición real de los almacenes detectados en cada cuadrante.
        Garantiza que el Salto se posicione sobre el PRIMER MURO exterior alineado
        con el almacén objetivo de ese cuadrante.
        """
        w, h = self.device.width, self.device.height
        default_sw = (int(0.415 * w), int(0.640 * h))
        default_se = (int(0.585 * w), int(0.640 * h))
        default_north = (int(0.500 * w), int(0.315 * h))
        default_core = (int(0.500 * w), int(0.500 * h))

        targets = []
        # Solo incluir saltos en cuadrantes que tengan almacenes reales detectados
        if quadrants["sw"]:
            c_x = int(np.mean([p[0] for p in quadrants["sw"]]))
            c_y = int(np.mean([p[1] for p in quadrants["sw"]]))
            muro_x = int(np.clip(c_x - 30, int(0.36 * w), int(0.45 * w)))
            muro_y = int(np.clip(c_y + 40, int(0.60 * h), int(0.70 * h)))
            targets.append(("Primer Muro Suroeste (Adaptativo)", (muro_x, muro_y)))
        elif not quadrants["all"]:
            targets.append(("Primer Muro Suroeste (Default)", default_sw))

        if quadrants["se"]:
            c_x = int(np.mean([p[0] for p in quadrants["se"]]))
            c_y = int(np.mean([p[1] for p in quadrants["se"]]))
            muro_x = int(np.clip(c_x + 30, int(0.55 * w), int(0.64 * w)))
            muro_y = int(np.clip(c_y + 40, int(0.60 * h), int(0.70 * h)))
            targets.append(("Primer Muro Sureste (Adaptativo)", (muro_x, muro_y)))
        elif not quadrants["all"]:
            targets.append(("Primer Muro Sureste (Default)", default_se))

        if quadrants["north"]:
            c_x = int(np.mean([p[0] for p in quadrants["north"]]))
            c_y = int(np.mean([p[1] for p in quadrants["north"]]))
            muro_x = int(np.clip(c_x, int(0.42 * w), int(0.58 * w)))
            muro_y = int(np.clip(c_y - 45, int(0.28 * h), int(0.38 * h)))
            targets.append(("Primer Muro Norte (Adaptativo)", (muro_x, muro_y)))
        elif not quadrants["all"]:
            targets.append(("Primer Muro Norte (Default)", default_north))

        # Solo lanzar en Núcleo si realmente hay almacenes en el núcleo (evitar desperdiciar salto en TH sin muros)
        if quadrants["core"]:
            c_x = int(np.mean([p[0] for p in quadrants["core"]]))
            c_y = int(np.mean([p[1] for p in quadrants["core"]]))
            targets.append(("Núcleo / Almacenes Centrales", (c_x, c_y)))

        return targets
