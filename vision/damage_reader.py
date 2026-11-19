"""
vision/damage_reader.py
Lector de porcentaje de destrucción en batalla para Clash of Clans.
Detecta el porcentaje (Overall Damage) en la zona inferior derecha de la pantalla (2400x1080)
mediante coincidencia de plantillas de dígitos de alta velocidad (<0.5ms) sin dependencias externas.
"""
import os
import cv2
import numpy as np
from core.config import PROJECT_ROOT
from core.logger import get_logger

log = get_logger("damage_reader")

# Coordenadas de referencia para resolución 2400x1080 (horizontal)
# Zona inferior derecha encima de la cinta de tropas:
# "Overall Damage" texto: Y ∈ [700, 740], X ∈ [2010, 2310]
# Porcentaje (número + %): Y ∈ [780, 840], X ∈ [2130, 2330]
DEFAULT_ROI_DAMAGE = (2130, 780, 200, 60)  # x, y, w, h en 2400x1080
BASE_WIDTH = 2400
BASE_HEIGHT = 1080


class DamageReader:
    """Lee el porcentaje de destrucción actual durante la batalla activa."""

    def __init__(self, templates_dir: str | None = None):
        if templates_dir is None:
            templates_dir = os.path.join(PROJECT_ROOT, "assets", "digits_damage")
        self.templates_dir = templates_dir
        self.digit_templates: dict[int, np.ndarray] = {}
        self.percent_template: np.ndarray | None = None
        self._load_templates()

    def _load_templates(self):
        """Carga las plantillas binarizadas de dígitos 0-9 y el símbolo %."""
        for d in range(10):
            p = os.path.join(self.templates_dir, f"{d}.png")
            if os.path.exists(p):
                tpl = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if tpl is not None:
                    _, bin_tpl = cv2.threshold(tpl, 128, 255, cv2.THRESH_BINARY)
                    self.digit_templates[d] = bin_tpl
            else:
                log.warning("No se encontró la plantilla de daño para el dígito %d en %s", d, p)

        pct_path = os.path.join(self.templates_dir, "percent.png")
        if os.path.exists(pct_path):
            pct_img = cv2.imread(pct_path, cv2.IMREAD_GRAYSCALE)
            if pct_img is not None:
                _, bin_pct = cv2.threshold(pct_img, 128, 255, cv2.THRESH_BINARY)
                self.percent_template = bin_pct
        else:
            log.warning("No se encontró la plantilla del símbolo %% en %s", pct_path)

    def _get_roi(self, h: int, w: int) -> tuple[int, int, int, int]:
        """Calcula el ROI adaptado a la resolución actual de la pantalla."""
        rx = w / float(BASE_WIDTH)
        ry = h / float(BASE_HEIGHT)
        bx, by, bw, bh = DEFAULT_ROI_DAMAGE
        x = int(round(bx * rx))
        y = int(round(by * ry))
        width = int(round(bw * rx))
        height = int(round(bh * ry))
        # Asegurar dentro de los límites
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        width = min(w - x, width)
        height = min(h - y, height)
        return x, y, width, height

    def read_destruction_percentage(self, screen: np.ndarray | None) -> int | None:
        """
        Lee el porcentaje de destrucción (0 a 100) desde la captura de pantalla de combate.
        Retorna int si se detecta con éxito, o None si no hay HUD de daño o no se reconoce.
        """
        if screen is None or screen.size == 0:
            return None

        h, w = screen.shape[:2]
        rx, ry, rw, rh = self._get_roi(h, w)
        crop = screen[ry:ry + rh, rx:rx + rw]

        if crop.size == 0:
            return None

        # Convertir a escala de grises y binarizar dígitos blancos brillantes
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # Píxeles de texto blanco en Supercell font son > 210
        _, bin_img = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale_factor = h / float(BASE_HEIGHT)
        min_h = int(round(18 * scale_factor))
        min_w = int(round(7 * scale_factor))

        boxes = [
            cv2.boundingRect(c)
            for c in cnts
            if cv2.boundingRect(c)[3] >= min_h and cv2.boundingRect(c)[2] >= min_w
        ]
        boxes.sort(key=lambda b: b[0])

        if not boxes:
            return None

        # Si el último componente es el símbolo '%', lo removemos para clasificar los números
        if self.percent_template is not None and len(boxes) >= 1:
            lx, ly, lw, lh = boxes[-1]
            if lw >= int(round(22 * scale_factor)):
                ch = bin_img[ly:ly + lh, lx:lx + lw]
                th, tw = self.percent_template.shape
                ch_res = cv2.resize(ch, (tw, th), interpolation=cv2.INTER_NEAREST)
                pct_score = np.sum(self.percent_template == ch_res) / float(self.percent_template.size)
                if pct_score > 0.70:
                    boxes = boxes[:-1]

        if not boxes:
            return None

        # Clasificar cada caja de dígito restante de izquierda a derecha
        num_str = ""
        for bx, by, bw, bh in boxes:
            ch = bin_img[by:by + bh, bx:bx + bw]
            best_digit = None
            best_score = -1.0

            for d, tpl in self.digit_templates.items():
                th, tw = tpl.shape
                ch_res = cv2.resize(ch, (tw, th), interpolation=cv2.INTER_NEAREST)
                score = np.sum(tpl == ch_res) / float(tpl.size)
                if score > best_score:
                    best_score = score
                    best_digit = d

            if best_score >= 0.75 and best_digit is not None:
                num_str += str(best_digit)
            else:
                return None

        if num_str.isdigit():
            val = int(num_str)
            if 0 <= val <= 100:
                return val

        return None
