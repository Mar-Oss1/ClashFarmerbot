"""
vision/resource_reader.py
Lector de recursos robusto con binarización adaptativa, banco de escalas precalculado,
validación de grupos de miles y registro de fallos en logs/fails/.
"""
import os
import re
import time
import cv2
import numpy as np
from core.config import SETTINGS, PROJECT_ROOT
from core.vision import Vision, save_debug_image_async
from core.logger import get_logger

log = get_logger("resource_reader")

SCALES = (0.78, 0.82, 0.86, 0.90, 0.94, 0.98, 1.0, 1.08)


class ResourceReader:
    def __init__(self, vision: Vision):
        self.vision = vision
        self.digit_templates = {d: f"assets/digits/{d}.png" for d in range(10)}
        self.enemy_rois = SETTINGS["rois"]["enemy_loot"]
        self.own_rois = SETTINGS["rois"]["own_resources"]
        self._bank: list[tuple[int, np.ndarray]] | None = None
        self._bank_by_scale: dict[float, list[tuple[int, np.ndarray]]] = {}
        self._preferred_scale: dict[str, float] = {}
        self.last_tooltip_box: tuple[int, int, int, int] | None = None

    @classmethod
    def _binarize_digits(cls, img: np.ndarray, adaptive: bool = False) -> np.ndarray:
        if img.ndim == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            white_mask = (hsv[:, :, 1] <= 65) & (hsv[:, :, 2] >= 185)
            yellow_mask = (
                (hsv[:, :, 0] >= 14)
                & (hsv[:, :, 0] <= 40)
                & (hsv[:, :, 1] >= 75)
                & (hsv[:, :, 2] >= 185)
            )
            return np.where(white_mask | yellow_mask, 255, 0).astype(np.uint8)

        # Si ya es escala de grises (2D)
        thr = 190
        if adaptive and img.size > 0:
            peak = float(np.percentile(img, 99))
            thr = int(np.clip(peak * 0.85, 150, 200))
        _, binary = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
        return binary

    # Alias para compatibilidad con código y tests previos
    _binarize_white_digits = _binarize_digits

    @classmethod
    def _prepare_scaled_template(cls, tpl_gray: np.ndarray, scale: float) -> np.ndarray | None:
        if scale != 1.0:
            th = max(1, int(round(tpl_gray.shape[0] * scale)))
            tw = max(1, int(round(tpl_gray.shape[1] * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            tpl_gray = cv2.resize(tpl_gray, (tw, th), interpolation=interp)

        tpl_bin = cls._binarize_white_digits(tpl_gray)
        coords = cv2.findNonZero(tpl_bin)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        cropped = tpl_bin[y:y + h, x:x + w]
        return cv2.copyMakeBorder(cropped, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    def _populate_banks(self) -> None:
        if self._bank_by_scale:
            return
        self._bank_by_scale = {s: [] for s in SCALES}
        self._bank = []
        for digit, path in self.digit_templates.items():
            cached = self.vision._load_template(path)
            if cached is None:
                continue
            tpl_gray, _ = cached
            for s in SCALES:
                t = self._prepare_scaled_template(tpl_gray, s)
                if t is not None:
                    self._bank_by_scale[s].append((digit, t))
                    self._bank.append((digit, t))

    def _get_bank(self) -> list[tuple[int, np.ndarray]]:
        if self._bank is None:
            self._populate_banks()
        return self._bank

    def _get_bank_for_scale(self, scale: float) -> list[tuple[int, np.ndarray]]:
        if not self._bank_by_scale:
            self._populate_banks()
        return self._bank_by_scale.get(scale, [])

    def _candidates(
        self,
        roi_bin: np.ndarray,
        floor: float = 0.65,
        bank: list[tuple[int, np.ndarray]] | None = None,
    ) -> list[tuple[int, float, int, int]]:
        rh, rw = roi_bin.shape[:2]
        out: list[tuple[int, float, int, int]] = []
        active_bank = bank if bank is not None else self._get_bank()
        for digit, tpl in active_bank:
            th, tw = tpl.shape[:2]
            if th >= rh or tw >= rw or th < 8 or tw < 4:
                continue
            res = cv2.matchTemplate(roi_bin, tpl, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= floor)
            out.extend([(int(x), float(res[y, x]), digit, tw) for y, x in zip(ys, xs)])
        return out

    @staticmethod
    def _nms(cands: list[tuple[int, float, int, int]], thr: float) -> list[tuple[int, int, int, float]]:
        kept: list[tuple[int, int, int, float]] = []  # (x, digit, tw, score)
        for x, score, digit, tw in sorted(cands, key=lambda c: c[1], reverse=True):
            if score < thr:
                break
            if all(max(0, min(x + tw, kx + ktw) - max(x, kx)) / min(tw, ktw) <= 0.35 for kx, _, ktw, _ in kept):
                kept.append((x, digit, tw, score))
        return sorted(kept, key=lambda k: k[0])

    @staticmethod
    def _truncate_trailing_detections(
        dets: list[tuple[int, int, int, float]],
    ) -> list[tuple[int, int, int, float]]:
        """
        Descarta dígitos espurios a la derecha (e.g. el '2' de '2x' tras el icono de pastelito)
        si hay un hueco horizontal > 1.8 * med_w entre dígitos consecutivos.
        dets está ordenado por coordenada x: (x, digit, tw, score).
        """
        if len(dets) <= 1:
            return dets
        med_w = float(np.median([d[2] for d in dets]))
        cutoff = 1.8 * med_w
        truncated = [dets[0]]
        for a, b in zip(dets, dets[1:]):
            gap = b[0] - (a[0] + a[2])
            if gap > cutoff:
                break
            truncated.append(b)
        return truncated

    @staticmethod
    def _valid_grouping(dets: list[tuple[int, int, int, float]]) -> bool:
        """Valida que todos los grupos de miles salvo el primero tengan exactamente 3 dígitos."""
        if not dets:
            return False
        if len(dets) == 1:
            return True
        med_w = float(np.median([d[2] for d in dets]))
        # Separador de miles en fuente Clash of Clans es de 7-11px; gaps entre dígitos son de -1 a 2px.
        sep_thresh = max(5.0, 0.25 * med_w)
        groups = [1]
        for a, b in zip(dets, dets[1:]):
            gap = b[0] - (a[0] + a[2])
            if gap > sep_thresh:  # Separador de miles
                groups.append(1)
            else:
                groups[-1] += 1
        return 1 <= groups[0] <= 3 and all(g == 3 for g in groups[1:])

    def _dump_fail(self, roi_img: np.ndarray, roi_bin: np.ndarray, key: str):
        fail_dir = os.path.join(PROJECT_ROOT, "logs", "fails")
        os.makedirs(fail_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        save_debug_image_async(os.path.join(fail_dir, f"{ts}_{key}_raw.png"), roi_img)
        save_debug_image_async(os.path.join(fail_dir, f"{ts}_{key}_bin.png"), roi_bin)

        # Rotación simple: no superar 200 archivos en logs/fails/
        try:
            files = [
                os.path.join(fail_dir, f)
                for f in os.listdir(fail_dir)
                if f.endswith(".png")
            ]
            if len(files) > 200:
                files.sort(key=os.path.getmtime)
                for old_f in files[: len(files) - 200]:
                    try:
                        os.remove(old_f)
                    except OSError:
                        pass
        except Exception as e_rot:
            log.debug("Error en rotación de logs/fails/: %s", e_rot)

    def _read_number_from_roi(self, roi_img: np.ndarray, key: str = "") -> int | None:
        roi_bin = self._binarize_digits(roi_img, adaptive=True)
        roi_bin = cv2.copyMakeBorder(roi_bin, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

        cat = "general"
        if "tooltip" in key:
            cat = "tooltip"
        elif "enemy" in key:
            cat = "enemy"
        elif "own" in key:
            cat = "own"

        # 1. Probar primero la escala preferida si ya se conoce para esta categoría
        pref_scale = self._preferred_scale.get(cat)
        if pref_scale is not None:
            cands_pref = self._candidates(roi_bin, floor=0.74, bank=self._get_bank_for_scale(pref_scale))
            dets_pref = self._nms(cands_pref, thr=0.74)
            dets_pref = self._truncate_trailing_detections(dets_pref)
            if self._valid_grouping(dets_pref):
                return int("".join(str(d[1]) for d in dets_pref))

        # 2. Fallback: probar todas las escalas y registrar la escala ganadora
        best_scale = None
        best_val = None
        best_metric = (-1, -1.0)

        for s in SCALES:
            cands_s = self._candidates(roi_bin, floor=0.65, bank=self._get_bank_for_scale(s))
            if not cands_s:
                continue
            for thr in (0.80, 0.74, 0.68, 0.65):
                dets_s = self._nms(cands_s, thr)
                dets_s = self._truncate_trailing_detections(dets_s)
                if self._valid_grouping(dets_s):
                    avg_score = float(np.mean([d[3] for d in dets_s]))
                    metric = (len(dets_s), avg_score)
                    if metric > best_metric:
                        best_metric = metric
                        best_scale = s
                        best_val = int("".join(str(d[1]) for d in dets_s))
                    break

        if best_val is not None:
            if best_scale is not None:
                self._preferred_scale[cat] = best_scale
            return best_val

        # 3. Fallback global sobre el banco combinado por si se requieren templates múltiples
        cands_global = self._candidates(roi_bin, floor=0.65)
        best_global_metric = (-1, -1.0)
        best_global_val = None
        for thr in (0.80, 0.74, 0.68, 0.65):
            dets = self._nms(cands_global, thr)
            dets = self._truncate_trailing_detections(dets)
            if self._valid_grouping(dets):
                avg_score = float(np.mean([d[3] for d in dets]))
                metric = (len(dets), avg_score)
                if metric > best_global_metric:
                    best_global_metric = metric
                    best_global_val = int("".join(str(d[1]) for d in dets))
        if best_global_val is not None:
            return best_global_val

        self._dump_fail(roi_img, roi_bin, key)
        return None

    def _extract_rois(self, screenshot, roi_map: dict, category: str = "") -> dict[str, int | None]:
        if isinstance(screenshot, np.ndarray) and screenshot.ndim in (2, 3):
            img = screenshot
        else:
            img = self.vision.to_gray(screenshot)

        h, w = img.shape[:2]
        result: dict[str, int | None] = {}

        for key, (ry1, ry2, rx1, rx2) in roi_map.items():
            roi = img[int(h * ry1):int(h * ry2), int(w * rx1):int(w * rx2)]
            composite_key = f"{category}_{key}" if category else key
            result[key] = self._read_number_from_roi(roi, composite_key)
        return result

        for key, (ry1, ry2, rx1, rx2) in roi_map.items():
            roi = screen_gray[int(h * ry1):int(h * ry2), int(w * rx1):int(w * rx2)]
            composite_key = f"{category}_{key}" if category else key
            result[key] = self._read_number_from_roi(roi, composite_key)
        return result

    @staticmethod
    def _format_resource(val: int | None) -> str:
        return f"{val:,}" if val is not None else "??"

    def read_enemy_loot(self, screenshot) -> dict[str, int | None]:
        loot = self._extract_rois(screenshot, self.enemy_rois, category="enemy")
        log.info(
            "🔎 Botín Enemigo -> Oro: %s | Elixir: %s | Oscuro: %s",
            self._format_resource(loot.get("gold")),
            self._format_resource(loot.get("elixir")),
            self._format_resource(loot.get("dark_elixir"))
        )
        return loot

    def read_enemy_loot_stable(self, screenshot_fn, needed: int = 2, timeout: float = 4.0, interval: float = 0.2) -> dict[str, int] | None:
        """
        Espera a que la lectura sea consistente en 2 capturas consecutivas
        para evitar lecturas durante la animación de conteo inicial.
        """
        t0 = time.monotonic()
        last = None
        streak = 0

        while time.monotonic() - t0 < timeout:
            loot = self._extract_rois(screenshot_fn(), self.enemy_rois, category="enemy")
            if all(v is not None for v in loot.values()):
                if loot == last:
                    streak += 1
                    if streak >= needed:
                        log.info(
                            "🔎 Botín Estable -> Oro: %s | Elixir: %s | Oscuro: %s",
                            f"{loot['gold']:,}", f"{loot['elixir']:,}", f"{loot['dark_elixir']:,}"
                        )
                        return loot  # type: ignore[return-value]
                else:
                    last = loot
                    streak = 1
            time.sleep(interval)

        log.warning("⚠️ No se obtuvo lectura estable del botín tras %.1fs.", timeout)
        return None

    def read_own_resources(self, screenshot) -> dict[str, int]:
        rss = self._extract_rois(screenshot, self.own_rois, category="own")
        cleaned = {k: (v if v is not None else 0) for k, v in rss.items()}
        log.info(
            "🏦 Recursos Propios -> Oro: %s | Elixir: %s | Oscuro: %s",
            self._format_resource(cleaned.get("gold")),
            self._format_resource(cleaned.get("elixir")),
            self._format_resource(cleaned.get("dark_elixir"))
        )
        return cleaned

    @staticmethod
    def meets_attack_criteria(loot: dict[str, int] | None, criteria_override: dict | None = None) -> bool:
        if not loot:
            return False
        cfg = criteria_override or SETTINGS.get("loot_criteria", {})
        min_gold = cfg.get("min_gold", 800000)
        min_elixir = cfg.get("min_elixir", 800000)
        min_dark = cfg.get("min_dark_elixir", 7000)
        req = cfg.get("conditions_required", 2)
        conds = [
            (loot.get("gold") or 0) >= min_gold,
            (loot.get("elixir") or 0) >= min_elixir,
            (loot.get("dark_elixir") or 0) >= min_dark,
        ]
        met = sum(conds)
        decision = met >= req
        log.info("⚖️ Condiciones cumplidas: %d/3 (req: %d) -> %s", met, req, "⚔️ ATACAR" if decision else "⏭️ SIGUIENTE")
        return decision

    # =========================================================================
    # LECTURA DE TOOLTIP DE CAPACIDAD REAL DE RECURSOS (INGLÉS)
    # =========================================================================

    @staticmethod
    def parse_tooltip_text(text: str) -> dict[str, int | None]:
        """
        Extrae los 3 valores numéricos del texto del tooltip de recursos (en inglés):
        - Total Capacity / Storage Capacity / Max: <número>
        - Production per hour
        - In Treasury
        """
        result: dict[str, int | None] = {
            "total_capacity": None,
            "production_per_hour": None,
            "in_treasury": None,
        }
        if not text:
            return result

        def _clean_number(val: str) -> int | None:
            digits = "".join(ch for ch in val if ch.isdigit())
            return int(digits) if digits else None

        # 1. Total Capacity / Storage Capacity / Max / Capacidad máx.
        m_cap = re.search(
            r"(?:Max(?:imum)?|Total|Storage|Capacidad(?:\s*m[aá]x(?:ima)?|\s*total)?)\s*(?:Capacity)?[^\d\n]*[:\-]?\s*([0-9][0-9\s,.]*)",
            text,
            re.IGNORECASE,
        )
        if not m_cap:
            m_cap = re.search(r"(?:Capacity|Capacidad)[^\d\n]*[:\-]?\s*([0-9][0-9\s,.]*)", text, re.IGNORECASE)
        if m_cap:
            result["total_capacity"] = _clean_number(m_cap.group(1))

        # 2. Production per hour / Producción por hora
        m_prod = re.search(
            r"(?:Production|Producci[oó]n)\s*(?:per\s*hour|por\s*hora)?[^\d\n]*[:\-]?\s*([0-9][0-9\s,.]*)",
            text,
            re.IGNORECASE,
        )
        if m_prod:
            result["production_per_hour"] = _clean_number(m_prod.group(1))

        # 3. In Treasury / En tesorería / Tesoro
        m_tres = re.search(
            r"(?:(?:In\s*)?Treasury|En\s*(?:la\s*)?tesorer[ií]a|Tesorer[ií]a|Tesoro)[^\d\n]*[:\-]?\s*([0-9][0-9\s,.]*)",
            text,
            re.IGNORECASE,
        )
        if m_tres:
            result["in_treasury"] = _clean_number(m_tres.group(1))

        return result

    def get_resource_icon_tap_pos(self, resource_type: str, width: int = 2400, height: int = 1080) -> tuple[int, int]:
        """
        Calcula la coordenada absoluta (X, Y) para tocar el icono o barra del recurso
        y abrir su tooltip informativo.
        Consulta SETTINGS['rois']['resource_tooltips']['tap_coordinates'] o calcula desde own_resources.
        """
        res = resource_type.lower()
        if res == "dark":
            res = "dark_elixir"

        tooltips_cfg = SETTINGS.get("rois", {}).get("resource_tooltips", {})
        coords = tooltips_cfg.get("tap_coordinates", {}).get(res)
        if coords and len(coords) == 2:
            ry, rx = coords[0], coords[1]
            return int(rx * width), int(ry * height)

        # Fallback al centro de la ROI de recursos propios
        own_roi = self.own_rois.get(res)
        if own_roi and len(own_roi) == 4:
            ry1, ry2, rx1, rx2 = own_roi
            cy = int(height * (ry1 + ry2) / 2.0)
            cx = int(width * (rx1 + rx2) / 2.0)
            return cx, cy

        # Valores estándar por defecto para resolución 2400x1080
        default_positions = {
            "gold": (int(width * 0.88), int(height * 0.05)),
            "elixir": (int(width * 0.88), int(height * 0.115)),
            "dark_elixir": (int(width * 0.88), int(height * 0.18)),
        }
        return default_positions.get(res, (int(width * 0.88), int(height * 0.05)))

    def get_tooltip_close_pos(self, width: int = 2400, height: int = 1080) -> tuple[int, int]:
        """
        Calcula la coordenada para tocar fuera del tooltip y cerrarlo limpiamente.
        Por defecto toca en el centro de la aldea (50%, 50%).
        """
        tooltips_cfg = SETTINGS.get("rois", {}).get("resource_tooltips", {})
        coords = tooltips_cfg.get("outside_tap_coordinate")
        if coords and len(coords) == 2:
            ry, rx = coords[0], coords[1]
            return int(rx * width), int(ry * height)
        return int(width * 0.50), int(height * 0.50)

    def find_tooltip_box(
        self,
        screenshot: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int, int, int] | None:
        """
        Detecta y retorna la caja delimitadora (x_left, y_top, bw, bh) del modal de tooltip en pantalla.
        Retorna None si no se detecta con suficiente certeza.
        """
        if screenshot is None or getattr(screenshot, "size", 0) == 0:
            return None

        screen_gray = screenshot if screenshot.ndim == 2 else cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        h, w = screen_gray.shape[:2]

        if roi is not None:
            y1, y2, x1, x2 = roi
            return (x1, y1, x2 - x1, y2 - y1)

        if w >= 1000:
            sub = screen_gray[50:int(h * 0.45), int(w * 0.65):w]
            ox, oy = int(w * 0.65), 50
        else:
            sub = screen_gray
            ox, oy = 0, 0

        # 1. Búsqueda principal por contornos rectangulares Canny
        edges = cv2.Canny(sub, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        canny_candidates = []
        for c in contours:
            cx, cy, cbw, cbh = cv2.boundingRect(c)
            if 300 <= cbw <= 680 and 110 <= cbh <= 220 and 2.0 <= (cbw / cbh) <= 4.5:
                canny_candidates.append((ox + cx, oy + cy, cbw, cbh))
        if canny_candidates:
            canny_candidates.sort(key=lambda b: abs((b[2] / b[3]) - 2.8))
            return canny_candidates[0]

        # 2. Respaldo por morfología de líneas horizontales paralelas del tooltip
        _, bin_sub = cv2.threshold(sub, 170, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
        lines = cv2.morphologyEx(bin_sub, cv2.MORPH_OPEN, kernel)
        row_sums = np.sum(lines, axis=1) / 255
        y_lines = np.where(row_sums > 140)[0]

        groups = []
        for y in y_lines:
            if not groups or y > groups[-1][-1] + 5:
                groups.append([y])
            else:
                groups[-1].append(y)

        line_info = []
        for g in groups:
            mid_y = int(np.mean(g))
            xs = np.where(lines[mid_y, :] > 0)[0]
            if len(xs) > 0:
                line_info.append((oy + mid_y, ox + xs[0], ox + xs[-1]))

        if len(line_info) >= 2:
            for i in range(len(line_info) - 1):
                y_top, x_l1, x_r1 = line_info[i]
                y_bot, x_l2, x_r2 = line_info[i + 1]
                dh = y_bot - y_top
                if 130 <= dh <= 170:
                    x_left = min(x_l1, x_l2)
                    x_right = max(x_r1, x_r2)
                    bw = x_right - x_left
                    if 350 <= bw <= 650:
                        return (x_left, y_top, bw, dh)

        return None

    def _extract_tooltip_cv(
        self,
        screenshot: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> dict[str, int | None]:
        """
        Extrae los 3 valores del popup de recursos usando visión artificial y el banco
        de plantillas de dígitos nativo (_read_number_from_roi), sin depender de Tesseract:
        - Total Capacity (fila Max:)
        - Production per hour
        - In Treasury

        Utiliza detección morfológica de líneas horizontales para identificar inequívocamente
        los bordes superior e inferior del popup (separación de ~145px) y rechazar capturas
        donde el tooltip no esté presente, evitando lecturas erróneas del fondo de la aldea.
        """
        result: dict[str, int | None] = {"total_capacity": None, "production_per_hour": None, "in_treasury": None}
        if screenshot is None or getattr(screenshot, "size", 0) == 0:
            return result

        screen_gray = screenshot if screenshot.ndim == 2 else cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        h, w = screen_gray.shape[:2]

        boxes_to_try = []
        if roi is not None:
            y1, y2, x1, x2 = roi
            boxes_to_try.append((x1, y1, x2 - x1, y2 - y1))
        else:
            if w >= 1000:
                sub = screen_gray[50:int(h * 0.45), int(w * 0.65):w]
                ox, oy = int(w * 0.65), 50
            else:
                sub = screen_gray
                ox, oy = 0, 0

            # 1. Búsqueda principal por contornos rectangulares Canny (detecta el contorno exterior completo del popup)
            edges = cv2.Canny(sub, 40, 120)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            canny_candidates = []
            for c in contours:
                cx, cy, cbw, cbh = cv2.boundingRect(c)
                if 300 <= cbw <= 680 and 110 <= cbh <= 220 and 2.0 <= (cbw / cbh) <= 4.5:
                    canny_candidates.append((ox + cx, oy + cy, cbw, cbh))
            if canny_candidates:
                canny_candidates.sort(key=lambda b: abs((b[2] / b[3]) - 2.8))
                boxes_to_try.extend(canny_candidates[:3])

            # 2. Respaldo por morfología de líneas horizontales paralelas del tooltip
            _, bin_sub = cv2.threshold(sub, 170, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
            lines = cv2.morphologyEx(bin_sub, cv2.MORPH_OPEN, kernel)
            row_sums = np.sum(lines, axis=1) / 255
            y_lines = np.where(row_sums > 140)[0]

            groups = []
            for y in y_lines:
                if not groups or y > groups[-1][-1] + 5:
                    groups.append([y])
                else:
                    groups[-1].append(y)

            line_info = []
            for g in groups:
                mid_y = int(np.mean(g))
                xs = np.where(lines[mid_y, :] > 0)[0]
                if len(xs) > 0:
                    line_info.append((oy + mid_y, ox + xs[0], ox + xs[-1]))

            if len(line_info) >= 2:
                for i in range(len(line_info) - 1):
                    y_top, x_l1, x_r1 = line_info[i]
                    y_bot, x_l2, x_r2 = line_info[i + 1]
                    dh = y_bot - y_top
                    if 130 <= dh <= 170:
                        x_left = min(x_l1, x_l2)
                        x_right = max(x_r1, x_r2)
                        bw = x_right - x_left
                        if 350 <= bw <= 650:
                            boxes_to_try.append((x_left, y_top, bw, dh))
                            break

        if not boxes_to_try:
            return result

        best_result = None
        best_box = None
        for box in boxes_to_try:
            x_left, y_top, bw, bh = box
            # Slices precisos de las 3 filas numéricas dentro de la caja delimitada:
            # Fila 1 (Max: <capacidad>)
            r1 = screen_gray[y_top + int(bh * 0.08) : y_top + int(bh * 0.38), x_left + int(bw * 0.30) : x_left + bw - 10]
            # Fila 2 (Production per hour: <producción>)
            r2 = screen_gray[y_top + int(bh * 0.38) : y_top + int(bh * 0.68), x_left + int(bw * 0.40) : x_left + bw - 10]
            # Fila 3 (In treasury: <tesorería>)
            r3 = screen_gray[y_top + int(bh * 0.68) : y_top + int(bh * 0.98), x_left + int(bw * 0.30) : x_left + bw - 10]

            cand = {"total_capacity": None, "production_per_hour": None, "in_treasury": None}
            if r1.size > 0:
                cand["total_capacity"] = self._read_number_from_roi(r1, "tooltip_max_cap")
            if r2.size > 0:
                cand["production_per_hour"] = self._read_number_from_roi(r2, "tooltip_prod")
            if r3.size > 0:
                cand["in_treasury"] = self._read_number_from_roi(r3, "tooltip_treasury")

            if cand.get("total_capacity") is not None and cand["total_capacity"] >= 10000:
                self.last_tooltip_box = box
                return cand
            if cand.get("total_capacity") is not None and best_result is None:
                best_result = cand
                best_box = box

        if best_result is not None:
            self.last_tooltip_box = best_box
            return best_result
        return result

    def extract_tooltip_values_from_image(
        self,
        screenshot: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> dict[str, int | None]:
        """
        Extrae los 3 valores del tooltip de recursos:
        1. Intenta extracción robusta con computer vision nativo y banco de plantillas de dígitos.
        2. Si falla o no detecta la capacidad, intenta pytesseract si está disponible.
        """
        result = {"total_capacity": None, "production_per_hour": None, "in_treasury": None}
        if screenshot is None or getattr(screenshot, "size", 0) == 0:
            return result

        # 1. Extracción nativa por Computer Vision (sin dependencias externas)
        try:
            cv_result = self._extract_tooltip_cv(screenshot, roi=roi)
            if cv_result.get("total_capacity") is not None:
                return cv_result
        except Exception as e:
            log.debug("Excepción en extracción CV de tooltip: %s", e)

        # 2. Fallback secundario con pytesseract si está configurado y disponible
        h, w = screenshot.shape[:2]
        if roi is not None:
            y1, y2, x1, x2 = roi
            crop = screenshot[y1:y2, x1:x2]
        else:
            crop = screenshot[0:int(h * 0.40), int(w * 0.65):int(w * 0.98)]

        if crop.size == 0:
            return result

        try:
            import pytesseract
            from core.config import TESSERACT_CMD
            if os.path.exists(TESSERACT_CMD):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

            text = pytesseract.image_to_string(crop, config="--psm 6")
            parsed = self.parse_tooltip_text(text)
            if parsed.get("total_capacity") is not None:
                return parsed

            text_full = pytesseract.image_to_string(crop, config="--psm 3")
            parsed_full = self.parse_tooltip_text(text_full)
            if parsed_full.get("total_capacity") is not None:
                return parsed_full

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            _, bin_crop = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            text_bin = pytesseract.image_to_string(bin_crop, config="--psm 6")
            parsed_bin = self.parse_tooltip_text(text_bin)
            if parsed_bin.get("total_capacity") is not None:
                return parsed_bin
        except Exception as e:
            log.debug("Excepción durante fallback OCR de tooltip: %s", e)

        return result

    def read_resource_tooltip(
        self,
        device,
        resource_type: str,
        delay_sec: float | None = None,
    ) -> dict[str, int | None] | None:
        """
        Flujo de lectura interactivo:
        1. Toca la barra o icono del recurso para desplegar su tooltip informativo.
        2. Espera el renderizado del tooltip y toma captura de pantalla.
        3. Toca en el centro de la ventanita modal detectada (o fallback interior) para cerrar el popup limpiamente.
        4. Extrae y retorna los 3 valores (Total Capacity, Production per hour, In Treasury).
        """
        tooltips_cfg = SETTINGS.get("rois", {}).get("resource_tooltips", {})
        open_delay = delay_sec if delay_sec is not None else float(tooltips_cfg.get("open_delay_sec", 0.85))
        close_delay = float(tooltips_cfg.get("close_delay_sec", 0.40))

        w, h = getattr(device, "width", 2400), getattr(device, "height", 1080)
        tap_pos = self.get_resource_icon_tap_pos(resource_type, width=w, height=h)

        self.last_tooltip_box = None
        log.info("🔎 Abriendo tooltip de recurso %s tocando en %s...", resource_type.upper(), tap_pos)
        device.tap(tap_pos[0], tap_pos[1], humanize=False)
        time.sleep(open_delay)

        screen = device.screenshot()
        values = self.extract_tooltip_values_from_image(screen)

        # Si el primer frame no detectó el popup (animación lenta o lag de renderizado), ventana de gracia
        if values.get("total_capacity") is None and getattr(device, "screenshot", None) is not None:
            time.sleep(0.35)
            screen_retry = device.screenshot()
            values_retry = self.extract_tooltip_values_from_image(screen_retry)
            if values_retry.get("total_capacity") is not None:
                screen = screen_retry
                values = values_retry

        # Guardar captura de depuración y auditoría para OCR y análisis de background
        try:
            save_dir = os.path.join(PROJECT_ROOT, "logs", "tooltips")
            os.makedirs(save_dir, exist_ok=True)
            save_debug_image_async(os.path.join(save_dir, f"tooltip_{resource_type}.png"), screen)
        except Exception as e_save:
            log.debug("Error guardando captura de tooltip: %s", e_save)

        # Cerrar el tooltip tocando el centro del modal detectado (o fallback modal interior)
        modal_box = getattr(self, "last_tooltip_box", None) or self.find_tooltip_box(screen)
        if modal_box is not None:
            bx, by, bw, bh = modal_box
            close_pos = (int(bx + bw // 2), int(by + bh // 2))
            log.info("🔒 Cerrando tooltip de %s tocando el centro del modal en %s...", resource_type.upper(), close_pos)
        else:
            close_pos = (int(tap_pos[0] - 150), int(tap_pos[1] + 110))
            log.info("🔒 Cerrando tooltip de %s usando tap fallback modal en %s...", resource_type.upper(), close_pos)

        try:
            device.tap(close_pos[0], close_pos[1], humanize=False)
            time.sleep(close_delay)
        except Exception as err:
            log.warning("⚠️ Error al cerrar tooltip de %s: %s", resource_type, err)

        if values.get("total_capacity") is not None:
            log.info(
                "✅ Tooltip %s leído con éxito -> Capacidad: %s | Producción: %s | Treasury: %s",
                resource_type.upper(),
                f"{values['total_capacity']:,}",
                f"{values['production_per_hour']:,}" if values.get("production_per_hour") is not None else "N/A",
                f"{values['in_treasury']:,}" if values.get("in_treasury") is not None else "N/A",
            )
            return values

        log.warning("⚠️ No se pudo leer la capacidad del tooltip para %s (valores: %s).", resource_type, values)
        return None

    def read_all_resource_tooltips(
        self,
        device,
    ) -> dict[str, dict[str, int | None] | None]:
        """
        Lee secuencialmente los tooltips de los 3 recursos principales (Gold, Elixir, Dark Elixir).
        Abre cada tooltip y lo cierra dando click de nuevo en la misma barra antes de pasar al siguiente.
        """
        results: dict[str, dict[str, int | None] | None] = {}
        for res in ("gold", "elixir", "dark_elixir"):
            results[res] = self.read_resource_tooltip(device, res)
            time.sleep(0.4)
        return results
