"""
core/vision.py
Motor de visión por computadora optimizado con ROI dinámico (porcentajes) y caché RAM.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image

from core.config import PROJECT_ROOT
from core.logger import get_logger

log = get_logger("vision")

_DEBUG_IMG_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="debug_img_saver")


def save_debug_image_async(path: str, img: np.ndarray, sync: bool = False) -> None:
    """Guarda imágenes de depuración y auditoría de forma asíncrona para no bloquear el hilo principal."""
    if img is None or getattr(img, "size", 0) == 0:
        return
    is_testing = "unittest" in sys.modules
    img_copy = img.copy()

    def _write():
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            cv2.imwrite(path, img_copy)
        except Exception as e:
            log.debug("Error guardando imagen debug %s: %s", path, e)

    if sync or is_testing:
        _write()
    else:
        _DEBUG_IMG_EXECUTOR.submit(_write)


class Vision:
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self._template_cache: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}

    def _load_template(self, template_path: str) -> tuple[np.ndarray, tuple[int, int]] | None:
        if template_path in self._template_cache:
            return self._template_cache[template_path]

        ruta_absoluta = os.path.join(PROJECT_ROOT, template_path)
        if not os.path.exists(ruta_absoluta):
            log.debug("Template no encontrado: %s", ruta_absoluta)
            return None

        template = cv2.imread(ruta_absoluta, cv2.IMREAD_GRAYSCALE)
        if template is None:
            log.debug("No se pudo decodificar: %s", ruta_absoluta)
            return None

        h, w = template.shape
        self._template_cache[template_path] = (template, (h, w))
        return self._template_cache[template_path]

    def preload_templates(self, template_dir: str = "assets"):
        ruta_abs = os.path.join(PROJECT_ROOT, template_dir)
        if not os.path.isdir(ruta_abs):
            return
        cargados = 0
        for root, dirs, files in os.walk(ruta_abs):
            dirs[:] = [d for d in dirs if not d.startswith(("_", "."))]
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    abs_file = os.path.join(root, fname)
                    rel_path = os.path.relpath(abs_file, PROJECT_ROOT).replace("\\", "/")
                    if self._load_template(rel_path):
                        cargados += 1
        log.info("📦 Precargados %d templates en caché RAM.", cargados)

    def clear_cache(self):
        self._template_cache.clear()

    @staticmethod
    def to_gray(image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return image
            elif image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        raise TypeError(f"Formato de imagen no soportado: {type(image)}")

    @staticmethod
    def _crop_zone(screen_gray: np.ndarray, zona: str) -> tuple[np.ndarray, int, int]:
        h, w = screen_gray.shape[:2]

        zonas_calc = {
            "COMPLETA": (0, h, 0, w),
            "CENTRO": (int(h * 0.15), int(h * 0.85), int(w * 0.20), int(w * 0.80)),
            "MITAD_INFERIOR": (int(h * 0.45), h, 0, w),
            "MITAD_SUPERIOR": (0, int(h * 0.55), 0, w),
            "MITAD_DERECHA": (0, h, int(w * 0.45), w),
            "MITAD_IZQUIERDA": (0, h, 0, int(w * 0.55)),
            "ESQUINA_INF_DER": (int(h * 0.45), h, int(w * 0.45), w),
            "ESQUINA_INF_IZQ": (int(h * 0.45), h, 0, int(w * 0.55)),
            "ESQUINA_SUP_DER": (0, int(h * 0.35), int(w * 0.50), w),
            "ESQUINA_SUP_IZQ": (0, int(h * 0.35), 0, int(w * 0.50)),
        }

        if not zona or zona not in zonas_calc:
            return screen_gray, 0, 0

        y1, y2, x1, x2 = zonas_calc[zona]
        return screen_gray[y1:y2, x1:x2], x1, y1

    def find_template(
        self,
        screenshot,
        template_path: str,
        threshold: float | None = None,
        zona: str = "COMPLETA"
    ) -> tuple[int, int] | None:
        threshold = threshold if threshold is not None else self.threshold
        cached = self._load_template(template_path)
        if cached is None:
            return None

        template_gray, (h, w) = cached
        screen_gray = self.to_gray(screenshot)

        search_area, offset_x, offset_y = self._crop_zone(screen_gray, zona)

        if search_area.shape[0] >= h and search_area.shape[1] >= w:
            res = cv2.matchTemplate(search_area, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
                cx = max_loc[0] + offset_x + w // 2
                cy = max_loc[1] + offset_y + h // 2
                return (cx, cy)

        return None

    def find_template_cached(
        self,
        screenshot,
        template_path: str,
        cache_name: str,
        ui_cache=None,
        threshold: float | None = None,
        zona: str = "COMPLETA",
    ) -> tuple[int, int] | None:
        """
        Intenta micro-verificación rápida (<1ms) si existe en ui_cache.
        Si no se detecta en la micro-ROI, recurre a find_template normal
        y auto-actualiza la posición en el caché.
        """
        threshold = threshold if threshold is not None else self.threshold
        if ui_cache is not None and getattr(ui_cache, "enabled", True) and getattr(ui_cache, "use_micro_verify", True):
            cached = self._load_template(template_path)
            if cached is not None:
                tpl_gray, _ = cached
                screen_gray = self.to_gray(screenshot)
                pos = ui_cache.micro_verify(screen_gray, cache_name, tpl_gray, threshold=threshold)
                if pos:
                    return pos

        pos = self.find_template(screenshot, template_path, threshold=threshold, zona=zona)
        if pos and ui_cache is not None and getattr(ui_cache, "enabled", True) and getattr(ui_cache, "learn_on_detect", True):
            h, w = (screenshot.shape[:2] if isinstance(screenshot, np.ndarray) else (1080, 2400))
            ui_cache.update_pos(cache_name, pos[0], pos[1], width=w, height=h)
        return pos

    def find_any(
        self,
        screenshot,
        template_paths: list[str],
        threshold: float | None = None,
        zona: str = "COMPLETA"
    ) -> tuple[str, int, int] | None:
        threshold = threshold if threshold is not None else self.threshold
        screen_gray = self.to_gray(screenshot)

        for path in template_paths:
            pos = self.find_template(screen_gray, path, threshold=threshold, zona=zona)
            if pos:
                return (path, pos[0], pos[1])
        return None

    def wait_for_template(
        self,
        screenshot_func,
        template_path: str,
        timeout: int = 12,
        threshold: float | None = None,
        interval: float = 0.25,
        zona: str = "COMPLETA",
        cache_name: str | None = None,
        ui_cache=None,
    ) -> tuple[int, int] | None:
        threshold = threshold if threshold is not None else self.threshold
        inicio = time.monotonic()

        while time.monotonic() - inicio < timeout:
            try:
                screen = screenshot_func()
                if cache_name and ui_cache:
                    pos = self.find_template_cached(
                        screen, template_path, cache_name, ui_cache=ui_cache, threshold=threshold, zona=zona
                    )
                else:
                    pos = self.find_template(screen, template_path, threshold, zona=zona)
                if pos:
                    return pos
            except Exception as e:
                log.warning("Error en wait_for_template: %s", e)
            time.sleep(interval)

        return None