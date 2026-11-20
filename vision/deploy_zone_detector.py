"""
vision/deploy_zone_detector.py
Detector de Zona de Despliegue (Deploy Zone Finder) para Clash of Clans.
Basado en el modelo de segmentación U-Net con encoder EfficientNet-B4
entrenado por nihatxp (clash-of-clans-vision).

Infiere con precisión sub-pixel dónde está prohibido desplegar tropas (área roja)
y dónde es legal desplegar (borde exterior y huecos desplegables).
"""
import os
import time
from typing import Optional, Tuple, List
import cv2
import numpy as np

from core.logger import get_logger
from core.config import PROJECT_ROOT, SETTINGS

log = get_logger("deploy_zone_detector")


class DeployZoneDetector:
    """
    Detector de zona de no-despliegue basado en ONNX Runtime.
    Toma capturas de batalla (2400x1080 o cualquier resolución) y produce:
    1. Máscara de no-despliegue (1 = prohibido, 0 = desplegable).
    2. Contornos del borde de despliegue legal.
    3. Validación y proyección O(1) de puntos de lanzamiento de tropas.
    """

    def __init__(
        self,
        model_path: str = "models/deploy_zone_finder.onnx",
        conf_thresh: float = 0.50,
        imgsz: int = 1280,
        use_tta: bool = False,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.use_tta = use_tta
        self.session = None
        self._ready = False
        self.last_mask: Optional[np.ndarray] = None
        self.last_prob: Optional[np.ndarray] = None
        self.last_contours: Optional[List[np.ndarray]] = None
        self._last_dist_map: Optional[np.ndarray] = None
        self._boundary_pts_cache: dict[tuple, np.ndarray] = {}

        # Parámetros de normalización ImageNet (idénticos al entrenamiento)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._init_model()

    def _init_model(self) -> None:
        resolved_path = (
            os.path.join(PROJECT_ROOT, self.model_path)
            if not os.path.isabs(self.model_path)
            else self.model_path
        )

        if not os.path.exists(resolved_path):
            log.warning("Modelo DeployZoneFinder no encontrado en: %s. Detector inactivo.", resolved_path)
            self._ready = False
            return

        try:
            import onnxruntime as ort
            # Inferencia óptima en CPU
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                resolved_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self._ready = True
            log.info("DeployZoneDetector listo (modelo: %s, threads: %d)", resolved_path, opts.intra_op_num_threads)
        except Exception as e:
            log.error("Error al inicializar sesión ONNX para DeployZoneDetector: %s", e)
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def predict_mask(self, screenshot: np.ndarray) -> Optional[np.ndarray]:
        """
        Ejecuta la inferencia de segmentación sobre la captura.
        Retorna una máscara binaria uint8 (2400x1080 o resolución nativa):
          - 1: Área de no-despliegue (prohibido desplegar / interior de aldea)
          - 0: Área legal de despliegue (césped exterior / huecos legales)
        """
        if not self._ready or screenshot is None:
            return None

        t0 = time.time()
        orig_img = screenshot
        if orig_img.ndim == 2:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        elif orig_img.shape[2] == 4:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)

        orig_h, orig_w = orig_img.shape[:2]

        # Escalar con lado mayor acotado a imgsz (múltiplo de 32 para UNet)
        scale = min(self.imgsz / max(orig_w, orig_h), 1.0)
        target_w = int(round(orig_w * scale / 32.0)) * 32
        target_h = int(round(orig_h * scale / 32.0)) * 32

        resized_bgr = cv2.resize(orig_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - self.mean) / self.std

        # Formato NCHW [1, 3, H, W]
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        try:
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            raw_out = outputs[0]  # [1, 3, target_h, target_w]

            # Softmax sobre los 3 canales de salida
            exp_out = np.exp(raw_out - np.max(raw_out, axis=1, keepdims=True))
            probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)

            # Canal 1: Probabilidad de área no-desplegable
            no_deploy_prob = probs[0, 1, :, :]

            # TTA opcional (flip horizontal promediado)
            if self.use_tta:
                tensor_flip = tensor[:, :, :, ::-1]
                outputs_flip = self.session.run([self.output_name], {self.input_name: tensor_flip})
                raw_flip = outputs_flip[0]
                exp_flip = np.exp(raw_flip - np.max(raw_flip, axis=1, keepdims=True))
                probs_flip = exp_flip / np.sum(exp_flip, axis=1, keepdims=True)
                no_deploy_prob = (no_deploy_prob + probs_flip[0, 1, :, ::-1]) / 2.0

            # Interpolar mapa de probabilidad al tamaño original
            prob_full = cv2.resize(no_deploy_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (prob_full >= self.conf_thresh).astype(np.uint8)

            self.last_prob = prob_full
            self.last_mask = binary_mask

            if getattr(self, "_boundary_pts_cache", None) is not None:
                self._boundary_pts_cache.clear()
            else:
                self._boundary_pts_cache = {}

            self._last_dist_map = cv2.distanceTransform(
                (binary_mask == 0).astype(np.uint8), cv2.DIST_L2, 5
            )

            # Extraer contornos del borde
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.last_contours = contours

            elapsed_ms = (time.time() - t0) * 1000
            log.debug("DeployZoneFinder inferencia completada en %.1f ms (pixeles no-deploy: %d)",
                      elapsed_ms, int(np.sum(binary_mask)))

            return binary_mask
        except Exception as e:
            log.error("Fallo durante inferencia de DeployZoneDetector: %s", e)
            return None

    def is_point_deployable(
        self,
        x: int,
        y: int,
        mask: Optional[np.ndarray] = None,
        safety_margin_px: int = 15
    ) -> bool:
        """
        Determina en O(1) si la coordenada (x, y) es legal para desplegar tropas.
        Opcionalmente verifica un margen de seguridad alrededor del punto.
        """
        active_mask = mask if mask is not None else self.last_mask
        if active_mask is None:
            return True  # Sin máscara, no se puede vetar

        h, w = active_mask.shape[:2]
        ix, iy = int(round(x)), int(round(y))

        # Verificar límites de pantalla
        if ix < 10 or ix >= w - 10 or iy < 10 or iy >= h - 10:
            return False

        # Si el punto exacto es zona prohibida
        if active_mask[iy, ix] == 1:
            return False

        # Si se requiere margen de seguridad, comprobar con radio euclidiano
        if safety_margin_px > 0:
            r = safety_margin_px
            y1 = max(0, iy - r)
            y2 = min(h, iy + r + 1)
            x1 = max(0, ix - r)
            x2 = min(w, ix + r + 1)
            roi = active_mask[y1:y2, x1:x2]
            if np.any(roi == 1):
                # Validar con distancia euclidiana exacta para no falsear por las esquinas del recuadro
                ys, xs = np.where(roi == 1)
                dists_sq = (xs + x1 - ix) ** 2 + (ys + y1 - iy) ** 2
                if np.any(dists_sq <= r * r):
                    return False

        return True

    def get_boundary_polygons(
        self,
        screenshot: np.ndarray,
        margin_px: int = 30
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calcula los polígonos exactos de la aldea directamente desde la segmentación IA:
        - pts_bld: Frontera exacta del área no desplegable (línea roja).
        - pts_grass: Perímetro exterior seguro de despliegue legal (línea verde),
                     expandido con el margen de seguridad especificado.
        """
        mask = self.predict_mask(screenshot)
        if mask is None:
            return None, None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        # Tomar el contorno principal más grande (aldea central)
        pts_bld = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(pts_bld))
        total_px = float(mask.shape[0] * mask.shape[1])
        if area < 50000.0 or area > 0.85 * total_px:
            # Contorno degenerado (pantalla negra en mocks o sin aldea reconocible), permitir fallback
            return None, None

        # Dilatar máscara para obtener el perímetro seguro de despliegue legal
        k_size = max(3, int(margin_px * 2 + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(mask, kernel)
        green_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if green_contours:
            pts_grass = max(green_contours, key=cv2.contourArea)
        else:
            pts_grass = pts_bld

        return pts_bld, pts_grass

    def find_closest_deployable_point(
        self,
        target_x: int,
        target_y: int,
        search_radius_px: int = 500,
        margin_px: int = 25,
        mask: Optional[np.ndarray] = None,
        safe_bounds: Optional[Tuple[int, int, int, int]] = (120, 140, 2280, 845)
    ) -> Tuple[int, int]:
        """
        Dada una estructura objetivo (almacén, TH, defensa) dentro de la base,
        encuentra el punto exterior legal más cercano para lanzar las tropas.
        Garantiza que el punto esté dentro de safe_bounds (zona activa de toque sin HUD ni barra).
        """
        active_mask = mask if mask is not None else self.last_mask
        if active_mask is None:
            if safe_bounds is not None:
                xmin, ymin, xmax, ymax = safe_bounds
                return int(np.clip(target_x, xmin, xmax)), int(np.clip(target_y, ymin, ymax))
            return (target_x, target_y)

        h, w = active_mask.shape[:2]

        # Si ya es desplegable y cumple safe_bounds
        if self.is_point_deployable(target_x, target_y, active_mask, margin_px):
            if safe_bounds is None:
                return (target_x, target_y)
            xmin, ymin, xmax, ymax = safe_bounds
            if xmin <= target_x <= xmax and ymin <= target_y <= ymax:
                return (target_x, target_y)

        # Asegurar diccionario de caché incluso si el detector fue creado con __new__
        if getattr(self, "_boundary_pts_cache", None) is None:
            self._boundary_pts_cache = {}

        # Reutilizar mapa de distancias si corresponde
        is_same_as_last = mask is None or (
            getattr(self, "last_mask", None) is not None and mask is self.last_mask
        )
        if (
            is_same_as_last
            and getattr(self, "_last_dist_map", None) is not None
            and self._last_dist_map.shape == active_mask.shape[:2]
        ):
            dist_map = self._last_dist_map
        else:
            deployable_map = (active_mask == 0).astype(np.uint8)
            dist_map = cv2.distanceTransform(deployable_map, cv2.DIST_L2, 5)
            if is_same_as_last:
                self._last_dist_map = dist_map

        cache_key = (margin_px, safe_bounds)
        boundary_pts = self._boundary_pts_cache.get(cache_key)

        if boundary_pts is None:
            # Máscara donde el margen de seguridad es válido
            safe_deployable = (dist_map >= margin_px).astype(np.uint8)

            # Filtrar límites de toque activo de pantalla para evitar barra de notificaciones y slots
            if safe_bounds is not None:
                xmin, ymin, xmax, ymax = safe_bounds
                safe_deployable[:ymin, :] = 0
                safe_deployable[ymax:, :] = 0
                safe_deployable[:, :xmin] = 0
                safe_deployable[:, xmax:] = 0

            # Extraer únicamente píxeles de frontera/contorno (frontera geométrica exterior)
            contours, _ = cv2.findContours(safe_deployable, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if contours:
                boundary_pts = np.vstack([c.reshape(-1, 2) for c in contours])
            else:
                boundary_pts = np.empty((0, 2), dtype=np.int32)

            self._boundary_pts_cache[cache_key] = boundary_pts

        if len(boundary_pts) == 0:
            if safe_bounds is not None:
                xmin, ymin, xmax, ymax = safe_bounds
                return int(np.clip(target_x, xmin, xmax)), int(np.clip(target_y, ymin, ymax))
            return (target_x, target_y)

        # Distancia euclidiana mínima contra el array compacto de puntos de frontera
        dists_sq = (boundary_pts[:, 0] - target_x) ** 2 + (boundary_pts[:, 1] - target_y) ** 2
        min_idx = np.argmin(dists_sq)

        best_x = int(boundary_pts[min_idx, 0])
        best_y = int(boundary_pts[min_idx, 1])
        return (best_x, best_y)

    def get_overlay_visualization(self, screenshot: np.ndarray) -> np.ndarray:
        """
        Genera una imagen con el overlay visual idéntico al portal oficial:
        - Zona prohibida resaltada en magenta traslúcido (#ec40c8).
        - Borde exterior trazado con línea blanca nítida.
        """
        if screenshot is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        mask = self.predict_mask(screenshot)
        overlay = screenshot.copy()
        if mask is None:
            return overlay

        # Mezcla alfa para área prohibida
        magenta_bgr = np.array([200, 64, 236], dtype=np.uint8)
        mask_idx = (mask == 1)
        overlay[mask_idx] = (0.55 * overlay[mask_idx] + 0.45 * magenta_bgr).astype(np.uint8)

        # Trazar contorno blanco
        if self.last_contours:
            cv2.drawContours(overlay, self.last_contours, -1, (255, 255, 255), 2, cv2.LINE_AA)

        return overlay


# Instancia singleton predeterminada
_default_detector: Optional[DeployZoneDetector] = None


def get_deploy_zone_detector() -> DeployZoneDetector:
    global _default_detector
    if _default_detector is None:
        cfg = SETTINGS.get("attack", {}).get("deploy_zone_ai", {})
        model_path = cfg.get("model_path", "models/deploy_zone_finder.onnx")
        conf = float(cfg.get("conf_thresh", 0.50))
        imgsz = int(cfg.get("imgsz", 1280))
        tta = bool(cfg.get("use_tta", False))
        _default_detector = DeployZoneDetector(
            model_path=model_path,
            conf_thresh=conf,
            imgsz=imgsz,
            use_tta=tta,
        )
    return _default_detector


def reset_deploy_zone_detector() -> None:
    """Restablece la instancia singleton de DeployZoneDetector (útil en tests)."""
    global _default_detector
    _default_detector = None


