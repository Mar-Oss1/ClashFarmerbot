"""
vision/wall_seg_detector.py
Detector de Segmentación de Muros (U-Net con MobileNetV2 encoder) para Clash of Clans.
Infiere sobre capturas completas a resolución nativa (2400x1080 o cualquiera),
convierte BGR->RGB, redimensiona a (768, 352) con INTER_AREA, normaliza según ImageNet,
aplica sigmoid a los logits de salida ONNX y umbraliza (>= 0.5) para devolver una
máscara binaria uint8 (0 / 255) escalada con INTER_NEAREST a la resolución original.
Soporta fallback transparente cuando el modelo no existe o está en entrenamiento.
"""
from __future__ import annotations
import math
import os
from typing import Optional
import cv2
import numpy as np

from core.logger import get_logger
from core.config import PROJECT_ROOT, SETTINGS

log = get_logger("wall_seg_detector")

# Constantes de resolución y normalización U-Net (MobileNetV2)
IMG_W: int = 768
IMG_H: int = 352
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)


from core.vision import save_debug_image_async
from vision.hud import get_battle_hud_mask


def build_cyan_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Genera una imagen de auditoría visual con overlay cian semitransparente
    sobre los muros detectados y contornos verdes en los bordes.
    Color cian en BGR: (255, 255, 0).
    Color verde de contorno: (0, 255, 0).
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    wall_pixels = mask > 0
    if np.any(wall_pixels):
        cyan_bgr = np.array([255, 255, 0], dtype=np.uint8)
        num_wall = int(np.count_nonzero(wall_pixels))
        cyan_tile = np.tile(cyan_bgr, (num_wall, 1))

        blended = cv2.addWeighted(
            image_bgr[wall_pixels], 1.0 - alpha,
            cyan_tile, alpha,
            0.0
        )
        overlay[wall_pixels] = blended

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

    return overlay


def _get_line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Genera un kernel morfológico lineal a lo largo del ángulo angle_deg."""
    rad = math.radians(angle_deg)
    half = length // 2
    k = np.zeros((length, length), dtype=np.uint8)
    for i in range(-half, half + 1):
        x = int(round(half + i * math.cos(rad)))
        y = int(round(half + i * math.sin(rad)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1
    return k


class WallSegDetector:
    """
    Detector de muros mediante segmentación semántica U-Net (MobileNetV2 / ONNX).
    Ejecuta inferencia CPU en onnxruntime con sesión optimizada.
    Soporta inferencia mockeada y fallback transparente cuando el modelo no existe.
    """

    def __init__(
        self,
        model_path: str = "models/wall_seg.onnx",
        conf_thresh: float = 0.50,
        imgsz: int = 1280,
        strict: bool = False,
        morph_closing_enabled: bool | None = None,
        morph_kernel_length: int | None = None,
        morph_angles: list[float] | None = None,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.strict = strict

        cfg = SETTINGS.get("attack", {}).get("wall_seg", SETTINGS.get("wall_seg", {}))
        self.morph_closing_enabled = bool(
            morph_closing_enabled if morph_closing_enabled is not None
            else cfg.get("morph_closing_enabled", True)
        )
        self.morph_kernel_length = int(
            morph_kernel_length if morph_kernel_length is not None
            else cfg.get("morph_kernel_length", 9)
        )
        self.morph_angles = list(
            morph_angles if morph_angles is not None
            else cfg.get("morph_angles", [36.0, 144.0])
        )

        self.session = None
        self.input_name: str | None = None
        self.output_names: list[str] = []
        self._ready: bool = False

        self._init_session()

    def _init_session(self) -> None:
        resolved_path = (
            os.path.join(PROJECT_ROOT, self.model_path)
            if not os.path.isabs(self.model_path)
            else self.model_path
        )

        if not os.path.exists(resolved_path):
            msg = (
                f"Modelo U-Net de segmentación de muros no encontrado en: '{resolved_path}'. "
                "Operando en modo fallback transparente."
            )
            if self.strict:
                raise FileNotFoundError(msg)
            log.info(msg)
            self._ready = False
            return

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                resolved_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            self._ready = True
            log.info("WallSegDetector listo (modelo: %s, threads: %d)", resolved_path, opts.intra_op_num_threads)
        except Exception as e:
            if self.strict:
                raise
            log.warning("No se pudo inicializar sesión ONNX para WallSegDetector (%s). Operando en modo fallback.", e)
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready and self.session is not None

    def predict(
        self,
        frame_bgr: np.ndarray,
        detections: list | None = None,
        deploy_zone_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Infiere sobre frame_bgr y produce una máscara binaria uint8 (0 / 255)
        en la resolución original del frame.
        Aplica la máscara de HUD de batalla y bordes.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        orig_h, orig_w = frame_bgr.shape[:2]
        hud_mask = get_battle_hud_mask(orig_h, orig_w)

        if not self.is_ready():
            return self._predict_fallback(frame_bgr, detections, deploy_zone_mask, hud_mask)

        try:
            inputs = self.session.get_inputs()
            input_shape = inputs[0].shape if inputs else []
            is_yolo = len(self.output_names) > 1 or (len(input_shape) >= 4 and input_shape[2] == 1280)

            if not is_yolo:
                # 1. Pipeline U-Net: BGR -> RGB obligatorio
                if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)
                else:
                    frame_rgb = frame_bgr

                resized = cv2.resize(frame_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
                img_float = resized.astype(np.float32) / 255.0
                norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
                tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

                outputs = self.session.run(None, {self.input_name: tensor})
                raw_out = outputs[0]
                if raw_out.ndim == 4:
                    logits = raw_out[0, 0]
                elif raw_out.ndim == 3:
                    logits = raw_out[0]
                else:
                    logits = raw_out

                if np.min(logits) < 0.0 or np.max(logits) > 1.0:
                    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
                else:
                    probs = logits

                mask_small = (probs >= self.conf_thresh).astype(np.uint8) * 255
                mask_native = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            else:
                # Pipeline YOLOv8-seg (compatibilidad con export previo)
                target_sz = 1280
                scale = min(target_sz / orig_w, target_sz / orig_h)
                nw, nh = int(round(orig_w * scale)), int(round(orig_h * scale))
                resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
                pad_w = target_sz - nw
                pad_h = target_sz - nh
                top, bottom = pad_h // 2, pad_h - (pad_h // 2)
                left, right = pad_w // 2, pad_w - (pad_w // 2)
                padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                tensor = np.transpose(blob, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

                outputs = self.session.run(None, {self.input_name: tensor})
                out0 = outputs[0]
                protos = outputs[1][0] if len(outputs) > 1 else None

                if out0.ndim == 3 and out0.shape[1] == 37:
                    preds = out0[0].T
                else:
                    preds = out0[0]

                scores = preds[:, 4]
                keep = scores >= self.conf_thresh
                if not np.any(keep) or protos is None:
                    mask_native = np.zeros((orig_h, orig_w), dtype=np.uint8)
                else:
                    coeffs = preds[keep, 5:]
                    proto_h, proto_w = protos.shape[1], protos.shape[2]
                    masks = 1.0 / (1.0 + np.exp(-(coeffs @ protos.reshape(32, -1))))
                    masks = masks.reshape(-1, proto_h, proto_w)
                    combined_mask = np.max(masks, axis=0)

                    proto_scale_x = proto_w / float(target_sz)
                    proto_scale_y = proto_h / float(target_sz)
                    p_top = int(round(top * proto_scale_y))
                    p_left = int(round(left * proto_scale_x))
                    p_h = int(round(nh * proto_scale_y))
                    p_w = int(round(nw * proto_scale_x))
                    cropped_proto = combined_mask[p_top : p_top + p_h, p_left : p_left + p_w]
                    mask_full = cv2.resize(cropped_proto, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    mask_native = (mask_full >= 0.5).astype(np.uint8) * 255

            # Post-procesado: deploy_zone_mask, cierre morfológico direccional y HUD mask
            if deploy_zone_mask is None:
                try:
                    from vision.deploy_zone_detector import get_deploy_zone_detector
                    dzd = get_deploy_zone_detector()
                    if dzd is not None and dzd.is_ready():
                        deploy_zone_mask = dzd.predict_mask(frame_bgr)
                except Exception as e:
                    log.debug("Auto deploy_zone_mask retrieval skipped: %s", e)

            if deploy_zone_mask is not None and isinstance(deploy_zone_mask, np.ndarray):
                if deploy_zone_mask.shape[:2] != (orig_h, orig_w):
                    deploy_zone_mask = cv2.resize(deploy_zone_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                # En deploy_zone_detector: 0 = exterior (césped legal), 1 = aldea (rojo)
                mask_native[deploy_zone_mask == 0] = 0

            if self.morph_closing_enabled and np.any(mask_native > 0):
                closed_layers = []
                for ang in self.morph_angles:
                    k = _get_line_kernel(self.morph_kernel_length, ang)
                    closed_layers.append(cv2.morphologyEx(mask_native, cv2.MORPH_CLOSE, k))
                if closed_layers:
                    combined_closed = closed_layers[0]
                    for layer in closed_layers[1:]:
                        combined_closed = cv2.bitwise_or(combined_closed, layer)
                    mask_native = combined_closed

            mask_native[hud_mask > 0] = 0
            if deploy_zone_mask is not None and isinstance(deploy_zone_mask, np.ndarray):
                mask_native[deploy_zone_mask == 0] = 0

            return mask_native
        except Exception as e:
            log.warning("Fallo en inferencia ONNX de WallSegDetector (%s). Usando fallback.", e)
            return self._predict_fallback(frame_bgr, detections, deploy_zone_mask, hud_mask)

    def predict_mask(self, screenshot: np.ndarray, **kwargs) -> np.ndarray:
        """Alias de compatibilidad para predict()."""
        return self.predict(screenshot, **kwargs)

    def _predict_fallback(
        self,
        screenshot: np.ndarray,
        detections: list | None,
        deploy_zone_mask: np.ndarray | None,
        hud_mask: np.ndarray,
    ) -> np.ndarray:
        """Fallback transparente con gradiente Lab y cierre morfológico direccional."""
        from vision.wall_compartments import build_clean_wall_canvas
        clean_canvas = build_clean_wall_canvas(screenshot, detections, deploy_zone_mask)
        edges = cv2.Canny(clean_canvas, 45, 135)

        closed_layers = []
        for ang in self.morph_angles:
            k = _get_line_kernel(self.morph_kernel_length, ang)
            closed_layers.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k))
        if closed_layers:
            wall_binary = closed_layers[0]
            for layer in closed_layers[1:]:
                wall_binary = cv2.bitwise_or(wall_binary, layer)
        else:
            wall_binary = edges

        contours, _ = cv2.findContours(wall_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(wall_binary)
        for c in contours:
            if cv2.contourArea(c) >= 50 or cv2.arcLength(c, False) >= 30:
                cv2.drawContours(mask, [c], -1, 255, -1)

        mask[hud_mask > 0] = 0
        return mask

    def build_overlay(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray | None = None,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Genera overlay cian de auditoría visual."""
        if mask is None:
            mask = self.predict(frame_bgr)
        return build_cyan_overlay(frame_bgr, mask, alpha=alpha)

    def save_debug_overlay(
        self,
        frame_bgr: np.ndarray,
        out_path: str | np.ndarray = "scratch/wall_seg_debug.png",
        mask: np.ndarray | str | None = None,
    ) -> str:
        """Guarda la imagen de auditoría visual con overlay cian en disco.
        Acepta tanto (frame, out_path, mask) como (frame, mask, out_path).
        """
        if isinstance(out_path, np.ndarray):
            actual_mask = out_path
            actual_out_path = str(mask) if isinstance(mask, str) else "scratch/wall_seg_debug.png"
        else:
            actual_mask = mask if isinstance(mask, np.ndarray) else None
            actual_out_path = str(out_path)

        overlay = self.build_overlay(frame_bgr, mask=actual_mask)
        resolved_out = (
            os.path.join(PROJECT_ROOT, actual_out_path)
            if not os.path.isabs(actual_out_path)
            else actual_out_path
        )
        os.makedirs(os.path.dirname(resolved_out), exist_ok=True)
        save_debug_image_async(resolved_out, overlay)
        log.info("📸 Debug visual de WallSegDetector guardado en: %s", resolved_out)
        return resolved_out


_DEFAULT_WALL_SEG_DETECTOR: Optional[WallSegDetector] = None


def get_wall_seg_detector(
    model_path: str | None = None,
    conf_thresh: float | None = None,
    morph_closing_enabled: bool | None = None,
    morph_kernel_length: int | None = None,
    morph_angles: list[float] | None = None,
) -> WallSegDetector:
    """Obtiene o inicializa el singleton predeterminado de WallSegDetector leyendo settings.json si no se pasan parámetros."""
    global _DEFAULT_WALL_SEG_DETECTOR
    cfg = SETTINGS.get("attack", {}).get("jump_optimizer", {})
    resolved_path = model_path or cfg.get("wall_seg_model_path", "models/wall_seg.onnx")
    resolved_conf = conf_thresh if conf_thresh is not None else float(cfg.get("wall_seg_conf", 0.50))

    if (
        _DEFAULT_WALL_SEG_DETECTOR is None
        or _DEFAULT_WALL_SEG_DETECTOR.model_path != resolved_path
        or _DEFAULT_WALL_SEG_DETECTOR.conf_thresh != resolved_conf
        or (morph_closing_enabled is not None and _DEFAULT_WALL_SEG_DETECTOR.morph_closing_enabled != morph_closing_enabled)
        or (morph_kernel_length is not None and _DEFAULT_WALL_SEG_DETECTOR.morph_kernel_length != morph_kernel_length)
        or (morph_angles is not None and _DEFAULT_WALL_SEG_DETECTOR.morph_angles != morph_angles)
    ):
        _DEFAULT_WALL_SEG_DETECTOR = WallSegDetector(
            model_path=resolved_path,
            conf_thresh=resolved_conf,
            morph_closing_enabled=morph_closing_enabled,
            morph_kernel_length=morph_kernel_length,
            morph_angles=morph_angles,
        )
    return _DEFAULT_WALL_SEG_DETECTOR


def reset_wall_seg_detector() -> None:
    """Reinicia el singleton de WallSegDetector (útil en pruebas unitarias)."""
    global _DEFAULT_WALL_SEG_DETECTOR
    _DEFAULT_WALL_SEG_DETECTOR = None
