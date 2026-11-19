"""
vision/yolo_detector.py
Módulo de inferencia YOLOv8 para detección de estructuras en Clash of Clans.
Soporta backend ONNX (onnxruntime) y PyTorch (.pt vía ultralytics).
"""
import os
from dataclasses import dataclass
import cv2
import numpy as np

from core.logger import get_logger
from core.config import SETTINGS, PROJECT_ROOT

log = get_logger("yolo_detector")

DEFAULT_CLASSES = [
    "storage_gold",
    "storage_elixir",
    "storage_dark",
    "town_hall",
    "collector_gold",
    "collector_elixir",
    "collector_dark",
]

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class _CallableBool(int):
    """Permite acceder como property booleana (d.is_finetuned) y como método (d.is_finetuned())."""
    def __call__(self):
        return bool(self)

    def __repr__(self):
        return repr(bool(self))

    def __str__(self):
        return str(bool(self))


@dataclass
class Detection:
    cls_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]  # coords en píxeles originales
    center: tuple[int, int]              # centro en píxeles originales


class YoloDetector:
    def __init__(
        self,
        model_path: str = "models/yolov8s.onnx",
        conf: float = 0.40,
        iou: float = 0.5,
        imgsz: int = 1280,
        filter_hud: bool = True,
    ):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.filter_hud = filter_hud
        self._ready = False
        self.backend = "none"
        self.session = None
        self.pt_model = None

        self.class_names = list(DEFAULT_CLASSES)
        self.model_nc = len(self.class_names)
        self._load_class_names()
        self._init_model()

    def _load_class_names(self):
        yaml_path = os.path.join(PROJECT_ROOT, "datasets", "coc", "data.yaml")
        if os.path.exists(yaml_path):
            try:
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and "names" in data:
                        self.class_names = list(data["names"])
            except Exception:
                pass
        self.model_nc = len(self.class_names)

    def _init_model(self):
        resolved_path = (
            os.path.join(PROJECT_ROOT, self.model_path)
            if not os.path.isabs(self.model_path)
            else self.model_path
        )
        if not os.path.exists(resolved_path):
            log.warning("Modelo no encontrado en disco: %s. El detector no estará activo.", resolved_path)
            self._ready = False
            return

        try:
            if resolved_path.lower().endswith(".onnx"):
                import onnxruntime as ort
                self.session = ort.InferenceSession(resolved_path, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                out_shape = self.session.get_outputs()[0].shape
                # En YOLOv8 ONNX: [1, 4 + nc, num_candidates] o [1, num_candidates, 4 + nc]
                nc_dim = out_shape[1] if (len(out_shape) > 1 and out_shape[1] < out_shape[-1]) else out_shape[-1]
                self.model_nc = int(nc_dim) - 4
                self.backend = "onnxruntime (CPUExecutionProvider)"
                self._ready = True
                log.info("YoloDetector inicializado con backend %s (modelo: %s, nc=%d)", self.backend, resolved_path, self.model_nc)
            elif resolved_path.lower().endswith(".pt"):
                from ultralytics import YOLO
                self.pt_model = YOLO(resolved_path)
                self.model_nc = len(self.pt_model.names)
                self.backend = "ultralytics (PyTorch)"
                self._ready = True
                log.info("YoloDetector inicializado con backend %s (modelo: %s, nc=%d)", self.backend, resolved_path, self.model_nc)
            else:
                log.warning("Formato de modelo desconocido: %s", resolved_path)
                self._ready = False
        except Exception as e:
            log.error("Fallo al inicializar sesión de modelo YOLO (%s): %s", resolved_path, e)
            self._ready = False

    @property
    def is_finetuned(self):
        is_ft = (self.model_nc == len(self.class_names))
        return _CallableBool(1 if is_ft else 0)

    def is_ready(self) -> bool:
        return self._ready

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        shape = img.shape[:2]  # (h, w)
        r = min(self.imgsz / shape[0], self.imgsz / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (self.imgsz - new_unpad[0]) / 2
        dh = (self.imgsz - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, r, (left, top)

    @staticmethod
    def _dedup_overlapping(detections: list[Detection], iou_thresh: float = 0.55) -> list[Detection]:
        """Fusiona cajas de la MISMA clase muy solapadas conservando la de mayor confianza."""
        def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix = max(0, min(ax2, bx2) - max(ax1, bx1))
            iy = max(0, min(ay2, by2) - max(ay1, by1))
            inter = float(ix * iy)
            area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
            area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
            return inter / (area_a + area_b - inter + 1e-9)

        kept: list[Detection] = []
        for d in sorted(detections, key=lambda x: -x.confidence):
            if all(_iou(d.bbox_xyxy, k.bbox_xyxy) < iou_thresh or d.cls_name != k.cls_name for k in kept):
                kept.append(d)
        return kept

    def detect(self, screenshot: np.ndarray) -> list[Detection]:
        if not self._ready or screenshot is None:
            return []

        try:
            orig_img = screenshot
            if orig_img.ndim == 2:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
            elif orig_img.shape[2] == 4:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)

            orig_h, orig_w = orig_img.shape[:2]

            # Fallback backend con PyTorch (.pt)
            if self.backend.startswith("ultralytics"):
                results = self.pt_model(orig_img, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
                detections: list[Detection] = []
                if results and len(results) > 0:
                    r = results[0]
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        x1, y1, x2, y2 = [int(round(v)) for v in coords]
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)

                        # Filtro de área de juego (150 < y < 900 en 1080p) para eliminar falsos positivos en HUD
                        if self.filter_hud:
                            min_y = int(0.138 * orig_h)
                            max_y = int(0.833 * orig_h)
                            if center[1] < min_y or center[1] > max_y:
                                continue

                        if self.is_finetuned():
                            raw_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                            cls_name = raw_name
                        else:
                            raw_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                            cls_name = f"[COCO] {raw_name}"
                        detections.append(Detection(cls_name, conf, (x1, y1, x2, y2), center))
                return self._dedup_overlapping(detections, iou_thresh=0.55)

            # Backend principal con ONNX
            padded, scale, (pad_w, pad_h) = self._letterbox(orig_img)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            blob = rgb.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

            outputs = self.session.run(None, {self.input_name: blob})
            preds = outputs[0]  # Shape: (1, 4 + nc, num_candidates)

            preds = np.transpose(preds[0], (1, 0))  # (num_candidates, 4 + nc)
            boxes = preds[:, :4]
            scores = preds[:, 4:]

            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)

            mask = confidences >= self.conf
            if not np.any(mask):
                return []

            boxes = boxes[mask]
            class_ids = class_ids[mask]
            confidences = confidences[mask]

            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            w = boxes[:, 2]
            h = boxes[:, 3]

            boxes_xywh = np.stack([x1, y1, w, h], axis=1).tolist()
            confidences_list = confidences.tolist()

            indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences_list, score_threshold=self.conf, nms_threshold=self.iou)
            if len(indices) == 0:
                return []

            indices = np.array(indices).flatten()
            detections: list[Detection] = []

            for idx in indices:
                bx1, by1, bw, bh = boxes_xywh[idx]
                bx2, by2 = bx1 + bw, by1 + bh

                orig_x1 = max(0, min(orig_w, int(round((bx1 - pad_w) / scale))))
                orig_y1 = max(0, min(orig_h, int(round((by1 - pad_h) / scale))))
                orig_x2 = max(0, min(orig_w, int(round((bx2 - pad_w) / scale))))
                orig_y2 = max(0, min(orig_h, int(round((by2 - pad_h) / scale))))

                cx = (orig_x1 + orig_x2) // 2
                cy = (orig_y1 + orig_y2) // 2

                # Filtro de área de juego (150 < y < 900 en 1080p)
                if self.filter_hud:
                    min_y = int(0.138 * orig_h)
                    max_y = int(0.833 * orig_h)
                    if cy < min_y or cy > max_y:
                        continue

                cid = int(class_ids[idx])
                if self.is_finetuned():
                    raw_name = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
                    cls_name = raw_name
                else:
                    raw_name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
                    cls_name = f"[COCO] {raw_name}"
                conf = float(confidences[idx])

                detections.append(Detection(cls_name, conf, (orig_x1, orig_y1, orig_x2, orig_y2), (cx, cy)))

            return self._dedup_overlapping(detections, iou_thresh=0.55)
        except Exception as e:
            log.error("Error durante detect() con YOLO: %s", e)
            return []


_DETECTOR_SINGLETON: YoloDetector | None = None


def get_detector() -> YoloDetector | None:
    """
    Instancia global lazy (singleton) que lee config/settings.json -> attack.use_yolo.
    Si está en false, devuelve None.
    """
    global _DETECTOR_SINGLETON
    attack_cfg = SETTINGS.get("attack", {})
    use_yolo = attack_cfg.get("use_yolo", False)
    if not use_yolo:
        return None

    if _DETECTOR_SINGLETON is None:
        model_path = attack_cfg.get("yolo_model_path", "models/yolov8s.onnx")
        conf = float(attack_cfg.get("yolo_conf", 0.40))
        iou = float(attack_cfg.get("yolo_iou", 0.5))
        imgsz = int(attack_cfg.get("yolo_imgsz", 1280))
        _DETECTOR_SINGLETON = YoloDetector(model_path=model_path, conf=conf, iou=iou, imgsz=imgsz)

    return _DETECTOR_SINGLETON
