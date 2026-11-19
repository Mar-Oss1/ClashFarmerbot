"""
tools/test_yolo.py
Script de prueba visual para inferencia YOLOv8 sin interactuar con el dispositivo.
"""
import argparse
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.yolo_detector import YoloDetector
from core.logger import get_logger

log = get_logger("test_yolo")

# Colores BGR para visualización:
# Verde para storages, azul para town_hall, amarillo para collectors
CLASS_COLORS = {
    "storage_gold": (0, 255, 0),        # Verde
    "storage_elixir": (0, 255, 128),     # Verde claro / menta
    "storage_dark": (0, 200, 0),        # Verde oscuro
    "town_hall": (255, 0, 0),           # Azul
    "collector_gold": (0, 255, 255),    # Amarillo
    "collector_elixir": (0, 215, 255),  # Amarillo oro
    "collector_dark": (128, 0, 128),    # Púrpura oscuro extractor oscuro
}
DEFAULT_COLOR = (255, 255, 255)         # Blanco para otras clases


def main():
    parser = argparse.ArgumentParser(description="Prueba visual de inferencia YOLO")
    parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen de prueba")
    parser.add_argument("--model", type=str, default="models/yolov8s.onnx", help="Ruta al modelo (.onnx o .pt)")
    parser.add_argument("--conf", type=float, default=0.40, help="Umbral de confianza")
    parser.add_argument("--imgsz", type=int, default=1280, help="Resolución de inferencia (default: 1280)")
    parser.add_argument("--output", type=str, default="logs/yolo_debug.png", help="Ruta de salida anotada")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        log.critical("Imagen no encontrada: %s", args.image)
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        log.critical("No se pudo decodificar la imagen: %s", args.image)
        sys.exit(1)

    log.info("Cargando detector con modelo: %s (conf=%.2f, imgsz=%d)...", args.model, args.conf, args.imgsz)
    detector = YoloDetector(args.model, conf=args.conf, imgsz=args.imgsz)
    if not detector.is_ready():
        log.warning("El detector no está completamente listo o el modelo no existe. Intentando inferencia de prueba...")

    detections = detector.detect(img)
    log.info("Total de detecciones: %d", len(detections))

    out_img = img.copy()

    for idx, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d.bbox_xyxy
        color = CLASS_COLORS.get(d.cls_name, DEFAULT_COLOR)
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)

        # Centro
        cx, cy = d.center
        cv2.circle(out_img, (cx, cy), 4, color, -1)

        label = f"{d.cls_name} {d.confidence:.2f}"
        log.info("  [%d] %-16s | conf: %.2f | bbox: (%d, %d, %d, %d) | center: (%d, %d)",
                 idx, d.cls_name, d.confidence, x1, y1, x2, y2, cx, cy)

        # Fondo del texto para legibilidad
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out_img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out_img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, out_img)
    log.info("Imagen con detecciones guardada en: %s", args.output)


if __name__ == "__main__":
    main()
