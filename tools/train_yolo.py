"""
tools/train_yolo.py
Script CLI para entrenar YOLOv8 con dataset CoC y exportar el mejor modelo a ONNX.
Configurado con hiperparámetros optimizados para Clash of Clans (imgsz=1280, luz fija sin fliplr/flipud).
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import get_logger

log = get_logger("train_yolo")


def check_dataset_ready(data_yaml_path: str = "datasets/coc/data.yaml") -> bool:
    train_dir = os.path.join("datasets", "coc", "images", "train")
    if not os.path.exists(train_dir):
        log.error("Directorio de imágenes de entrenamiento no encontrado: %s", train_dir)
        return False

    valid_images = [
        f for f in os.listdir(train_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ]
    if not valid_images:
        log.error(
            "El directorio de entrenamiento '%s' está vacío o no contiene imágenes válidas. "
            "Asegúrate de agregar imágenes y etiquetas en 'datasets/coc/' antes de entrenar "
            "(puedes usar 'python tools/split_dataset.py' para repartir las capturas crudas).",
            train_dir,
        )
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo YOLOv8 para ClashFarmerbot")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8s.pt" if os.path.exists("models/yolov8s.pt") else "yolov8s.pt",
        help="Pesos base para fine-tuning (default: models/yolov8s.pt o yolov8s.pt)"
    )
    parser.add_argument("--data", type=str, default="datasets/coc/data.yaml", help="Ruta al archivo data.yaml")
    parser.add_argument("--epochs", type=int, default=80, help="Número de épocas de entrenamiento (default: 80)")
    parser.add_argument("--batch", type=int, default=8, help="Tamaño de batch (default: 8)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Tamaño de imagen (default: 1280)")
    parser.add_argument("--patience", type=int, default=25, help="Paciencia de early stopping (default: 25)")
    parser.add_argument("--device", type=str, default="", help="Dispositivo para entrenamiento (ej: 0, cpu)")
    parser.add_argument("--project", type=str, default="runs", help="Directorio raíz del proyecto")
    parser.add_argument("--name", type=str, default="coc_v1", help="Nombre del experimento (default: coc_v1)")
    args = parser.parse_args()

    if not check_dataset_ready(args.data):
        log.critical("Entrenamiento abortado: Dataset no preparado.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        log.critical("No se pudo importar 'ultralytics'. Instala las dependencias con pip: %s", e)
        sys.exit(1)

    log.info("Cargando modelo base: %s", args.model)
    model = YOLO(args.model)

    # Hiperparámetros ajustados a Clash of Clans:
    # - fliplr=0.0, flipud=0.0: La iluminación isométrica es fija (sol siempre desde el mismo ángulo).
    # - scale=0.25, translate=0.08, degrees=0.0: Evitar rotaciones destructivas en vista isométrica fija.
    # - mosaic=1.0, close_mosaic=10: Ayuda a detectar edificios pequeños en aldeas densas.
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "scale": 0.25,
        "translate": 0.08,
        "degrees": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.35,
        "project": args.project,
        "name": args.name,
    }
    if args.device:
        train_kwargs["device"] = args.device

    log.info("Iniciando entrenamiento con parámetros: %s", train_kwargs)
    results = model.train(**train_kwargs)

    best_pt_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_pt_path):
        log.info("Entrenamiento finalizado exitosamente. Exportando '%s' a ONNX (imgsz=%d)...", best_pt_path, args.imgsz)
        best_model = YOLO(best_pt_path)
        best_model.export(format="onnx", imgsz=args.imgsz, opset=17)
        log.info("Exportación ONNX completada.")
    else:
        log.warning("No se encontró 'best.pt' en '%s'. Revisa los resultados del entrenamiento.", best_pt_path)


if __name__ == "__main__":
    main()
