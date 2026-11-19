"""
tools/train_wall_seg.py
Script CLI para entrenar el modelo ligero de segmentación de muros (YOLOv8n-seg)
con restricciones isométricas específicas para Clash of Clans y auto-exportación a ONNX.

Restricciones isométricas obligatorias:
- degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0
- fliplr=0.5 (muros en ángulos fijos 36°/144° con proyección 2.5D)

Configuración base:
- Base: yolov8n-seg.pt
- data: datasets/coc_walls/data.yaml
- epochs: 80, imgsz: 1280, batch: 8, device: cpu, patience: 20, max_det: 300, overlap_mask: True
- Auto-export: exporta best.pt a ONNX (format=onnx, imgsz=1280, simplify=True) -> models/wall_seg.onnx
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# Agregar raíz del proyecto al path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.logger import get_logger

log = get_logger("train_wall_seg")


def resolve_base_model(model_arg: str | None = None) -> str:
    """Resuelve la ruta de los pesos base (yolov8n-seg.pt)."""
    candidates = [
        model_arg,
        os.path.join(PROJECT_ROOT, "models", "yolov8n-seg.pt"),
        os.path.join(PROJECT_ROOT, "yolov8n-seg.pt"),
        "yolov8n-seg.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    # Si ninguno existe localmente, devolver el nombre para que ultralytics lo descargue
    return model_arg or "yolov8n-seg.pt"


def check_dataset_ready(data_yaml_path: str = "datasets/coc_walls/data.yaml") -> bool:
    """Verifica si el dataset de segmentación de muros existe y tiene imágenes."""
    resolved_yaml = data_yaml_path if os.path.isabs(data_yaml_path) else os.path.join(PROJECT_ROOT, data_yaml_path)
    if not os.path.exists(resolved_yaml):
        log.error("Archivo data.yaml no encontrado: %s", resolved_yaml)
        return False

    base_dir = os.path.dirname(resolved_yaml)
    train_dir = os.path.join(base_dir, "images", "train")
    if not os.path.exists(train_dir):
        # Alternativa relativa a PROJECT_ROOT
        train_dir = os.path.join(PROJECT_ROOT, "datasets", "coc_walls", "images", "train")

    if not os.path.exists(train_dir):
        log.error("Directorio de imágenes de entrenamiento no encontrado: %s", train_dir)
        return False

    valid_images = [
        f for f in os.listdir(train_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ]
    if not valid_images:
        log.error("No se encontraron imágenes válidas en el directorio: %s", train_dir)
        return False

    log.info("Dataset verificado correctamente: %d imágenes encontradas en %s", len(valid_images), train_dir)
    return True


def export_model_to_onnx(
    pt_path: str,
    export_dest: str = "models/wall_seg.onnx",
    imgsz: int = 1280,
    simplify: bool = True,
    opset: int = 17,
) -> bool:
    """
    Exporta el modelo YOLOv8-seg de PyTorch a formato ONNX optimizado
    y lo copia en la ubicación de destino para inferencia en CPU.
    """
    if not os.path.exists(pt_path):
        log.error("No se encontró el archivo de pesos PyTorch a exportar: %s", pt_path)
        return False

    try:
        from ultralytics import YOLO
    except ImportError as e:
        log.critical("No se pudo importar 'ultralytics'. Instala las dependencias: %s", e)
        return False

    log.info("Exportando modelo '%s' a ONNX (imgsz=%d, simplify=%s, opset=%d)...", pt_path, imgsz, simplify, opset)
    try:
        model = YOLO(pt_path)
        exported = model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=simplify,
            opset=opset,
        )
        log.info("Ultralytics export retornó: %s", exported)

        exported_path = str(exported)
        if not os.path.exists(exported_path):
            stem = Path(pt_path).stem
            candidate = os.path.join(os.path.dirname(pt_path), f"{stem}.onnx")
            if os.path.exists(candidate):
                exported_path = candidate
            else:
                log.error("No se encontró el archivo ONNX exportado en %s ni en %s", exported_path, candidate)
                return False

        # Asegurar ruta absoluta para el destino
        dest_abs = export_dest if os.path.isabs(export_dest) else os.path.join(PROJECT_ROOT, export_dest)
        dest_dir = os.path.dirname(dest_abs)
        os.makedirs(dest_dir, exist_ok=True)

        shutil.copy2(exported_path, dest_abs)
        file_size = os.path.getsize(dest_abs)
        log.info("Modelo ONNX copiado exitosamente a: %s (%d bytes)", dest_abs, file_size)

        # Validación con onnxruntime
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(dest_abs, providers=["CPUExecutionProvider"])
            inputs = [inp.name for inp in sess.get_inputs()]
            outputs = [out.name for out in sess.get_outputs()]
            log.info("Validación con onnxruntime exitosa. Inputs: %s, Outputs: %s", inputs, outputs)
        except Exception as e:
            log.warning("No se pudo verificar con onnxruntime: %s", e)

        return True
    except Exception as e:
        log.exception("Error crítico durante la exportación a ONNX: %s", e)
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de Segmentación de Muros (YOLOv8-seg) con restricciones isométricas y exportación ONNX"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pesos base para segmentación (default: models/yolov8n-seg.pt o yolov8n-seg.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/coc_walls/data.yaml",
        help="Ruta al archivo data.yaml del dataset de muros (default: datasets/coc_walls/data.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Número de épocas de entrenamiento (default: 80)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Tamaño de batch (default: 8)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Resolución de entrada cuadrada (default: 1280)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Dispositivo de entrenamiento, ej. 'cpu', '0', 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Paciencia para early stopping (default: 20)",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Máximo número de detecciones por imagen (default: 300)",
    )
    parser.add_argument(
        "--overlap-mask",
        action="store_true",
        default=True,
        help="Superponer máscaras durante el entrenamiento (default: True)",
    )
    # Restricciones isométricas obligatorias
    parser.add_argument(
        "--degrees",
        type=float,
        default=0.0,
        help="Rotación máxima en grados - debe ser 0.0 por perspectiva isométrica fija (default: 0.0)",
    )
    parser.add_argument(
        "--shear",
        type=float,
        default=0.0,
        help="Corte/cizalla - debe ser 0.0 por perspectiva isométrica fija (default: 0.0)",
    )
    parser.add_argument(
        "--perspective",
        type=float,
        default=0.0,
        help="Perspectiva - debe ser 0.0 por perspectiva isométrica fija (default: 0.0)",
    )
    parser.add_argument(
        "--flipud",
        type=float,
        default=0.0,
        help="Flip vertical - debe ser 0.0 para preservar iluminación 2.5D (default: 0.0)",
    )
    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="Flip horizontal (espejo) permitido en isometría simétrica 36/144 deg (default: 0.5)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Directorio base para guardar ejecuciones (default: runs)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="wall_seg",
        help="Nombre del experimento (default: wall_seg)",
    )
    parser.add_argument(
        "--export-dest",
        type=str,
        default="models/wall_seg.onnx",
        help="Destino final del modelo ONNX exportado (default: models/wall_seg.onnx)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Si se activa, no exporta automáticamente a ONNX al finalizar",
    )
    parser.add_argument(
        "--export-only",
        type=str,
        default=None,
        help="Exporta a ONNX los pesos PyTorch indicados y sale sin entrenar",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida configuración, pesos base y dataset sin ejecutar épocas de entrenamiento",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Si se pide solo exportar un modelo existente
    if args.export_only:
        log.info("Modo export-only: Exportando '%s' a '%s'...", args.export_only, args.export_dest)
        success = export_model_to_onnx(
            pt_path=args.export_only,
            export_dest=args.export_dest,
            imgsz=args.imgsz,
            simplify=True,
        )
        return 0 if success else 1

    base_model_path = resolve_base_model(args.model)
    log.info("Pesos base resueltos: %s", base_model_path)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        log.critical("Dependencia 'ultralytics' no disponible: %s", e)
        return 1

    # Modo dry-run para validar pipeline antes de que prelabeler termine
    if args.dry_run:
        log.info("Modo dry-run: Validando modelo y parámetros...")
        try:
            model = YOLO(base_model_path)
            log.info("Modelo YOLO cargado exitosamente: %s", model.model_name or "YOLOv8n-seg")
        except Exception as e:
            log.error("Fallo al instanciar modelo con '%s': %s", base_model_path, e)
            return 1

        dataset_ok = check_dataset_ready(args.data)
        log.info("Estado del dataset: %s", "Listo" if dataset_ok else "Pendiente / Incompleto")
        log.info("Configuración de entrenamiento validada correctamente.")
        return 0

    # Validar que el dataset esté listo para entrenar
    if not check_dataset_ready(args.data):
        log.critical("Entrenamiento cancelado: El dataset '%s' no está preparado aún.", args.data)
        return 1

    log.info("Cargando modelo YOLOv8n-seg desde: %s", base_model_path)
    model = YOLO(base_model_path)

    # Hiperparámetros con restricciones isométricas obligatorias
    train_kwargs = {
        "data": args.data if os.path.isabs(args.data) else os.path.join(PROJECT_ROOT, args.data),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "patience": args.patience,
        "max_det": args.max_det,
        "overlap_mask": args.overlap_mask,
        # Restricciones isométricas:
        "degrees": args.degrees,          # 0.0: Muros mantienen ángulos fijos 36/144
        "shear": args.shear,              # 0.0: Proyección 2.5D fija
        "perspective": args.perspective,  # 0.0: Sin distorsión de cámara
        "flipud": args.flipud,            # 0.0: La iluminación isométrica proviene de un cuadrante fijo
        "fliplr": args.fliplr,            # 0.5: Espejo horizontal permitido
        "mosaic": 1.0,
        "close_mosaic": 10,
        "scale": 0.25,
        "translate": 0.08,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "save": True,
    }

    log.info("Iniciando entrenamiento de segmentación de muros...")
    log.info("Parámetros: %s", train_kwargs)

    try:
        results = model.train(**train_kwargs)
        log.info("Entrenamiento finalizado exitosamente.")
    except Exception as e:
        log.exception("Error durante el entrenamiento: %s", e)
        return 1

    if args.no_export:
        log.info("Exportación omitida por flag --no-export.")
        return 0

    # Auto-exportar best.pt a ONNX
    best_pt_candidates = [
        getattr(results, "save_dir", None) and os.path.join(results.save_dir, "weights", "best.pt"),
        os.path.join(args.project, args.name, "weights", "best.pt"),
        os.path.join(PROJECT_ROOT, args.project, args.name, "weights", "best.pt"),
    ]

    best_pt_path = None
    for cand in best_pt_candidates:
        if cand and os.path.exists(cand):
            best_pt_path = cand
            break

    if not best_pt_path:
        log.warning("No se encontró 'best.pt' en las rutas esperadas. Buscando el archivo .pt más reciente...")
        run_dir = os.path.join(PROJECT_ROOT, args.project, args.name, "weights")
        if os.path.exists(run_dir):
            pts = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith(".pt")]
            if pts:
                best_pt_path = max(pts, key=os.path.getmtime)

    if not best_pt_path or not os.path.exists(best_pt_path):
        log.error("No se pudo localizar 'best.pt' para la exportación a ONNX.")
        return 1

    log.info("Mejor modelo encontrado en: %s", best_pt_path)
    export_ok = export_model_to_onnx(
        pt_path=best_pt_path,
        export_dest=args.export_dest,
        imgsz=args.imgsz,
        simplify=True,
    )

    if export_ok:
        log.info("Pipeline completo finalizado exitosamente. Modelo listo en: %s", args.export_dest)
        return 0
    else:
        log.error("El entrenamiento concluyó pero falló la exportación a ONNX.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
