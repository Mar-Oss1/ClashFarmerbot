"""
tools/json_to_wall_mask.py
Convierte anotaciones poligonales de X-AnyLabeling / LabelMe (JSON, label "wall")
en máscaras binarias PNG para entrenar el modelo U-Net (WallSegDetector).

Pipeline de conversión:
1. Lee pares imagen (.png/.jpg) + anotación (.json) desde directorio de entrada.
2. Rasteriza polígonos (cv2.fillPoly), rectángulos y linestrips (~8px fijo).
3. Aplica cv2.dilate (kernel 3x3, 1 iteración) ANTES de cualquier redimensionamiento,
   preservando tramos delgados de muro frente al submuestreo.
4. Aplica cierre morfológico (cv2.MORPH_CLOSE, kernel configurable) para sellar micro-fisuras.
5. Enmascara zona del HUD inferior de batalla: y >= round(h * 888 / 1080) a 0.
6. Divide el conjunto en train/val (ratio configurable, min 1 en val si >= 5 imágenes).
7. Guarda datasets/coc_walls_unet/{train,val}/{images,masks}/*.png.
8. Genera vistas previas de auditoría en scratch/wall_unet_mask_preview/ con overlay cian y métricas.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configurar path raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import get_logger

log = get_logger("json_to_wall_mask")

# Extensiones de imagen soportadas
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def build_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.40,
    draw_contour: bool = True,
    contour_color: Tuple[int, int, int] = (0, 255, 0),
    contour_thickness: int = 2,
    overlay_color_bgr: Tuple[int, int, int] = (255, 255, 0),  # Cian en BGR
) -> np.ndarray:
    """
    Crea un overlay visual semitransparente cian con contorno verde sobre la imagen original.
    Reutilizable tanto para auditoría de preetiquetado como para inferencia en tiempo real.
    """
    if image_bgr is None or mask is None:
        raise ValueError("image_bgr y mask no pueden ser None")

    h, w = image_bgr.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    bin_mask = (mask > 0).astype(np.uint8)

    # Capa de color cian sobre los píxeles de muro
    colored_overlay = image_bgr.copy()
    colored_overlay[bin_mask == 1] = overlay_color_bgr

    # Fusión semitransparente
    blended = cv2.addWeighted(colored_overlay, alpha, image_bgr, 1.0 - alpha, 0)

    # Dibujar contornos verdes en los bordes del muro
    if draw_contour and np.any(bin_mask):
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, contour_color, thickness=contour_thickness)

    return blended


def render_wall_mask_preview(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    split: str = "train",
    filename: str = "",
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Genera la imagen de verificación visual completa, incluyendo el overlay cian/verde
    y una barra de encabezado con métricas (split, nombre, píxeles de muro y porcentaje).
    """
    preview = build_overlay(image_bgr, mask, alpha=alpha, draw_contour=True)
    h, w = preview.shape[:2]

    # Calcular estadísticas de muro
    wall_pixels = int(np.count_nonzero(mask > 0))
    total_pixels = int(mask.size)
    wall_pct = (wall_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    # Barra superior oscura semitransparente para legibilidad de texto
    banner_h = max(36, int(h * 0.04))
    banner = preview[:banner_h, :].copy()
    black_bar = np.zeros_like(banner)
    cv2.addWeighted(black_bar, 0.70, banner, 0.30, 0, banner)
    preview[:banner_h, :] = banner

    # Texto descriptivo
    text = (
        f"[{split.upper()}] {filename} | Muro: {wall_pixels:,} px ({wall_pct:.2f}%) | "
        f"Res: {w}x{h}"
    )
    font_scale = max(0.5, min(1.0, w / 2400.0 * 0.75))
    font_thickness = max(1, int(round(font_scale * 2)))
    text_y = int(banner_h * 0.68)
    cv2.putText(
        preview,
        text,
        (15, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),  # Amarillo claro
        font_thickness,
        cv2.LINE_AA,
    )

    return preview


def rasterize_shape(
    mask: np.ndarray,
    shape: dict,
    linestrip_thickness: int = 8,
    target_label: str = "wall",
) -> bool:
    """
    Rasteriza una forma geométrica de LabelMe en la máscara binaria.
    Soporta: 'polygon', 'rectangle', 'linestrip' y 'line'.
    Retorna True si la forma fue procesada, False si fue ignorada.
    """
    label = str(shape.get("label", "")).strip().lower()
    if label != target_label.lower():
        return False

    shape_type = str(shape.get("shape_type", "polygon")).lower()
    raw_points = shape.get("points", [])
    if not raw_points:
        return False

    pts = np.array(raw_points, dtype=np.float32)

    if shape_type == "polygon":
        if len(pts) >= 3:
            int_pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [int_pts], color=255)
            return True

    elif shape_type == "rectangle":
        if len(pts) >= 2:
            x1 = int(round(min(pts[0][0], pts[1][0])))
            y1 = int(round(min(pts[0][1], pts[1][1])))
            x2 = int(round(max(pts[0][0], pts[1][0])))
            y2 = int(round(max(pts[0][1], pts[1][1])))
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)
            return True

    elif shape_type in ("linestrip", "line"):
        if len(pts) >= 2:
            int_pts = np.round(pts).astype(np.int32)
            cv2.polylines(
                mask,
                [int_pts],
                isClosed=False,
                color=255,
                thickness=int(linestrip_thickness),
            )
            return True

    elif shape_type == "circle":
        if len(pts) >= 2:
            cx, cy = int(round(pts[0][0])), int(round(pts[0][1]))
            radius = int(round(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])))
            cv2.circle(mask, (cx, cy), radius, color=255, thickness=-1)
            return True

    else:
        # Fallback genérico a polígono si tiene >= 3 puntos
        if len(pts) >= 3:
            int_pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [int_pts], color=255)
            return True

    return False


def convert_json_to_mask(
    json_path: Path,
    expected_shape: Tuple[int, int],
    close_kernel: int = 3,
    linestrip_thickness: int = 8,
    target_label: str = "wall",
) -> np.ndarray:
    """
    Carga un archivo JSON LabelMe y produce una máscara binaria (H, W) uint8 (0 o 255):
    1. Rasteriza todas las figuras con label 'wall'.
    2. Aplica cv2.dilate (kernel 3x3, 1 iteración) ANTES de cualquier resize.
    3. Aplica cv2.MORPH_CLOSE (kernel close_kernel).
    4. Enmascara zona inferior HUD de batalla: y >= round(h * 888 / 1080) a 0.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determinar dimensiones de imagen (desde JSON o desde la imagen real leída)
    json_h = data.get("imageHeight")
    json_w = data.get("imageWidth")
    if json_h and json_w:
        h, w = int(json_h), int(json_w)
    else:
        h, w = expected_shape

    mask = np.zeros((h, w), dtype=np.uint8)

    shapes = data.get("shapes", [])
    valid_shapes = 0
    for shape in shapes:
        if rasterize_shape(mask, shape, linestrip_thickness=linestrip_thickness, target_label=target_label):
            valid_shapes += 1

    # 1. cv2.dilate (3x3, 1 iter) ANTES de cualquier resize para engrosar trazos delgados
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, k_dilate, iterations=1)

    # 2. Cierre morfológico para sellar micro-grietas entre polígonos adyacentes
    if close_kernel > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (int(close_kernel), int(close_kernel)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # 3. Enmascarar HUD inferior (y >= round(h * 888 / 1080) puesto a 0)
    y_hud = int(round(h * 888.0 / 1080.0))
    if 0 <= y_hud < h:
        mask[y_hud:, :] = 0

    return mask


def find_image_json_pairs(input_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Escanea input_dir buscando pares (imagen, json) válidos.
    Falla con error descriptivo si no hay JSONs o si ningún JSON tiene imagen coincidente.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"El directorio de entrada especificado no existe: '{input_dir.resolve()}'"
        )

    json_files = sorted(list(input_dir.glob("*.json")))
    if not json_files:
        raise FileNotFoundError(
            f"No se encontraron archivos JSON en '{input_dir.resolve()}'. "
            "Asegúrate de exportar las anotaciones de X-AnyLabeling/LabelMe en dicho directorio."
        )

    pairs: List[Tuple[Path, Path]] = []
    missing_images: List[str] = []

    for json_path in json_files:
        stem = json_path.stem
        # 1. Buscar imagen con mismo stem y extensión común
        image_path: Optional[Path] = None
        for ext in IMAGE_EXTENSIONS:
            candidate = input_dir / f"{stem}{ext}"
            if candidate.is_file():
                image_path = candidate
                break

        # 2. Si no se encuentra por stem, intentar leer imagePath dentro del JSON
        if image_path is None:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    rel_img = meta.get("imagePath")
                    if rel_img:
                        cand = input_dir / Path(rel_img).name
                        if cand.is_file():
                            image_path = cand
            except Exception:
                pass

        if image_path is not None and image_path.is_file():
            pairs.append((image_path, json_path))
        else:
            missing_images.append(json_path.name)

    if not pairs:
        raise FileNotFoundError(
            f"Se encontraron {len(json_files)} JSON(s) en '{input_dir.resolve()}', pero ninguno "
            f"tiene un archivo de imagen correspondiente ({', '.join(IMAGE_EXTENSIONS)})."
        )

    if missing_images:
        log.warning(
            "Se omitieron %d JSON(s) sin imagen coincidente: %s",
            len(missing_images),
            missing_images[:5],
        )

    return pairs


def split_dataset(
    pairs: List[Tuple[Path, Path]],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Divide los pares en train y val de forma reproducible:
    - ratio configurable por val_ratio (default 0.15).
    - mínimo 1 imagen en val si el dataset tiene >= 5 imágenes.
    """
    total = len(pairs)
    if total == 0:
        return [], []

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    if total >= 5:
        n_val = max(1, int(round(total * val_ratio)))
    else:
        n_val = int(round(total * val_ratio))

    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]

    # Asegurar que train no quede vacío si hay al menos 2 imágenes
    if len(train_pairs) == 0 and total >= 2:
        train_pairs = [val_pairs.pop()]

    return train_pairs, val_pairs


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    preview_dir: Path,
    val_ratio: float = 0.15,
    close_kernel: int = 3,
    linestrip_thickness: int = 8,
    seed: int = 42,
    clean: bool = True,
) -> Dict[str, object]:
    """
    Ejecuta el procesamiento integral del dataset:
    - Encuentra pares imagen+json.
    - Divide en train/val.
    - Convierte cada JSON a máscara PNG aplicando dilatación y cierre.
    - Genera previews de auditoría en preview_dir.
    - Retorna diccionario de estadísticas.
    """
    pairs = find_image_json_pairs(input_dir)
    log.info("Pares válidos imagen+JSON encontrados: %d", len(pairs))

    train_pairs, val_pairs = split_dataset(pairs, val_ratio=val_ratio, seed=seed)
    log.info("División de dataset: %d en train, %d en val", len(train_pairs), len(val_pairs))

    # Preparar directorios de salida
    dirs = {
        "train_img": output_dir / "train" / "images",
        "train_mask": output_dir / "train" / "masks",
        "val_img": output_dir / "val" / "images",
        "val_mask": output_dir / "val" / "masks",
    }
    for d in dirs.values():
        if clean and d.exists():
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()
        d.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, object] = {
        "total_pairs": len(pairs),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "train_wall_pixels": 0,
        "val_wall_pixels": 0,
        "items": [],
    }

    splits = [("train", train_pairs), ("val", val_pairs)]

    for split_name, split_list in splits:
        out_img_dir = dirs[f"{split_name}_img"]
        out_mask_dir = dirs[f"{split_name}_mask"]

        for img_path, json_path in split_list:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("No se pudo cargar la imagen: %s. Omitiendo.", img_path)
                continue

            h, w = img.shape[:2]

            # Convertir JSON a máscara binaria
            mask = convert_json_to_mask(
                json_path=json_path,
                expected_shape=(h, w),
                close_kernel=close_kernel,
                linestrip_thickness=linestrip_thickness,
            )

            # Destinos
            dest_img = out_img_dir / f"{img_path.stem}.png"
            dest_mask = out_mask_dir / f"{img_path.stem}.png"

            # Guardar imagen (PNG sin pérdida)
            if img_path.suffix.lower() == ".png":
                shutil.copy2(img_path, dest_img)
            else:
                cv2.imwrite(str(dest_img), img)

            # Guardar máscara (PNG monomodal 0 o 255)
            cv2.imwrite(str(dest_mask), mask)

            # Generar preview de auditoría visual
            preview_img = render_wall_mask_preview(
                image_bgr=img,
                mask=mask,
                split=split_name,
                filename=img_path.name,
            )
            preview_path = preview_dir / f"{split_name}_{img_path.stem}_preview.png"
            cv2.imwrite(str(preview_path), preview_img)

            wall_px = int(np.count_nonzero(mask > 0))
            if split_name == "train":
                stats["train_wall_pixels"] = int(stats["train_wall_pixels"]) + wall_px
            else:
                stats["val_wall_pixels"] = int(stats["val_wall_pixels"]) + wall_px

            stats["items"].append({
                "split": split_name,
                "filename": img_path.name,
                "wall_pixels": wall_px,
                "dimensions": (w, h),
            })

            log.info(
                "[%s] %s -> máscara guardada (%d px de muro)",
                split_name,
                img_path.name,
                wall_px,
            )

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte anotaciones LabelMe JSON en máscaras PNG binarias para U-Net (ClashFarmerbot)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets/coc_walls_raw",
        help="Directorio con capturas e imágenes JSON de X-AnyLabeling / LabelMe.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/coc_walls_unet",
        help="Directorio destino del dataset U-Net (train/val con images/masks).",
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        default="scratch/wall_unet_mask_preview",
        help="Directorio donde guardar previews visuales con overlay cian.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fracción del dataset asignada a validación (default: 0.15, min 1 si >= 5 imágenes).",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=3,
        help="Tamaño de kernel para cv2.MORPH_CLOSE (default: 3).",
    )
    parser.add_argument(
        "--linestrip-thickness",
        type=int,
        default=8,
        help="Grosor en píxeles para shape_type linestrip/line (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para partición reproducible train/val (default: 42).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        default=False,
        help="No limpia los directorios de salida antes de escribir nuevos datos.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    preview_dir = PROJECT_ROOT / args.preview_dir

    try:
        stats = process_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            preview_dir=preview_dir,
            val_ratio=args.val_ratio,
            close_kernel=args.close_kernel,
            linestrip_thickness=args.linestrip_thickness,
            seed=args.seed,
            clean=not args.no_clean,
        )
    except FileNotFoundError as e:
        log.error("ERROR: %s", e)
        return 1
    except Exception as e:
        log.exception("Error inesperado durante la conversión de máscaras: %s", e)
        return 2

    log.info("=================================================================")
    log.info("Conversión a máscaras U-Net completada con éxito.")
    log.info("Total imágenes procesadas : %d", stats["total_pairs"])
    log.info("  - Train                 : %d imágenes (%d px de muro totales)", stats["train_count"], stats["train_wall_pixels"])
    log.info("  - Val                   : %d imágenes (%d px de muro totales)", stats["val_count"], stats["val_wall_pixels"])
    log.info("Estructura generada en    : %s", output_dir)
    log.info("Previews de auditoría en  : %s", preview_dir)
    log.info("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
