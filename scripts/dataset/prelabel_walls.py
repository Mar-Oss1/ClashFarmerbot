"""
scripts/dataset/prelabel_walls.py
Pre-etiquetador automático de muros para acelerar la creación del dataset U-Net.
Lee capturas desde datasets/walls_pool/unlabeled/, corre WallSegDetector (models/wall_seg.onnx),
limpia interferencias mediante hud_mask y deploy_zone_mask, aplica cierre morfológico suave
para unir tramos continuos, simplifica los contornos con cv2.approxPolyDP y los exporta a
formato JSON compatible con LabelMe y X-AnyLabeling (clase "wall") junto a cada imagen
en datasets/walls_pool/prelabeled/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configurar path raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import get_logger
from vision.deploy_zone_detector import DeployZoneDetector
from vision.hud import get_battle_hud_mask
from vision.wall_seg_detector import WallSegDetector, build_cyan_overlay

log = get_logger("prelabel_walls")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _break_mask_holes(
    mask: np.ndarray,
    cut_thickness: int = 3,
    max_iterations: int = 20,
) -> np.ndarray:
    """
    Detecta huecos interiores en la máscara binaria usando cv2.findContours con cv2.RETR_CCOMP
    (contornos con hierarchy[i][3] != -1) y traza cortes limpios de 2-3 px en negro hacia el
    exterior a través de la pared más delgada usando distanceTransform, abriendo los
    compartimentos cerrados para que LabelMe no los interprete como bloques sólidos.
    """
    if mask is None or mask.size == 0 or not np.any(mask > 0):
        return mask

    m = mask.copy()
    h, w = m.shape[:2]

    for _ in range(max_iterations):
        cnts, hie = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hie is None or len(cnts) == 0:
            break

        hole_indices = [i for i, info in enumerate(hie[0]) if info[3] != -1]
        if not hole_indices:
            break

        # Identificar el exterior mediante floodFill desde los bordes de la imagen
        padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flood = padded.copy()
        cv2.floodFill(flood, None, (0, 0), 128)
        exterior = (flood[1:-1, 1:-1] == 128)

        # Matriz de distancia hacia el exterior legal
        non_exterior = (~exterior).astype(np.uint8)
        dist, labels = cv2.distanceTransformWithLabels(
            non_exterior, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )

        for idx in hole_indices:
            pts = cnts[idx][:, 0, :]
            # Distancias de los puntos del contorno del hueco hacia el exterior más cercano
            dists = dist[pts[:, 1], pts[:, 0]]
            best_pt_idx = int(np.argmin(dists))
            hx, hy = int(pts[best_pt_idx][0]), int(pts[best_pt_idx][1])

            # Pixel exterior más cercano
            lbl = int(labels[hy, hx])
            ey, ex = int(lbl // w), int(lbl % w)

            # Trazar corte limpio en negro (2-3 px) a través de la pared más delgada
            cv2.line(m, (hx, hy), (ex, ey), 0, thickness=int(cut_thickness))

    return m


def simplify_and_build_shapes(
    mask: np.ndarray,
    epsilon_px: float = 1.5,
    min_area: float = 40.0,
    target_label: str = "wall",
    break_holes: bool = True,
    cut_thickness: int = 3,
) -> Tuple[List[Dict[str, object]], int]:
    """
    Extrae contornos de la máscara binaria, filtra ruidos menores a min_area,
    aplica cv2.approxPolyDP para reducir vértices redundantes y construye
    la lista de shapes en formato LabelMe.
    Retorna (shapes, total_raw_points).
    """
    if break_holes:
        mask = _break_mask_holes(mask, cut_thickness=cut_thickness)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes: List[Dict[str, object]] = []
    total_raw_points = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        total_raw_points += len(cnt)
        approx = cv2.approxPolyDP(cnt, epsilon=epsilon_px, closed=True)
        if len(approx) < 3:
            continue

        # Formatear puntos como lista de coordenadas flotantes [[x, y], ...]
        points = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]

        shape_entry = {
            "label": target_label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "attributes": {},
            "difficult": False,
            "review_status": "pending",
            "score": None,
            "kie_linking": [],
        }
        shapes.append(shape_entry)

    return shapes, total_raw_points


def build_labelme_json(
    image_name: str,
    shapes: List[Dict[str, object]],
    height: int,
    width: int,
    version: str = "5.2.1",
    checked: bool = False,
    tags: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Construye la estructura de metadatos JSON estándar de LabelMe / X-AnyLabeling.
    """
    if tags is None:
        tags = ["wall"]
    return {
        "version": version,
        "flags": {},
        "checked": checked,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "tags": tags,
    }


def prelabel_image(
    image_path: Path,
    output_dir: Path,
    wall_detector: WallSegDetector,
    deploy_detector: Optional[DeployZoneDetector] = None,
    epsilon_px: float = 1.5,
    min_area: float = 40.0,
    close_kernel_sz: int = 5,
    preview_dir: Optional[Path] = None,
) -> Optional[Dict[str, object]]:
    """
    Procesa una imagen individual:
    1. Carga la imagen BGR.
    2. Ejecuta WallSegDetector para obtener la máscara cruda.
    3. Limpia la máscara con hud_mask.
    4. Limpia la máscara con deploy_zone_mask (zona no-desplegable = aldea/muros).
    5. Aplica cierre morfológico suave (MORPH_CLOSE).
    6. Simplifica contornos con approxPolyDP.
    7. Exporta imagen + JSON a output_dir.
    8. Opcionalmente guarda vista previa con overlay cian en preview_dir.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        log.warning("No se pudo cargar la imagen: %s. Omitiendo.", image_path)
        return None

    h, w = img.shape[:2]

    # 1. Inferencia del detector de muros (WallSegDetector)
    raw_wall_mask = wall_detector.predict(img)

    # 2. Limpieza estricta de HUD de batalla (barra inferior, botón siguiente, medidores)
    hud_mask = get_battle_hud_mask(h, w)
    cleaned_mask = raw_wall_mask.copy()
    cleaned_mask[hud_mask > 0] = 0

    # 3. Limpieza de zona exterior de despliegue (DeployZoneDetector)
    # En DeployZoneDetector: 1 = no-despliegue (aldea/muros), 0 = césped exterior desplegable
    # Los muros solo pueden estar en la zona de no-despliegue (d_mask == 1)
    if deploy_detector is not None and deploy_detector.is_ready():
        d_mask = deploy_detector.predict_mask(img)
        if d_mask is not None:
            cleaned_mask[d_mask == 0] = 0

    # 4. Cierre morfológico suave para unir micro-grietas y sellar polígonos
    if close_kernel_sz > 1:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(close_kernel_sz), int(close_kernel_sz))
        )
        final_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, k_close)
    else:
        final_mask = cleaned_mask

    # Garantizar HUD en cero
    final_mask[hud_mask > 0] = 0

    # 4.5. Romper huecos interiores para evitar que compartimentos cerrados se exporten sólidos
    final_mask = _break_mask_holes(final_mask, cut_thickness=3)

    # 5. Extracción y simplificación de contornos poligonales (approxPolyDP)
    shapes, total_raw_pts = simplify_and_build_shapes(
        final_mask,
        epsilon_px=epsilon_px,
        min_area=min_area,
        target_label="wall",
        break_holes=False,  # Ya aplicado arriba sobre final_mask
    )

    # 6. Guardar imagen y JSON en output_dir
    dest_img = output_dir / image_path.name
    dest_json = output_dir / f"{image_path.stem}.json"

    # Copiar imagen si el origen y destino difieren
    if image_path.resolve() != dest_img.resolve():
        shutil.copy2(image_path, dest_img)

    # Construir y guardar JSON
    labelme_data = build_labelme_json(
        image_name=image_path.name,
        shapes=shapes,
        height=h,
        width=w,
    )
    with open(dest_json, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)

    # 7. Vista previa opcional con overlay cian
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_img = build_cyan_overlay(img, final_mask, alpha=0.40)
        # Añadir banner informativo superior
        banner_h = max(32, int(h * 0.035))
        banner = preview_img[:banner_h, :].copy()
        black_bar = np.zeros_like(banner)
        cv2.addWeighted(black_bar, 0.70, banner, 0.30, 0, banner)
        preview_img[:banner_h, :] = banner
        info_txt = f"{image_path.name} | Poligonos: {len(shapes)} | Vertices: {sum(len(s['points']) for s in shapes)}"
        cv2.putText(
            preview_img,
            info_txt,
            (12, int(banner_h * 0.70)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        preview_path = preview_dir / f"{image_path.stem}_preview.png"
        cv2.imwrite(str(preview_path), preview_img)

    total_simplified_pts = sum(len(s["points"]) for s in shapes)
    wall_px = int(np.count_nonzero(final_mask > 0))

    return {
        "filename": image_path.name,
        "polygons_count": len(shapes),
        "raw_points": total_raw_pts,
        "simplified_points": total_simplified_pts,
        "reduction_pct": round(
            (1.0 - total_simplified_pts / float(total_raw_pts)) * 100.0, 1
        ) if total_raw_pts > 0 else 0.0,
        "wall_pixels": wall_px,
        "json_path": str(dest_json),
    }


def run_prelabeling(
    input_dir: Path,
    output_dir: Path,
    model_path: str = "models/wall_seg.onnx",
    deploy_model_path: str = "models/deploy_zone_finder.onnx",
    conf_thresh: float = 0.50,
    limit: Optional[int] = None,
    epsilon_px: float = 1.5,
    min_area: float = 40.0,
    close_kernel_sz: int = 5,
    preview_dir: Optional[Path] = None,
    clean_output: bool = False,
    skip_existing_json: bool = True,
) -> Dict[str, object]:
    """
    Ejecuta el pipeline de preetiquetado sobre las imágenes en input_dir.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"El directorio de entrada especificado no existe: '{input_dir.resolve()}'"
        )

    # Buscar imágenes en input_dir
    candidates: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        candidates.extend(input_dir.glob(f"*{ext}"))
        candidates.extend(input_dir.glob(f"*{ext.upper()}"))
    candidates = sorted(list(set(candidates)))

    if skip_existing_json:
        # Filtrar únicamente imágenes que NO tienen ya un archivo .json manual en input_dir
        candidates = [img for img in candidates if not img.with_suffix(".json").exists()]

    if not candidates:
        log.warning("No se encontraron imágenes sin etiquetar en: %s", input_dir)
        return {
            "total_found": 0,
            "total_processed": 0,
            "items": [],
        }

    # Aplicar límite configurado
    if limit is not None and limit > 0:
        targets = candidates[:limit]
    else:
        targets = candidates

    log.info("Imágenes encontradas: %d. Procesando lote de %d...", len(candidates), len(targets))

    # Inicializar detectores
    log.info("Cargando detector WallSegDetector (%s)...", model_path)
    wall_detector = WallSegDetector(model_path=model_path, conf_thresh=conf_thresh)

    log.info("Cargando detector DeployZoneDetector (%s)...", deploy_model_path)
    deploy_detector = DeployZoneDetector(model_path=deploy_model_path)

    if clean_output and output_dir.exists():
        for f in output_dir.glob("*"):
            if f.is_file():
                f.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    t_start = time.time()

    for idx, img_path in enumerate(targets, 1):
        t0 = time.time()
        res = prelabel_image(
            image_path=img_path,
            output_dir=output_dir,
            wall_detector=wall_detector,
            deploy_detector=deploy_detector,
            epsilon_px=epsilon_px,
            min_area=min_area,
            close_kernel_sz=close_kernel_sz,
            preview_dir=preview_dir,
        )
        elapsed = time.time() - t0
        if res:
            res["elapsed_s"] = round(elapsed, 2)
            results.append(res)
            log.info(
                "[%d/%d] Pre-etiquetado %s -> %d polígonos (%d vértices, -%.1f%%) en %.2fs",
                idx,
                len(targets),
                img_path.name,
                res["polygons_count"],
                res["simplified_points"],
                res["reduction_pct"],
                elapsed,
            )

    total_time = time.time() - t_start
    total_polygons = sum(int(r["polygons_count"]) for r in results)

    return {
        "total_found": len(candidates),
        "total_processed": len(results),
        "total_polygons": total_polygons,
        "total_time_s": round(total_time, 2),
        "avg_time_per_image_s": round(total_time / len(results), 2) if results else 0.0,
        "output_dir": str(output_dir),
        "preview_dir": str(preview_dir) if preview_dir else None,
        "items": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-etiquetador automático de muros (LabelMe/X-AnyLabeling JSON) para ClashFarmerbot."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets/coc_walls_raw",
        help="Directorio con imágenes de batalla sin etiquetar (default: datasets/coc_walls_raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/coc_walls_prelabeled",
        help="Directorio destino para imágenes y archivos JSON de anotación (default: datasets/coc_walls_prelabeled).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Número máximo de imágenes a procesar (default: 0 = sin límite, procesar todas).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/wall_seg.onnx",
        help="Ruta al modelo U-Net ONNX de muros.",
    )
    parser.add_argument(
        "--deploy-model-path",
        type=str,
        default="models/deploy_zone_finder.onnx",
        help="Ruta al modelo U-Net ONNX de zona de no-despliegue.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Umbral de confianza para binarizar la predicción de muros (default: 0.50).",
    )
    parser.add_argument(
        "--epsilon-px",
        type=float,
        default=1.5,
        help="Épsilon en píxeles para cv2.approxPolyDP (default: 1.5).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=40.0,
        help="Área mínima de contorno en px^2 para filtrar ruido (default: 40.0).",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=5,
        help="Tamaño de kernel morfológico para cierre suave (default: 5).",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        default=False,
        help="Genera además imágenes de verificación visual con overlay cian en scratch/wall_prelabels_preview.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Limpia el directorio destino antes de guardar nuevos datos.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        default=False,
        help="No saltar imágenes que ya tienen un archivo .json manual en input-dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    preview_dir = (PROJECT_ROOT / "scratch/wall_prelabels_preview") if args.save_preview else None

    try:
        summary = run_prelabeling(
            input_dir=input_dir,
            output_dir=output_dir,
            model_path=args.model_path,
            deploy_model_path=args.deploy_model_path,
            conf_thresh=args.conf,
            limit=args.limit,
            epsilon_px=args.epsilon_px,
            min_area=args.min_area,
            close_kernel_sz=args.close_kernel,
            preview_dir=preview_dir,
            clean_output=args.clean,
            skip_existing_json=not args.no_skip_existing,
        )
    except FileNotFoundError as e:
        log.error("ERROR: %s", e)
        return 1
    except Exception as e:
        log.exception("Error durante la ejecución de prelabel_walls: %s", e)
        return 2

    print("\n" + "=" * 65)
    print("RESUMEN DE PRE-ETIQUETADO AUTOMÁTICO DE MUROS")
    print("=" * 65)
    print(f"Total imágenes encontradas : {summary['total_found']}")
    print(f"Total procesadas en el lote: {summary['total_processed']}")
    print(f"Total polígonos generados  : {summary['total_polygons']}")
    print(f"Tiempo total               : {summary['total_time_s']}s (promedio: {summary['avg_time_per_image_s']}s/img)")
    print(f"Directorio de salida       : {summary['output_dir']}")
    if summary["preview_dir"]:
        print(f"Previews de auditoría      : {summary['preview_dir']}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
