"""
scripts/dataset/triage_images.py
Escanea carpetas de capturas disponibles (excluyendo assets/),
filtra únicamente capturas limpias de batalla en vista completa
(descarta overlays debug con cajas/líneas ya dibujadas, pantallas de aldea propia,
popups y duplicados usando hash perceptual/thumbnail).

Separa las válidas en datasets/walls_pool/unlabeled/ y genera un CSV con métricas básicas
(resolución, brillo/escenario estimado, dHash) para elegir variedad de niveles de muro y sceneries.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Configurar path raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import get_logger
from core.vision import Vision

log = get_logger("triage_images")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

EXCLUDE_DIRS = {
    "assets",
    ".git",
    ".venv",
    ".idea",
    ".gemini",
    "_quarantine",
    "_backups",
    "backups",
    "walls_pool",
}

DEBUG_FILENAME_KEYWORDS = (
    "debug",
    "overlay",
    "preview",
    "_mask",
    "_pred",
    "_draw",
    "annotated",
    "_box",
    "gt_comparison",
    "crop",
    "_sm.",
    "tooltip",
    "labels.jpg",
    "batch",
    "slot",
    "card",
    "drop_debug",
    "template",
    "bar25_",
)


def compute_dhash(image_bgr: np.ndarray, hash_size: int = 8) -> Tuple[int, str]:
    """
    Calcula el Difference Hash (dHash) perceptual de 64 bits.
    Retorna (hash_int, hash_hex).
    """
    resized = cv2.resize(image_bgr, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    val = 0
    for b in diff.flatten():
        val = (val << 1) | int(b)
    hex_str = f"{val:016x}"
    return val, hex_str


def hamming_distance(h1: int, h2: int) -> int:
    """Calcula la distancia de Hamming entre dos hashes de 64 bits."""
    return bin(h1 ^ h2).count("1")


def analyze_scenery(image_bgr: np.ndarray) -> Dict[str, object]:
    """
    Analiza brillo, contraste, saturación, matiz y porcentaje de verde para estimar
    el tipo de escenario (scenery) y métricas de calidad de la captura.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    hue_channel = hsv[:, :, 0]
    sat_channel = hsv[:, :, 1]
    val_channel = hsv[:, :, 2]

    hue_mean = float(np.mean(hue_channel))
    sat_mean = float(np.mean(sat_channel))
    val_mean = float(np.mean(val_channel))

    # Píxeles con tono verde característico de césped (Hue 35 a 85, Saturación >= 40)
    green_mask = (hue_channel >= 35) & (hue_channel <= 85) & (sat_channel >= 40)
    green_ratio = float(np.count_nonzero(green_mask) / (h * w))

    # Heurística de clasificación de escenario
    if brightness_mean > 130 and sat_mean < 55:
        estimated_scenery = "Snow / Frost / Winter"
    elif green_ratio >= 0.22:
        estimated_scenery = "Classic Grass"
    elif val_mean < 75 or brightness_mean < 75:
        estimated_scenery = "Dark / Shadow / Night"
    elif (hue_mean < 25 or hue_mean > 155) and sat_mean > 45:
        estimated_scenery = "Inferno / Autumn / Lava"
    elif 85 < hue_mean < 140 and sat_mean > 35:
        estimated_scenery = "Fantasy / Magic / Cosmic"
    else:
        estimated_scenery = "Alternative / Custom"

    return {
        "brightness_mean": round(brightness_mean, 2),
        "contrast_std": round(contrast_std, 2),
        "hue_mean": round(hue_mean, 2),
        "sat_mean": round(sat_mean, 2),
        "green_ratio": round(green_ratio, 4),
        "estimated_scenery": estimated_scenery,
    }


class ImageTriager:
    """
    Filtra y clasifica capturas de batalla para el dataset de muros.
    """

    def __init__(
        self,
        vision: Optional[Vision] = None,
        max_hash_dist: int = 4,
        require_battle_hud: bool = True,
    ):
        self.vision = vision or Vision()
        self.vision.preload_templates("assets/buttons")
        self.max_hash_dist = max_hash_dist
        self.require_battle_hud = require_battle_hud

        # Templates de verificación
        self.tpl_own_village_atk = [
            "assets/buttons/attack_btn.png",
            "assets/buttons/attack_btn_event.png",
            "assets/buttons/battle_text.png",
        ]
        self.tpl_own_village_build = "assets/buttons/build_btn.png"

        self.tpl_popups_sup = [
            "assets/buttons/red_x_close.png",
            "assets/buttons/blue_x_close.png",
            "assets/buttons/close_btn.png",
            "assets/buttons/train_close_btn.png",
        ]
        self.tpl_popups_modal = [
            "assets/buttons/okay_btn.png",
            "assets/buttons/dialog_confirm_btn.png",
        ]

        self.tpl_battle_results = [
            "assets/buttons/return_home.png",
            "assets/buttons/continue_btn.png",
            "assets/buttons/continue_text.png",
        ]

        self.tpl_battle_next = "assets/buttons/next_btn.png"
        self.tpl_battle_surrender = [
            "assets/buttons/end_battle.png",
            "assets/buttons/surrender_battle.png",
        ]

        # Registro de hashes aceptados para deduplicación: list of (hash_int, filename)
        self.accepted_hashes: List[Tuple[int, str]] = []

    def check_clean_battle(
        self, image_bgr: np.ndarray, file_path: Path
    ) -> Tuple[bool, str]:
        """
        Evalúa si la captura es una imagen limpia de batalla en vista completa.
        Retorna (is_clean, reason).
        Optimizado en orden de descarte de menor a mayor coste computacional.
        """
        # 1. Filtro rápido por nombre de archivo (0.01ms)
        stem_lower = file_path.name.lower()
        for kw in DEBUG_FILENAME_KEYWORDS:
            if kw in stem_lower:
                return False, f"filename_debug_indicator ({kw})"

        # 2. Comprobación geométrica: relación de aspecto y resolución mínima (0.01ms)
        h, w = image_bgr.shape[:2]
        if w < 800 or h < 400:
            return False, f"resolution_too_small ({w}x{h})"

        aspect_ratio = w / float(h)
        if aspect_ratio < 1.60:
            return False, f"aspect_ratio_not_landscape ({aspect_ratio:.2f})"

        # 3. Comprobación de imagen negra / carga / pantalla en blanco (0.5ms)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        b_mean = float(np.mean(gray))
        c_std = float(np.std(gray))
        if b_mean < 25.0 or c_std < 15.0:
            return False, f"blank_or_loading_screen (mean={b_mean:.1f}, std={c_std:.1f})"

        # 4. Detección de overlays debug por colores puros artificiales de dibujo OpenCV (>250 px)
        # Cian puro (overlay de segmentación U-Net: B>=245, G>=245, R<=10)
        cyan_count = np.count_nonzero(
            (image_bgr[:, :, 0] >= 245) & (image_bgr[:, :, 1] >= 245) & (image_bgr[:, :, 2] <= 10)
        )
        if cyan_count > 250:
            return False, f"cyan_debug_overlay ({cyan_count} px)"

        # Verde puro (contornos o cajas YOLO: B<=10, G>=245, R<=10)
        green_box_count = np.count_nonzero(
            (image_bgr[:, :, 0] <= 10) & (image_bgr[:, :, 1] >= 245) & (image_bgr[:, :, 2] <= 10)
        )
        if green_box_count > 250:
            return False, f"green_debug_contour ({green_box_count} px)"

        # Rojo puro (cajas o puntos de drop debug: B<=10, G<=10, R>=245)
        red_box_count = np.count_nonzero(
            (image_bgr[:, :, 0] <= 10) & (image_bgr[:, :, 1] <= 10) & (image_bgr[:, :, 2] >= 245)
        )
        if red_box_count > 250:
            return False, f"red_debug_boxes ({red_box_count} px)"

        # Amarillo puro (anotaciones cv2: B<=10, G>=245, R>=245)
        yellow_box_count = np.count_nonzero(
            (image_bgr[:, :, 0] <= 10) & (image_bgr[:, :, 1] >= 245) & (image_bgr[:, :, 2] >= 245)
        )
        if yellow_box_count > 250:
            return False, f"yellow_debug_boxes ({yellow_box_count} px)"

        # Magenta puro (anotaciones cv2: B>=245, G<=10, R>=245)
        magenta_box_count = np.count_nonzero(
            (image_bgr[:, :, 0] >= 245) & (image_bgr[:, :, 1] <= 10) & (image_bgr[:, :, 2] >= 245)
        )
        if magenta_box_count > 250:
            return False, f"magenta_debug_boxes ({magenta_box_count} px)"

        # 5. Confirmación RÁPIDA de HUD de Batalla mediante ROIs locales (~15-25ms)
        # En batalla activa existe el botón 'Next' (derecha) o 'End/Surrender' (izquierda)
        has_next = bool(
            self.vision.find_template(
                image_bgr, self.tpl_battle_next, threshold=0.70, zona="ESQUINA_INF_DER"
            )
        )
        has_surrender = False
        if not has_next:
            has_surrender = bool(
                self.vision.find_any(
                    image_bgr, self.tpl_battle_surrender, threshold=0.70, zona="ESQUINA_INF_IZQ"
                )
            )

        if not (has_next or has_surrender):
            if self.require_battle_hud:
                return False, "no_battle_hud_detected"

        # 6. Detección de Aldea Propia (Home Village) mediante ROI local
        if not has_next:
            # Botón Attack en esquina inferior izquierda
            if self.vision.find_any(
                image_bgr, self.tpl_own_village_atk, threshold=0.72, zona="ESQUINA_INF_IZQ"
            ):
                return False, "own_village_attack_button"

        if not has_surrender:
            # Botón Build en esquina inferior derecha
            if self.vision.find_template(
                image_bgr, self.tpl_own_village_build, threshold=0.72, zona="ESQUINA_INF_DER"
            ):
                return False, "own_village_build_button"

        # 7. Detección de Popups / Modales / Diálogos mediante ROIs superiores
        if self.vision.find_any(
            image_bgr, self.tpl_popups_sup, threshold=0.72, zona="ESQUINA_SUP_DER"
        ) or self.vision.find_any(
            image_bgr, self.tpl_popups_sup, threshold=0.72, zona="MITAD_SUPERIOR"
        ):
            return False, "popup_close_button"

        if self.vision.find_any(
            image_bgr, self.tpl_popups_modal, threshold=0.72, zona="CENTRO"
        ):
            return False, "popup_dialog_button"

        # 8. Detección de Fin de Batalla / Recompensa (Return Home, Continue)
        if self.vision.find_any(
            image_bgr, self.tpl_battle_results, threshold=0.72, zona="MITAD_INFERIOR"
        ):
            return False, "battle_results_or_chest"

        return True, "clean_battle_view"

    def check_and_register_duplicate(
        self, dhash_val: int, filename: str
    ) -> Optional[str]:
        """
        Compara con los hashes existentes. Retorna el nombre del archivo duplicado
        si la distancia de Hamming <= max_hash_dist, o None si es único.
        """
        for existing_hash, existing_name in self.accepted_hashes:
            if hamming_distance(dhash_val, existing_hash) <= self.max_hash_dist:
                return existing_name
        self.accepted_hashes.append((dhash_val, filename))
        return None


def collect_candidate_images(
    source_dirs: List[Path],
    recursive: bool = True,
) -> List[Path]:
    """
    Recolecta todos los archivos de imagen bajo source_dirs, ignorando
    directorios de exclusión como assets, .venv, git, etc.
    """
    candidates: List[Path] = []
    seen_paths: Set[Path] = set()

    for sdir in source_dirs:
        if not sdir.exists():
            log.warning("Directorio de búsqueda no encontrado: %s", sdir)
            continue

        if sdir.is_file():
            if sdir.suffix.lower() in IMAGE_EXTENSIONS and sdir not in seen_paths:
                candidates.append(sdir)
                seen_paths.add(sdir)
            continue

        for root, dirs, files in os.walk(sdir):
            # Filtrar carpetas excluidas in-place
            dirs[:] = [
                d for d in dirs
                if d not in EXCLUDE_DIRS and not d.startswith((".", "_"))
            ]

            for fname in files:
                p = Path(root) / fname
                if p.suffix.lower() in IMAGE_EXTENSIONS and p not in seen_paths:
                    # Comprobación de seguridad adicional: no procesar assets/
                    parts = p.resolve().parts
                    if any(ex in parts for ex in EXCLUDE_DIRS):
                        continue
                    candidates.append(p)
                    seen_paths.add(p)

    return sorted(candidates)


def run_triage(
    source_dirs: List[Path],
    output_dir: Path,
    csv_path: Path,
    max_hash_dist: int = 4,
    require_battle_hud: bool = True,
    limit: Optional[int] = None,
    clean_output: bool = False,
) -> Dict[str, object]:
    """
    Ejecuta el pipeline completo de triage y exporta resultados a output_dir y csv_path.
    """
    log.info("Iniciando escaneo de capturas...")
    candidates = collect_candidate_images(source_dirs)
    log.info("Total de imágenes candidatas encontradas: %d", len(candidates))

    if clean_output and output_dir.exists():
        for f in output_dir.glob("*"):
            if f.is_file():
                f.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    triager = ImageTriager(
        max_hash_dist=max_hash_dist,
        require_battle_hud=require_battle_hud,
    )

    accepted_records: List[Dict[str, object]] = []
    rejection_counts: Dict[str, int] = {}
    total_processed = 0

    for idx, img_path in enumerate(candidates, 1):
        if limit is not None and len(accepted_records) >= limit:
            log.info("Límite alcanzado (%d imágenes aceptadas). Finalizando triage.", limit)
            break

        total_processed += 1
        img = cv2.imread(str(img_path))
        if img is None:
            reason = "unreadable_image"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue

        # 1. Comprobación de pantalla de batalla limpia
        is_clean, reason = triager.check_clean_battle(img, img_path)
        if not is_clean:
            cat = reason.split()[0]
            rejection_counts[cat] = rejection_counts.get(cat, 0) + 1
            continue

        # 2. Cálculo de dHash y deduplicación perceptual
        dhash_int, dhash_hex = compute_dhash(img)
        dup_match = triager.check_and_register_duplicate(dhash_int, img_path.name)
        if dup_match:
            reason = f"duplicate_hash (similar to {dup_match})"
            rejection_counts["duplicate_hash"] = rejection_counts.get("duplicate_hash", 0) + 1
            continue

        # 3. Métricas de escenario y resolución
        scenery_info = analyze_scenery(img)
        h, w = img.shape[:2]

        # 4. Nombre destino único para evitar colisiones
        dest_filename = img_path.name
        dest_path = output_dir / dest_filename
        if dest_path.exists() and dest_path.resolve() != img_path.resolve():
            dest_filename = f"{img_path.stem}_{dhash_hex[:6]}{img_path.suffix}"
            dest_path = output_dir / dest_filename

        # Copiar imagen a la carpeta de no etiquetadas
        if img_path.resolve() != dest_path.resolve():
            shutil.copy2(img_path, dest_path)

        record = {
            "filename": dest_filename,
            "source_path": str(img_path.relative_to(PROJECT_ROOT) if img_path.is_relative_to(PROJECT_ROOT) else img_path),
            "width": w,
            "height": h,
            "aspect_ratio": round(w / float(h), 3),
            "brightness_mean": scenery_info["brightness_mean"],
            "contrast_std": scenery_info["contrast_std"],
            "hue_mean": scenery_info["hue_mean"],
            "sat_mean": scenery_info["sat_mean"],
            "green_ratio": scenery_info["green_ratio"],
            "estimated_scenery": scenery_info["estimated_scenery"],
            "dhash": dhash_hex,
        }
        accepted_records.append(record)

        if len(accepted_records) % 10 == 0 or len(accepted_records) == 1:
            log.info(
                "[%d] Aceptada: %s | %s | Res: %dx%d | Escenario: %s",
                len(accepted_records),
                dest_filename,
                record["source_path"],
                w,
                h,
                record["estimated_scenery"],
            )

    # 5. Exportar CSV con metadata
    fieldnames = [
        "filename",
        "source_path",
        "width",
        "height",
        "aspect_ratio",
        "brightness_mean",
        "contrast_std",
        "hue_mean",
        "sat_mean",
        "green_ratio",
        "estimated_scenery",
        "dhash",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accepted_records)

    # Resumen de escenarios
    scenery_summary: Dict[str, int] = {}
    for r in accepted_records:
        sc = str(r["estimated_scenery"])
        scenery_summary[sc] = scenery_summary.get(sc, 0) + 1

    return {
        "total_scanned": total_processed,
        "total_accepted": len(accepted_records),
        "rejections": rejection_counts,
        "scenery_summary": scenery_summary,
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filtra y organiza capturas limpias de batalla para el dataset de muros (ClashFarmerbot)."
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=[
            "datasets/coc_walls_raw",
            "datasets/coc/images/raw",
            "logs/yolo",
            "logs/runs",
            "scratch",
            "ejemplos",
        ],
        help="Directorios donde buscar capturas (excluye assets/ automáticamente).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/walls_pool/unlabeled",
        help="Directorio destino para capturas limpias no etiquetadas.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="datasets/walls_pool/unlabeled_metadata.csv",
        help="Ruta para el archivo CSV de métricas y escenarios.",
    )
    parser.add_argument(
        "--max-hash-dist",
        type=int,
        default=4,
        help="Distancia de Hamming máxima para considerar dos imágenes duplicadas (default: 4).",
    )
    parser.add_argument(
        "--no-require-hud",
        action="store_true",
        default=False,
        help="No exige detección de botón Next o Surrender si la imagen pasa los demás filtros.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Límite máximo de imágenes a aceptar y copiar (opcional).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Limpia el directorio destino antes de copiar nuevas imágenes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_dirs = [PROJECT_ROOT / d for d in args.source_dirs]
    output_dir = PROJECT_ROOT / args.output_dir
    csv_path = PROJECT_ROOT / args.csv_path

    try:
        results = run_triage(
            source_dirs=source_dirs,
            output_dir=output_dir,
            csv_path=csv_path,
            max_hash_dist=args.max_hash_dist,
            require_battle_hud=not args.no_require_hud,
            limit=args.limit,
            clean_output=args.clean,
        )
    except Exception as e:
        log.exception("Error durante la ejecución de triage_images: %s", e)
        return 1

    print("\n" + "=" * 65)
    print("RESUMEN DE TRIAGE DE CAPTURAS DE BATALLA")
    print("=" * 65)
    print(f"Total imágenes analizadas : {results['total_scanned']}")
    print(f"Total imágenes aceptadas  : {results['total_accepted']}")
    print(f"Directorio de salida      : {results['output_dir']}")
    print(f"Archivo de métricas CSV   : {results['csv_path']}")
    print("\nDesglose de motivos de descarte:")
    for reason, cnt in sorted(results["rejections"].items(), key=lambda x: -x[1]):
        print(f"  - {reason:<32}: {cnt}")
    print("\nVariedad de escenarios aceptados:")
    for sc, cnt in sorted(results["scenery_summary"].items(), key=lambda x: -x[1]):
        print(f"  - {sc:<32}: {cnt} imágenes")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
