"""
tools/annotate_best_jump.py
Herramienta de anotación de saltos óptimos (Fase 1.1 - Ground Truth).

Permite etiquetar interactivamente sobre capturas reales de aldeas:
- Superpone segmentos de muros detectados globalmente (detect_global_wall_segments).
- Permite seleccionar hasta 4 puntos óptimos de hechizo de salto en orden con el ratón.
- Soporta marcar 'sin salto' ('s'), marcar 'botin en colectores' ('k'), deshacer ('u'), avanzar ('n'), retroceder ('b'), limpiar ('c') y salir ('q').
- Guarda las anotaciones en datasets/coc/jump_gt/<image_name>.json con el esquema:
    {"puntos": [[x, y], ...], "sin_salto": bool, "botin_en_colectores": bool, "notas": str}
- Permite reanudar donde se quedó.
- Soporta modo headless y dry-run para pruebas automatizadas y CI sin bloqueo de GUI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Asegurar raíz del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from core.logger import get_logger
from vision.wall_raycast import detect_global_wall_segments, reset_iso_angles_cache

log = get_logger("annotate_best_jump")


class JumpAnnotator:
    """Gestiona la carga de imágenes, detección de muros, captura de clics y guardado de ground truth."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        raw_dir: str | Path | None = None,
        attack_dir: str | Path | None = None,
        gt_dir: str | Path | None = None,
        jump_radius_px: int = 90,
        headless: bool = False,
    ):
        self.root = Path(project_root) if project_root else PROJECT_ROOT
        self.raw_dir = Path(raw_dir) if raw_dir else self.root / "datasets" / "coc" / "images" / "raw"
        self.attack_dir = Path(attack_dir) if attack_dir else self.root / "logs" / "yolo"
        self.gt_dir = Path(gt_dir) if gt_dir else self.root / "datasets" / "coc" / "jump_gt"
        self.jump_radius_px = jump_radius_px
        self.headless = headless
        self.max_jumps = 4

        # Asegurar directorio de ground truth
        self.gt_dir.mkdir(parents=True, exist_ok=True)

        # Estado interactivo
        self.images: list[Path] = []
        self.current_idx: int = 0
        self.current_img: np.ndarray | None = None
        self.current_wall_segments: list[tuple[int, int, int, int]] = []
        self.current_iso_angles: tuple[float, float] = (36.0, 144.0)
        self.current_puntos: list[list[int]] = []
        self.current_sin_salto: bool = False
        self.current_botin_en_colectores: bool = False
        self.current_notas: str = ""
        self.current_modified: bool = False
        self.status_message: str = ""

        # Colores para los saltos (BGR): hasta 4 saltos
        self.jump_colors = [
            (0, 255, 120),    # Salto #1: Verde esmeralda
            (0, 215, 255),    # Salto #2: Amarillo dorado
            (255, 100, 255),  # Salto #3: Magenta brillante
            (0, 140, 255),    # Salto #4: Naranja intenso
        ]

    def find_all_images(self) -> list[Path]:
        """
        Descubre todas las imágenes elegibles:
        1. datasets/coc/images/raw/*.png y *.jpg
        2. logs/yolo/attack_*/screen_raw.png
        """
        discovered: list[Path] = []

        if self.raw_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                discovered.extend(sorted(self.raw_dir.glob(ext)))

        if self.attack_dir.exists():
            attack_screens = sorted(self.attack_dir.glob("attack_*/screen_raw.png"))
            discovered.extend(attack_screens)

        # Eliminar duplicados manteniendo orden
        seen: set[str] = set()
        unique_images: list[Path] = []
        for p in discovered:
            resolved = str(p.resolve())
            if resolved not in seen:
                seen.add(resolved)
                unique_images.append(p)

        self.images = unique_images
        log.info(
            "Descubiertas %d imágenes en total (%d raw, %d capturas de ataque)",
            len(unique_images),
            len([p for p in unique_images if "raw" in p.parts]),
            len([p for p in unique_images if "attack_" in str(p)]),
        )
        return self.images

    @staticmethod
    def get_image_id(path: Path) -> str:
        """
        Retorna un identificador único y determinista para la imagen:
        - Para 'logs/yolo/attack_20261113_171514/screen_raw.png' -> 'attack_20261113_171514'
        - Para 'datasets/coc/images/raw/auto_20261104_104900_0001.png' -> 'auto_20261104_104900_0001'
        """
        path = Path(path)
        if path.name == "screen_raw.png" and path.parent.name.startswith("attack_"):
            return path.parent.name
        return path.stem

    def get_gt_path(self, image_path: Path) -> Path:
        """Retorna la ruta destino del JSON ground truth para la imagen."""
        img_id = self.get_image_id(image_path)
        return self.gt_dir / f"{img_id}.json"

    def load_annotation(self, image_path: Path) -> dict[str, Any]:
        """
        Carga la anotación previa si existe en datasets/coc/jump_gt/, o retorna la estructura vacía.
        Esquema: {puntos: [[x, y], ...], sin_salto: bool, botin_en_colectores: bool, notas: str}
        """
        gt_path = self.get_gt_path(image_path)

        # Buscar también variantes alternativas en caso de convención previa
        alt_paths = [
            gt_path,
            self.gt_dir / f"{image_path.stem}.json",
            self.gt_dir / f"{image_path.parent.name}_{image_path.stem}.json",
        ]

        for p in alt_paths:
            if p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    puntos = [[int(pt[0]), int(pt[1])] for pt in data.get("puntos", [])]
                    sin_salto = bool(data.get("sin_salto", False))
                    botin_en_colectores = bool(data.get("botin_en_colectores", False))
                    notas = str(data.get("notas", ""))
                    return {
                        "puntos": puntos,
                        "sin_salto": sin_salto,
                        "botin_en_colectores": botin_en_colectores,
                        "notas": notas,
                        "image": image_path.name,
                        "exists": True,
                        "file_path": str(p),
                    }
                except Exception as e:
                    log.warning("Error leyendo archivo de anotación previo %s: %s", p, e)

        return {
            "puntos": [],
            "sin_salto": False,
            "botin_en_colectores": False,
            "notas": "",
            "image": image_path.name,
            "exists": False,
            "file_path": str(gt_path),
        }

    def save_annotation(
        self,
        image_path: Path,
        puntos: list[list[int]],
        sin_salto: bool,
        notas: str = "",
        botin_en_colectores: bool = False,
    ) -> Path:
        """
        Guarda la anotación con el esquema estricto:
        {
            "puntos": [[x, y], ...],
            "sin_salto": bool,
            "botin_en_colectores": bool,
            "notas": str
        }
        """
        gt_path = self.get_gt_path(image_path)
        gt_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "image": image_path.name,
            "puntos": [[int(pt[0]), int(pt[1])] for pt in puntos[:self.max_jumps]],
            "sin_salto": bool(sin_salto),
            "botin_en_colectores": bool(botin_en_colectores),
            "notas": str(notas or ""),
        }

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        log.info(
            "💾 Anotación guardada en %s | saltos: %d | sin_salto: %s | colectores: %s",
            gt_path.name,
            len(payload["puntos"]),
            payload["sin_salto"],
            payload["botin_en_colectores"],
        )
        return gt_path

    def compute_wall_segments(self, img: np.ndarray) -> tuple[tuple[float, float], list[tuple[int, int, int, int]]]:
        """Calcula los segmentos de muro y ángulos isométricos con detect_global_wall_segments."""
        reset_iso_angles_cache()
        iso_angles, segments = detect_global_wall_segments(img, force_refresh=True)
        return iso_angles, segments

    def find_first_unannotated_idx(self) -> int:
        """Retorna el índice de la primera imagen sin anotar para reanudar trabajo."""
        for idx, p in enumerate(self.images):
            gt_path = self.get_gt_path(p)
            if not gt_path.exists():
                return idx
        return 0

    def load_image_at_index(self, idx: int) -> bool:
        """Carga la imagen y su anotación en el índice indicado."""
        if not (0 <= idx < len(self.images)):
            return False

        self.current_idx = idx
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("No se pudo cargar la imagen: %s", img_path)
            return False

        self.current_img = img
        self.current_iso_angles, self.current_wall_segments = self.compute_wall_segments(img)

        ann = self.load_annotation(img_path)
        self.current_puntos = [list(pt) for pt in ann["puntos"]]
        self.current_sin_salto = ann["sin_salto"]
        self.current_botin_en_colectores = ann.get("botin_en_colectores", False)
        self.current_notas = ann["notas"]
        self.current_modified = False

        status = "Existente" if ann["exists"] else "Nueva"
        self.status_message = f"[{idx+1}/{len(self.images)}] {img_path.name} ({status})"
        log.info(
            "Cargada imagen %d/%d: %s | Muros detectados: %d | Saltos previos: %d | Sin salto: %s | Colectores: %s",
            idx + 1,
            len(self.images),
            img_path.name,
            len(self.current_wall_segments),
            len(self.current_puntos),
            self.current_sin_salto,
            self.current_botin_en_colectores,
        )
        return True

    def render_overlay(self) -> np.ndarray:
        """Renderiza la imagen actual con superposición de muros, saltos, banner y HUD."""
        if self.current_img is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        canvas = self.current_img.copy()
        h, w = canvas.shape[:2]

        # 1. Superponer segmentos de muro detectados en cian suave
        wall_color = (240, 180, 50)  # BGR
        for x1, y1, x2, y2 in self.current_wall_segments:
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), wall_color, thickness=2)

        # 2. Superponer saltos marcados (hasta self.max_jumps)
        for i, pt in enumerate(self.current_puntos[:self.max_jumps]):
            px, py = int(pt[0]), int(pt[1])
            col = self.jump_colors[i % len(self.jump_colors)]

            # Círculo del radio de salto
            cv2.circle(canvas, (px, py), radius=self.jump_radius_px, color=col, thickness=2)
            # Punto central
            cv2.circle(canvas, (px, py), radius=6, color=col, thickness=-1)
            cv2.circle(canvas, (px, py), radius=8, color=(0, 0, 0), thickness=1)

            # Etiqueta con caja de fondo para legibilidad
            label = f"SALTO #{i+1} ({px}, {py})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.60
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            tx = max(10, px - text_w // 2)
            ty = max(30, py - self.jump_radius_px - 10)
            cv2.rectangle(
                canvas,
                (tx - 4, ty - text_h - 4),
                (tx + text_w + 4, ty + baseline + 2),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.rectangle(
                canvas,
                (tx - 4, ty - text_h - 4),
                (tx + text_w + 4, ty + baseline + 2),
                col,
                thickness=1,
            )
            cv2.putText(canvas, label, (tx, ty), font, font_scale, col, thickness)

        # 3. Banner 'SIN SALTO' si aplica
        if self.current_sin_salto:
            banner_text = ">>> [SIN SALTO MARCAD0] <<<"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(banner_text, font, 1.1, 3)
            bx1 = (w - tw) // 2 - 20
            bx2 = (w + tw) // 2 + 20
            by1 = 90
            by2 = by1 + th + 24
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 180), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            cv2.putText(canvas, banner_text, (bx1 + 20, by2 - 12), font, 1.1, (255, 255, 255), 3)

        # 4. Badge 'BOTIN EN COLECTORES' si aplica
        if self.current_botin_en_colectores:
            badge_text = ">>> [BOTIN EN COLECTORES / ALMACENES VACIOS] <<<"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(badge_text, font, 0.85, 2)
            bx1 = 20
            bx2 = bx1 + tw + 20
            by1 = h - 55
            by2 = by1 + th + 16
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 100, 200), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 220, 255), 2)
            cv2.putText(canvas, badge_text, (bx1 + 10, by2 - 8), font, 0.85, (255, 255, 255), 2)

        # 5. HUD superior
        hud_h = 75
        hud_overlay = canvas[:hud_h, :w].copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (25, 25, 25), -1)
        cv2.addWeighted(hud_overlay, 0.85, canvas[:hud_h, :w], 0.15, 0, canvas[:hud_h, :w])

        # Línea de estado
        img_name = self.images[self.current_idx].name if self.images else "N/A"
        img_id = self.get_image_id(self.images[self.current_idx]) if self.images else "N/A"
        saved_status = "MODIFICADO" if self.current_modified else "SIN CAMBIOS"
        gt_exists = self.get_gt_path(self.images[self.current_idx]).exists() if self.images else False
        persisted = "GUARDADO EN DISCO" if gt_exists else "PENDIENTE"

        hud_line1 = (
            f"[{self.current_idx+1}/{len(self.images)}] {img_name} (ID: {img_id}) | "
            f"Muros: {len(self.current_wall_segments)} | "
            f"Saltos: {len(self.current_puntos)}/{self.max_jumps} | "
            f"Sin Salto: {'SI' if self.current_sin_salto else 'NO'} | "
            f"Colectores: {'SI' if self.current_botin_en_colectores else 'NO'} | "
            f"{persisted} [{saved_status}]"
        )
        cv2.putText(canvas, hud_line1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        hud_line2 = (
            "Controles: [Clic Izq]: Marcar salto | [Clic Der / 'u']: Deshacer | "
            "['s']: Sin Salto | ['k']: Botin Colectores | ['n' / Espacio]: Siguiente | ['b']: Anterior | "
            "['c']: Limpiar | ['q']: Guardar y Salir"
        )
        cv2.putText(canvas, hud_line2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return canvas

    def add_point(self, x: int, y: int) -> bool:
        """Añade un punto de salto (máximo self.max_jumps). Si sin_salto estaba activo, lo desactiva."""
        if len(self.current_puntos) >= self.max_jumps:
            log.warning(
                "Máximo de %d puntos de salto alcanzado. Presione 'u' para deshacer o 'c' para limpiar.",
                self.max_jumps,
            )
            return False

        self.current_sin_salto = False
        self.current_puntos.append([int(x), int(y)])
        self.current_modified = True
        log.info("Añadido salto #%d en (%d, %d)", len(self.current_puntos), x, y)
        return True

    def undo_point(self) -> bool:
        """Deshace el último punto o desactiva sin_salto."""
        if self.current_puntos:
            removed = self.current_puntos.pop()
            self.current_modified = True
            log.info("Deshecho salto en (%d, %d). Restantes: %d", removed[0], removed[1], len(self.current_puntos))
            return True
        elif self.current_sin_salto:
            self.current_sin_salto = False
            self.current_modified = True
            log.info("Desactivada marca 'sin_salto'.")
            return True
        return False

    def toggle_sin_salto(self) -> bool:
        """Alterna el flag sin_salto. Si se activa, vacía la lista de puntos."""
        self.current_sin_salto = not self.current_sin_salto
        if self.current_sin_salto:
            self.current_puntos.clear()
        self.current_modified = True
        log.info("Flag 'sin_salto' establecido en: %s", self.current_sin_salto)
        return self.current_sin_salto

    def toggle_botin_en_colectores(self) -> bool:
        """Alterna el flag 'botin_en_colectores' (almacenes vacíos, botín principal en colectores)."""
        self.current_botin_en_colectores = not self.current_botin_en_colectores
        self.current_modified = True
        log.info("Flag 'botin_en_colectores' establecido en: %s", self.current_botin_en_colectores)
        return self.current_botin_en_colectores

    def clear_points(self) -> None:
        """Limpia los puntos y flags de la imagen actual."""
        self.current_puntos.clear()
        self.current_sin_salto = False
        self.current_modified = True
        log.info("Puntos limpiados para la imagen actual.")

    def save_current_if_modified(self) -> None:
        """Guarda la anotación actual en disco si hubo cambios o si contiene datos."""
        if not self.images:
            return
        img_path = self.images[self.current_idx]
        # Guardar si fue modificada o si tiene saltos / sin_salto / colectores definidos
        if (
            self.current_modified
            or self.current_puntos
            or self.current_sin_salto
            or self.current_botin_en_colectores
        ):
            self.save_annotation(
                img_path,
                self.current_puntos,
                self.current_sin_salto,
                self.current_notas,
                botin_en_colectores=self.current_botin_en_colectores,
            )
            self.current_modified = False

    def run_interactive(self, start_idx: int | None = None, auto_resume: bool = True) -> None:
        """Bucle principal de la GUI interactiva con OpenCV."""
        if self.headless:
            log.warning("Ejecutando en modo headless. Use run_headless() para procesamiento por lotes.")
            return

        self.find_all_images()
        if not self.images:
            log.warning("No se encontraron imágenes en las rutas especificadas.")
            return

        if start_idx is not None and 0 <= start_idx < len(self.images):
            idx = start_idx
        elif auto_resume:
            idx = self.find_first_unannotated_idx()
            log.info("Reanudando en primera imagen pendiente: índice %d (%s)", idx, self.images[idx].name)
        else:
            idx = 0

        window_name = "Anotador de Saltos Optimos - CoC (Fase 1.1)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                if y > 75:  # Evitar clics sobre la barra de HUD superior
                    self.add_point(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.undo_point()

        cv2.setMouseCallback(window_name, on_mouse)

        if not self.load_image_at_index(idx):
            log.error("Fallo al inicializar primera imagen.")
            cv2.destroyAllWindows()
            return

        log.info("══ INICIANDO INTERFAZ DE ANOTACIÓN ══")
        log.info("Presione 's' para sin salto, 'k' para botin colectores, 'u' para deshacer, 'n' siguiente, 'b' anterior, 'q' salir.")

        try:
            while True:
                canvas = self.render_overlay()
                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(25) & 0xFF

                if key in (ord("q"), 27):  # 'q' o ESC
                    self.save_current_if_modified()
                    log.info("Saliendo de la herramienta de anotación.")
                    break

                elif key in (ord("n"), 32, 13):  # 'n', Espacio o Enter -> Siguiente
                    self.save_current_if_modified()
                    if self.current_idx + 1 < len(self.images):
                        self.load_image_at_index(self.current_idx + 1)
                    else:
                        log.info("¡Has alcanzado la última imagen de la lista!")

                elif key == ord("b"):  # 'b' -> Anterior
                    self.save_current_if_modified()
                    if self.current_idx > 0:
                        self.load_image_at_index(self.current_idx - 1)
                    else:
                        log.info("Ya estás en la primera imagen.")

                elif key == ord("u"):  # 'u' -> Deshacer
                    self.undo_point()

                elif key == ord("s"):  # 's' -> Sin salto
                    self.toggle_sin_salto()

                elif key == ord("k"):  # 'k' -> Alternar Botín en Colectores
                    self.toggle_botin_en_colectores()

                elif key == ord("c"):  # 'c' -> Limpiar
                    self.clear_points()

        finally:
            cv2.destroyAllWindows()

    def run_headless(
        self,
        limit: int | None = 3,
        dry_run: bool = True,
        simulate_sample_annotations: bool = False,
    ) -> dict[str, Any]:
        """
        Ejecución sin entorno gráfico para tests automatizados y CI.
        Procesa hasta 'limit' imágenes, calcula segmentos de muro y valida el guardado.
        """
        self.find_all_images()
        if not self.images:
            log.warning("No se encontraron imágenes para procesar.")
            return {"total": 0, "processed": 0, "annotated": 0}

        target_images = self.images[:limit] if limit else self.images
        processed_count = 0
        annotated_count = 0

        log.info("Iniciando procesamiento headless sobre %d imágenes (dry_run=%s)", len(target_images), dry_run)

        for idx, img_path in enumerate(target_images):
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("Headless: no se pudo leer %s", img_path)
                continue

            iso_angles, segments = self.compute_wall_segments(img)
            ann = self.load_annotation(img_path)
            processed_count += 1

            if ann["exists"]:
                annotated_count += 1

            log.info(
                "Headless [%d/%d] %s -> Muros: %d, Ángulos: (%.1f°, %.1f°), Anotación previa: %s",
                idx + 1,
                len(target_images),
                img_path.name,
                len(segments),
                iso_angles[0],
                iso_angles[1],
                ann["exists"],
            )

            # Si se solicita simular anotación para prueba de persistencia
            if simulate_sample_annotations and not dry_run:
                # Marcar 2 saltos sintéticos o sin salto
                sample_pts = [[1000 + idx * 50, 500 + idx * 30], [1200, 600]]
                self.save_annotation(
                    img_path,
                    sample_pts,
                    sin_salto=False,
                    notas="Anotación headless sintética",
                    botin_en_colectores=False,
                )
                annotated_count += 1

        summary = {
            "total": len(self.images),
            "processed": processed_count,
            "annotated": annotated_count,
            "dry_run": dry_run,
        }
        log.info("Procesamiento headless completado: %s", summary)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Herramienta de anotación de saltos óptimos CoC (Fase 1.1)")
    parser.add_argument("--raw-dir", default=None, help="Ruta a imágenes raw (default: datasets/coc/images/raw)")
    parser.add_argument("--attack-dir", default=None, help="Ruta a ataques YOLO (default: logs/yolo)")
    parser.add_argument("--gt-dir", default=None, help="Ruta de salida JSON (default: datasets/coc/jump_gt)")
    parser.add_argument("--radius", type=int, default=90, help="Radio en px del hechizo de salto (default: 90)")
    parser.add_argument("--start-idx", type=int, default=None, help="Índice inicial 0-based")
    parser.add_argument("--no-resume", action="store_true", help="No saltar automáticamente a la primera pendiente")
    parser.add_argument("--headless", action="store_true", help="Modo sin interfaz gráfica (para tests/CI)")
    parser.add_argument("--dry-run", action="store_true", help="No escribe cambios en disco en modo headless")
    parser.add_argument("--limit", type=int, default=3, help="Límite de imágenes en modo headless (default: 3)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotator = JumpAnnotator(
        raw_dir=args.raw_dir,
        attack_dir=args.attack_dir,
        gt_dir=args.gt_dir,
        jump_radius_px=args.radius,
        headless=args.headless,
    )

    if args.headless:
        annotator.run_headless(limit=args.limit, dry_run=args.dry_run)
    else:
        annotator.run_interactive(start_idx=args.start_idx, auto_resume=not args.no_resume)


if __name__ == "__main__":
    main()
