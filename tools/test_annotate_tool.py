"""
tools/test_annotate_tool.py
Suite de pruebas unitarias para la herramienta de anotación de saltos óptimos (Fase 1.1).

Verifica:
1. Descubrimiento de capturas (raw village images y attack screens).
2. Detección y superposición de segmentos de muros sobre 3 imágenes en modo headless.
3. Guardado y recarga de anotaciones respetando el esquema estricto:
   {"puntos": [[x, y], ...], "sin_salto": bool, "botin_en_colectores": bool, "notas": str}
   y compatibilidad hacia atrás con anotaciones existentes de 3 saltos sin flag de colectores.
4. Lógica de reanudación automática (resume where left off).
5. Máquina de estados: adición de hasta 4 saltos, deshacer ('u'), 'sin salto' ('s'), 'botin colectores' ('k') y limpiar ('c').
6. Invocación de CLI en modo headless / dry-run sin bloqueo de GUI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Asegurar raíz del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from core.logger import get_logger
from tools.annotate_best_jump import JumpAnnotator

log = get_logger("test_annotate_tool")


class TestAnnotateBestJump(unittest.TestCase):
    """Pruebas unitarias de JumpAnnotator y persistencia de ground truth."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_jump_gt_")
        self.gt_dir = Path(self.temp_dir)
        self.annotator = JumpAnnotator(
            project_root=PROJECT_ROOT,
            gt_dir=self.gt_dir,
            headless=True,
        )

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_image_discovery(self) -> None:
        """Verifica que se descubran imágenes tanto en raw como en logs/yolo/attack_*."""
        images = self.annotator.find_all_images()
        self.assertGreaterEqual(len(images), 3, "Debe descubrir al menos 3 imágenes para pruebas")

        raw_count = sum(1 for p in images if "raw" in p.parts)
        attack_count = sum(1 for p in images if "attack_" in str(p))

        log.info(
            "Test 01: Descubiertas %d imágenes (%d raw, %d attack)",
            len(images),
            raw_count,
            attack_count,
        )
        self.assertGreater(raw_count, 0, "Debe incluir imágenes del dataset raw")
        self.assertGreater(attack_count, 0, "Debe incluir capturas de logs/yolo/attack_*")

    def test_02_headless_load_and_detect_walls_3_images(self) -> None:
        """Verifica carga y cálculo de segmentos de muro sobre 3 imágenes reales sin bloquear GUI."""
        images = self.annotator.find_all_images()
        self.assertGreaterEqual(len(images), 3)

        sample_images = images[:3]
        for idx, img_path in enumerate(sample_images):
            img = cv2.imread(str(img_path))
            self.assertIsNotNone(img, f"La imagen {img_path} no pudo ser leída por OpenCV")
            self.assertEqual(len(img.shape), 3)
            self.assertEqual(img.shape[0], 1080)
            self.assertEqual(img.shape[1], 2400)

            iso_angles, segments = self.annotator.compute_wall_segments(img)
            self.assertEqual(len(iso_angles), 2, "Debe retornar 2 ángulos isométricos")
            self.assertGreater(len(segments), 10, f"Debe detectar segmentos de muro en {img_path.name}")

            # Validar formato de segmento: (x1, y1, x2, y2)
            for seg in segments[:5]:
                self.assertEqual(len(seg), 4)

            log.info(
                "Test 02: Imagen %d/3 (%s) validada con %d segmentos de muro",
                idx + 1,
                img_path.name,
                len(segments),
            )

    def test_03_save_and_load_schema_3_images(self) -> None:
        """
        Verifica guardado y carga de imágenes con casos representativos:
        1. Aldea con 4 saltos y botin_en_colectores=True
        2. Aldea con 1 salto y botin_en_colectores=False
        3. Aldea marcada con 'sin_salto' y botin_en_colectores=False
        4. Compatibilidad hacia atrás: archivo legado con 3 saltos sin campo 'botin_en_colectores'.
        Valida rigurosamente el esquema JSON: {puntos: [[x, y], ...], sin_salto: bool, botin_en_colectores: bool, notas: str}.
        """
        images = self.annotator.find_all_images()[:3]
        self.assertEqual(len(images), 3)

        test_cases = [
            {
                "puntos": [[1100, 520], [1350, 680], [950, 400], [1200, 800]],
                "sin_salto": False,
                "botin_en_colectores": True,
                "notas": "Entrada principal norte, núcleo y colectores llenos",
            },
            {
                "puntos": [[980, 410]],
                "sin_salto": False,
                "botin_en_colectores": False,
                "notas": "Salto único entre compartimento oscuro",
            },
            {
                "puntos": [],
                "sin_salto": True,
                "botin_en_colectores": False,
                "notas": "Aldea abierta, no se requiere hechizo de salto",
            },
        ]

        for idx, (img_path, tc) in enumerate(zip(images, test_cases)):
            saved_path = self.annotator.save_annotation(
                image_path=img_path,
                puntos=tc["puntos"],
                sin_salto=tc["sin_salto"],
                notas=tc["notas"],
                botin_en_colectores=tc["botin_en_colectores"],
            )

            self.assertTrue(saved_path.exists(), f"El archivo {saved_path} debe existir")

            # Lectura directa del archivo crudo para validar formato JSON
            with open(saved_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)

            self.assertIn("puntos", raw_json)
            self.assertIn("sin_salto", raw_json)
            self.assertIn("botin_en_colectores", raw_json)
            self.assertIn("notas", raw_json)

            self.assertEqual(raw_json["puntos"], tc["puntos"])
            self.assertEqual(raw_json["sin_salto"], tc["sin_salto"])
            self.assertEqual(raw_json["botin_en_colectores"], tc["botin_en_colectores"])
            self.assertEqual(raw_json["notas"], tc["notas"])

            # Lectura a través del método de carga del anotador
            loaded = self.annotator.load_annotation(img_path)
            self.assertTrue(loaded["exists"])
            self.assertEqual(loaded["puntos"], tc["puntos"])
            self.assertEqual(loaded["sin_salto"], tc["sin_salto"])
            self.assertEqual(loaded["botin_en_colectores"], tc["botin_en_colectores"])
            self.assertEqual(loaded["notas"], tc["notas"])

            log.info("Test 03: Validación de esquema JSON superada para caso %d (%s)", idx + 1, saved_path.name)

        # 4. Caso de prueba de compatibilidad hacia atrás: archivo preexistente de 3 saltos sin flag de colectores
        legacy_file = self.gt_dir / "legacy_test_village.json"
        legacy_payload = {
            "image": "legacy_test_village.png",
            "puntos": [[1000, 500], [1100, 600], [1200, 700]],
            "sin_salto": False,
            "notas": "Anotación previa de 3 saltos sin flag de colectores",
        }
        with open(legacy_file, "w", encoding="utf-8") as f:
            json.dump(legacy_payload, f, indent=2)

        loaded_legacy = self.annotator.load_annotation(Path("dummy/legacy_test_village.png"))
        self.assertTrue(loaded_legacy["exists"])
        self.assertEqual(len(loaded_legacy["puntos"]), 3)
        self.assertFalse(loaded_legacy["botin_en_colectores"], "Debe tener por defecto False si no existe en JSON")
        self.assertFalse(loaded_legacy["sin_salto"])
        log.info("Test 03: Compatibilidad hacia atrás con anotaciones de 3 saltos verificada exitosamente")

    def test_04_resume_functionality(self) -> None:
        """Verifica que find_first_unannotated_idx salte las imágenes ya anotadas."""
        images = self.annotator.find_all_images()
        self.assertGreaterEqual(len(images), 3)

        # Inicialmente ninguna anotada en el directorio temporal
        first_idx = self.annotator.find_first_unannotated_idx()
        self.assertEqual(first_idx, 0)

        # Anotar imagen 0
        self.annotator.save_annotation(images[0], [[1000, 500]], sin_salto=False, notas="Anotado 0")
        resumed_idx = self.annotator.find_first_unannotated_idx()
        self.assertEqual(resumed_idx, 1, "Debe avanzar automáticamente a la imagen 1")

        # Anotar imagen 1
        self.annotator.save_annotation(images[1], [], sin_salto=True, notas="Anotado 1")
        resumed_idx_2 = self.annotator.find_first_unannotated_idx()
        self.assertEqual(resumed_idx_2, 2, "Debe avanzar automáticamente a la imagen 2")

        log.info("Test 04: Reanudación automática validada con éxito")

    def test_05_state_machine_controls(self) -> None:
        """Verifica adición de saltos (máximo 4), undo, toggle sin_salto, toggle botin_en_colectores y clear."""
        # 1. Cargar imagen de prueba
        self.annotator.find_all_images()
        success = self.annotator.load_image_at_index(0)
        self.assertTrue(success)

        # 2. Agregar hasta 4 puntos (nuevo tope para ejército de 4 saltos)
        self.assertTrue(self.annotator.add_point(100, 200))
        self.assertTrue(self.annotator.add_point(300, 400))
        self.assertTrue(self.annotator.add_point(500, 600))
        self.assertTrue(self.annotator.add_point(700, 800))
        self.assertEqual(len(self.annotator.current_puntos), 4)

        # 3. Intentar agregar un 5to punto debe fallar
        self.assertFalse(self.annotator.add_point(900, 1000))
        self.assertEqual(len(self.annotator.current_puntos), 4)

        # 4. Probar tecla 'k' (toggle botin_en_colectores)
        self.assertFalse(self.annotator.current_botin_en_colectores)
        self.assertTrue(self.annotator.toggle_botin_en_colectores())
        self.assertTrue(self.annotator.current_botin_en_colectores)
        self.assertFalse(self.annotator.toggle_botin_en_colectores())
        self.assertFalse(self.annotator.current_botin_en_colectores)

        # 5. Deshacer punto ('u')
        self.assertTrue(self.annotator.undo_point())
        self.assertEqual(len(self.annotator.current_puntos), 3)
        self.assertEqual(self.annotator.current_puntos[-1], [500, 600])

        # 6. Activar 'sin_salto' ('s') -> debe vaciar los puntos
        is_sin_salto = self.annotator.toggle_sin_salto()
        self.assertTrue(is_sin_salto)
        self.assertEqual(len(self.annotator.current_puntos), 0)

        # 7. Deshacer cuando sin_salto está activo debe desactivarlo
        self.assertTrue(self.annotator.undo_point())
        self.assertFalse(self.annotator.current_sin_salto)

        # 8. Agregar punto y luego limpiar ('c')
        self.annotator.add_point(900, 950)
        self.assertEqual(len(self.annotator.current_puntos), 1)
        self.annotator.clear_points()
        self.assertEqual(len(self.annotator.current_puntos), 0)
        self.assertFalse(self.annotator.current_sin_salto)

        log.info("Test 05: Controles interactivos y límites (máx 4, undo, sin_salto, botin colectores 'k', clear) validados")

    def test_06_cli_headless_dry_run(self) -> None:
        """Verifica ejecución de tools/annotate_best_jump.py desde línea de comandos en headless dry-run."""
        script_path = PROJECT_ROOT / "tools" / "annotate_best_jump.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--headless",
            "--dry-run",
            "--limit",
            "3",
            "--gt-dir",
            str(self.gt_dir),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"El CLI headless falló con código {result.returncode}.\nStderr: {result.stderr}\nStdout: {result.stdout}",
        )
        self.assertIn("Procesamiento headless completado", result.stdout + result.stderr)
        log.info("Test 06: Ejecución por CLI en modo headless --dry-run completada con retorno 0")


def main() -> None:
    log.info("Ejecutando suite de pruebas para annotate_best_jump...")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAnnotateBestJump)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
