"""
tests/test_dataset_pipeline.py
Tests unitarios para los scripts de triage y preetiquetado de dataset de muros:
- scripts/dataset/triage_images.py
- scripts/dataset/prelabel_walls.py
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from scripts.dataset.triage_images import (
    ImageTriager,
    analyze_scenery,
    compute_dhash,
    hamming_distance,
)
from scripts.dataset.prelabel_walls import (
    _break_mask_holes,
    build_labelme_json,
    prelabel_image,
    simplify_and_build_shapes,
)


class TestTriageImages(unittest.TestCase):
    def setUp(self):
        self.triager = ImageTriager(require_battle_hud=False)

    def test_compute_dhash_and_hamming(self):
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()
        h1, hex1 = compute_dhash(img1)
        h2, hex2 = compute_dhash(img2)
        self.assertEqual(h1, h2)
        self.assertEqual(hex1, hex2)
        self.assertEqual(hamming_distance(h1, h2), 0)

        # Modificar imagen para cambiar gradiente
        img3 = img1.copy()
        img3[:, 50:] = 255
        h3, hex3 = compute_dhash(img3)
        self.assertNotEqual(h1, h3)
        self.assertGreater(hamming_distance(h1, h3), 0)

    def test_analyze_scenery_classic_grass(self):
        # Crear imagen con verde predominante (BGR: 40, 160, 40 -> HSV H~60, S~190)
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        img[:, :] = (40, 160, 40)
        metrics = analyze_scenery(img)
        self.assertIn("estimated_scenery", metrics)
        self.assertEqual(metrics["estimated_scenery"], "Classic Grass")
        self.assertGreaterEqual(metrics["green_ratio"], 0.22)

    def test_analyze_scenery_dark(self):
        # Imagen oscura
        img = np.full((200, 400, 3), 30, dtype=np.uint8)
        metrics = analyze_scenery(img)
        self.assertEqual(metrics["estimated_scenery"], "Dark / Shadow / Night")

    def test_filter_resolution_and_aspect_ratio(self):
        # Resolución demasiado pequeña
        small_img = np.full((300, 600, 3), 120, dtype=np.uint8)
        is_clean, reason = self.triager.check_clean_battle(small_img, Path("test.png"))
        self.assertFalse(is_clean)
        self.assertIn("resolution_too_small", reason)

        # Aspect ratio cuadrado (no landscape CoC)
        square_img = np.full((1000, 1000, 3), 120, dtype=np.uint8)
        is_clean, reason = self.triager.check_clean_battle(square_img, Path("test.png"))
        self.assertFalse(is_clean)
        self.assertIn("aspect_ratio_not_landscape", reason)

    def test_filter_blank_loading_screen(self):
        blank_img = np.zeros((1080, 2400, 3), dtype=np.uint8)
        is_clean, reason = self.triager.check_clean_battle(blank_img, Path("test.png"))
        self.assertFalse(is_clean)
        self.assertIn("blank_or_loading_screen", reason)

    def test_filter_debug_overlays_cyan_and_green(self):
        # Base válida (1080x2400) con textura y contraste natural
        rng = np.random.default_rng(42)
        base_img = rng.integers(50, 180, size=(1080, 2400, 3), dtype=np.uint8)

        # Overlay cian puro (>250 px B>=245, G>=245, R<=10)
        cyan_img = base_img.copy()
        cyan_img[100:130, 100:130] = (255, 255, 0)
        is_clean, reason = self.triager.check_clean_battle(cyan_img, Path("test.png"))
        self.assertFalse(is_clean)
        self.assertIn("cyan_debug_overlay", reason)

        # Contornos verdes puros (>250 px B<=10, G>=245, R<=10)
        green_img = base_img.copy()
        green_img[100:130, 100:130] = (0, 255, 0)
        is_clean, reason = self.triager.check_clean_battle(green_img, Path("test.png"))
        self.assertFalse(is_clean)
        self.assertIn("green_debug_contour", reason)

    def test_filter_filename_debug_indicator(self):
        valid_img = np.full((1080, 2400, 3), 120, dtype=np.uint8)
        for kw in ["sample_preview.png", "attack_debug.png", "yolo_box.png", "gt_comparison.png"]:
            is_clean, reason = self.triager.check_clean_battle(valid_img, Path(kw))
            self.assertFalse(is_clean)
            self.assertIn("filename_debug_indicator", reason)

    def test_duplicate_registration(self):
        triager = ImageTriager()
        h1 = 0x1234567890ABCDEF
        dup = triager.check_and_register_duplicate(h1, "img1.png")
        self.assertIsNone(dup)

        # Mismo hash
        dup = triager.check_and_register_duplicate(h1, "img2.png")
        self.assertEqual(dup, "img1.png")

        # Hash muy diferente (distancia > 4)
        h2 = 0xFFFFFFFFFFFFFFFF
        dup2 = triager.check_and_register_duplicate(h2, "img3.png")
        self.assertIsNone(dup2)


class TestPrelabelWalls(unittest.TestCase):
    def test_break_mask_holes_opens_closed_compartments(self):
        """Verifica que _break_mask_holes rompa el anillo de un compartimento cerrado."""
        mask = np.zeros((400, 400), dtype=np.uint8)
        # Anillo cerrado de muro de 20px de grosor: exterior 100..300, hueco 120..280
        cv2.rectangle(mask, (100, 100), (300, 300), 255, thickness=-1)
        cv2.rectangle(mask, (120, 120), (280, 280), 0, thickness=-1)

        # Comprobar que inicialmente hay 1 hueco interior con RETR_CCOMP
        cnts_before, hie_before = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes_before = [i for i, h in enumerate(hie_before[0]) if h[3] != -1]
        self.assertEqual(len(holes_before), 1)

        # Romper hueco con _break_mask_holes
        m_broken = _break_mask_holes(mask, cut_thickness=3)

        # Comprobar que ya no queda ningún hueco interior
        cnts_after, hie_after = cv2.findContours(m_broken, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes_after = [i for i, h in enumerate(hie_after[0]) if h[3] != -1]
        self.assertEqual(len(holes_after), 0)

        # Comprobar que findContours con RETR_EXTERNAL solo traza la cinta del muro
        # en vez de rellenar el compartimento sólido (~40000 px)
        shapes, _ = simplify_and_build_shapes(m_broken, break_holes=False)
        self.assertEqual(len(shapes), 1)
        poly_pts = np.array(shapes[0]["points"], dtype=np.int32)
        area = cv2.contourArea(poly_pts)
        self.assertLess(area, 20000.0, "El área no debe incluir el interior del compartimento cerrado")

    def test_simplify_and_build_shapes(self):
        mask = np.zeros((1000, 1000), dtype=np.uint8)
        # Dibujar un rectángulo de muro 100x40
        cv2.rectangle(mask, (200, 200), (300, 240), 255, -1)

        shapes, raw_pts = simplify_and_build_shapes(mask, epsilon_px=1.5, min_area=40.0)
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["label"], "wall")
        self.assertEqual(shapes[0]["shape_type"], "polygon")
        self.assertGreaterEqual(len(shapes[0]["points"]), 4)

        # Ruido muy pequeño (< 40 px) debe ser ignorado
        cv2.rectangle(mask, (10, 10), (14, 14), 255, -1)
        shapes_filtered, _ = simplify_and_build_shapes(mask, epsilon_px=1.5, min_area=40.0)
        self.assertEqual(len(shapes_filtered), 1)

    def test_build_labelme_json(self):
        shapes = [{
            "label": "wall",
            "points": [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0]],
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
        }]
        res = build_labelme_json("test.png", shapes, 1080, 2400)
        self.assertEqual(res["version"], "5.2.1")
        self.assertEqual(res["imagePath"], "test.png")
        self.assertIsNone(res["imageData"])
        self.assertEqual(res["imageHeight"], 1080)
        self.assertEqual(res["imageWidth"], 2400)
        self.assertEqual(len(res["shapes"]), 1)

    def test_prelabel_image_pipeline_mock(self):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            in_dir = temp_dir / "in"
            out_dir = temp_dir / "out"
            in_dir.mkdir()
            out_dir.mkdir()

            test_img_path = in_dir / "test_atk.png"
            fake_img = np.full((1080, 2400, 3), 100, dtype=np.uint8)
            cv2.imwrite(str(test_img_path), fake_img)

            # Mock detector
            mock_wall_det = MagicMock()
            fake_wall_mask = np.zeros((1080, 2400), dtype=np.uint8)
            cv2.rectangle(fake_wall_mask, (500, 400), (600, 450), 255, -1)
            mock_wall_det.predict.return_value = fake_wall_mask

            mock_deploy_det = MagicMock()
            mock_deploy_det.is_ready.return_value = True
            # Deploy zone: 1 inside, 0 outside
            fake_deploy_mask = np.ones((1080, 2400), dtype=np.uint8)
            fake_deploy_mask[:200, :] = 0  # Exterior grass
            mock_deploy_det.predict_mask.return_value = fake_deploy_mask

            res = prelabel_image(
                image_path=test_img_path,
                output_dir=out_dir,
                wall_detector=mock_wall_det,
                deploy_detector=mock_deploy_det,
                epsilon_px=1.5,
                min_area=40.0,
            )

            self.assertIsNotNone(res)
            self.assertEqual(res["polygons_count"], 1)

            # Verificar que se crearon los archivos esperados
            dest_img = out_dir / "test_atk.png"
            dest_json = out_dir / "test_atk.json"
            self.assertTrue(dest_img.exists())
            self.assertTrue(dest_json.exists())

            # Verificar que el JSON es válido
            with open(dest_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["imageHeight"], 1080)
            self.assertEqual(data["imageWidth"], 2400)
            self.assertEqual(len(data["shapes"]), 1)
            self.assertEqual(data["shapes"][0]["label"], "wall")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
