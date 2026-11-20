"""
tests/test_wall_unet_pipeline.py
Suite de pruebas unitarias para el pipeline de segmentación semántica U-Net (MobileNetV2):
1. tools/json_to_wall_mask.py (conversión de anotaciones LabelMe a máscaras U-Net, dilate, morph close, HUD mask).
2. tools/train_wall_unet.py (constantes 768x352, augmentación isométrica, WallMaskDataset, loss combinado).
3. vision/wall_seg_detector.py (WallSegDetector ONNX inference, BGR->RGB, resize INTER_AREA/INTER_NEAREST, error si modelo no existe).
4. vision/wall_compartments.py (modos "unet" vs "raycast" y fallback transparente ante modelo ausente).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from core.config import SETTINGS

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Componente 1: json_to_wall_mask
from tools.json_to_wall_mask import (
    build_overlay,
    render_wall_mask_preview,
    rasterize_shape,
    convert_json_to_mask,
    find_image_json_pairs,
    process_dataset,
)

# Componente 2: train_wall_unet
from tools.train_wall_unet import (
    IMG_W as TRAIN_IMG_W,
    IMG_H as TRAIN_IMG_H,
    IMAGENET_MEAN as TRAIN_IMAGENET_MEAN,
    IMAGENET_STD as TRAIN_IMAGENET_STD,
    apply_augmentations,
    WallMaskDataset,
)

# Componente 3: wall_seg_detector
from vision.wall_seg_detector import (
    IMG_W as DET_IMG_W,
    IMG_H as DET_IMG_H,
    IMAGENET_MEAN as DET_IMAGENET_MEAN,
    IMAGENET_STD as DET_IMAGENET_STD,
    build_cyan_overlay,
    WallSegDetector,
)

# Componente 4: wall_compartments
from vision.wall_compartments import (
    optimize_jumps,
    get_wall_seg_detector,
    reset_wall_seg_detector,
    build_clean_wall_canvas,
)


class TestJsonToWallMask(unittest.TestCase):
    """Pruebas unitarias para tools/json_to_wall_mask.py."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "raw"
        self.output_dir = Path(self.temp_dir) / "coc_walls_unet"
        self.preview_dir = Path(self.temp_dir) / "preview"
        self.input_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rasterize_shape_types(self):
        """Valida rasterización de polígonos, rectángulos y linestrips."""
        h, w = 400, 600
        mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Polígono
        poly_shape = {
            "label": "wall",
            "shape_type": "polygon",
            "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
        }
        res_poly = rasterize_shape(mask, poly_shape)
        self.assertTrue(res_poly)
        self.assertEqual(mask[150, 150], 255)

        # 2. Rectángulo
        rect_shape = {
            "label": "wall",
            "shape_type": "rectangle",
            "points": [[300, 100], [400, 200]],
        }
        res_rect = rasterize_shape(mask, rect_shape)
        self.assertTrue(res_rect)
        self.assertEqual(mask[150, 350], 255)

        # 3. Linestrip
        line_shape = {
            "label": "wall",
            "shape_type": "linestrip",
            "points": [[100, 300], [500, 300]],
        }
        res_line = rasterize_shape(mask, line_shape, linestrip_thickness=8)
        self.assertTrue(res_line)
        self.assertEqual(mask[300, 300], 255)

        # 4. Label no relevante (debe ser ignorado)
        other_shape = {
            "label": "town_hall",
            "shape_type": "rectangle",
            "points": [[10, 10], [50, 50]],
        }
        res_other = rasterize_shape(mask, other_shape)
        self.assertFalse(res_other)

    def test_convert_json_to_mask_dilate_close_and_hud(self):
        """Valida que convert_json_to_mask aplique dilate, morph close y enmascare HUD."""
        h, w = 1080, 2400
        json_path = self.input_dir / "sample.json"
        data = {
            "imageWidth": w,
            "imageHeight": h,
            "shapes": [
                # Muro delgado superior
                {"label": "wall", "shape_type": "linestrip", "points": [[200, 400], [600, 400]]},
                # Muro dentro del área del HUD inferior (y >= 888)
                {"label": "wall", "shape_type": "rectangle", "points": [[200, 950], [600, 1020]]},
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        mask = convert_json_to_mask(json_path, expected_shape=(h, w), close_kernel=3)
        self.assertEqual(mask.shape, (h, w))
        self.assertEqual(mask.dtype, np.uint8)

        # El muro superior debe existir y tener grosor >= 8 + dilate 3x3
        self.assertEqual(mask[400, 400], 255)
        # El muro en el HUD (y >= 888) DEBE haber sido puesto a 0
        hud_y = int(round(h * 888.0 / 1080.0))
        self.assertTrue(np.all(mask[hud_y:, :] == 0))

    def test_process_dataset_end_to_end(self):
        """Valida el pipeline completo de conversión y particionado train/val."""
        h, w = 1080, 2400
        # Crear 6 pares imagen + json para garantizar que val reciba al menos 1 muestra
        for i in range(6):
            stem = f"base_{i:03d}"
            img = np.full((h, w, 3), 120, dtype=np.uint8)
            cv2.imwrite(str(self.input_dir / f"{stem}.png"), img)

            json_data = {
                "imageWidth": w,
                "imageHeight": h,
                "shapes": [
                    {"label": "wall", "shape_type": "polygon", "points": [[100 + i * 20, 200], [300, 200], [300, 400]]}
                ],
            }
            with open(self.input_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump(json_data, f)

        summary = process_dataset(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            preview_dir=self.preview_dir,
            val_ratio=0.20,
        )

        self.assertEqual(summary["total_pairs"], 6)
        self.assertGreaterEqual(summary["train_count"], 4)
        self.assertGreaterEqual(summary["val_count"], 1)

        # Verificar estructura en disco
        self.assertTrue((self.output_dir / "train" / "images").is_dir())
        self.assertTrue((self.output_dir / "train" / "masks").is_dir())
        self.assertTrue((self.output_dir / "val" / "images").is_dir())
        self.assertTrue((self.output_dir / "val" / "masks").is_dir())

        # Verificar que se generaron previews
        preview_files = list(self.preview_dir.glob("*.png"))
        self.assertEqual(len(preview_files), 6)

    def test_find_image_json_pairs_missing_raises(self):
        """Valida que un directorio sin JSONs levante FileNotFoundError con mensaje claro."""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            find_image_json_pairs(empty_dir)


class TestTrainWallUnet(unittest.TestCase):
    """Pruebas unitarias para tools/train_wall_unet.py."""

    def test_constants_and_normalization_specs(self):
        """Verifica que las constantes coincidan exactamente con la especificación de tarea.md."""
        self.assertEqual(TRAIN_IMG_W, 768)
        self.assertEqual(TRAIN_IMG_H, 352)
        self.assertEqual(TRAIN_IMAGENET_MEAN, (0.485, 0.456, 0.406))
        self.assertEqual(TRAIN_IMAGENET_STD, (0.229, 0.224, 0.225))

        # Verificar coherencia con detector
        self.assertEqual(TRAIN_IMG_W, DET_IMG_W)
        self.assertEqual(TRAIN_IMG_H, DET_IMG_H)
        self.assertTrue(np.allclose(TRAIN_IMAGENET_MEAN, DET_IMAGENET_MEAN))
        self.assertTrue(np.allclose(TRAIN_IMAGENET_STD, DET_IMAGENET_STD))

    def test_apply_augmentations_constraints(self):
        """Verifica que las augmentaciones respeten dimensiones y tipo uint8 sin volteo vertical."""
        img_rgb = np.random.randint(0, 256, (352, 768, 3), dtype=np.uint8)
        mask = np.random.choice([0, 255], size=(352, 768)).astype(np.uint8)

        aug_img, aug_mask = apply_augmentations(img_rgb, mask)
        self.assertEqual(aug_img.shape, (352, 768, 3))
        self.assertEqual(aug_mask.shape, (352, 768))
        self.assertEqual(aug_img.dtype, np.uint8)
        self.assertEqual(aug_mask.dtype, np.uint8)

    def test_wall_mask_dataset_loader(self):
        """Valida WallMaskDataset cargando tensores normalizados."""
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp)
            images_dir = split_dir / "images"
            masks_dir = split_dir / "masks"
            images_dir.mkdir()
            masks_dir.mkdir()

            # Guardar 2 muestras sintéticas a resolución original (1080x2400)
            for i in range(2):
                img = np.full((1080, 2400, 3), 100 + i * 20, dtype=np.uint8)
                mask = np.zeros((1080, 2400), dtype=np.uint8)
                mask[400:600, 800:1200] = 255
                cv2.imwrite(str(images_dir / f"sample_{i}.png"), img)
                cv2.imwrite(str(masks_dir / f"sample_{i}.png"), mask)

            dataset = WallMaskDataset(str(split_dir), augment=False)
            self.assertEqual(len(dataset), 2)

            img_tensor, mask_tensor = dataset[0]
            # Validar dimensiones: (3, 352, 768) y (1, 352, 768)
            self.assertEqual(img_tensor.shape, (3, 352, 768))
            self.assertEqual(mask_tensor.shape, (1, 352, 768))
            # La máscara normalizada debe estar en [0.0, 1.0]
            self.assertGreaterEqual(float(mask_tensor.min()), 0.0)
            self.assertLessEqual(float(mask_tensor.max()), 1.0)


class TestWallSegDetector(unittest.TestCase):
    """Pruebas unitarias para vision/wall_seg_detector.py."""

    def test_missing_model_raises_descriptive_error(self):
        """Valida que intentar cargar un modelo ONNX inexistente levante FileNotFoundError descriptivo con strict=True."""
        with self.assertRaises(FileNotFoundError) as ctx:
            WallSegDetector(model_path="models/non_existent_wall_seg_9999.onnx", strict=True)
        self.assertIn("Modelo U-Net de segmentación de muros no encontrado", str(ctx.exception))

        # En modo no estricto (strict=False), opera en modo fallback sin romper el bot
        det_fallback = WallSegDetector(model_path="models/non_existent_wall_seg_9999.onnx", strict=False)
        self.assertFalse(det_fallback.is_ready())

    def test_predict_mocked_onnx_pipeline(self):
        """Valida el pipeline predict(): BGR->RGB, resize INTER_AREA, inferencia, sigmoid y resize nativo."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "logits"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]

        # Logits simulados: canal único (1, 1, 352, 768)
        logits = np.full((1, 1, 352, 768), -5.0, dtype=np.float32)
        # Activar una banda central con logit positivo (+5.0 -> sigmoid ~0.993 > 0.5)
        logits[0, 0, 100:200, 200:400] = 5.0
        mock_session.run.return_value = [logits]

        with patch("os.path.exists", return_value=True), \
             patch("onnxruntime.InferenceSession", return_value=mock_session), \
             patch("vision.deploy_zone_detector.get_deploy_zone_detector", return_value=None):
            detector = WallSegDetector(model_path="models/wall_seg.onnx", conf_thresh=0.5)
            self.assertTrue(detector.is_ready())

            # Frame BGR nativo 1080x2400
            frame_bgr = np.zeros((1080, 2400, 3), dtype=np.uint8)
            frame_bgr[300:500, 600:1000] = (200, 200, 200)

            mask = detector.predict(frame_bgr)
            # Debe volver a la resolución original del frame
            self.assertEqual(mask.shape, (1080, 2400))
            self.assertEqual(mask.dtype, np.uint8)
            # Debe contener valores binarios 0 o 255
            unique_vals = set(np.unique(mask))
            self.assertTrue(unique_vals.issubset({0, 255}))
            self.assertIn(255, unique_vals)

            # Validar overlay cian de debug
            overlay = detector.build_overlay(frame_bgr, mask)
            self.assertEqual(overlay.shape, (1080, 2400, 3))
            self.assertEqual(overlay.dtype, np.uint8)

    def test_predict_postprocessing_deploy_zone_and_morph(self):
        """Valida que predict() ponga en 0 el césped exterior vía deploy_zone_mask y selle fisuras."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = [1, 3, 352, 768]
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [MagicMock(name="output")]

        # Logits con señal positiva general
        logits = np.full((1, 1, 352, 768), 5.0, dtype=np.float32)
        mock_session.run.return_value = [logits]

        with patch("os.path.exists", return_value=True), \
             patch("onnxruntime.InferenceSession", return_value=mock_session), \
             patch("vision.deploy_zone_detector.get_deploy_zone_detector", return_value=None):
            detector = WallSegDetector(model_path="models/wall_seg.onnx", conf_thresh=0.5)

            frame_bgr = np.zeros((1080, 2400, 3), dtype=np.uint8)
            # Deploy zone mask: solo la mitad derecha es aldea (1), la izquierda es césped (0)
            deploy_mask = np.zeros((1080, 2400), dtype=np.uint8)
            deploy_mask[:, 1200:] = 1

            mask = detector.predict(frame_bgr, deploy_zone_mask=deploy_mask)
            # La mitad izquierda (deploy_zone_mask == 0) debe estar estrictamente en 0
            self.assertEqual(np.count_nonzero(mask[:, :1200]), 0)
            # La mitad derecha (aldea) fuera del HUD debe contener píxeles de muro (255)
            self.assertTrue(np.any(mask[:, 1200:] == 255))

    def test_get_wall_seg_detector_reads_conf_from_settings(self):
        """Valida que get_wall_seg_detector() resuelva wall_seg_conf desde settings.json y permita overrides."""
        reset_wall_seg_detector()
        # 1. Por defecto, lee wall_seg_conf desde SETTINGS["attack"]["jump_optimizer"] (0.3)
        expected_conf = float(SETTINGS.get("attack", {}).get("jump_optimizer", {}).get("wall_seg_conf", 0.50))
        det = get_wall_seg_detector()
        self.assertEqual(det.conf_thresh, expected_conf)

        # 2. Con configuración modificada vía patch
        with patch.dict(SETTINGS, {"attack": {"jump_optimizer": {"wall_seg_conf": 0.35}}}):
            reset_wall_seg_detector()
            det_patched = get_wall_seg_detector()
            self.assertEqual(det_patched.conf_thresh, 0.35)

        # 3. Con override explícito por parámetro
        reset_wall_seg_detector()
        det_override = get_wall_seg_detector(conf_thresh=0.42)
        self.assertEqual(det_override.conf_thresh, 0.42)
        reset_wall_seg_detector()


class TestWallCompartmentsIntegration(unittest.TestCase):
    """Pruebas unitarias de integración en vision/wall_compartments.py con modos 'unet' y 'raycast'."""

    def setUp(self):
        reset_wall_seg_detector()

    def tearDown(self):
        reset_wall_seg_detector()

    def test_optimize_jumps_raycast_mode(self):
        """Valida que optimize_jumps funcione correctamente en modo 'raycast' tradicional."""
        shape = (600, 600)
        walls = [
            (100, 100, 500, 100),
            (500, 100, 500, 500),
            (500, 500, 100, 500),
            (100, 500, 100, 100),
        ]
        target = [{"cls_name": "storage_dark", "center": (300, 300), "bbox_xyxy": (280, 280, 320, 320)}]
        drop_points = [(50, 300)]

        res = optimize_jumps(
            img_shape=shape,
            wall_segments=walls,
            detections=target,
            drop_points=drop_points,
            loot_weights={"storage_dark": 1.0},
            wall_detection_mode="raycast",
            max_jumps=1,
            beam_width=2,
        )
        self.assertEqual(res.initial_cost, 1.0)
        self.assertEqual(res.final_cost, 0.0)
        self.assertEqual(len(res.jumps), 1)

    def test_optimize_jumps_unet_mode_with_mocked_detector(self):
        """Valida que en modo 'unet', optimize_jumps consuma la máscara devuelta por WallSegDetector."""
        shape = (600, 600)
        synth_mask = np.zeros(shape, dtype=np.uint8)
        # Anillo cerrado de muros en synth_mask
        cv2.rectangle(synth_mask, (150, 150), (450, 450), color=255, thickness=8)

        mock_detector = MagicMock()
        mock_detector.is_ready.return_value = True
        mock_detector.predict.return_value = synth_mask

        dummy_frame = np.full((600, 600, 3), 100, dtype=np.uint8)
        target = [{"cls_name": "storage_gold", "center": (300, 300), "bbox_xyxy": (285, 285, 315, 315)}]
        drop_points = [(50, 50)]

        with patch("vision.wall_compartments.get_wall_seg_detector", return_value=mock_detector):
            res = optimize_jumps(
                img_shape=shape,
                base_img=dummy_frame,
                wall_segments=[],  # Sin segmentos raycast: debe usar la máscara U-Net
                detections=target,
                drop_points=drop_points,
                loot_weights={"storage_gold": 1.0},
                wall_detection_mode="unet",
                max_jumps=1,
                beam_width=2,
            )

            mock_detector.predict.assert_called_once()
            # La máscara de U-Net debe haber creado 2 compartimentos (exterior e interior)
            self.assertEqual(res.num_compartments, 2)
            self.assertEqual(res.initial_cost, 1.0)
            self.assertEqual(res.final_cost, 0.0)

    def test_optimize_jumps_unet_fallback_when_model_missing(self):
        """Valida que si el modelo U-Net no existe en disco, haya fallback transparente a raycast."""
        shape = (600, 600)
        walls = [
            (200, 200, 400, 200),
            (400, 200, 400, 400),
            (400, 400, 200, 400),
            (200, 400, 200, 200),
        ]
        target = [{"cls_name": "storage_gold", "center": (300, 300), "bbox_xyxy": (285, 285, 315, 315)}]
        drop_points = [(50, 50)]

        # Simular que get_wall_seg_detector() lanza FileNotFoundError porque models/wall_seg.onnx aún no existe
        with patch("vision.wall_compartments.get_wall_seg_detector", side_effect=FileNotFoundError("Model missing")):
            res = optimize_jumps(
                img_shape=shape,
                base_img=np.zeros((600, 600, 3), dtype=np.uint8),
                wall_segments=walls,
                detections=target,
                drop_points=drop_points,
                loot_weights={"storage_gold": 1.0},
                wall_detection_mode="unet",
                max_jumps=1,
                beam_width=2,
            )

            # No debe romperse: debe caer en fallback a raycast y resolver los muros
            self.assertEqual(res.initial_cost, 1.0)
            self.assertEqual(res.final_cost, 0.0)


if __name__ == "__main__":
    unittest.main()
