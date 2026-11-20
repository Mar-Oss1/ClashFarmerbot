"""
Tests Unitarios para el Modelo de Compartimentos de Muros y Optimización de Saltos.
Fase 2 de plan_saltos_optimos_antigravity.md.
Puro numpy / OpenCV, sin ONNX ni modelos pesados.
"""
from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from vision.wall_compartments import (
    rasterize_wall_mask,
    build_barrier_mask,
    compute_compartments,
    build_compartment_graph,
    map_targets_to_compartments,
    compute_total_cost,
    get_compartments_in_footprint,
    sample_jump_candidates,
    optimize_jumps,
    render_debug_overlay,
    CompartmentAnalysisResult,
    JumpCandidate,
    WallSegDetector,
    get_wall_seg_detector,
    reset_wall_seg_detector,
    is_valid_wall_centered_jump,
    DEFAULT_JUMP_RX,
    DEFAULT_JUMP_RY,
    DEFAULT_FUSION_RADIUS,
    DEFAULT_CORE_RADIUS,
    DEFAULT_MIN_WALL_PIXELS,
    DEFAULT_BEAM_WIDTH,
)


class TestWallCompartments(unittest.TestCase):
    """Batería de pruebas sintéticas para el optimizador de saltos por compartimentos."""

    def test_two_concentric_rings_bridges_both(self):
        """
        Test 1: Dos anillos concéntricos de muro con un objetivo en el centro.
        Un salto colocado en la zona intermedia entre ambos anillos tiene suficiente
        radio para cubrir tanto el anillo exterior como el interior, reduciendo
        el costo de 2 capas a 0 (ganancia marginal = 2.0).
        """
        shape = (600, 600)

        # Anillo exterior: 400x400 centrado en (300, 300), de x=100 a x=500
        outer_walls = [
            (100, 100, 500, 100),
            (500, 100, 500, 500),
            (500, 500, 100, 500),
            (100, 500, 100, 100),
        ]

        # Anillo interior: 200x200 centrado en (300, 300), de x=200 a x=400
        inner_walls = [
            (200, 200, 400, 200),
            (400, 200, 400, 400),
            (400, 400, 200, 400),
            (200, 400, 200, 200),
        ]
        all_walls = outer_walls + inner_walls

        # Objetivo de botín en el compartimento central
        target = [{
            "cls_name": "storage_dark",
            "center": (300, 300),
            "bbox_xyxy": (285, 285, 315, 315),
        }]

        # Punto de entrada de tropas en el exterior
        drop_points = [(50, 300)]

        # Huella circular con radio 65px (diámetro 130px, suficiente para cruzar
        # los 100px que separan el muro exterior del interior)
        result = optimize_jumps(
            img_shape=shape,
            wall_segments=all_walls,
            detections=target,
            drop_points=drop_points,
            jump_footprint=65.0,
            max_jumps=1,
            loot_weights={"storage_dark": 1.0},
            endpoint_tol=0.0,
            morph_kernel_size=3,
            wall_thickness=6,
        )

        # Verificaciones
        self.assertEqual(result.num_compartments, 3, "Debe haber 3 compartimentos: exterior, anillo intermedio y centro")
        self.assertEqual(result.initial_cost, 2.0, "Costo inicial sin saltos debe ser 2 capas de muro")
        self.assertGreaterEqual(len(result.jumps), 1, "Debe seleccionar al menos 1 salto")

        best_jump = result.jump_details[0]
        self.assertEqual(best_jump.gain, 2.0, "El salto óptimo debe puentear ambos anillos con ganancia de 2 capas")
        self.assertEqual(result.final_cost, 0.0, "El costo final hacia el objetivo debe ser 0")
        self.assertEqual(result.total_gain, 2.0, "La ganancia total debe ser 2.0")

    def test_ring_gap_blocked_by_building_and_extension(self):
        """
        Test 2: Anillo con hueco tapado por edificio.
        - Sin edificio y con extensión insuficiente: el espacio interior fuga al exterior (1 solo compartimento, costo 0).
        - Con buildings_block=True: el bounding box del edificio sella el hueco, creando 2 compartimentos y evitando la fuga.
        - Con endpoint_tol=35: la prolongación de extremos a lo largo de la dirección isométrica sella el hueco.
        """
        shape = (600, 600)

        # Anillo con hueco de 60px en la pared superior (de x=270 a x=330 en y=200)
        walls_with_gap = [
            (200, 200, 270, 200),
            (330, 200, 400, 200),
            (400, 200, 400, 400),
            (400, 400, 200, 400),
            (200, 400, 200, 200),
        ]

        # Edificio que tapa el hueco (ej: cañón o recolector)
        building = [{
            "cls_name": "cannon",
            "center": (300, 200),
            "bbox_xyxy": (265, 185, 335, 215),
        }]

        # Almacén de oro en el centro
        target = [{
            "cls_name": "storage_gold",
            "center": (300, 300),
            "bbox_xyxy": (285, 285, 315, 315),
        }]

        drop_points = [(50, 50)]

        # Caso A: buildings_block=False y sin extensión -> fuga al exterior
        res_leak = optimize_jumps(
            img_shape=shape,
            wall_segments=walls_with_gap,
            detections=building + target,
            drop_points=drop_points,
            buildings_block=False,
            endpoint_tol=5.0,
            morph_kernel_size=3,
            loot_weights={"storage_gold": 1.0},
        )
        self.assertEqual(res_leak.num_compartments, 1, "Sin barrera de edificio el muro fuga al exterior (1 compartimento)")
        self.assertEqual(res_leak.initial_cost, 0.0, "Costo inicial es 0 debido a la fuga")

        # Caso B: buildings_block=True -> la barrera del edificio sella el compartimento
        res_sealed = optimize_jumps(
            img_shape=shape,
            wall_segments=walls_with_gap,
            detections=building + target,
            drop_points=drop_points,
            buildings_block=True,
            endpoint_tol=5.0,
            morph_kernel_size=3,
            loot_weights={"storage_gold": 1.0},
        )
        self.assertEqual(res_sealed.num_compartments, 2, "La barrera del edificio sella el compartimento (2 compartimentos)")
        self.assertEqual(res_sealed.initial_cost, 1.0, "Costo inicial sellado es 1 capa de muro")

        # Caso C: Sin edificio, pero prolongación isométrica endpoint_tol=35 sella el hueco de 60px
        res_ext = optimize_jumps(
            img_shape=shape,
            wall_segments=walls_with_gap,
            detections=target,
            drop_points=drop_points,
            buildings_block=False,
            endpoint_tol=35.0,
            morph_kernel_size=5,
            iso_angles=(0.0, 90.0),
            loot_weights={"storage_gold": 1.0},
        )
        self.assertEqual(res_ext.num_compartments, 2, "La prolongación isométrica debe sellar el hueco")
        self.assertEqual(res_ext.initial_cost, 1.0, "Costo debe ser 1 capa con el hueco sellado por extensión")

    def test_village_without_walls_zero_jumps(self):
        """
        Test 3: Aldea sin muros.
        Debe devolver 0 saltos, costo inicial 0 y ganancia 0.
        """
        shape = (600, 600)
        target = [{
            "cls_name": "storage_gold",
            "center": (300, 300),
            "bbox_xyxy": (285, 285, 315, 315),
        }]
        drop_points = [(50, 50)]

        res = optimize_jumps(
            img_shape=shape,
            wall_segments=[],
            detections=target,
            drop_points=drop_points,
        )

        self.assertEqual(res.jumps, [], "Aldea sin muros debe devolver lista vacía de saltos")
        self.assertEqual(res.initial_cost, 0.0, "Costo inicial sin muros debe ser 0")
        self.assertEqual(res.final_cost, 0.0, "Costo final debe ser 0")
        self.assertEqual(res.total_gain, 0.0, "Ganancia total debe ser 0")

    def test_two_targets_separate_compartments_no_duplicate(self):
        """
        Test 4: Dos objetivos en compartimentos distintos y distantes.
        La selección voraz abre el primer compartimento y en la siguiente iteración
        NO duplica la apertura anterior, sino que selecciona el segundo compartimento.
        """
        shape = (600, 600)

        # Sala 1 (izquierda): x de 50 a 180, y de 200 a 400
        room1 = [
            (50, 200, 180, 200),
            (180, 200, 180, 400),
            (180, 400, 50, 400),
            (50, 400, 50, 200),
        ]

        # Sala 2 (derecha, a 240px de distancia): x de 420 a 550, y de 200 a 400
        room2 = [
            (420, 200, 550, 200),
            (550, 200, 550, 400),
            (550, 400, 420, 400),
            (420, 400, 420, 200),
        ]
        walls = room1 + room2

        targets = [
            {"cls_name": "storage_gold", "center": (115, 300), "bbox_xyxy": (100, 285, 130, 315)},
            {"cls_name": "storage_elixir", "center": (485, 300), "bbox_xyxy": (470, 285, 500, 315)},
        ]
        drop_points = [(300, 100)]

        # Radio de salto de 50px (insuficiente para alcanzar ambas salas a la vez)
        res = optimize_jumps(
            img_shape=shape,
            wall_segments=walls,
            detections=targets,
            drop_points=drop_points,
            jump_footprint=50.0,
            max_jumps=3,
            loot_weights={"storage_gold": 1.0, "storage_elixir": 1.0},
            endpoint_tol=0.0,
            morph_kernel_size=3,
        )

        self.assertEqual(res.num_compartments, 3, "Debe haber 3 compartimentos: exterior, sala 1 y sala 2")
        self.assertEqual(res.initial_cost, 2.0, "Costo inicial debe ser 2 (1 capa por cada sala)")
        self.assertEqual(len(res.jumps), 2, "Debe seleccionar exactamente 2 saltos (uno para cada sala)")

        # Verificar que ambos saltos tienen ganancia 1.0 y abrieron compartimentos distintos
        jump1 = res.jump_details[0]
        jump2 = res.jump_details[1]
        self.assertEqual(jump1.gain, 1.0)
        self.assertEqual(jump2.gain, 1.0)
        self.assertNotEqual(
            set(jump1.bridged_compartments),
            set(jump2.bridged_compartments),
            "El segundo salto no debe duplicar los compartimentos puenteados por el primero",
        )
        self.assertEqual(res.final_cost, 0.0, "El costo final debe ser 0 al abrir ambas salas")

    def test_isometric_ellipse_footprint(self):
        """
        Test 5: Soporte de huella elíptica isométrica.
        Verifica que una elipse (rx=70, ry=45) colocada en el plano isométrico
        puentee de forma óptima las capas concéntricas con ganancia 2.0.
        """
        shape = (600, 600)
        outer_walls = [
            (100, 100, 500, 100),
            (500, 100, 500, 500),
            (500, 500, 100, 500),
            (100, 500, 100, 100),
        ]
        inner_walls = [
            (200, 200, 400, 200),
            (400, 200, 400, 400),
            (400, 400, 200, 400),
            (200, 400, 200, 200),
        ]
        target = [{"cls_name": "storage_dark", "center": (300, 300), "bbox_xyxy": (285, 285, 315, 315)}]
        drop_points = [(50, 300)]

        ellipse_fp = {"type": "ellipse", "rx": 70.0, "ry": 45.0, "angle": 0.0}

        res = optimize_jumps(
            img_shape=shape,
            wall_segments=outer_walls + inner_walls,
            detections=target,
            drop_points=drop_points,
            jump_footprint=ellipse_fp,
            max_jumps=1,
            loot_weights={"storage_dark": 1.0},
            endpoint_tol=0.0,
            morph_kernel_size=3,
            wall_thickness=6,
        )

        self.assertGreaterEqual(len(res.jumps), 1, "Debe seleccionar un salto elíptico")
        self.assertEqual(res.total_gain, 2.0, "La huella elíptica debe lograr ganancia de 2 capas")
        self.assertEqual(res.final_cost, 0.0)

    def test_execution_time_budget(self):
        """
        Test 6: Presupuesto de tiempo de ejecución < 300 ms.
        """
        shape = (1080, 1920)
        # Generar 20 segmentos de muro sintéticos
        segments = []
        for i in range(10):
            segments.append((300 + i * 50, 300, 300 + i * 50, 700))
            segments.append((300, 300 + i * 40, 800, 300 + i * 40))

        targets = [
            {"cls_name": "storage_gold", "center": (500, 500), "bbox_xyxy": (480, 480, 520, 520)},
            {"cls_name": "storage_dark", "center": (600, 550), "bbox_xyxy": (580, 530, 620, 570)},
        ]
        drop_points = [(100, 500)]

        res = optimize_jumps(
            img_shape=shape,
            wall_segments=segments,
            detections=targets,
            drop_points=drop_points,
            jump_footprint=90.0,
            max_jumps=3,
            candidate_step=30,
            time_budget_sec=0.300,
        )

        self.assertLess(res.execution_time_ms, 300.0, f"El tiempo ({res.execution_time_ms:.1f} ms) debe ser < 300 ms")

    def test_wall_seg_detector_inference_and_fallback(self):
        """
        Test 7: WallSegDetector - inferencia mockeada y fallback transparente cuando
        models/wall_seg.onnx no existe.
        """
        reset_wall_seg_detector()

        # 1. Fallback transparente cuando el archivo no existe
        detector_fallback = WallSegDetector(model_path="models/non_existent_wall_seg.onnx")
        self.assertFalse(detector_fallback.is_ready())

        h, w = 1080, 1920
        dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
        dummy_img[300:320, 200:800] = 180

        mask_fallback = detector_fallback.predict_mask(dummy_img)
        self.assertEqual(mask_fallback.shape, (h, w))
        self.assertEqual(mask_fallback.dtype, np.uint8)
        # Verificar que el HUD de batalla (y >= 888) está enmascarado
        self.assertTrue(np.all(mask_fallback[888:, :] == 0))

        # 2. Inferencia ONNX mockeada
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "images"
        mock_session.get_inputs.return_value = [mock_input]

        mock_out0 = MagicMock()
        mock_out0.name = "output0"
        mock_out1 = MagicMock()
        mock_out1.name = "output1"
        mock_session.get_outputs.return_value = [mock_out0, mock_out1]

        # Simular tensor YOLOv8-seg: out0 (1, 37, 8400), protos (1, 32, 160, 160)
        out0_tensor = np.zeros((1, 37, 8400), dtype=np.float32)
        out0_tensor[0, 4, 0] = 0.95
        out0_tensor[0, 5:, 0] = 1.0

        protos_tensor = np.zeros((1, 32, 160, 160), dtype=np.float32)
        protos_tensor[0, :, 50:110, 50:110] = 2.0

        mock_session.run.return_value = [out0_tensor, protos_tensor]

        with patch("os.path.exists", return_value=True), \
             patch("onnxruntime.InferenceSession", return_value=mock_session):
            detector_mock = WallSegDetector(model_path="models/wall_seg.onnx")
            self.assertTrue(detector_mock.is_ready())

            mask_onnx = detector_mock.predict_mask(dummy_img)
            self.assertEqual(mask_onnx.shape, (h, w))
            self.assertEqual(mask_onnx.dtype, np.uint8)
            self.assertTrue(np.any(mask_onnx > 0))
            self.assertTrue(np.all(mask_onnx[888:, :] == 0))

    def test_physical_calibration_and_wall_density_rejection(self):
        """
        Test 8: Calibración física real (rx=76, ry=57, radius=72) y filtro de
        densidad de muro (is_valid_wall_centered_jump).
        """
        self.assertEqual(DEFAULT_JUMP_RX, 76.0)
        self.assertEqual(DEFAULT_JUMP_RY, 57.0)
        self.assertEqual(DEFAULT_FUSION_RADIUS, 72.0)
        self.assertEqual(DEFAULT_CORE_RADIUS, 42.0)
        self.assertEqual(DEFAULT_MIN_WALL_PIXELS, 30)

        h, w = 600, 600
        wall_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(wall_mask, (100, 300), (500, 300), 255, 8)

        # Punto centrado sobre muro: aceptado
        pt_on_wall = (300, 300)
        self.assertTrue(
            is_valid_wall_centered_jump(
                pt_on_wall,
                wall_mask,
                core_radius=DEFAULT_CORE_RADIUS,
                min_wall_pixels=DEFAULT_MIN_WALL_PIXELS,
            )
        )

        # Punto en el centro vacío del compartimento: rechazado
        pt_empty_center = (300, 150)
        self.assertFalse(
            is_valid_wall_centered_jump(
                pt_empty_center,
                wall_mask,
                core_radius=DEFAULT_CORE_RADIUS,
                min_wall_pixels=DEFAULT_MIN_WALL_PIXELS,
            )
        )

        # Punto con densidad insuficiente (< 30 píxeles en núcleo)
        sparse_mask = np.zeros((h, w), dtype=np.uint8)
        sparse_mask[295:305, 300] = 255
        self.assertFalse(
            is_valid_wall_centered_jump(
                (300, 300),
                sparse_mask,
                core_radius=DEFAULT_CORE_RADIUS,
                min_wall_pixels=DEFAULT_MIN_WALL_PIXELS,
            )
        )

    def test_beam_search_bridges_zero_marginal_gain_intermediate(self):
        """
        Test 9: Beam Search resolviendo una cadena de saltos donde un salto
        intermedio aporta ganancia marginal 0.0.
        """
        # 1. Validación matemática del grafo
        graph = {
            0: {1, 2},
            1: {0, 4},
            2: {0, 3},
            3: {2, 4},
            4: {1, 3},
        }
        entry_comps = {0}
        targets = [{"comp": 4, "weight": 1.0}]

        init_cost, _ = compute_total_cost(graph, entry_comps, targets, jump_bridges=[])
        self.assertEqual(init_cost, 2.0)

        # Salto intermedio en Ruta B: ganancia marginal = 0.0
        cost_jump1, _ = compute_total_cost(graph, entry_comps, targets, jump_bridges=[{0, 2}])
        marginal_gain_jump1 = init_cost - cost_jump1
        self.assertEqual(marginal_gain_jump1, 0.0)

        # Salto 2 completando la cadena: costo final = 0.0
        cost_jump2, _ = compute_total_cost(graph, entry_comps, targets, jump_bridges=[{0, 2}, {2, 3, 4}])
        self.assertEqual(cost_jump2, 0.0)
        self.assertEqual(init_cost - cost_jump2, 2.0)

        # 2. Validación end-to-end con optimize_jumps
        shape = (700, 700)
        wall1 = [(150, 200, 550, 200), (550, 200, 550, 500), (550, 500, 150, 500), (150, 500, 150, 200)]
        wall2 = [(250, 280, 450, 280), (450, 280, 450, 420), (450, 420, 250, 420), (250, 420, 250, 280)]
        all_walls = wall1 + wall2

        target = [{
            "cls_name": "storage_dark",
            "center": (350, 350),
            "bbox_xyxy": (330, 330, 370, 370),
        }]
        drop_points = [(50, 350)]

        res = optimize_jumps(
            img_shape=shape,
            wall_segments=all_walls,
            detections=target,
            drop_points=drop_points,
            max_jumps=2,
            beam_width=4,
            jump_footprint={"type": "ellipse", "rx": 76.0, "ry": 57.0, "radius": 72.0, "angle": 0.0},
            loot_weights={"storage_dark": 1.0},
            endpoint_tol=0.0,
            morph_kernel_size=3,
            wall_thickness=6,
        )

        self.assertGreaterEqual(len(res.jumps), 1)
        self.assertEqual(res.final_cost, 0.0)
        self.assertEqual(res.total_gain, res.initial_cost)

    def test_render_debug_overlay_scales_jumps_properly(self):
        """
        Test 10: Valida que render_debug_overlay aplique grid_scale a los saltos.
        En resoluciones HD (ej. 2400x1080) con grid_scale=2, el lienzo se reduce a 1200x540.
        Los saltos con coordenadas nativas (ej. (1600, 800)) deben dibujarse escalados
        en (800, 400) con su marcador rojo (0, 0, 255) dentro de los límites del canvas.
        """
        h_scaled, w_scaled = 540, 1200
        comp_labels = np.zeros((h_scaled, w_scaled), dtype=np.int32)
        base_img = np.zeros((1080, 2400, 3), dtype=np.uint8)
        jump = JumpCandidate(
            x=1600,
            y=800,
            gain=2.0,
            cost_before=3.0,
            cost_after=1.0,
            bridged_compartments=[1, 2],
        )

        canvas = render_debug_overlay(
            comp_labels=comp_labels,
            base_img=base_img,
            wall_mask=None,
            selected_jumps=[jump],
            footprint=72.0,
            grid_scale=2,
        )

        self.assertEqual(canvas.shape, (540, 1200, 3))
        # El marcador central rojo cv2.circle(canvas, (jx, jy), 5, (0, 0, 255), -1)
        # debe ubicarse exactamente en (800, 400)
        np.testing.assert_array_equal(canvas[400, 800], [0, 0, 255])
        # Las coordenadas originales 1600 estarían fuera del ancho de 1200 si no se escalaran
        self.assertGreater(jump.x, w_scaled)

    def test_beam_search_40_segments_performance_within_300ms(self):
        """
        Test 11: Benchmark de rendimiento en 1080p con 40 segmentos de muro.
        Beam Search con beam_width=4 y max_jumps=4 debe completar todos los niveles
        respetando estrictamente el presupuesto < 300 ms (time_budget_sec=0.300).
        """
        # Generar 10 anillos concéntricos = 40 segmentos de muro en 2400x1080
        segments = []
        cx, cy = 1200, 540
        for i in range(1, 11):
            half_w = 60 * i
            half_h = 35 * i
            x1, y1 = cx - half_w, cy - half_h
            x2, y2 = cx + half_w, cy + half_h
            segments.append((x1, y1, x2, y1))
            segments.append((x2, y1, x2, y2))
            segments.append((x2, y2, x1, y2))
            segments.append((x1, y2, x1, y1))

        self.assertEqual(len(segments), 40)

        targets = [
            {"cls_name": "storage_gold", "center": (cx, cy), "bbox_xyxy": (cx - 20, cy - 20, cx + 20, cy + 20)},
            {"cls_name": "storage_elixir", "center": (cx + 50, cy + 20), "bbox_xyxy": (cx + 30, cy, cx + 70, cy + 40)},
            {"cls_name": "storage_dark", "center": (cx - 50, cy - 20), "bbox_xyxy": (cx - 70, cy - 40, cx - 30, cy)},
            {"cls_name": "town_hall", "center": (cx + 20, cy - 30), "bbox_xyxy": (cx, cy - 50, cx + 40, cy - 10)},
        ]
        drop_points = [(100, 540)]

        res = optimize_jumps(
            img_shape=(1080, 2400),
            wall_segments=segments,
            detections=targets,
            drop_points=drop_points,
            beam_width=4,
            max_jumps=4,
            time_budget_sec=0.300,
        )

        self.assertLess(res.execution_time_ms, 300.0, f"Beam Search excedió el presupuesto: {res.execution_time_ms} ms")
        self.assertIn(len(res.jumps), (3, 4), "Debe haber completado los niveles de saltos necesarios")
        self.assertEqual(res.final_cost, 0.0, "Los saltos deben permitir acceso total a los objetivos")


if __name__ == "__main__":
    unittest.main()
