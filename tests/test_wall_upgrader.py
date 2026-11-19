"""
tests/test_wall_upgrader.py
Suite de validación exhaustiva para el módulo de mejora de muros generalizado:
1. Validación de la fuente única de verdad oficial (WALL_LEVEL_COSTS) niveles 1 a 19.
2. Validación de activación de Elixir estrictamente desde nivel 5 (None en 1 a 4).
3. Validación de can_afford() y get_required_resources().
4. Validación del flag is_extrapolated (True para niveles >= 20, False para <= 19).
5. Validación de transiciones (5->6, 10->11, 18->19 y dinámicas N->N+1).
6. Validación de filtro combinado (RSS + Permanente) para cobertura del 100%.
7. Validación de la plantilla global reutilizable agnóstica de nivel y fallback sin regresión 18->19.
8. Validación de evaluación dinámica de costos (blanco vs rojo).
9. Validación de integración dinámica con Ledger y SETTINGS.
"""
import os
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from core.wall_cost_table import (
    WALL_LEVEL_COSTS,
    get_required_resources,
    can_afford,
)
from core.wall_upgrader import (
    WallUpgradeStrategy,
    WallTransition,
    WallUpgrader,
)
from core.ledger import Ledger
from core.config import PROJECT_ROOT


class TestOfficialWallCostsAndValidation(unittest.TestCase):
    """
    Pruebas rigurosas sobre la fuente oficial de verdad y reglas de recursos:
    - Niveles 1-4: elixir_cost is None.
    - Nivel 5 en adelante: requiere Oro y Elixir simultáneamente según la tabla oficial.
    - Validación can_afford y get_required_resources.
    - Flag is_extrapolated para niveles >= 20.
    """

    def test_official_table_levels_1_to_4_no_elixir(self):
        """Verifica que los niveles 1 a 4 no requieren elixir (elixir_cost is None)."""
        expected_gold = {
            1: 0,
            2: 1_000,
            3: 5_000,
            4: 10_000,
        }
        for lvl in (1, 2, 3, 4):
            self.assertIn(lvl, WALL_LEVEL_COSTS)
            row = WALL_LEVEL_COSTS[lvl]
            self.assertIsNone(row["elixir_cost"], f"Nivel {lvl} no debe requerir elixir")
            self.assertEqual(row["gold_cost"], expected_gold[lvl])

            req = get_required_resources(lvl)
            self.assertEqual(req["gold"], expected_gold[lvl])
            self.assertIsNone(req["elixir"])

    def test_official_table_levels_5_to_19_exact_costs(self):
        """
        Verifica que del nivel 5 al 19 se requiere Oro y Elixir simultáneamente
        con los valores exactos estipulados por Supercell.
        """
        official_expected = {
            5:  {"gold": 20_000,     "elixir": 20_000,     "th": 5},
            6:  {"gold": 30_000,     "elixir": 30_000,     "th": 6},
            7:  {"gold": 50_000,     "elixir": 50_000,     "th": 7},
            8:  {"gold": 75_000,     "elixir": 75_000,     "th": 8},
            9:  {"gold": 100_000,    "elixir": 100_000,    "th": 9},
            10: {"gold": 200_000,    "elixir": 200_000,    "th": 9},
            11: {"gold": 500_000,    "elixir": 500_000,    "th": 10},
            12: {"gold": 1_000_000,  "elixir": 1_000_000,  "th": 11},
            13: {"gold": 1_500_000,  "elixir": 1_500_000,  "th": 12},
            14: {"gold": 2_000_000,  "elixir": 2_000_000,  "th": 13},
            15: {"gold": 3_000_000,  "elixir": 3_000_000,  "th": 14},
            16: {"gold": 4_000_000,  "elixir": 4_000_000,  "th": 15},
            17: {"gold": 5_000_000,  "elixir": 5_000_000,  "th": 16},
            18: {"gold": 7_000_000,  "elixir": 7_000_000,  "th": 17},
            19: {"gold": 10_000_000, "elixir": 10_000_000, "th": 18},
        }

        for lvl, expected in official_expected.items():
            self.assertIn(lvl, WALL_LEVEL_COSTS)
            row = WALL_LEVEL_COSTS[lvl]
            self.assertEqual(row["gold_cost"], expected["gold"], f"Oro incorrecto para nivel {lvl}")
            self.assertEqual(row["elixir_cost"], expected["elixir"], f"Elixir incorrecto para nivel {lvl}")
            self.assertEqual(row["th_required"], expected["th"], f"TH incorrecto para nivel {lvl}")

            req = get_required_resources(lvl)
            self.assertEqual(req["gold"], expected["gold"])
            self.assertEqual(req["elixir"], expected["elixir"])

    def test_can_afford_levels_1_to_4_ignores_elixir(self):
        """Verifica que para niveles 1-4 no se rechace por falta de elixir."""
        # Nivel 2 cuesta 1,000 Oro y 0 Elixir
        self.assertTrue(can_afford(target_level=2, available_gold=1_000, available_elixir=0))
        self.assertTrue(can_afford(target_level=2, available_gold=5_000, available_elixir=0))
        # Rechaza si falta oro
        self.assertFalse(can_afford(target_level=2, available_gold=999, available_elixir=100_000))

        # Nivel 4 cuesta 10,000 Oro y 0 Elixir
        self.assertTrue(can_afford(target_level=4, available_gold=10_000, available_elixir=0))
        self.assertFalse(can_afford(target_level=4, available_gold=9_999, available_elixir=100_000))

    def test_can_afford_level_5_plus_accepts_if_either_gold_or_elixir_sufficient(self):
        """
        Verifica que can_afford() acepte correctamente una mejora de nivel 5+
        si alcanza Oro O si alcanza Elixir (lógica OR de Clash of Clans).
        Rechaza únicamente si ambos recursos son insuficientes.
        Cubre los casos benchmark solicitados: lvl 5 (20k), lvl 10 (200k), lvl 18 (7M), lvl 19 (10M).
        """
        # Nivel 5 pide 20,000 Oro o 20,000 Elixir
        # Caso 1: Oro de sobra (100M), pero Elixir insuficiente (19,999) -> ACEPTADO (alcanza con Oro)
        self.assertTrue(
            can_afford(target_level=5, available_gold=100_000_000, available_elixir=19_999),
            "Debe aceptar nivel 5 si alcanza oro aunque falte elixir"
        )
        # Caso 2: Oro insuficiente (19,999), pero Elixir de sobra (100M) -> ACEPTADO (alcanza con Elixir)
        self.assertTrue(can_afford(target_level=5, available_gold=19_999, available_elixir=100_000_000))
        # Caso 3: Elixir exacto y Oro exacto -> ACEPTADO
        self.assertTrue(can_afford(target_level=5, available_gold=20_000, available_elixir=20_000))
        # Caso 4: Ambos insuficientes -> RECHAZADO
        self.assertFalse(can_afford(target_level=5, available_gold=19_999, available_elixir=19_999))

        # Nivel 10 pide 200,000 Oro o 200,000 Elixir
        self.assertTrue(
            can_afford(target_level=10, available_gold=50_000_000, available_elixir=199_999),
            "Debe aceptar nivel 10 si alcanza oro aunque falte elixir"
        )
        self.assertTrue(can_afford(target_level=10, available_gold=199_999, available_elixir=50_000_000))
        self.assertTrue(can_afford(target_level=10, available_gold=200_000, available_elixir=200_000))
        self.assertFalse(can_afford(target_level=10, available_gold=199_999, available_elixir=199_999))

        # Nivel 12 pide 1,000,000 Oro o 1,000,000 Elixir
        self.assertTrue(can_afford(target_level=12, available_gold=50_000_000, available_elixir=999_999))
        self.assertTrue(can_afford(target_level=12, available_gold=999_999, available_elixir=50_000_000))
        self.assertTrue(can_afford(target_level=12, available_gold=1_000_000, available_elixir=1_000_000))
        self.assertFalse(can_afford(target_level=12, available_gold=999_999, available_elixir=999_999))

        # Nivel 18 pide 7,000,000 Oro o 7,000,000 Elixir
        self.assertTrue(
            can_afford(target_level=18, available_gold=100_000_000, available_elixir=6_999_999),
            "Debe aceptar nivel 18 si alcanza oro aunque falte elixir"
        )
        self.assertTrue(can_afford(target_level=18, available_gold=6_999_999, available_elixir=100_000_000))
        self.assertTrue(can_afford(target_level=18, available_gold=7_000_000, available_elixir=7_000_000))
        self.assertFalse(can_afford(target_level=18, available_gold=6_999_999, available_elixir=6_999_999))

        # Nivel 19 pide 10,000,000 Oro o 10,000,000 Elixir
        self.assertTrue(
            can_afford(target_level=19, available_gold=20_000_000, available_elixir=9_999_999),
            "Debe aceptar nivel 19 si alcanza oro aunque falte elixir"
        )
        self.assertTrue(can_afford(target_level=19, available_gold=9_999_999, available_elixir=20_000_000))
        self.assertTrue(can_afford(target_level=19, available_gold=10_000_000, available_elixir=10_000_000))
        self.assertTrue(can_afford(target_level=19, available_gold=10_000_000, available_elixir=0))
        self.assertTrue(can_afford(target_level=19, available_gold=0, available_elixir=10_000_000))
        self.assertFalse(can_afford(target_level=19, available_gold=5_000_000, available_elixir=5_000_000))
        self.assertFalse(can_afford(target_level=19, available_gold=9_999_999, available_elixir=9_999_999))

        # Opcional: require_both=True
        self.assertFalse(can_afford(target_level=19, available_gold=10_000_000, available_elixir=0, require_both=True))
        self.assertTrue(can_afford(target_level=19, available_gold=10_000_000, available_elixir=10_000_000, require_both=True))

    def test_benchmark_levels_exact_costs_in_table(self):
        """
        Valida explícitamente los niveles benchmark requeridos en la especificación:
        - Nivel 5: 20k oro, 20k elixir
        - Nivel 10: 200k oro, 200k elixir
        - Nivel 18: 7M oro, 7M elixir
        - Nivel 19: 10M oro, 10M elixir
        """
        benchmarks = {
            5: (20_000, 20_000),
            10: (200_000, 200_000),
            18: (7_000_000, 7_000_000),
            19: (10_000_000, 10_000_000),
        }
        for lvl, (gold, elixir) in benchmarks.items():
            self.assertEqual(WALL_LEVEL_COSTS[lvl]["gold_cost"], gold)
            self.assertEqual(WALL_LEVEL_COSTS[lvl]["elixir_cost"], elixir)
            req = get_required_resources(lvl)
            self.assertEqual(req["gold"], gold)
            self.assertEqual(req["elixir"], elixir)

    def test_is_extrapolated_flag_transition_and_strategy(self):
        """
        Verifica que el flag is_extrapolated sea False para niveles <= 19
        y True para niveles >= 20 en WallTransition y WallUpgradeStrategy.
        """
        strategy = WallUpgradeStrategy()

        # Niveles tabulados oficialmente (1 a 19): is_extrapolated = False
        for tgt in (2, 5, 6, 11, 19):
            src = tgt - 1
            tr = strategy.get_transition(src, tgt)
            self.assertFalse(tr.is_extrapolated, f"Nivel {tgt} no debe ser extrapolado")
            self.assertFalse(strategy.is_extrapolated, f"Strategy no debe marcar extrapolado para {tgt}")

        # Nivel 20 (fuera de tabla oficial): is_extrapolated = True
        tr20 = strategy.get_transition(19, 20)
        self.assertTrue(tr20.is_extrapolated, "Nivel 20 debe marcar is_extrapolated = True")
        self.assertTrue(strategy.is_extrapolated, "Strategy debe marcar is_extrapolated = True")

        # Nivel 25: is_extrapolated = True
        tr25 = strategy.get_transition(24, 25)
        self.assertTrue(tr25.is_extrapolated)
        self.assertTrue(strategy.is_extrapolated_level(25))
        self.assertFalse(strategy.is_extrapolated_level(19))


class TestWallUpgradeStrategy(unittest.TestCase):
    """Pruebas unitarias de la lógica y estrategia dinámica de muros."""

    def setUp(self):
        self.strategy = WallUpgradeStrategy()

    def test_transitions_for_three_distinct_levels(self):
        """Verifica al menos 3 niveles distintos: 5->6, 10->11 y 18->19."""
        # 1. Nivel 5 -> 6 (Target 6: 30,000 Oro, 30,000 Elixir, TH6)
        tr5 = self.strategy.get_transition(5, 6)
        self.assertEqual(tr5.source_level, 5)
        self.assertEqual(tr5.target_level, 6)
        self.assertEqual(tr5.cost_gold, 30_000)
        self.assertEqual(tr5.cost_elixir, 30_000)
        self.assertTrue(tr5.can_use_elixir, "Nivel 6 permite Elixir según la tabla oficial")
        self.assertEqual(tr5.required_th, 6)
        self.assertEqual(tr5.upgrade_time_sec, 0, "Tiempo de muro siempre es instantáneo (0s)")
        self.assertFalse(tr5.is_extrapolated)

        # 2. Nivel 10 -> 11 (Target 11: 500,000 Oro, 500,000 Elixir, TH10)
        tr10 = self.strategy.get_transition(10, 11)
        self.assertEqual(tr10.source_level, 10)
        self.assertEqual(tr10.target_level, 11)
        self.assertEqual(tr10.cost_gold, 500_000)
        self.assertEqual(tr10.cost_elixir, 500_000)
        self.assertTrue(tr10.can_use_elixir, "Nivel 11 permite Elixir")
        self.assertEqual(tr10.required_th, 10)
        self.assertEqual(tr10.upgrade_time_sec, 0)
        self.assertFalse(tr10.is_extrapolated)

        # 3. Nivel 18 -> 19 (Target 19: 10,000,000 Oro, 10,000,000 Elixir, TH18)
        tr18 = self.strategy.get_transition(18, 19)
        self.assertEqual(tr18.source_level, 18)
        self.assertEqual(tr18.target_level, 19)
        self.assertEqual(tr18.cost_gold, 10_000_000)
        self.assertEqual(tr18.cost_elixir, 10_000_000)
        self.assertTrue(tr18.can_use_elixir, "Nivel 19 permite Elixir")
        self.assertEqual(tr18.required_th, 18)
        self.assertEqual(tr18.upgrade_time_sec, 0)
        self.assertFalse(tr18.is_extrapolated)

    def test_dynamic_arbitrary_transition_n_to_n_plus_one(self):
        """Valida que niveles más allá del 19 se calculan dinámicamente y se marcan como extrapolados."""
        tr20 = self.strategy.get_transition(19, 20)
        self.assertEqual(tr20.source_level, 19)
        self.assertEqual(tr20.target_level, 20)
        self.assertEqual(tr20.cost_gold, 11_000_000)
        self.assertEqual(tr20.cost_elixir, 11_000_000)
        self.assertTrue(tr20.can_use_elixir)
        self.assertTrue(tr20.is_extrapolated)
        self.assertEqual(tr20.upgrade_time_sec, 0)

        tr25 = self.strategy.get_transition(24, 25)
        self.assertEqual(tr25.source_level, 24)
        self.assertEqual(tr25.target_level, 25)
        self.assertTrue(tr25.is_extrapolated)
        self.assertGreaterEqual(tr25.required_th, 18)

    def test_resource_permissions_and_costs(self):
        """Valida que Oro siempre es válido y Elixir es válido únicamente desde nivel 5."""
        # Oro siempre permitido
        for src in (1, 4, 5, 10, 18, 20):
            self.assertTrue(self.strategy.can_use_resource(src, "gold"))
            self.assertGreater(self.strategy.get_cost(src, "gold"), 0)

        # Elixir prohibido para transiciones que culminan en niveles 1 a 4 (e.g. 1->2, 2->3, 3->4)
        for src in (1, 2, 3):
            self.assertFalse(self.strategy.can_use_resource(src, "elixir"))
            with self.assertRaises(ValueError):
                self.strategy.get_cost(src, "elixir")

        # Elixir permitido para transiciones que culminan en nivel 5 o superior (e.g. 4->5, 5->6, 10->11, 18->19)
        for src in (4, 5, 6, 8, 10, 14, 18):
            self.assertTrue(self.strategy.can_use_resource(src, "elixir"))
            self.assertGreater(self.strategy.get_cost(src, "elixir"), 0)

    def test_dynamic_batch_costs(self):
        """Verifica el cálculo dinámico de costo para lotes de múltiples muros consultando la tabla."""
        # 5 muros de nivel 5->6 (target 6, 30,000 c/u) -> 5 * 30,000 = 150,000
        self.assertEqual(self.strategy.calculate_batch_cost(5, 5, "gold"), 150_000)
        self.assertEqual(self.strategy.calculate_batch_cost(5, 5, "elixir"), 150_000)

        # 2 muros de nivel 10->11 (target 11, 500,000 c/u) -> 2 * 500,000 = 1,000,000
        self.assertEqual(self.strategy.calculate_batch_cost(10, 2, "elixir"), 1_000_000)

        # 2 muros de nivel 18->19 (target 19, 10,000,000 c/u) -> 2 * 10,000,000 = 20,000,000
        self.assertEqual(self.strategy.calculate_batch_cost(18, 2, "gold"), 20_000_000)

        # Consulta separada de ambos recursos (resource_type="both" o None)
        batch_both = self.strategy.calculate_batch_cost(5, 5, resource_type="both")
        self.assertEqual(batch_both, {"gold": 150_000, "elixir": 150_000})

        # Consulta separada para nivel bajo (1->2, elixir es None)
        batch_low = self.strategy.calculate_batch_cost(1, 3, resource_type=None)
        self.assertEqual(batch_low, {"gold": 3_000, "elixir": None})

    def test_dynamic_min_rss_threshold(self):
        """Verifica que el umbral de activación se adapta dinámicamente según el nivel."""
        # Nivel 5 (5->6, 30k c/u): 2 muros = 60,000
        self.assertEqual(self.strategy.get_min_rss_threshold(5, batch_size=2), 60_000)
        # Nivel 10 (10->11, 500k c/u): 2 muros = 1_000_000
        self.assertEqual(self.strategy.get_min_rss_threshold(10, batch_size=2), 1_000_000)
        # Nivel 18 (18->19, 10M c/u): 2 muros = 20_000_000
        self.assertEqual(self.strategy.get_min_rss_threshold(18, batch_size=2), 20_000_000)
        # Sin nivel (None): usa el fallback
        self.assertEqual(self.strategy.get_min_rss_threshold(None, fallback=21_000_000), 21_000_000)

    def test_estimate_affordable_walls(self):
        """Verifica la estimación de muros costeables según recursos."""
        # Con 150,000 en nivel 5->6 (30,000 c/u) -> 5 muros
        self.assertEqual(self.strategy.estimate_affordable_walls(150_000, 5, "gold"), 5)
        # Con 2,500,000 en nivel 10->11 (500,000 c/u) -> 5 muros
        self.assertEqual(self.strategy.estimate_affordable_walls(2_500_000, 10, "elixir"), 5)
        # Con 25,000,000 en nivel 18->19 (10,000,000 c/u) -> 2 muros
        self.assertEqual(self.strategy.estimate_affordable_walls(25_000_000, 18, "gold"), 2)


class TestWallUpgraderFilters(unittest.TestCase):
    """Pruebas del filtro combinado (RSS + Permanente) para cobertura del 100%."""

    def setUp(self):
        self.mock_device = MagicMock()
        self.mock_device.width = 1920
        self.mock_device.height = 1080
        self.mock_device.screenshot.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)

        self.mock_vision = MagicMock()
        self.upgrader = WallUpgrader(self.mock_device, self.mock_vision)

        self._patch_sleep = patch("time.sleep", return_value=None)
        self._patch_sleep.start()
        self._patch_delay = patch("core.wall_upgrader.HumanBehavior.reaction_delay", return_value=None)
        self._patch_delay.start()

    def tearDown(self):
        self._patch_sleep.stop()
        self._patch_delay.stop()

    def test_apply_filter_activates_both_rss_and_permanent(self):
        """
        Verifica que _apply_filter busca y activa tanto el filtro de recurso
        como el filtro permanente, garantizando la cobertura del 100%.
        """
        self.mock_vision.find_template.return_value = (1800, 100)
        self.mock_vision.wait_for_template.side_effect = [
            (1500, 300),  # target_res_tpl (gold)
            (1500, 450),  # filter_perm_off
        ]

        self.upgrader._apply_filter("gold")

        self.assertEqual(self.mock_device.tap.call_count, 4)
        taps = [call[0] for call in self.mock_device.tap.call_args_list]
        self.assertEqual(taps[0], (1800, 100))
        self.assertEqual(taps[1], (1500, 300))
        self.assertEqual(taps[2], (1500, 450))
        self.assertEqual(taps[3], (1800, 100))

    def test_apply_filter_elixir_and_permanent(self):
        """Verifica la activación combinada con Elixir y Permanente."""
        self.mock_vision.find_template.return_value = (1800, 100)
        self.mock_vision.wait_for_template.side_effect = [
            (1500, 380),  # target_res_tpl (elixir)
            (1500, 450),  # filter_perm_off
        ]

        self.upgrader._apply_filter("elixir")

        taps = [call[0] for call in self.mock_device.tap.call_args_list]
        self.assertEqual(taps[1], (1500, 380))
        self.assertEqual(taps[2], (1500, 450))

    def test_apply_filter_already_active_filters(self):
        """Si los filtros ya están encendidos (no detecta _off), solo abre y cierra el menú."""
        self.mock_vision.find_template.return_value = (1800, 100)
        self.mock_vision.wait_for_template.return_value = None

        self.upgrader._apply_filter("gold")

        self.assertEqual(self.mock_device.tap.call_count, 2)
        taps = [call[0] for call in self.mock_device.tap.call_args_list]
        self.assertEqual(taps[0], (1800, 100))
        self.assertEqual(taps[1], (1800, 100))


class TestWallUpgraderTemplateAndMatching(unittest.TestCase):
    """Pruebas de la plantilla global reutilizable y detección agnóstica de nivel."""

    def setUp(self):
        self._patch_sleep = patch("time.sleep", return_value=None)
        self._patch_sleep.start()
        self._patch_delay = patch("core.wall_upgrader.HumanBehavior.reaction_delay", return_value=None)
        self._patch_delay.start()

    def tearDown(self):
        self._patch_sleep.stop()
        self._patch_delay.stop()

    def test_generic_template_file_exists(self):
        """Verifica la existencia física de la plantilla global reutilizable."""
        gen_path = os.path.join(PROJECT_ROOT, "assets/buttons/wall_row_generic.png")
        self.assertTrue(os.path.exists(gen_path), "assets/buttons/wall_row_generic.png debe existir")

        img = cv2.imread(gen_path)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[0], 38, "La altura debe ser 38px")
        self.assertEqual(img.shape[1], 75, "El ancho debe ser 75px (palabra 'Wall ')")

    def test_generic_template_matches_legacy_item(self):
        """Verifica que la plantilla global hace match perfecto (1.0) con la fila original."""
        gen_path = os.path.join(PROJECT_ROOT, "assets/buttons/wall_row_generic.png")
        leg_path = os.path.join(PROJECT_ROOT, "assets/buttons/wall_row_item.png")
        gen_img = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)
        leg_img = cv2.imread(leg_path, cv2.IMREAD_GRAYSCALE)

        res = cv2.matchTemplate(leg_img, gen_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        self.assertGreaterEqual(max_val, 0.99, f"Debe coincidir con la fila original (max_val={max_val})")
        self.assertEqual(max_loc, (0, 0), "El match debe situarse al inicio de la fila")

    def test_find_and_tap_uses_generic_template_first(self):
        """Verifica que _find_and_tap_wall_in_list utiliza la plantilla genérica primero."""
        mock_device = MagicMock()
        mock_device.screenshot.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_vision = MagicMock()

        def mock_find(screen, tpl, threshold=0.74, zona=None):
            if "wall_row_generic" in tpl:
                return (1600, 420)
            return None

        mock_vision.find_template.side_effect = mock_find
        upgrader = WallUpgrader(mock_device, mock_vision)

        found = upgrader._find_and_tap_wall_in_list()
        self.assertTrue(found)
        mock_device.tap.assert_called_once_with(1600, 420)

    def test_find_and_tap_fallback_to_legacy(self):
        """Verifica que si la genérica no coincide, el fallback a legacy (18->19) funciona."""
        mock_device = MagicMock()
        mock_device.screenshot.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_vision = MagicMock()

        def mock_find(screen, tpl, threshold=0.74, zona=None):
            if "wall_row_item" in tpl:
                return (1600, 420)
            return None

        mock_vision.find_template.side_effect = mock_find
        upgrader = WallUpgrader(mock_device, mock_vision)

        found = upgrader._find_and_tap_wall_in_list()
        self.assertTrue(found, "Fallback legacy debe encontrar el muro")
        mock_device.tap.assert_called_once_with(1600, 420)

    def test_find_and_tap_swipes_when_not_initially_visible(self):
        """Verifica que si el muro no está en pantalla inicialmente, desliza la lista."""
        mock_device = MagicMock()
        mock_device.width = 1920
        mock_device.height = 1080
        mock_device.screenshot.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_vision = MagicMock()

        calls = {"count": 0}

        def mock_find(screen, tpl, threshold=0.74, zona=None):
            calls["count"] += 1
            if calls["count"] >= 3:
                return (1600, 500)
            return None

        mock_vision.find_template.side_effect = mock_find
        upgrader = WallUpgrader(mock_device, mock_vision)

        found = upgrader._find_and_tap_wall_in_list()
        self.assertTrue(found)
        self.assertTrue(mock_device.swipe.called, "Debe haber deslizado la lista")


class TestCostEvaluation(unittest.TestCase):
    """Pruebas de la evaluación dinámica de costeabilidad en pantalla."""

    def setUp(self):
        mock_device = MagicMock()
        mock_vision = MagicMock()
        self.upgrader = WallUpgrader(mock_device, mock_vision)

    def test_is_cost_affordable_white_text(self):
        """Genera imagen con texto blanco brillante (costeable) -> debe retornar True."""
        screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bx, by = 1030, 842
        y1, y2 = by - 95, by - 65
        x1, x2 = bx - 60, bx + 60
        screen[y1 + 5:y2 - 5, x1 + 10:x2 - 10] = [220, 220, 220]

        is_affordable = self.upgrader._is_cost_affordable(screen, (bx, by))
        self.assertTrue(is_affordable, "El texto blanco debe indicar costeable")

    def test_is_cost_affordable_red_text(self):
        """Genera imagen con texto rojo (costo insuficiente) -> debe retornar False."""
        screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bx, by = 1030, 842
        y1, y2 = by - 95, by - 65
        x1, x2 = bx - 60, bx + 60
        screen[y1 + 5:y2 - 5, x1 + 10:x2 - 10] = [20, 20, 240]

        is_affordable = self.upgrader._is_cost_affordable(screen, (bx, by))
        self.assertFalse(is_affordable, "El texto rojo debe indicar insuficiente")


class TestUpgradeWallFlow(unittest.TestCase):
    """Pruebas del flujo completo para distintos niveles sin regresiones."""

    def setUp(self):
        self.mock_device = MagicMock()
        self.mock_device.width = 1920
        self.mock_device.height = 1080
        self.mock_device.screenshot.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.mock_vision = MagicMock()
        self.upgrader = WallUpgrader(self.mock_device, self.mock_vision)

        self._patch_sleep = patch("time.sleep", return_value=None)
        self._patch_sleep.start()
        self._patch_delay = patch("core.wall_upgrader.HumanBehavior.reaction_delay", return_value=None)
        self._patch_delay.start()

    def tearDown(self):
        self._patch_sleep.stop()
        self._patch_delay.stop()

    def test_upgrade_wall_prevents_elixir_on_level_1_to_4(self):
        """Comprueba que intentar mejorar niveles 1-4 con Elixir es rechazado preventivamente."""
        # Intento de mejorar nivel 1 a 2 con Elixir
        result = self.upgrader.upgrade_wall("elixir", source_level=1, target_level=2)
        self.assertFalse(result, "Nivel 1->2 no debe permitir Elixir")
        self.assertFalse(self.mock_device.tap.called, "No debe ejecutar ninguna pulsación")

        # Intento de mejorar nivel 3 a 4 con Elixir
        result3 = self.upgrader.upgrade_wall("elixir", source_level=3, target_level=4)
        self.assertFalse(result3, "Nivel 3->4 no debe permitir Elixir")

    def test_upgrade_wall_allows_elixir_on_level_5_plus(self):
        """Comprueba que intentar mejorar nivel 5 a 6 con Elixir es permitido si es costeable."""
        self.mock_vision.find_template.side_effect = lambda screen, tpl, threshold=None, zona=None: (
            (1850, 950) if "build_btn" in tpl
            else (1800, 100) if "build_filter_btn" in tpl
            else (1600, 400) if "wall_row" in tpl
            else (835, 848) if "add_wall_1" in tpl
            else (418, 850) if "remove_wall_1" in tpl
            else (960, 540) if "dialog_confirm" in tpl
            else None
        )

        self.mock_vision.wait_for_template.side_effect = lambda fn, tpl, timeout=None, threshold=None, zona=None, **kw: (
            (1500, 380) if "filter_elixir_off" in tpl
            else (1500, 450) if "filter_perm_off" in tpl
            else (960, 850) if "upgrade_more" in tpl
            else None
        )

        with patch.object(self.upgrader, "_is_cost_affordable", return_value=True), \
             patch.object(self.upgrader, "_find_upgrade_buttons_left_to_right", return_value=[(1030, 842), (1240, 840)]):
            success = self.upgrader.upgrade_wall("elixir", max_walls=2, source_level=5, target_level=6)
            self.assertTrue(success, "Nivel 5->6 debe permitir y ejecutar mejora con Elixir")

    def test_upgrade_wall_18_to_19_regression_free(self):
        """
        Verifica que para 18->19 el flujo se ejecuta sin regresiones:
        Build -> Filtro combinado -> Wall -> Upgrade More -> Multi-selección -> Upgrade -> Confirmar.
        """
        self.mock_vision.find_template.side_effect = lambda screen, tpl, threshold=None, zona=None: (
            (1850, 950) if "build_btn" in tpl
            else (1800, 100) if "build_filter_btn" in tpl
            else (1600, 400) if "wall_row" in tpl
            else (835, 848) if "add_wall_1" in tpl
            else (418, 850) if "remove_wall_1" in tpl
            else (960, 540) if "dialog_confirm" in tpl
            else None
        )

        self.mock_vision.wait_for_template.side_effect = lambda fn, tpl, timeout=None, threshold=None, zona=None, **kw: (
            (1500, 380) if "filter_elixir_off" in tpl
            else (1500, 450) if "filter_perm_off" in tpl
            else (960, 850) if "upgrade_more" in tpl
            else None
        )

        with patch.object(self.upgrader, "_is_cost_affordable", return_value=True), \
             patch.object(self.upgrader, "_find_upgrade_buttons_left_to_right", return_value=[(1030, 842), (1240, 840)]):
            success = self.upgrader.upgrade_wall("elixir", max_walls=2, source_level=18, target_level=19)
            self.assertTrue(success, "La mejora 18->19 debe completarse exitosamente")


class TestLedgerDynamicIntegration(unittest.TestCase):
    """Pruebas de la integración dinámica de Ledger con la estrategia de muros."""

    def test_ledger_dynamic_threshold_for_level_1(self):
        """Verifica que con nivel 1 (1->2) prohíbe Elixir y selecciona Oro."""
        ledger = Ledger()
        ledger.rss_current = {"gold": 50_000, "elixir": 10_000_000}

        with patch("core.ledger.SETTINGS", {
            "economy": {
                "enable_wall_upgrade": True,
                "wall_level_source": 1,
                "wall_batch_size": 2,
                "min_rss_to_spend": 21_000_000,
            }
        }):
            # Aunque Elixir tenga 10M, nivel 1->2 NO admite Elixir -> debe usar Oro
            res = ledger.get_wall_upgrade_resource()
            self.assertEqual(res, "gold", "Nivel 1 debe seleccionar Oro porque Elixir no es admitido")

    def test_ledger_dynamic_threshold_for_level_5(self):
        """Verifica que con nivel 5 (5->6) compara y prioriza el recurso más alto que supere 60k."""
        ledger = Ledger()
        # Caso 1: Elixir más alto que Oro -> prioriza Elixir
        ledger.rss_current = {"gold": 50_000, "elixir": 100_000}
        with patch("core.ledger.SETTINGS", {
            "economy": {
                "enable_wall_upgrade": True,
                "wall_level_source": 5,
                "wall_batch_size": 2,
                "min_rss_to_spend": 21_000_000,
            }
        }):
            res = ledger.get_wall_upgrade_resource()
            self.assertEqual(res, "elixir", "Nivel 5 debe seleccionar Elixir al ser el recurso más alto")

            # Caso 2: Oro más alto que Elixir -> prioriza Oro
            ledger.rss_current = {"gold": 120_000, "elixir": 70_000}
            res_gold = ledger.get_wall_upgrade_resource()
            self.assertEqual(res_gold, "gold", "Nivel 5 debe seleccionar Oro al ser el recurso más alto")

    def test_ledger_dynamic_threshold_for_level_10(self):
        """Verifica que con nivel 10 (10->11) compara y prioriza el recurso más alto que supere 1M."""
        ledger = Ledger()
        # Caso 1: Elixir más alto que Oro -> prioriza Elixir
        ledger.rss_current = {"gold": 1_200_000, "elixir": 3_000_000}
        with patch("core.ledger.SETTINGS", {
            "economy": {
                "enable_wall_upgrade": True,
                "wall_level_source": 10,
                "wall_batch_size": 2,
                "min_rss_to_spend": 21_000_000,
            }
        }):
            res = ledger.get_wall_upgrade_resource()
            self.assertEqual(res, "elixir", "Nivel 10 debe seleccionar Elixir cuando Elixir es mayor")

            # Caso 2: Oro más alto que Elixir -> prioriza Oro
            ledger.rss_current = {"gold": 5_000_000, "elixir": 1_500_000}
            res_gold = ledger.get_wall_upgrade_resource()
            self.assertEqual(res_gold, "gold", "Nivel 10 debe seleccionar Oro cuando Oro es mayor")

    def test_ledger_legacy_18_to_19_threshold(self):
        """Verifica que sin nivel especificado (None) mantiene el umbral estándar de 21M."""
        ledger = Ledger()
        ledger.rss_current = {"gold": 22_000_000, "elixir": 15_000_000}

        with patch("core.ledger.SETTINGS", {
            "economy": {
                "enable_wall_upgrade": True,
                "wall_level_source": None,
                "min_rss_to_spend": 21_000_000,
            }
        }):
            # Elixir tiene 15M (< 21M), Oro tiene 22M (>= 21M) -> Oro
            res = ledger.get_wall_upgrade_resource()
            self.assertEqual(res, "gold")


if __name__ == "__main__":
    unittest.main()


class TestComparisonAndSortingSystem(unittest.TestCase):
    """
    Suite de pruebas para el mini sistema de comparación y ordenamiento:
    1. Comparación en WallUpgradeStrategy (niveles 1-4 solo devuelven ['gold']).
    2. Comparación en WallUpgradeStrategy (niveles >= 5 ordenan según recurso más alto).
    3. Comparación en WallUpgrader delegando a la estrategia.
    4. Secuencia de mejora en WallUpgrader ejecutando en orden de prioridad.
    """

    def setUp(self):
        self.strategy = WallUpgradeStrategy()

    def test_strategy_compare_and_order_levels_1_to_4_only_gold(self):
        """Niveles destino 2 a 4 solo permiten Oro; Elixir debe ser excluido aun con saldo mayor."""
        for src in (1, 2, 3):
            order = self.strategy.compare_and_order_resources(
                available_gold=100_000,
                available_elixir=10_000_000,
                source_level=src,
            )
            self.assertEqual(order, ["gold"], f"Transición {src}->{src+1} solo debe admitir Oro")

    def test_strategy_compare_and_order_level_5_plus_gold_higher(self):
        """Nivel >= 5 con Oro > Elixir debe ordenar ['gold', 'elixir']."""
        order = self.strategy.compare_and_order_resources(
            available_gold=15_000_000,
            available_elixir=8_000_000,
            source_level=10,
        )
        self.assertEqual(order, ["gold", "elixir"])

    def test_strategy_compare_and_order_level_5_plus_elixir_higher(self):
        """Nivel >= 5 con Elixir > Oro debe ordenar ['elixir', 'gold']."""
        order = self.strategy.compare_and_order_resources(
            available_gold=5_000_000,
            available_elixir=18_000_000,
            source_level=18,
        )
        self.assertEqual(order, ["elixir", "gold"])

    def test_wall_upgrader_delegates_comparison(self):
        """WallUpgrader delega la comparación correctamente a la estrategia."""
        mock_device = MagicMock()
        mock_vision = MagicMock()
        upgrader = WallUpgrader(device=mock_device, vision=mock_vision, source_level=18)
        order = upgrader.compare_and_order_resources(available_gold=20_000_000, available_elixir=10_000_000)
        self.assertEqual(order, ["gold", "elixir"])

        order_e = upgrader.compare_and_order_resources(available_gold=5_000_000, available_elixir=15_000_000)
        self.assertEqual(order_e, ["elixir", "gold"])

    def test_wall_upgrader_sequence_without_resources_arg(self):
        """upgrade_walls_sequence sin resources explícito ordena automáticamente y ejecuta."""
        mock_device = MagicMock()
        mock_vision = MagicMock()
        upgrader = WallUpgrader(device=mock_device, vision=mock_vision, source_level=18)
        with patch.object(upgrader, "upgrade_wall", return_value=True) as mock_upgrade:
            results = upgrader.upgrade_walls_sequence(
                available_gold=22_000_000,
                available_elixir=10_000_000,
            )
            self.assertEqual(mock_upgrade.call_count, 2)
            first_call_res = mock_upgrade.call_args_list[0][1]["resource_type"]
            second_call_res = mock_upgrade.call_args_list[1][1]["resource_type"]
            self.assertEqual(first_call_res, "gold", "Oro debe ejecutarse primero por ser mayor")
            self.assertEqual(second_call_res, "elixir")
