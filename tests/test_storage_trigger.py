"""
tests/test_storage_trigger.py
Suite de validación unitaria para el sistema de lectura de tooltips y trigger de muros por capacidad de almacenes:
1. Lectura exitosa simulada de tooltip (Total Capacity, Production per hour, In Treasury) y cierre de popup.
2. Fallback oficial a la tabla de storages cuando el tooltip falla (TH18 con almacenes TH17 = 24M Oro/Elixir y 420k Oscuro).
3. Cálculo correcto e independiente de gold_pct, elixir_pct, dark_pct.
4. Confirmación estricta de que Dark Elixir NUNCA dispara mejoras de muro aunque dark_pct >= 0.90.
5. Comportamiento ante umbrales < 0.90 (no acción).
6. Comportamiento cuando solo Oro >= 0.90 (prioriza Oro).
7. Comportamiento cuando solo Elixir >= 0.90 (prioriza Elixir).
8. Comportamiento cuando ambos (Oro y Elixir) >= 0.90 (doble tanda secuencial: Elixir + Oro).
9. Verificación de que "In Treasury" se almacena como dato informativo y no altera la capacidad de almacenamiento normal.
10. Verificación de no acción (solo log) cuando no hay muros pendientes de subir.
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from core.storage_cost_table import (
    GOLD_ELIXIR_STORAGE_LEVELS,
    DARK_ELIXIR_STORAGE_LEVELS,
    GOLD_ELIXIR_STORAGE_COUNTS_BY_TH,
    DARK_ELIXIR_STORAGE_COUNTS_BY_TH,
    FALLBACK_TH18_CAPACITIES,
    get_storage_capacity,
    get_storage_count_for_th,
    calculate_estimated_storage_capacity,
    get_fallback_storage_capacities,
)
from core.ledger import Ledger
from core.config import SETTINGS, PROJECT_ROOT
from vision.resource_reader import ResourceReader


class TestStorageCostTable(unittest.TestCase):
    """Valida la tabla oficial de datos de almacenes y funciones de fallback."""

    def test_th17_storages_th18_fallback_exact_values(self):
        """Verifica que el fallback canónico sea TH18 con almacenes TH17: 24M Oro/Elixir y 420k Oscuro."""
        fallback = get_fallback_storage_capacities(th_level=18, use_th17_storages=True)
        self.assertEqual(fallback["gold"], 24_000_000)
        self.assertEqual(fallback["elixir"], 24_000_000)
        self.assertEqual(fallback["dark_elixir"], 420_000)

        # Validación directa de la constante canónica
        self.assertEqual(FALLBACK_TH18_CAPACITIES["gold"], 24_000_000)
        self.assertEqual(FALLBACK_TH18_CAPACITIES["elixir"], 24_000_000)
        self.assertEqual(FALLBACK_TH18_CAPACITIES["dark_elixir"], 420_000)

    def test_storage_counts_by_th(self):
        """Valida que la cantidad de almacenes por TH sea la oficial."""
        self.assertEqual(get_storage_count_for_th("gold", 1), 1)
        self.assertEqual(get_storage_count_for_th("gold", 8), 3)
        self.assertEqual(get_storage_count_for_th("gold", 17), 4)
        self.assertEqual(get_storage_count_for_th("gold", 18), 4)

        self.assertEqual(get_storage_count_for_th("dark_elixir", 6), 0)
        self.assertEqual(get_storage_count_for_th("dark_elixir", 7), 1)
        self.assertEqual(get_storage_count_for_th("dark_elixir", 18), 1)

    def test_storage_capacity_levels(self):
        """Valida capacidades clave de almacenes."""
        # Nivel 18 de Gold/Elixir Storage = 6,000,000 (disponible en TH17)
        self.assertEqual(get_storage_capacity("gold", 18), 6_000_000)
        self.assertEqual(get_storage_capacity("elixir", 18), 6_000_000)
        # Nivel 19 de Gold/Elixir Storage = 6,500,000 (disponible en TH18)
        self.assertEqual(get_storage_capacity("gold", 19), 6_500_000)

        # Nivel 12 de Dark Elixir Storage = 420,000 (disponible en TH17)
        self.assertEqual(get_storage_capacity("dark_elixir", 12), 420_000)
        # Nivel 13 de Dark Elixir Storage = 450,000 (disponible en TH18)
        self.assertEqual(get_storage_capacity("dark_elixir", 13), 450_000)


class TestTooltipParsingAndReading(unittest.TestCase):
    """Valida el parsing de texto y lectura interactiva de tooltips."""

    def setUp(self):
        self.mock_vision = MagicMock()
        self.reader = ResourceReader(self.mock_vision)

    def test_parse_tooltip_text_english_labels(self):
        """Verifica la extracción precisa de los 3 valores desde texto en inglés."""
        sample_text = """
        Total Capacity: 24,000,000
        Production per hour: 84,000
        In Treasury: 6,000,000
        """
        parsed = self.reader.parse_tooltip_text(sample_text)
        self.assertEqual(parsed["total_capacity"], 24_000_000)
        self.assertEqual(parsed["production_per_hour"], 84_000)
        self.assertEqual(parsed["in_treasury"], 6_000_000)

    def test_parse_tooltip_text_storage_capacity_variant(self):
        """Verifica la compatibilidad con 'Storage Capacity' y separadores alternativos."""
        sample_text = """
        Storage Capacity - 420 000
        Production per hour: 0
        In Treasury: 0
        """
        parsed = self.reader.parse_tooltip_text(sample_text)
        self.assertEqual(parsed["total_capacity"], 420_000)
        self.assertEqual(parsed["production_per_hour"], 0)
        self.assertEqual(parsed["in_treasury"], 0)

    def test_parse_tooltip_text_empty_or_corrupt(self):
        """Verifica que textos corruptos o vacíos retornen None sin excepciones."""
        parsed = self.reader.parse_tooltip_text("")
        self.assertIsNone(parsed["total_capacity"])
        self.assertIsNone(parsed["production_per_hour"])
        self.assertIsNone(parsed["in_treasury"])

    def test_parse_tooltip_text_max_variant(self):
        """Verifica la compatibilidad con 'Max: 21 000 000' observado en el juego real."""
        sample_text = """
        Max: 21 000 000
        Production per hour: 68 600
        In treasury: 1 105 000
        """
        parsed = self.reader.parse_tooltip_text(sample_text)
        self.assertEqual(parsed["total_capacity"], 21_000_000)
        self.assertEqual(parsed["production_per_hour"], 68_600)
        self.assertEqual(parsed["in_treasury"], 1_105_000)

    def test_extract_tooltip_values_from_real_fixture(self):
        """Verifica que la visión por computador extraiga exactamente los 3 valores desde la captura real."""
        import cv2
        import os
        from core.vision import Vision
        fixture_path = os.path.join(PROJECT_ROOT, "tests", "fixtures", "sample_resource_tooltip.png")
        if not os.path.exists(fixture_path):
            self.skipTest("Fixture no encontrada en tests/fixtures")

        img = cv2.imread(fixture_path)
        self.assertIsNotNone(img)

        # Usar ResourceReader con Vision real para cargar el banco de plantillas de dígitos
        real_reader = ResourceReader(Vision())

        # 1. Extracción directa sobre el recorte del tooltip
        extracted = real_reader.extract_tooltip_values_from_image(img)
        self.assertEqual(extracted["total_capacity"], 21_000_000)
        self.assertEqual(extracted["production_per_hour"], 68_600)
        self.assertEqual(extracted["in_treasury"], 1_105_000)

        # 2. Extracción montada sobre un canvas completo 2400x1080
        full = np.zeros((1080, 2400, 3), dtype=np.uint8)
        full[0:img.shape[0], 1741:1741+img.shape[1]] = img
        extracted_full = real_reader.extract_tooltip_values_from_image(full)
        self.assertEqual(extracted_full["total_capacity"], 21_000_000)
        self.assertEqual(extracted_full["production_per_hour"], 68_600)
        self.assertEqual(extracted_full["in_treasury"], 1_105_000)

    def test_read_all_resource_tooltips_flow_and_clean_taps(self):
        """Verifica que read_all_resource_tooltips lea secuencialmente los 3 recursos."""
        mock_device = MagicMock()
        mock_device.width = 2400
        mock_device.height = 1080

        with patch.object(self.reader, "read_resource_tooltip", return_value={"total_capacity": 21_000_000}) as mock_read:
            results = self.reader.read_all_resource_tooltips(mock_device)

        self.assertEqual(len(results), 3)
        self.assertIn("gold", results)
        self.assertIn("elixir", results)
        self.assertIn("dark_elixir", results)
        self.assertEqual(mock_read.call_count, 3)

    def test_read_resource_tooltip_flow_and_close(self):
        """Verifica que se toque el icono para abrir, se capture screenshot y se vuelva a tocar para cerrar (toggle)."""
        mock_device = MagicMock()
        mock_device.width = 2400
        mock_device.height = 1080
        mock_device.screenshot.return_value = np.zeros((1080, 2400, 3), dtype=np.uint8)

        sample_parsed = {
            "total_capacity": 24_000_000,
            "production_per_hour": 84_000,
            "in_treasury": 5_000_000,
        }

        with patch.object(self.reader, "extract_tooltip_values_from_image", return_value=sample_parsed):
            result = self.reader.read_resource_tooltip(mock_device, "gold", delay_sec=0.01)

        self.assertIsNotNone(result)
        self.assertEqual(result["total_capacity"], 24_000_000)
        self.assertEqual(result["production_per_hour"], 84_000)
        self.assertEqual(result["in_treasury"], 5_000_000)

        # Verificar que device.tap fue llamado al menos 2 veces: abrir icono y cerrar tocando de nuevo
        self.assertEqual(mock_device.tap.call_count, 2)
        # Ambos toques deben ser a la coordenada de la barra de oro (2112, 54) para abrir y cerrar
        first_call = mock_device.tap.call_args_list[0]
        second_call = mock_device.tap.call_args_list[1]
        self.assertEqual(first_call[0], (2112, 54))
        self.assertEqual(second_call[0], (2112, 54))


class TestLedgerCapacityAndTrigger(unittest.TestCase):
    """Valida el almacenamiento en Ledger, porcentajes independientes y reglas de trigger."""

    def setUp(self):
        self.ledger = Ledger()
        # Asegurar valores de capacidad iniciales limpios
        self.ledger.set_fallback_capacities(th_level=18)

    def test_record_tooltip_data_rejects_bogus_small_capacities(self):
        """Verifica que capacidades absurdas (<5M en Oro/Elixir) sean rechazadas y se use el fallback canónico."""
        bogus_data = {
            "gold": {"total_capacity": 1_769_363, "production_per_hour": 0, "in_treasury": 0},
            "elixir": {"total_capacity": 4_688_127, "production_per_hour": 0, "in_treasury": 0},
            "dark_elixir": {"total_capacity": 320_000, "production_per_hour": 720, "in_treasury": 5525},
        }
        self.ledger.record_tooltip_data(bogus_data)
        # Oro y Elixir deben haber sido rechazados y puestos en fallback (24,000,000)
        self.assertEqual(self.ledger.max_capacities["gold"], 24_000_000)
        self.assertEqual(self.ledger.max_capacities["elixir"], 24_000_000)
        # Elixir Oscuro es válido (320,000) y debe aceptarse
        self.assertEqual(self.ledger.max_capacities["dark_elixir"], 320_000)

        # Con 2,862,913 de Oro y 24,000,000 de capacidad, el porcentaje es ~11.9% (<90%) -> SIN TRIGGER
        self.ledger.rss_current = {"gold": 2_862_913, "elixir": 4_680_070, "dark_elixir": 320_000}
        self.assertEqual(self.ledger.get_storage_trigger_resources(), [])

    def test_independent_percentage_calculation(self):
        """Verifica que gold_pct, elixir_pct y dark_pct se calculen de forma independiente."""
        self.ledger.max_capacities = {
            "gold": 24_000_000,
            "elixir": 24_000_000,
            "dark_elixir": 420_000,
        }
        self.ledger.rss_current = {
            "gold": 12_000_000,     # 50%
            "elixir": 21_600_000,   # 90%
            "dark_elixir": 378_000, # 90%
        }

        pcts = self.ledger.calculate_resource_percentages()
        self.assertAlmostEqual(pcts["gold_pct"], 0.50, places=3)
        self.assertAlmostEqual(pcts["elixir_pct"], 0.90, places=3)
        self.assertAlmostEqual(pcts["dark_pct"], 0.90, places=3)

    def test_in_treasury_does_not_affect_storage_capacity(self):
        """Verifica que In Treasury sea informativo y no altere las capacidades ni porcentajes."""
        self.ledger.max_capacities = {"gold": 24_000_000, "elixir": 24_000_000, "dark_elixir": 420_000}
        self.ledger.in_treasury = {"gold": 6_000_000, "elixir": 6_000_000, "dark_elixir": 40_000}
        self.ledger.rss_current = {"gold": 12_000_000, "elixir": 12_000_000, "dark_elixir": 210_000}

        pcts = self.ledger.calculate_resource_percentages()
        # Debe mantenerse exactamente al 50%, no alterado por los 6M de tesorería
        self.assertAlmostEqual(pcts["gold_pct"], 0.50, places=3)
        self.assertAlmostEqual(pcts["elixir_pct"], 0.50, places=3)

    def test_dark_elixir_never_triggers_wall_upgrade(self):
        """
        REGLA CRÍTICA: Dark Elixir NUNCA debe disparar mejoras de muro aunque dark_pct >= 0.90.
        """
        self.ledger.max_capacities = {"gold": 24_000_000, "elixir": 24_000_000, "dark_elixir": 420_000}
        # Solo Oscuro al 100%, Oro y Elixir bajos (<90%)
        self.ledger.rss_current = {"gold": 5_000_000, "elixir": 5_000_000, "dark_elixir": 420_000}

        self.assertGreaterEqual(self.ledger.dark_pct, 0.90)
        triggers = self.ledger.get_storage_trigger_resources()
        self.assertEqual(triggers, [], "Dark Elixir NUNCA debe disparar mejoras de muro")
        self.assertNotIn("dark_elixir", triggers)

    def test_threshold_not_reached_no_action(self):
        """Verifica que con porcentajes < 90% no se active el trigger."""
        self.ledger.max_capacities = {"gold": 24_000_000, "elixir": 24_000_000, "dark_elixir": 420_000}
        self.ledger.rss_current = {"gold": 21_000_000, "elixir": 21_000_000, "dark_elixir": 300_000}
        # 21M / 24M = 87.5% (< 90%)
        self.assertLess(self.ledger.gold_pct, 0.90)
        self.assertLess(self.ledger.elixir_pct, 0.90)
        triggers = self.ledger.get_storage_trigger_resources()
        self.assertEqual(triggers, [])

    def test_only_gold_triggers_gold(self):
        """Verifica que si solo Oro >= 90%, devuelva ['gold']."""
        self.ledger.max_capacities = {"gold": 24_000_000, "elixir": 24_000_000, "dark_elixir": 420_000}
        self.ledger.rss_current = {"gold": 22_000_000, "elixir": 10_000_000, "dark_elixir": 100_000}
        # Oro: 22M / 24M = 91.7% (>= 90%)
        triggers = self.ledger.get_storage_trigger_resources()
        self.assertEqual(triggers, ["gold"])

    def test_only_elixir_triggers_elixir(self):
        """Verifica que si solo Elixir >= 90%, devuelva ['elixir']."""
        self.ledger.max_capacities = {"gold": 24_000_000, "elixir": 24_000_000, "dark_elixir": 420_000}
        self.ledger.rss_current = {"gold": 10_000_000, "elixir": 22_500_000, "dark_elixir": 100_000}
        # Elixir: 22.5M / 24M = 93.75% (>= 90%)
        triggers = self.ledger.get_storage_trigger_resources()
        self.assertEqual(triggers, ["elixir"])

    def test_both_gold_and_elixir_trigger_sequential_double_batch(self):
        """
        Verifica que si ambos superan el 90%, devuelva ['elixir', 'gold']
        cuando Elixir es mayor que Oro.
        """
        self.ledger.max_capacities = {"gold": 24_000_000, "elixir": 24_000_000, "dark_elixir": 420_000}
        self.ledger.rss_current = {"gold": 22_000_000, "elixir": 23_000_000, "dark_elixir": 400_000}

        self.assertGreaterEqual(self.ledger.gold_pct, 0.90)
        self.assertGreaterEqual(self.ledger.elixir_pct, 0.90)

        triggers = self.ledger.get_storage_trigger_resources()
        self.assertEqual(triggers, ["elixir", "gold"])

    def test_both_gold_and_elixir_trigger_prioritizes_gold_when_gold_higher(self):
        """
        Verifica que si ambos superan el 90% y Oro > Elixir, el mini sistema de comparación
        y ordenamiento prioriza Oro primero: ['gold', 'elixir'].
        """
        self.ledger.max_capacities = {"gold": 24_000_000, "elixir": 24_000_000, "dark_elixir": 420_000}
        self.ledger.rss_current = {"gold": 23_800_000, "elixir": 22_000_000, "dark_elixir": 400_000}

        self.assertGreaterEqual(self.ledger.gold_pct, 0.90)
        self.assertGreaterEqual(self.ledger.elixir_pct, 0.90)
        self.assertGreater(self.ledger.gold_pct, self.ledger.elixir_pct)

        triggers = self.ledger.get_storage_trigger_resources()
        self.assertEqual(triggers, ["gold", "elixir"], "Oro debe colocarse primero al tener mayor llenado")

    def test_compare_and_order_resources_direct_method(self):
        """Verifica compare_and_order_resources tanto para Oro mayor como Elixir mayor."""
        self.ledger.max_capacities = {"gold": 24_000_000, "elixir": 24_000_000, "dark_elixir": 420_000}
        
        # Oro mayor
        self.ledger.rss_current = {"gold": 18_000_000, "elixir": 12_000_000}
        self.assertEqual(self.ledger.compare_and_order_resources(), ["gold", "elixir"])

        # Elixir mayor
        self.ledger.rss_current = {"gold": 10_000_000, "elixir": 20_000_000}
        self.assertEqual(self.ledger.compare_and_order_resources(), ["elixir", "gold"])

        # Nivel 1 a 4 solo retorna Oro sin importar el saldo de Elixir
        self.assertEqual(self.ledger.compare_and_order_resources(source_level=3), ["gold"])

    def test_storage_trigger_levels_1_to_4_disables_elixir(self):
        """Si el nivel origen es <= 4, Elixir nunca dispara aunque esté al 99%."""
        self.ledger.max_capacities = {"gold": 24_000_000, "elixir": 24_000_000, "dark_elixir": 420_000}
        self.ledger.rss_current = {"gold": 5_000_000, "elixir": 23_500_000, "dark_elixir": 100_000}
        triggers = self.ledger.get_storage_trigger_resources(source_level=3)
        self.assertEqual(triggers, [], "Niveles <= 4 no admiten mejoras con Elixir")

    def test_record_tooltip_data_with_fallback_on_missing_resource(self):
        """Verifica que si falla la lectura de un recurso, se aplique el fallback transparente."""
        incomplete_tooltip_data = {
            "gold": {"total_capacity": 22_000_000, "production_per_hour": 75_000, "in_treasury": 4_000_000},
            "elixir": None,  # Falló lectura
            "dark_elixir": {"total_capacity": None, "production_per_hour": None, "in_treasury": None},  # Falló OCR
        }

        self.ledger.record_tooltip_data(incomplete_tooltip_data)
        self.assertEqual(self.ledger.max_capacities["gold"], 22_000_000)
        # Elixir y Oscuro deben haber recibido los valores fallback de TH18
        self.assertEqual(self.ledger.max_capacities["elixir"], 24_000_000)
        self.assertEqual(self.ledger.max_capacities["dark_elixir"], 420_000)
        self.assertTrue(self.ledger.tooltip_cached)

    def test_custom_threshold_configuration(self):
        """Verifica que el umbral sea configurable dinámicamente desde settings."""
        self.ledger.max_capacities = {"gold": 24_000_000, "elixir": 24_000_000, "dark_elixir": 420_000}
        self.ledger.rss_current = {"gold": 20_000_000, "elixir": 10_000_000, "dark_elixir": 100_000}
        # 20M / 24M = 83.33%

        with patch.dict(SETTINGS["economy"], {"storage_upgrade_trigger_threshold": 0.80}):
            triggers = self.ledger.get_storage_trigger_resources()
            self.assertEqual(triggers, ["gold"])

        with patch.dict(SETTINGS["economy"], {"storage_upgrade_trigger_threshold": 0.85}):
            triggers = self.ledger.get_storage_trigger_resources()
            self.assertEqual(triggers, [])

    def test_zero_capacity_edge_case_no_exception(self):
        """Verifica que capacidades en cero o no inicializadas no provoquen ZeroDivisionError."""
        self.ledger.max_capacities = {"gold": 0, "elixir": 0, "dark_elixir": 0}
        self.ledger.rss_current = {"gold": 5_000, "elixir": 5_000, "dark_elixir": 100}
        pcts = self.ledger.calculate_resource_percentages()
        self.assertEqual(pcts["gold_pct"], 0.0)
        self.assertEqual(pcts["elixir_pct"], 0.0)
        self.assertEqual(pcts["dark_pct"], 0.0)


class TestStateMachineTooltipIntegration(unittest.TestCase):
    """Valida la integración de ensure_tooltip_cached en ClashStateMachine."""

    def test_ensure_tooltip_cached_called_only_once(self):
        """Verifica que la lectura de tooltips solo se ejecute una vez por sesión y no en cada tick."""
        from core.state_machine import ClashStateMachine
        mock_dev = MagicMock()
        mock_vis = MagicMock()
        sm = ClashStateMachine(mock_dev, mock_vis)

        mock_data = {
            "gold": {"total_capacity": 24_000_000, "production_per_hour": 80_000, "in_treasury": 5_000_000},
            "elixir": {"total_capacity": 24_000_000, "production_per_hour": 80_000, "in_treasury": 5_000_000},
            "dark_elixir": {"total_capacity": 420_000, "production_per_hour": 5_000, "in_treasury": 30_000},
        }

        with patch.object(sm.reader, "read_all_resource_tooltips", return_value=mock_data) as mock_read:
            sm.ensure_tooltip_cached()
            self.assertEqual(mock_read.call_count, 1)
            self.assertTrue(sm.ledger.tooltip_cached)
            self.assertEqual(sm.ledger.max_capacities["gold"], 24_000_000)

            # Segunda llamada: no debe llamar a read_all_resource_tooltips porque ya está en caché
            sm.ensure_tooltip_cached()
            self.assertEqual(mock_read.call_count, 1)


class TestWallUpgraderSequenceAndNoWalls(unittest.TestCase):
    """Valida upgrade_walls_sequence y comportamiento ante ausencia de muros pendientes."""

    def test_upgrade_walls_sequence_double_run(self):
        """Verifica la ejecución secuencial doble de elixir y luego oro."""
        from core.wall_upgrader import WallUpgrader
        mock_dev = MagicMock()
        mock_vis = MagicMock()
        upgrader = WallUpgrader(mock_dev, mock_vis)

        with patch.object(upgrader, "upgrade_wall", side_effect=[True, True]) as mock_upg:
            results = upgrader.upgrade_walls_sequence(["elixir", "gold"])

        self.assertEqual(results, {"elixir": True, "gold": True})
        self.assertEqual(mock_upg.call_count, 2)
        mock_upg.assert_any_call(resource_type="elixir", max_walls=5, source_level=None, target_level=None)
        mock_upg.assert_any_call(resource_type="gold", max_walls=5, source_level=None, target_level=None)

    def test_no_walls_pending_logs_and_returns_false_safely(self):
        """Verifica que si no hay muros pendientes, se cierre el menú y retorne False sin error."""
        from core.wall_upgrader import WallUpgrader
        mock_dev = MagicMock()
        mock_dev.width = 2400
        mock_dev.height = 1080
        mock_vis = MagicMock()
        # Build button encontrado
        mock_vis.find_template.side_effect = [(2300, 950), None, None]
        upgrader = WallUpgrader(mock_dev, mock_vis)

        # Simular que _find_and_tap_wall_in_list retorna False (no hay muros)
        with patch.object(upgrader, "_apply_filter"):
            with patch.object(upgrader, "_find_and_tap_wall_in_list", return_value=False):
                success = upgrader.upgrade_wall("gold")

        self.assertFalse(success)
        # Debe haber tocado fuera para cerrar el menú
        mock_dev.tap.assert_called_with(int(2400 * 0.25), int(1080 * 0.50))


if __name__ == "__main__":
    unittest.main()
