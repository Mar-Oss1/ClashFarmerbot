"""
tests/test_core_stability.py
Suite de verificación de estabilidad del core y corrección de bugs críticos:
1. Rendimiento y eliminación del congelamiento de 5s por tick en ProcessManager.
2. Unicidad de RotatingFileHandler en logger centralizado (Windows safety).
3. Respeto de SEARCH_ONLY en ClashStateMachine sin ejecutar ataques.
4. Regla de negocio oficial can_afford() (Oro O Elixir para muros nivel 5+).
"""
import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

from core.device import Device
from core.vision import Vision
from core.process_manager import ProcessManager
from core.logger import get_logger, _ROOT_LOGGER_NAME
from core.state_machine import ClashStateMachine, BotState
from core.wall_cost_table import can_afford
from core import config


class TestProcessManagerPerformance(unittest.TestCase):
    """Verifica que ProcessManager.handle_popup() no congele el bucle en pantallas limpias."""

    def setUp(self):
        self.mock_device = MagicMock(spec=Device)
        self.mock_device.width = 2400
        self.mock_device.height = 1080
        # Pantalla limpia (negra)
        self.clean_screen = np.zeros((1080, 2400, 3), dtype=np.uint8)
        self.mock_device.screenshot.return_value = self.clean_screen

        self.mock_vision = MagicMock(spec=Vision)
        # En pantalla limpia ningún template hace match
        self.mock_vision.find_template.return_value = None
        self.mock_vision.find_any.return_value = None

        self.proc_mgr = ProcessManager(self.mock_device, self.mock_vision)

    def test_handle_popup_clean_screen_executes_under_200ms(self):
        """
        ProcessManager.handle_popup() en pantalla limpia (imagen negra)
        debe terminar en < 0.2 segundos (antes tardaba > 5.0 segundos por el bucle de cofres).
        """
        t_start = time.perf_counter()
        result = self.proc_mgr.handle_popup(self.clean_screen)
        duration = time.perf_counter() - t_start

        self.assertFalse(result, "En pantalla limpia no debe detectar popups ni cofres")
        self.assertLess(
            duration,
            0.20,
            f"handle_popup() tardó {duration:.4f}s; debe terminar en menos de 0.2s en pantalla limpia",
        )
        # No debe solicitar capturas ADB adicionales si ya se pasó screenshot y no hay cofre
        self.assertEqual(self.mock_device.screenshot.call_count, 0)

    def test_handle_chest_reward_sequence_wait_for_chest_parameter(self):
        """Verifica la semántica de wait_for_chest=False vs wait_for_chest=True."""
        # 1. Con wait_for_chest=False en pantalla limpia: retorno inmediato False
        t_start = time.perf_counter()
        res_no_wait = self.proc_mgr.handle_chest_reward_sequence(
            self.clean_screen, wait_for_chest=False
        )
        dur_no_wait = time.perf_counter() - t_start
        self.assertFalse(res_no_wait)
        self.assertLess(dur_no_wait, 0.05)

        # 2. Con wait_for_chest=True y aldea lista: no debe activar fallback y retorna False
        with patch.object(self.proc_mgr, "is_village_ready", return_value=True):
            with patch("time.sleep", return_value=None):
                with patch("time.monotonic", side_effect=[0.0, 1.0, 3.0, 5.5, 6.0]):
                    res_wait = self.proc_mgr.handle_chest_reward_sequence(
                        self.clean_screen, wait_for_chest=True
                    )
                    self.assertFalse(res_wait)
                    # Debe haber intentado tomar capturas periódicas esperando que aparezca el cofre
                    self.assertGreaterEqual(self.mock_device.screenshot.call_count, 1)

    def test_handle_chest_reward_sequence_fallback_when_not_village_ready(self):
        """
        Si wait_for_chest=True y tras 3.0s no aparece cofre ni Continue pero la aldea no está lista,
        debe ejecutar los 4 taps de fallback al centro y esperar Continue.
        """
        with patch.object(self.proc_mgr, "is_village_ready", return_value=False):
            def fake_find_any(img, tpls, **kwargs):
                # Continue solo aparece tras dar los 4 taps al cofre
                if tpls == self.proc_mgr.tpl_continue_btns and self.mock_device.tap.call_count >= 4:
                    return ("continue", 1200, 800)
                return None

            self.mock_vision.find_any.side_effect = fake_find_any

            time_val = 0.0
            def fake_monotonic():
                nonlocal time_val
                t = time_val
                time_val += 1.0
                return t

            with patch("time.sleep", return_value=None):
                with patch("time.monotonic", side_effect=fake_monotonic):
                    res = self.proc_mgr.handle_chest_reward_sequence(
                        self.clean_screen, wait_for_chest=True
                    )
                    self.assertTrue(res, "Debe ejecutar fallback y reportar True")
                    cx = int(self.mock_device.width * 0.50)
                    cy = int(self.mock_device.height * 0.48)
                    center_taps = [
                        call for call in self.mock_device.tap.call_args_list
                        if call[0] == (cx, cy)
                    ]
                    self.assertEqual(len(center_taps), 4, "Debe haber dado 4 taps al centro")
                    self.mock_device.tap.assert_any_call(1200, 800)

    def test_chained_chest_wait_timeout_capped_at_2_5s(self):
        """
        Tras abrir un primer cofre (handled_any=True), la espera para cofres
        encadenados (cofre_idx >= 2) debe ser de máx 2.5s aun si wait_for_chest=True.
        """
        call_num = 0
        def fake_find_any(img, tpls, **kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                return None  # initial cont_match
            if call_num == 2:
                return ("chest", 1200, 500)  # cofre 1 chest_match
            if call_num == 3:
                return ("continue", 1200, 800)  # cofre 1 continue button
            return None  # all subsequent checks return None

        self.mock_vision.find_any.side_effect = fake_find_any

        # Time moves from 0 to 3.0 seconds during cofre 2 wait (exceeding 2.5s timeout)
        time_val = 0.0
        def fake_monotonic():
            nonlocal time_val
            t = time_val
            time_val += 0.8
            return t

        with patch("time.sleep", return_value=None):
            with patch("time.monotonic", side_effect=fake_monotonic):
                result = self.proc_mgr.handle_chest_reward_sequence(
                    self.clean_screen, wait_for_chest=True
                )
                self.assertTrue(result, "Debe reportar True al haber abierto el cofre 1")


class TestLoggerWindowsSafeRotation(unittest.TestCase):
    """Verifica que get_logger use un único logger raíz con exactamente 1 RotatingFileHandler."""

    def test_shared_rotating_file_handler(self):
        """
        get_logger('mod_a') y get_logger('mod_b') deben compartir exactamente 1 solo
        RotatingFileHandler en el logger padre, y ninguno duplicado en los hijos.
        """
        logger_a = get_logger("mod_a")
        logger_b = get_logger("mod_b")

        # Los loggers hijos deben propagar hacia el padre clashbot
        self.assertTrue(logger_a.propagate)
        self.assertTrue(logger_b.propagate)

        # Los hijos no deben tener handlers propios (evita handles duplicados abiertos en Windows)
        self.assertEqual(len(logger_a.handlers), 0)
        self.assertEqual(len(logger_b.handlers), 0)

        # El logger padre debe existir y tener exactamente 1 RotatingFileHandler y 1 StreamHandler
        parent_logger = logging.getLogger(_ROOT_LOGGER_NAME)
        file_handlers = [
            h for h in parent_logger.handlers if isinstance(h, RotatingFileHandler)
        ]
        stream_handlers = [
            h
            for h in parent_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, RotatingFileHandler)
        ]

        self.assertEqual(
            len(file_handlers),
            1,
            f"Se esperaba exactamente 1 RotatingFileHandler en el padre, pero se encontraron {len(file_handlers)}",
        )
        self.assertEqual(
            len(stream_handlers),
            1,
            f"Se esperaba exactamente 1 StreamHandler en el padre, pero se encontraron {len(stream_handlers)}",
        )

        # El nombre canónico de los loggers hijos debe estar bajo el namespace del padre
        self.assertEqual(logger_a.name, f"{_ROOT_LOGGER_NAME}.mod_a")
        self.assertEqual(logger_b.name, f"{_ROOT_LOGGER_NAME}.mod_b")

    def test_logger_empty_and_none_name(self):
        """get_logger('') o get_logger(None) retornan el logger raíz clashbot sin errores de namespace."""
        logger_empty = get_logger("")
        logger_none = get_logger(None)
        self.assertEqual(logger_empty.name, _ROOT_LOGGER_NAME)
        self.assertEqual(logger_none.name, _ROOT_LOGGER_NAME)


class TestStateMachineCoreStability(unittest.TestCase):
    """Verifica la corrección de inconsistencias lógicas en ClashStateMachine."""

    def setUp(self):
        self.mock_device = MagicMock(spec=Device)
        self.mock_device.width = 2400
        self.mock_device.height = 1080
        self.mock_device.screenshot.return_value = np.zeros(
            (1080, 2400, 3), dtype=np.uint8
        )

        self.mock_vision = MagicMock(spec=Vision)
        self.mock_vision.find_template.return_value = None
        self.mock_vision.find_any.return_value = None
        self.mock_vision.find_template_cached.return_value = None

        self.sm = ClashStateMachine(self.mock_device, self.mock_vision)

    def test_search_only_mode_stops_without_attacking(self):
        """
        ClashStateMachine respeta SEARCH_ONLY=True sin invocar execute_attack().
        """
        self.sm.state = BotState.EVALUATING_LOOT
        self.sm.deployer = MagicMock()
        self.sm.edrag_deployer = MagicMock()

        # Mockear lectura de botín que cumple los criterios
        self.sm.reader.read_enemy_loot_stable = MagicMock(
            return_value={"gold": 1_000_000, "elixir": 1_000_000, "dark_elixir": 10_000}
        )
        self.sm.reader.meets_attack_criteria = MagicMock(return_value=True)

        with patch.object(config, "SEARCH_ONLY", True):
            self.sm._tick()

        # Debe solicitar detención limpia
        self.assertTrue(self.sm._stop_requested)
        # NO debe pasar al estado ATTACKING
        self.assertNotEqual(self.sm.state, BotState.ATTACKING)
        # NO debe haber llamado al método execute_attack de ningún deployer
        self.sm.deployer.execute_attack.assert_not_called()
        self.sm.edrag_deployer.execute_attack.assert_not_called()

    def test_search_only_false_proceeds_to_attacking(self):
        """Con SEARCH_ONLY=False y botín apto, transiciona con normalidad a ATTACKING."""
        self.sm.state = BotState.EVALUATING_LOOT
        self.sm._stop_requested = False
        self.sm.reader.read_enemy_loot_stable = MagicMock(
            return_value={"gold": 1_000_000, "elixir": 1_000_000, "dark_elixir": 10_000}
        )
        self.sm.reader.meets_attack_criteria = MagicMock(return_value=True)

        with patch.object(config, "SEARCH_ONLY", False):
            self.sm._tick()

        self.assertFalse(self.sm._stop_requested)
        self.assertEqual(self.sm.state, BotState.ATTACKING)

    def test_returning_home_dead_code_eliminated(self):
        """Verifica que BotState no contenga RETURNING_HOME."""
        state_names = [s.name for s in BotState]
        self.assertNotIn("RETURNING_HOME", state_names)

    def test_return_homes_templates_only_existing(self):
        """Verifica que self.tpl_return_homes contenga solo rutas existentes en disco."""
        import os

        for tpl_path in self.sm.tpl_return_homes:
            self.assertTrue(
                os.path.exists(tpl_path),
                f"El template configurado en tpl_return_homes no existe físicamente: {tpl_path}",
            )

    def test_finish_battle_sequence_claim_reward_triggers_wait_and_tap_return_home(self):
        """
        Al detectar Claim Reward de evento, finish_battle_sequence debe reclamar el cofre
        Y acto seguido esperar y pulsar Return Home para garantizar volver a la aldea.
        """
        self.sm.proc_mgr.handle_chest_reward_sequence = MagicMock(return_value=True)
        self.sm._wait_and_tap_return_home = MagicMock(return_value=True)

        # Simular detección de Claim Reward en pantalla inicial
        self.mock_vision.find_any.side_effect = [
            ("claim_reward", 1200, 850),  # match_reward en screen
        ]

        self.sm.finish_battle_sequence(allow_surrender=False)

        # Debe haber pulsado Claim Reward
        self.mock_device.tap.assert_called_with(1200, 850, humanize=False)
        # Debe haber gestionado los cofres con wait_for_chest=True
        self.sm.proc_mgr.handle_chest_reward_sequence.assert_called_once_with(wait_for_chest=True)
        # Debe haber llamado a _wait_and_tap_return_home para volver a la aldea
        self.sm._wait_and_tap_return_home.assert_called_once()

    def test_wait_and_tap_return_home_exits_immediately_if_village_ready(self):
        """
        _wait_and_tap_return_home debe salir de inmediato si la aldea ya está lista,
        evitando pulsaciones innecesarias o taps fantasma del UI cache.
        """
        self.sm.proc_mgr.is_village_ready = MagicMock(return_value=True)
        ret = self.sm._wait_and_tap_return_home()
        self.assertTrue(ret)
        self.mock_device.tap.assert_not_called()

    def test_sneaky_farming_natural_end_does_not_surrender(self):
        """
        En WAITING_BATTLE_END (sneaky_farming), si la batalla termina naturalmente
        (aparece Return Home / Claim Reward), finish_battle_sequence debe llamarse
        con allow_surrender=False para no pulsar botones a ciegas en la pantalla de resultados.
        """
        self.sm.active_army_type = "sneaky_farming"
        self.sm.state = BotState.WAITING_BATTLE_END
        self.sm.finish_battle_sequence = MagicMock()

        # Simular que al primer chequeo ya aparece Return Home (fin natural)
        self.mock_vision.find_any.return_value = ("return_home", 1200, 950)

        with patch("time.sleep", return_value=None):
            self.sm._tick()

        # allow_surrender debe ser False en fin natural
        self.sm.finish_battle_sequence.assert_called_once_with(allow_surrender=False)
        self.assertEqual(self.sm.state, BotState.COLLECTING_LOOT)

    def test_run_forever_popup_handled_continues_without_tick(self):
        """
        En run_forever(), si handle_popup() maneja un popup (retorna True),
        debe continuar el bucle para estabilizar la pantalla y omitir _tick() en esa vuelta.
        """
        self.sm.proc_mgr.handle_popup = MagicMock(side_effect=[True, False])
        self.sm._tick = MagicMock()

        # Detener tras 2 vueltas
        call_count = 0
        def fake_tick():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                self.sm._stop_requested = True

        self.sm._tick.side_effect = fake_tick

        with patch("time.sleep", return_value=None):
            self.sm.run_forever(max_attacks=10)

        # En la primera iteración handle_popup fue True -> no llamó _tick()
        # En la segunda iteración handle_popup fue False -> llamó _tick()
        self.assertEqual(self.sm._tick.call_count, 1)

    def test_run_forever_resets_stop_requested(self):
        """run_forever() restablece _stop_requested=False al inicio para permitir reejecución."""
        self.sm._stop_requested = True
        self.sm.proc_mgr.handle_popup = MagicMock(return_value=False)
        self.sm._tick = MagicMock(side_effect=lambda: setattr(self.sm, "_stop_requested", True))

        self.sm.run_forever()
        self.assertEqual(self.sm._tick.call_count, 1)


class TestWallCostTableCanAfford(unittest.TestCase):
    """Valida la regla oficial de Clash of Clans en can_afford() (Oro O Elixir para niveles >= 5)."""

    def test_acceptance_criteria_level_19(self):
        """
        Criterio explícito de aceptación:
        - can_afford(19, 10_000_000, 0) retorna True
        - can_afford(19, 0, 10_000_000) retorna True
        - can_afford(19, 5_000_000, 5_000_000) retorna False
        """
        self.assertTrue(can_afford(19, 10_000_000, 0))
        self.assertTrue(can_afford(19, 0, 10_000_000))
        self.assertFalse(can_afford(19, 5_000_000, 5_000_000))

    def test_levels_1_to_4_gold_only(self):
        """Niveles 1 a 4 solo permiten Oro; Elixir no es válido ni sustituto."""
        # Nivel 2 cuesta 1,000 Oro
        self.assertTrue(can_afford(2, 1_000, 0))
        self.assertTrue(can_afford(2, 5_000, 0))
        # Si falta oro, no se puede pagar con Elixir
        self.assertFalse(can_afford(2, 999, 100_000_000))

    def test_levels_5_plus_or_logic(self):
        """Niveles 5+ aceptan Oro O Elixir; rechazan solo si ambos son insuficientes."""
        # Nivel 5: 20k Oro o 20k Elixir
        self.assertTrue(can_afford(5, 20_000, 0))
        self.assertTrue(can_afford(5, 0, 20_000))
        self.assertFalse(can_afford(5, 19_999, 19_999))

        # Nivel 10: 200k Oro o 200k Elixir
        self.assertTrue(can_afford(10, 200_000, 0))
        self.assertTrue(can_afford(10, 0, 200_000))
        self.assertFalse(can_afford(10, 199_999, 199_999))

        # Nivel 18: 7M Oro o 7M Elixir
        self.assertTrue(can_afford(18, 7_000_000, 0))
        self.assertTrue(can_afford(18, 0, 7_000_000))
        self.assertFalse(can_afford(18, 6_999_999, 6_999_999))

    def test_require_both_flag(self):
        """require_both=True permite forzar validación estricta de ambos recursos si fuera necesario."""
        self.assertFalse(can_afford(19, 10_000_000, 0, require_both=True))
        self.assertTrue(can_afford(19, 10_000_000, 10_000_000, require_both=True))

    def test_wall_upgrade_strategy_delegates_require_both(self):
        """WallUpgradeStrategy.can_afford acepta y propaga require_both correctamente."""
        from core.wall_upgrader import WallUpgradeStrategy
        strat = WallUpgradeStrategy()
        self.assertFalse(strat.can_afford(19, 10_000_000, 0, require_both=True))
        self.assertTrue(strat.can_afford(19, 10_000_000, 10_000_000, require_both=True))
        self.assertTrue(strat.can_afford(19, 10_000_000, 0, require_both=False))


if __name__ == "__main__":
    unittest.main()
