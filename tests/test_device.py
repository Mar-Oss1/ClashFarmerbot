"""
tests/test_device.py
Pruebas unitarias para la abstracción de Device y su espejo MockDevice.
Fase 4: Abstracción de Device, Paneo por Flancos y Toque Directo.
"""
from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch

from core.device import Device
from tests import MockDevice


class TestDeviceAbstraction(unittest.TestCase):
    """Verifica los métodos enriquecidos en core/device.py y tests/__init__.py."""

    def test_is_mock_flags(self):
        """Verifica que Device defina is_mock=False y MockDevice defina is_mock=True."""
        dev = Device()
        mock_dev = MockDevice()
        self.assertFalse(dev.is_mock)
        self.assertFalse(Device.is_mock)
        self.assertTrue(mock_dev.is_mock)
        self.assertTrue(MockDevice.is_mock)

    def test_raw_tap_disables_humanize(self):
        """Verifica que raw_tap invoque tap con humanize=False."""
        dev = Device()
        dev.tap = MagicMock()
        dev.raw_tap(120, 340)
        dev.tap.assert_called_once_with(120, 340, humanize=False)

    def test_press_back_delegates_to_u2_with_reconnect(self):
        """Verifica que press_back delegue a self.d.press('back')."""
        dev = Device()
        dev.d = MagicMock()
        dev.press_back()
        dev.d.press.assert_called_once_with("back")

    def test_pan_flank_coordinates_and_directions(self):
        """Verifica que pan_flank calcule coordenadas relativas calibradas en 2400x1080."""
        dev = Device()
        dev.width = 2400
        dev.height = 1080
        dev.swipe = MagicMock()

        # 1. Flanco Norte: (0.50, 0.222) -> (0.50, 0.759)
        dev.pan_flank("north", reverse=False, duration_ms=500)
        dev.swipe.assert_called_with(1200, 240, 1200, 820, duration=0.5)

        # 2. Flanco Norte con reverse=True: (0.50, 0.759) -> (0.50, 0.222)
        dev.pan_flank("north", reverse=True, duration_ms=400)
        dev.swipe.assert_called_with(1200, 820, 1200, 240, duration=0.4)

        # 3. Flanco Sur: (0.50, 0.722) -> (0.50, 0.222)
        dev.pan_flank("south", reverse=False, duration_ms=480)
        dev.swipe.assert_called_with(1200, 780, 1200, 240, duration=0.48)

        # 4. Flanco Este: (0.646, 0.50) -> (0.354, 0.50)
        dev.pan_flank("east", reverse=False, duration_ms=480)
        dev.swipe.assert_called_with(1550, 540, 850, 540, duration=0.48)

        # 5. Flanco Oeste: (0.354, 0.50) -> (0.646, 0.50)
        dev.pan_flank("west", reverse=False, duration_ms=480)
        dev.swipe.assert_called_with(850, 540, 1550, 540, duration=0.48)

        # 6. Flanco desconocido: no debe llamar swipe
        dev.swipe.reset_mock()
        dev.pan_flank("unknown_flank")
        dev.swipe.assert_not_called()

    def test_mock_device_all_enriched_methods(self):
        """Verifica que MockDevice registre limpiamente el historial de todas las acciones enriquecidas."""
        mock_dev = MockDevice(width=2400, height=1080)

        # raw_tap
        mock_dev.raw_tap(450, 600)
        self.assertEqual(mock_dev.tap_history, [(450, 600)])

        # press_back
        mock_dev.press_back()
        self.assertEqual(mock_dev.press_back_calls, 1)

        # zoom_half
        mock_dev.zoom_half()
        self.assertEqual(mock_dev.zoom_half_calls, 1)

        # hold_tap
        mock_dev.hold_tap(800, 900, duration=1.2)
        self.assertEqual(mock_dev.hold_tap_history, [(800, 900, 1.2)])
        self.assertEqual(mock_dev.tap_history[-1], (800, 900))

        # pan_flank
        mock_dev.pan_flank("north", reverse=False, duration_ms=450)
        self.assertEqual(mock_dev.pan_flank_history, [("north", False, 450)])
        self.assertEqual(mock_dev.swipe_history[-1], (1200, 240, 1200, 820, 0.45))

        # wait_camera_settled
        self.assertTrue(mock_dev.wait_camera_settled())

        # clear_history
        mock_dev.clear_history()
        self.assertEqual(mock_dev.tap_history, [])
        self.assertEqual(mock_dev.hold_tap_history, [])
        self.assertEqual(mock_dev.swipe_history, [])
        self.assertEqual(mock_dev.pan_flank_history, [])
        self.assertEqual(mock_dev.press_back_calls, 0)
        self.assertEqual(mock_dev.zoom_half_calls, 0)


if __name__ == "__main__":
    unittest.main()
