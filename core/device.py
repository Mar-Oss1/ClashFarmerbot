"""
core/device.py
Wrapper uiautomator2 con soporte de gestos, paneo de cámara y trazo continuo seguro.
"""
import os
import random
import tempfile
import time
import cv2
import numpy as np
import uiautomator2 as u2
from PIL import Image

from core.config import DEVICE_SERIAL, APP_PACKAGE, SETTINGS
from core.logger import get_logger
from humanizer.behavior import HumanBehavior

log = get_logger("device")

_MAX_RECONNECT = 3
_RECONNECT_WAIT = 5


class Device:
    def __init__(self, serial: str = DEVICE_SERIAL):
        self.serial = serial
        self.d: u2.Device | None = None
        self.width: int = 2400
        self.height: int = 1080

    def connect(self) -> bool:
        try:
            log.info("Conectando al dispositivo %s...", self.serial)
            self.d = u2.connect(self.serial)
            self.d.implicitly_wait(5)
            info = self.d.info
            w = info.get("displayWidth", 2400)
            h = info.get("displayHeight", 1080)
            self.width = max(w, h)
            self.height = min(w, h)
            log.info("ADB conectado. Resolucion: %dx%d", self.width, self.height)
            # Control de punteros visuales: desactivados por defecto para optimizar GPU y evitar lag
            show_trace = SETTINGS.get("device", {}).get("show_touch_trace", False)
            if show_trace:
                try:
                    self.d.show_touch_trace(pointer_location=True, show_touches=True)
                    log.info("Visualizacion de toques y puntero activada en pantalla.")
                except Exception as e:
                    log.warning("No se pudo activar show_touch_trace: %s", e)
            else:
                try:
                    self.d.show_touch_trace(pointer_location=False, show_touches=False)
                except Exception:
                    pass
            return True
        except Exception as e:
            log.error("Fallo conexion ADB: %s", e)
            return False

    def _reconnect(self) -> bool:
        for intento in range(1, _MAX_RECONNECT + 1):
            log.warning("Reintentando conexion ADB (%d/%d)...", intento, _MAX_RECONNECT)
            time.sleep(_RECONNECT_WAIT)
            try:
                self.d = u2.connect(self.serial)
                self.d.implicitly_wait(5)
                log.info("Reconexion ADB exitosa.")
                return True
            except Exception as e:
                log.error("Error en reconexion: %s", e)
        log.critical("No se pudo reconectar con ADB tras %d intentos.", _MAX_RECONNECT)
        return False

    def screenshot(self, target: str = "opencv") -> np.ndarray:
        save_path = None
        fmt = target
        if target.lower().endswith((".png", ".jpg", ".jpeg")):
            save_path = target
            fmt = "opencv"

        for intento in range(_MAX_RECONNECT):
            try:
                img = self.d.screenshot(format=fmt)
                if isinstance(img, np.ndarray) and img.shape[1] > 0:
                    self.height, self.width = img.shape[:2]
                if save_path is not None and isinstance(img, np.ndarray):
                    cv2.imwrite(save_path, img)
                return img
            except Exception as e:
                log.warning("Error en screenshot (intento %d): %s", intento + 1, e)
                if not self._reconnect():
                    raise RuntimeError("ADB caido al tomar captura.") from e
        raise RuntimeError("Imposible obtener captura.")

    def tap(self, x: int | float, y: int | float, humanize: bool = True):
        tx, ty = int(x), int(y)
        if humanize:
            tx, ty = HumanBehavior.jitter_point(tx, ty)
        for intento in range(_MAX_RECONNECT):
            try:
                self.d.click(tx, ty)
                return
            except Exception as e:
                log.warning("Error en tap(%d, %d): %s", tx, ty, e)
                if not self._reconnect():
                    raise RuntimeError("ADB caido en tap.") from e

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        for intento in range(_MAX_RECONNECT):
            try:
                self.d.swipe(int(x1), int(y1), int(x2), int(y2), duration=duration)
                return
            except Exception as e:
                log.warning("Error en swipe: %s", e)
                if not self._reconnect():
                    raise RuntimeError("ADB caido en swipe.") from e

    def touch_down(self, x: int | float, y: int | float):
        """Mantiene pulsado un punto (ACTION_DOWN) en la pantalla sin soltarlo."""
        tx, ty = int(x), int(y)
        for intento in range(_MAX_RECONNECT):
            try:
                self.d.touch.down(tx, ty)
                return
            except Exception as e:
                log.warning("Error en touch_down(%d, %d): %s", tx, ty, e)
                if not self._reconnect():
                    raise RuntimeError("ADB caido en touch_down.") from e

    def touch_up(self, x: int | float, y: int | float):
        """Libera el punto pulsado (ACTION_UP) en la pantalla."""
        tx, ty = int(x), int(y)
        for intento in range(_MAX_RECONNECT):
            try:
                self.d.touch.up(tx, ty)
                return
            except Exception as e:
                log.warning("Error en touch_up(%d, %d): %s", tx, ty, e)
                if not self._reconnect():
                    raise RuntimeError("ADB caido en touch_up.") from e

    def deploy_heroes_held_anchor(self, anchor_pt: tuple[int | float, int | float], hero_slots: list[tuple[int, int]], slot_delay: float = 0.10):
        """
        Despliega los héroes según la técnica pro de Clash of Clans:
        1. Selecciona el 1er héroe en la barra de batalla.
        2. Mantiene pulsado el mapa (touch_down) en el punto de drop exterior (anchor_pt),
           desplegando inmediatamente al primer héroe.
        3. Con el dedo anclado en el mapa, pulsa sucesivamente los héroes restantes en la barra,
           los cuales se despliegan instantáneamente en ese mismo punto de anclaje.
        4. Libera el toque del mapa (touch_up).
        """
        if not hero_slots:
            return
        ax, ay = int(anchor_pt[0]), int(anchor_pt[1])
        first_hero = hero_slots[0]
        remaining_heroes = hero_slots[1:]

        # 1. Seleccionar el primer héroe en la barra
        log.info("👑 [1/%d] Seleccionando 1er héroe en la barra en (%d, %d)...", len(hero_slots), first_hero[0], first_hero[1])
        self.tap(first_hero[0], first_hero[1], humanize=False)
        time.sleep(0.08)

        # 2. Dejar click sostenido en el anclaje del mapa para soltar el 1er héroe
        log.info("👑 Anclando toque sostenido en (%d, %d) (suelta 1er héroe y prepara restantes)...", ax, ay)
        try:
            self.touch_down(ax, ay)
            time.sleep(0.12)
            # 3. Pulsar sucesivamente los héroes restantes en la barra mientras el anclaje sigue activo
            for idx_h, (hx, hy) in enumerate(remaining_heroes, 2):
                log.info("   -> [%d/%d] Pulsando ranura de heroe restante en (%d, %d) con anclaje activo...", idx_h, len(hero_slots), hx, hy)
                self.tap(hx, hy, humanize=False)
                time.sleep(slot_delay)
            time.sleep(0.08)
        finally:
            try:
                self.touch_up(ax, ay)
            except Exception:
                pass

    def hold_tap(self, x: int | float, y: int | float, duration: float = 0.85):
        """
        Ejecuta un toque sostenido (press and hold) en (x, y) durante 'duration' segundos.
        Permite el despliegue continuo y masivo de tropas (EDrags) a alta velocidad en una sola tirada.
        """
        tx, ty = int(x), int(y)
        tx, ty = HumanBehavior.jitter_point(tx, ty, max_offset=4)
        for intento in range(_MAX_RECONNECT):
            try:
                self.d.touch.down(tx, ty).sleep(duration).up(tx, ty)
                return
            except Exception as e:
                log.warning("Error en touch.down/up hold_tap(%d, %d): %s", tx, ty, e)
                try:
                    dur_ms = int(duration * 1000)
                    self.d.shell(f"input swipe {tx} {ty} {tx+1} {ty+1} {dur_ms}")
                    return
                except Exception as e2:
                    log.warning("Fallback input swipe falló: %s", e2)
                if not self._reconnect():
                    raise RuntimeError("ADB caído en hold_tap.") from e

    def stroke_points(self, points: list[tuple[int, int]], duration: float = 1.3, anchor_sec: float = 0.60):
        """
        Ejecuta un barrido continuo con click inicial sostenido (~600ms, reducido un 33%)
        para activar de forma garantizada el modo despliegue continuo de tropas en Clash of Clans,
        seguido de un arrastre pausado y lento (duration) a lo largo de la trayectoria.
        Usa el método nativo UiDevice.swipePoints (jsonrpc.swipePoints) compatible con dispositivos
        Xiaomi / MIUI / HyperOS y emuladores, con fallback directo por hardware a adb shell input.
        """
        if not points or len(points) < 2:
            return

        start_x, start_y = points[0]
        num_move_segments = max(1, len(points) - 1)

        # En Android UiDevice.swipe, cada paso (step) toma ~10.5ms (5ms pause interno + overhead de eventos)
        step_duration = 0.0105
        steps_per_segment = max(2, int(round(duration / (num_move_segments * step_duration))))

        # Calcular los puntos duplicados necesarios para sostener el click inicial exactamente anchor_sec (~900ms)
        seg_time = steps_per_segment * step_duration
        num_anchor_segments = max(1, int(round(anchor_sec / seg_time)))
        anchor_count = num_anchor_segments + 1

        flat_pts = [int(start_x), int(start_y)] * anchor_count
        for px, py in points[1:]:
            flat_pts.extend([int(px), int(py)])

        try:
            self.d.jsonrpc.swipePoints(flat_pts, steps_per_segment)
        except Exception as e:
            log.warning("swipePoints fallo (%s), ejecutando fallback por adb shell...", e)
            try:
                end_x, end_y = points[-1]
                dur_ms = int((duration + anchor_sec) * 1000)
                self.d.shell(f"input swipe {int(start_x)} {int(start_y)} {int(end_x)} {int(end_y)} {dur_ms}")
            except Exception as e2:
                log.error("Fallo critico en inyeccion de sweep: %s", e2)

    def pan_camera(self, x1_rel: float, y1_rel: float, x2_rel: float, y2_rel: float, duration: float = 0.9):
        sx1, sy1 = int(x1_rel * self.width), int(y1_rel * self.height)
        sx2, sy2 = int(x2_rel * self.width), int(y2_rel * self.height)

        self.swipe(sx1, sy1, sx2, sy2, duration=duration)
        time.sleep(0.5)

    def wait_camera_settled(self, timeout=2.5, interval=0.10, max_shift=0.8, max_diff=25.0, need=2) -> bool:
        prev, ok, t0 = None, 0, time.monotonic()
        while time.monotonic() - t0 < timeout:
            img = self.screenshot()
            if img is None:
                time.sleep(interval)
                continue
            h = img.shape[0]
            g = cv2.cvtColor(cv2.resize(img[int(h * .12):int(h * .80)], (640, 300)), cv2.COLOR_BGR2GRAY).astype(
                np.float32)
            if prev is not None:
                (dx, dy), _ = cv2.phaseCorrelate(prev, g)
                diff = float(cv2.absdiff(prev, g).mean())
                # En combate o aldea activa, héroes, tropas y animaciones cambian píxeles locales, pero la traslación global de cámara (dx, dy) es nula
                is_settled = (abs(dx) < max_shift and abs(dy) < max_shift) and (diff < max_diff)
                ok = ok + 1 if is_settled else 0
                if ok >= need:
                    return True
            prev = g
            time.sleep(interval)
        log.warning("La camara NO se estabilizo en %.1fs", timeout)
        return False

    def zoom_out(self):
        """
        Ejecuta zoom out completo obligatorio centrado en la pantalla (percent=85, steps=35).
        Mantiene los dedos dentro de la zona segura alejada del notch (94px) y bordes de navegación,
        asegurando tope máximo de alejamiento con doble pinch suave.
        """
        log.info("🔍 Ejecutando Zoom Out centrado en zona segura...")
        try:
            self.d.pinch_in(percent=85, steps=35)
        except Exception:
            try:
                self.d().pinch_in(percent=85, steps=35)
            except Exception as e2:
                log.warning("Pinch_in global falló (%s).", e2)
        time.sleep(0.35)
        # Segundo pinch suave de confirmación para asegurar encuadre total sin zonas recortadas
        try:
            self.d.pinch_in(percent=65, steps=25)
        except Exception:
            pass
        time.sleep(0.5)
        self.wait_camera_settled(timeout=2.0)
        log.info("🔍 Zoom Out finalizado y mapa estabilizado.")

    def zoom_half(self):
        """
        Ejecuta zoom moderado (~mitad disponible) para mantener amplia superficie
        de césped exterior legal (evita toques en zona roja al desplegar tropas masivas).
        """
        log.info("🔍 Ejecutando Zoom Moderado (mitad disponible para mayor superficie)...")
        try:
            self.d().pinch_in(percent=35, steps=20)
        except Exception:
            try:
                self.d.pinch_in(percent=35, steps=20)
            except Exception as e2:
                log.warning("Pinch_in moderado falló (%s).", e2)
        time.sleep(0.35)
        self.wait_camera_settled(timeout=1.5)
        log.info("🔍 Zoom Moderado finalizado y cámara estabilizada.")

    def is_app_running(self, package: str = APP_PACKAGE) -> bool:
        try:
            return self.d.app_current().get("package") == package
        except Exception:
            return False

    def open_app(self):
        self.d.app_start(APP_PACKAGE)

    def restart_app(self):
        log.warning("Reiniciando %s...", APP_PACKAGE)
        try:
            self.d.app_stop(APP_PACKAGE)
            time.sleep(2.0)
            self.d.app_start(APP_PACKAGE)
        except Exception as e:
            log.error("Error al reiniciar aplicacion: %s", e)
            if self._reconnect():
                self.d.app_start(APP_PACKAGE)