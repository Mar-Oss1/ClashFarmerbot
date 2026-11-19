"""
tools/auto_capture_villages.py
Script CLI standalone para captura automática y masiva de aldeas para dataset CoC.
Navega de aldea en aldea continuamente usando el botón 'Next' sin salir al menú principal,
haciendo zoom out y extrayendo botín/metadata en cada aldea, sin desplegar tropas, hechizos ni héroes.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from core.device import Device
from core.config import SETTINGS, load_settings
from core.vision import Vision
from core.process_manager import ProcessManager
from core.logger import get_logger
from humanizer.behavior import HumanBehavior
from vision.army_checker import ArmyChecker
from vision.resource_reader import ResourceReader

# Alias para compatibilidad
load_config = load_settings
VisionEngine = Vision

log = get_logger("auto_capture")

# ====================================================================
# SEGURIDAD DURA (HARDCODED, NO CONFIGURABLE)
# ====================================================================
DEPLOY_ENABLED = False


def safe_tap(device: Device, x: int | float, y: int | float, description: str = "", in_battle: bool = False):
    """
    Toque seguro que garantiza que JAMÁS se toque la barra inferior de despliegue
    de tropas, hechizos o héroes durante la batalla.
    """
    ix, iy = int(x), int(y)
    deployment_bar_y = int(device.height * 0.82)
    # La franja de cartas de despliegue de tropas/hechizos solo existe durante el combate
    is_in_deployment_strip = in_battle and (iy >= deployment_bar_y) and (int(device.width * 0.12) <= ix <= int(device.width * 0.88))

    if not DEPLOY_ENABLED and is_in_deployment_strip:
        log.critical("SEGURIDAD: Intento de toque en la barra de tropas (%s): (%d, %d). ABORTANDO.", description, ix, iy)
        raise RuntimeError("Deployment disabled in capture mode")

    device.tap(ix, iy)


class AutoCaptureRunner:
    def __init__(self, serial: str | None = None, output_dir: str = "datasets/coc/images/raw"):
        self.serial = serial or SETTINGS.get("device", {}).get("serial", "5fba8618")
        self.output_dir = output_dir
        self.device = Device(serial=self.serial)
        self.vision = Vision(threshold=0.76)
        self.vision.preload_templates("assets")
        self.proc_mgr = ProcessManager(self.device, self.vision)
        self.army = ArmyChecker(self.vision)
        self.reader = ResourceReader(self.vision)
        self.in_battle = False

        # Templates de navegación
        self.tpl_attack = "assets/buttons/attack_btn.png"
        self.tpl_find_match = "assets/buttons/find_match_btn.png"
        self.tpl_attack_confirms = [
            "assets/buttons/attack_btn_confirm.png",
            "assets/buttons/confirm_attack_btn.png",
        ]
        self.tpl_next = "assets/buttons/next_btn.png"
        self.tpl_surrender = "assets/buttons/surrender_battle.png"
        self.tpl_end_battle = "assets/buttons/end_battle.png"
        self.tpl_confirm_end = "assets/buttons/confirm_end.png"
        self.tpl_okay = "assets/buttons/okay_btn.png"
        self.tpl_return_home = "assets/buttons/return_home.png"

    def is_in_match(self) -> bool:
        """Verifica si actualmente estamos en la pantalla de batalla/búsqueda con Next visible."""
        screen = self.device.screenshot()
        next_pos = self.vision.find_template(screen, self.tpl_next, threshold=0.74, zona="ESQUINA_INF_DER")
        if not next_pos:
            next_pos = self.vision.find_template(screen, self.tpl_next, threshold=0.68, zona="ESQUINA_INF_DER")
        return next_pos is not None

    def surrender_and_return_home(self) -> bool:
        """
        Finaliza la búsqueda / batalla y regresa de forma limpia a la aldea principal:
        1. Comprueba si ya está en aldea principal o si hay Return Home.
        2. Pulsa End Battle (o Surrender).
        3. Confirma finalización (Confirm End / Okay).
        4. Espera confirmación de aldea principal o Return Home.
        """
        log.info("🏳️ Finalizando combate / búsqueda y retornando a la aldea principal...")
        screen = self.device.screenshot()

        # Si ya estamos en la aldea principal
        if self.vision.find_template(screen, self.tpl_attack, threshold=0.74, zona="ESQUINA_INF_IZQ"):
            log.info("🏰 Ya nos encontramos en la aldea principal.")
            self.in_battle = False
            return True

        # Si ya está Return Home en pantalla (ej. batalla finalizada)
        ret_pos = self.vision.find_template(screen, self.tpl_return_home, zona="MITAD_INFERIOR")
        if ret_pos:
            HumanBehavior.reaction_delay()
            safe_tap(self.device, *ret_pos, description="return_home", in_battle=False)
            self.in_battle = False
            time.sleep(3.0)
            return True

        # 1. End Battle / Surrender en esquina inferior izquierda
        end_pos = self.vision.find_template(screen, self.tpl_end_battle, threshold=0.72, zona="ESQUINA_INF_IZQ")
        if not end_pos:
            end_pos = self.vision.find_template(screen, self.tpl_surrender, threshold=0.72, zona="ESQUINA_INF_IZQ")
        if not end_pos:
            end_pos = self.vision.wait_for_template(
                self.device.screenshot, self.tpl_end_battle, timeout=3, threshold=0.72, zona="MITAD_INFERIOR"
            )
        if not end_pos:
            end_pos = self.vision.wait_for_template(
                self.device.screenshot, self.tpl_surrender, timeout=3, threshold=0.72, zona="MITAD_INFERIOR"
            )

        if end_pos:
            HumanBehavior.reaction_delay()
            safe_tap(self.device, *end_pos, description="end_or_surrender_btn")
            time.sleep(0.8)
        else:
            log.warning("Botón de fin de batalla no detectado por template. Probando tecla BACK...")
            self.device.d.press("back")
            time.sleep(0.8)

        # 2. Confirmación de finalización / rendición
        confirm_pos = self.vision.wait_for_template(
            self.device.screenshot, self.tpl_confirm_end, timeout=4, threshold=0.70, zona="CENTRO"
        )
        if not confirm_pos:
            confirm_pos = self.vision.wait_for_template(
                self.device.screenshot, self.tpl_okay, timeout=2, threshold=0.70, zona="CENTRO"
            )

        if confirm_pos:
            HumanBehavior.reaction_delay()
            safe_tap(self.device, *confirm_pos, description="confirm_end_btn")
            time.sleep(1.0)

        # 3. Esperar confirmación de retorno: puede aparecer Return Home o volver directo a la aldea
        t0 = time.monotonic()
        while time.monotonic() - t0 < 8:
            s = self.device.screenshot()
            # Si aparece botón Return Home
            ret_pos = self.vision.find_template(s, self.tpl_return_home, zona="MITAD_INFERIOR")
            if ret_pos:
                HumanBehavior.reaction_delay()
                safe_tap(self.device, *ret_pos, description="return_home", in_battle=False)
                self.in_battle = False
                time.sleep(3.0)
                return True
            # Si ya se cargó la aldea principal (botón Attack visible)
            if self.vision.find_template(s, self.tpl_attack, threshold=0.74, zona="ESQUINA_INF_IZQ"):
                log.info("🏰 Aldea principal confirmada tras salir de búsqueda.")
                self.in_battle = False
                return True
            time.sleep(0.5)

        self.in_battle = False
        log.warning("No se confirmó Return Home ni aldea principal de forma inmediata.")
        return False

    def verify_army_ready(self, timeout_sec: int = 60) -> bool:
        """Verifica que el ejército esté listo con timeout de seguridad."""
        log.info("Verificando estado del ejército...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_sec:
            screen = self.device.screenshot()
            if self.army.can_start_battle(screen):
                log.info("Ejército confirmado listo para batalla.")
                return True
            time.sleep(2.0)
        log.warning("Timeout (%ds) esperando ejército listo.", timeout_sec)
        return False

    def _wait_for_first_rival(self, timeout: int = 90) -> bool:
        log.info("Esperando botón Next (despeje de nubes e ingreso a batalla)...")
        next_pos = self.vision.wait_for_template(
            self.device.screenshot, self.tpl_next, timeout=timeout, threshold=0.74, zona="ESQUINA_INF_DER"
        )
        if not next_pos:
            log.error("Timeout (%ds) esperando botón Next / ingreso a batalla.", timeout)
            self.surrender_and_return_home()
            return False

        self.in_battle = True
        log.info("🎯 Primera aldea rival cargada (Next visible en %s). MODO COMBATE ACTIVO.", next_pos)
        time.sleep(0.6)
        return True

    def enter_matchmaking(self) -> bool:
        """
        Inicia la búsqueda de rivales desde la aldea principal (o menús intermedios)
        hasta que carga la primera aldea rival con botón Next visible.
        """
        if self.is_in_match():
            log.info("🎯 Ya se encuentra en pantalla de rival (Next visible). Continuando...")
            self.in_battle = True
            return True

        log.info("Iniciando emparejamiento para combate...")
        screen = self.device.screenshot()

        # Caso A: ¿Ya está visible el botón de confirmar ataque?
        match_confirm = self.vision.find_any(screen, self.tpl_attack_confirms, threshold=0.72, zona="MITAD_INFERIOR")
        if match_confirm:
            log.info("🎯 Botón Confirm Attack ya visible en (%d, %d). Confirmando...", match_confirm[1], match_confirm[2])
            HumanBehavior.reaction_delay()
            safe_tap(self.device, match_confirm[1], match_confirm[2], description="confirm_attack_btn")
            return self._wait_for_first_rival()

        # Caso B: ¿Ya está abierto el menú de ataque (Find Match visible)?
        find_pos = self.vision.find_template(screen, self.tpl_find_match, threshold=0.72, zona="COMPLETA")
        if not find_pos:
            # Caso C: Estamos en la aldea principal
            if not self.proc_mgr.ensure_game_ready(timeout=25):
                log.error("Fallo ensure_game_ready antes de buscar rivales.")
                return False

            if not self.verify_army_ready(timeout_sec=30):
                log.error("Ejército no disponible para iniciar búsqueda.")
                return False

            screen = self.device.screenshot()
            attack_pos = self.vision.find_template(screen, self.tpl_attack, threshold=0.74, zona="ESQUINA_INF_IZQ")
            if not attack_pos:
                log.error("Botón Attack no detectado en aldea principal.")
                return False

            HumanBehavior.reaction_delay()
            safe_tap(self.device, *attack_pos, description="attack_btn")

            find_pos = self.vision.wait_for_template(
                self.device.screenshot, self.tpl_find_match, timeout=5, zona="COMPLETA"
            )
            if not find_pos:
                # Reintento de toque por si no registró
                log.info("Reintentando toque en botón Attack...")
                safe_tap(self.device, *attack_pos, description="attack_btn_retry")
                find_pos = self.vision.wait_for_template(
                    self.device.screenshot, self.tpl_find_match, timeout=6, zona="COMPLETA"
                )

        if not find_pos:
            log.error("Botón Find Match no detectado tras abrir menú de ataque.")
            self.surrender_and_return_home()
            return False

        HumanBehavior.reaction_delay()
        safe_tap(self.device, *find_pos, description="find_match_btn")
        time.sleep(0.8)

        # Esperar y confirmar búsqueda
        confirm_pos = None
        t_start = time.monotonic()
        while time.monotonic() - t_start < 4.5:
            s_confirm = self.device.screenshot()
            match = self.vision.find_any(s_confirm, self.tpl_attack_confirms, threshold=0.72, zona="MITAD_INFERIOR")
            if match:
                confirm_pos = (match[1], match[2])
                break
            time.sleep(0.25)

        if confirm_pos:
            HumanBehavior.reaction_delay()
            safe_tap(self.device, *confirm_pos, description="confirm_attack_btn")
        else:
            log.info("Clic de confirmación en posición estándar tras find_match...")
            HumanBehavior.reaction_delay()
            safe_tap(self.device, int(self.device.width * 0.814), int(self.device.height * 0.900), description="confirm_attack_fallback")

        return self._wait_for_first_rival()

    def capture_current_village(self, idx: int, timestamp_str: str) -> bool:
        """
        Ejecuta zoom out completo, estabiliza cámara, toma captura y guarda metadata de botín
        de la aldea rival actualmente visible.
        """
        # 1. Zoom out completo centrado y estabilización
        self.device.zoom_out()
        self.device.wait_camera_settled(timeout=2.0)

        # 2. Captura de pantalla
        cap_screen = self.device.screenshot()
        filename_png = f"auto_{timestamp_str}_{idx:04d}.png"
        filename_json = f"auto_{timestamp_str}_{idx:04d}.json"
        path_png = os.path.join(self.output_dir, filename_png)
        path_json = os.path.join(self.output_dir, filename_json)

        cv2.imwrite(path_png, cap_screen)
        log.info("💾 Imagen [%04d] guardada: %s (%dx%d)", idx, path_png, cap_screen.shape[1], cap_screen.shape[0])

        # 3. Lectura de metadata de botín
        loot_info = {}
        try:
            loot_info = self.reader.read_enemy_loot(cap_screen)
        except Exception as e:
            log.warning("No se pudo parsear botín de la captura: %s", e)

        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "index": idx,
            "image_file": filename_png,
            "resolution": [int(cap_screen.shape[1]), int(cap_screen.shape[0])],
            "loot": loot_info,
            "troops_deployed": False,
        }
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        log.info("📄 Metadata [%04d] guardada: %s", idx, path_json)

        return True

    def next_village(self, timeout: int = 90) -> bool:
        """
        Pulsa el botón 'Next' para avanzar a la siguiente aldea y espera a que las nubes se despejen.
        No sale al menú principal.
        """
        screen = self.device.screenshot()
        next_pos = self.vision.find_template(screen, self.tpl_next, threshold=0.74, zona="ESQUINA_INF_DER")
        if not next_pos:
            next_pos = self.vision.find_template(screen, self.tpl_next, threshold=0.68, zona="ESQUINA_INF_DER")

        if not next_pos:
            # Si el temporizador de 30s expiró, aparece surrender en lugar de end_battle y Next desaparece
            is_surrender = self.vision.find_template(screen, self.tpl_surrender, threshold=0.70, zona="ESQUINA_INF_IZQ")
            if is_surrender:
                log.warning("El temporizador de exploración expiró (combate iniciado sin tropas). Forzando rendición y re-entrada...")
                self.surrender_and_return_home()
                return False

            # Fallback en coordenadas conocidas para 2400x1080 si estamos en combate
            log.warning("Botón Next no detectado por template. Probando coordenadas conocidas (2051, 782)...")
            next_pos = (int(self.device.width * 0.854), int(self.device.height * 0.724))

        log.info("⏩ Pulsando botón Next en %s...", next_pos)
        HumanBehavior.reaction_delay()
        safe_tap(self.device, *next_pos, description="next_btn", in_battle=True)

        # Esperar a que entren las nubes (Next desaparece)
        time.sleep(1.2)

        # Esperar a que se despejen las nubes y aparezca el botón Next de la nueva aldea
        log.info("Esperando que disipen las nubes y cargue la nueva aldea...")
        new_next = self.vision.wait_for_template(
            self.device.screenshot, self.tpl_next, timeout=timeout, threshold=0.74, zona="ESQUINA_INF_DER"
        )
        if not new_next:
            new_next = self.vision.find_template(self.device.screenshot(), self.tpl_next, threshold=0.68, zona="ESQUINA_INF_DER")

        if not new_next:
            log.warning("Timeout (%ds) esperando nueva aldea tras pulsar Next.", timeout)
            return False

        log.info("🎯 Nueva aldea rival cargada (Next visible en %s).", new_next)
        time.sleep(0.6)
        return True

    def execute_capture_cycle(self, idx: int, timestamp_str: str) -> bool:
        """
        Método de compatibilidad para capturar la aldea actual.
        """
        if not self.is_in_match():
            if not self.enter_matchmaking():
                return False

        if not self.capture_current_village(idx, timestamp_str):
            return False

        return True


def get_connected_devices() -> list[str]:
    """Obtiene la lista de dispositivos conectados por ADB."""
    try:
        import adbutils
        devs = [d.serial for d in adbutils.adb.device_list()]
        if devs:
            return devs
    except Exception:
        pass
    try:
        res = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)
        devs = []
        for line in res.stdout.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device" and parts[0] != "List":
                devs.append(parts[0])
        return devs
    except Exception:
        return []


def print_banner(count: int):
    cost_est = count * 750
    print("=" * 70)
    print(" AUTO-CAPTURA MASIVA DE ALDEAS (MODO CONTINUO 'NEXT')")
    print("=" * 70)
    print(" Navega de aldea en aldea mediante botón 'Next' sin volver al menú.")
    print(" NO se despliegan tropas, hechizos ni héroes.")
    print(f" Coste ~500-1000 oro/aldea por Next. Total estimado: ~{cost_est:,} oro.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Auto-captura masiva de aldeas para dataset CoC")
    parser.add_argument("--count", type=int, default=100, help="Número de aldeas a capturar (default: 100)")
    parser.add_argument("--serial", type=str, default=None, help="Serial ADB del dispositivo")
    parser.add_argument("--output", type=str, default="datasets/coc/images/raw", help="Directorio de salida")
    parser.add_argument("--delay-min", type=float, default=2.5, help="Pausa mínima entre aldeas en segundos (default: 2.5)")
    parser.add_argument("--delay-max", type=float, default=5.0, help="Pausa máxima entre aldeas en segundos (default: 5.0)")
    parser.add_argument("--max-fails", type=int, default=3, help="Máximo de fallos consecutivos antes de abortar")
    parser.add_argument("--dry-run", action="store_true", help="Imprime plan y sale sin conectar ni tocar el dispositivo")
    parser.add_argument("--resume", action="store_true", help="Continúa desde el conteo de capturas ya existentes")
    args = parser.parse_args()

    print_banner(args.count)

    os.makedirs(args.output, exist_ok=True)

    connected_devs = get_connected_devices()
    target_serial = args.serial or SETTINGS.get("device", {}).get("serial", "5fba8618")
    has_target_device = target_serial in connected_devs or len(connected_devs) > 0

    if args.dry_run:
        print("\n--- PLAN DE EJECUCIÓN (--dry-run activo) ---")
        print(f"  • Objetivo de capturas : {args.count}")
        print(f"  • Carpeta de salida    : {os.path.abspath(args.output)}")
        print(f"  • Serial configurado   : {target_serial}")
        print(f"  • Dispositivos ADB     : {connected_devs if connected_devs else 'Ninguno detectado'}")
        print(f"  • Estado dispositivo   : {'Conectado' if has_target_device else 'No detectado'}")
        print(f"  • Rango de pausa Next  : {args.delay_min}s - {args.delay_max}s")
        print(f"  • Máximo de fallos     : {args.max_fails}")
        print(f"  • Modo despliegue      : DESACTIVADO (DEPLOY_ENABLED = {DEPLOY_ENABLED})")
        print("Saliendo sin conectar ni interactuar con el dispositivo.\n")
        return

    # Contar capturas existentes si resume está activo
    existing_files = [f for f in os.listdir(args.output) if f.startswith("auto_") and f.endswith(".png")]
    start_idx = 1
    if args.resume:
        start_idx = len(existing_files) + 1
        log.info("Modo --resume activo: Se encontraron %d capturas previas. Iniciando en índice %d.", len(existing_files), start_idx)
        if start_idx > args.count:
            log.info("El número de capturas existentes (%d) ya alcanza o supera el objetivo (%d). Nada que hacer.", len(existing_files), args.count)
            return

    if not has_target_device:
        log.critical("No se detectó el dispositivo ADB '%s'. Conecta el dispositivo o verifica 'adb devices'.", target_serial)
        sys.exit(1)

    runner = AutoCaptureRunner(serial=target_serial, output_dir=args.output)
    if not runner.device.connect():
        log.critical("No se pudo establecer conexión ADB con el dispositivo %s.", target_serial)
        sys.exit(1)

    ts_session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    consecutive_fails = 0
    successful_captures = 0

    log.info("═" * 60)
    log.info("🚀 INICIANDO AUTO-CAPTURA CONTINUA (Aldea en aldea vía botón Next)")
    log.info("   Meta: %d capturas | Pausa entre aldeas: %.1fs - %.1fs", args.count, args.delay_min, args.delay_max)
    log.info("═" * 60)

    try:
        # Asegurar ingreso inicial a la búsqueda de rivales si aún no estamos en combate
        if not runner.is_in_match():
            if not runner.enter_matchmaking():
                log.critical("No se pudo ingresar a la búsqueda de rivales. Abortando.")
                return

        current_idx = start_idx
        while current_idx <= args.count and consecutive_fails < args.max_fails:
            log.info("═" * 60)
            log.info("📸 CAPTURANDO ALDEA [%04d/%04d]...", current_idx, args.count)
            log.info("═" * 60)

            # Si salimos de batalla por algún motivo (desconexión, etc.), reingresar
            if not runner.is_in_match():
                log.warning("No estamos en modo búsqueda de rivales. Reingresando...")
                if not runner.enter_matchmaking():
                    consecutive_fails += 1
                    log.warning("Fallo al reingresar a búsqueda (%d/%d fallos consecutivos).", consecutive_fails, args.max_fails)
                    time.sleep(2.0)
                    continue

            # 1. Capturar aldea actual (zoom out + screenshot + metadata)
            try:
                ok_cap = runner.capture_current_village(current_idx, ts_session)
                if ok_cap:
                    successful_captures += 1
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1
                    log.warning("Fallo al capturar aldea %d (%d/%d fallos).", current_idx, consecutive_fails, args.max_fails)
                    continue
            except Exception as e:
                consecutive_fails += 1
                log.error("Excepción en captura [%04d]: %s (%d/%d fallos)", current_idx, e, consecutive_fails, args.max_fails)
                continue

            # Si ya se alcanzó la última captura solicitada, salimos del bucle (no necesitamos pulsar Next)
            if current_idx >= args.count:
                log.info("🎯 Se alcanzó el total solicitado de %d capturas.", args.count)
                break

            # 2. Pausa humanizada antes de pasar a la siguiente aldea
            log.info("⏩ Avanzando a la siguiente aldea usando botón 'Next'...")
            HumanBehavior.sleep_range(args.delay_min, args.delay_max)

            # 3. Pulsar Next y esperar a que disipen las nubes
            ok_next = runner.next_village(timeout=90)
            if ok_next:
                current_idx += 1
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                log.warning("Fallo al avanzar con Next (%d/%d fallos consecutivos).", consecutive_fails, args.max_fails)
                # Revisar si apareció un popup (ej. sin oro suficiente, reconexión)
                runner.proc_mgr.handle_popup()

    except KeyboardInterrupt:
        log.warning("\n⏹️ Interrupción manual por usuario (Ctrl+C).")
    except Exception as e:
        log.critical("Error fatal no controlado en bucle de captura: %s", e)
    finally:
        # Al finalizar la sesión completa o en caso de interrupción/error, volver a la aldea principal
        log.info("🏳️ Finalizando sesión: retornando a la aldea principal...")
        try:
            runner.surrender_and_return_home()
        except Exception as e:
            log.warning("No se pudo completar retorno a casa automático: %s", e)

    log.info("═" * 60)
    log.info("🏆 AUTO-CAPTURA FINALIZADA: %d/%d exitosas guardadas en %s", successful_captures, args.count, args.output)
    log.info("═" * 60)


if __name__ == "__main__":
    main()
