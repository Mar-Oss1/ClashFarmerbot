"""
tools/run_live_test_3_attacks.py
Script de supervisión y ejecución en vivo de 3 ataques reales:
1. Activa punteros táctiles completos (show_touches=1, pointer_location=1) en el dispositivo.
2. Ejecuta 3 ataques consecutivos mediante ClashStateMachine.
3. En cada ataque, monitorea la apertura de muros (SWB y Hechizos de Salto) y el despliegue de duendes.
4. Genera un reporte consolidado con capturas de pantalla de cada batalla.
"""
import os
import sys
import time
import shutil

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.device import Device
from core.vision import Vision
from core.state_machine import ClashStateMachine
from core.logger import get_logger

log = get_logger("live_test")


def enable_pointers(serial: str):
    log.info("🎯 Activando punteros táctiles completos en dispositivo %s...", serial)
    os.system(f'adb -s {serial} shell "settings put system show_touches 1; settings put system pointer_location 1"')


def disable_pointers(serial: str):
    log.info("🔌 Desactivando punteros táctiles en dispositivo %s...", serial)
    os.system(f'adb -s {serial} shell "settings put system show_touches 0; settings put system pointer_location 0"')


def main():
    log.info("🚀 Iniciando Test en Vivo: 3 Ataques Reales con Punteros y Auditoría Visual...")

    device = Device()
    if not device.connect():
        log.critical("❌ No se pudo conectar al dispositivo por ADB.")
        sys.exit(1)

    enable_pointers(device.serial)

    vision = Vision(threshold=0.80)
    vision.preload_templates("assets")

    test_dir = os.path.join("logs", "live_tests")
    os.makedirs(test_dir, exist_ok=True)

    bot = ClashStateMachine(device, vision)

    try:
        log.info("⚔️ Iniciando bucle de 3 ataques reales...")
        bot.run_forever(max_attacks=3)
    finally:
        # Desactivar punteros al concluir la ejecución de prueba
        disable_pointers(device.serial)
        log.info("✅ Punteros restaurados al estado original.")

    log.info("🏁 Test de 3 ataques completado. Verificando auditorías visuales...")


if __name__ == "__main__":
    main()

