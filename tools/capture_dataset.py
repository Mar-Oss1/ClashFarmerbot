"""
tools/capture_dataset.py
Toma capturas del dispositivo conectándose por ADB para armar dataset sin etiquetar.
"""
import argparse
import os
import sys
import time
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.device import Device
from core.logger import get_logger
from humanizer.behavior import HumanBehavior

log = get_logger("capture_dataset")


def main():
    parser = argparse.ArgumentParser(description="Captura imágenes raw del dispositivo para dataset CoC")
    parser.add_argument("--count", type=int, default=50, help="Número de capturas a tomar (default: 50)")
    parser.add_argument("--out_dir", type=str, default="datasets/coc/images/raw", help="Directorio de salida")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dev = Device()
    if not dev.connect():
        log.critical("No se pudo conectar al dispositivo por ADB.")
        return

    log.info("Iniciando captura de %d imágenes en %s...", args.count, args.out_dir)

    for i in range(1, args.count + 1):
        try:
            screen = dev.screenshot()
            ts = int(time.time() * 1000)
            filename = f"cap_{ts}.png"
            out_path = os.path.join(args.out_dir, filename)
            cv2.imwrite(out_path, screen)
            log.info("[%d/%d] Captura guardada: %s", i, args.count, out_path)
        except Exception as e:
            log.error("[%d/%d] Error tomando captura: %s", i, args.count, e)

        if i < args.count:
            # Pausa con jitter entre capturas
            HumanBehavior.sleep_range(1.5, 3.5)

    log.info("Proceso de captura finalizado. Total guardadas en %s", args.out_dir)


if __name__ == "__main__":
    main()
