import argparse
import sys
from core.device import Device
from core.vision import Vision
from core.state_machine import ClashStateMachine
from core.logger import get_logger

log = get_logger("main")


def main():
    parser = argparse.ArgumentParser(description="ClashFarmerbot Runner")
    parser.add_argument("--attacks", type=int, default=None, help="Número máximo de ataques antes de detenerse")
    args = parser.parse_args()

    device = Device()
    if not device.connect():
        log.critical("No se pudo establecer conexión ADB con el dispositivo. Abortando.")
        sys.exit(1)

    vision = Vision(threshold=0.80)
    vision.preload_templates("assets")

    bot = ClashStateMachine(device, vision)
    bot.run_forever(max_attacks=args.attacks)


if __name__ == "__main__":
    main()

