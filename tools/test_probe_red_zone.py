"""
tools/test_probe_red_zone.py
Herramienta de diagnóstico para capturar y analizar la retícula roja de exclusión
que dibuja Clash of Clans al tocar en zona prohibida con tropas seleccionadas.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import numpy as np
from core.device import Device
from core.vision import Vision
from core.logger import get_logger
from vision.army_checker import ArmyChecker

log = get_logger("probe_red_zone")


def run_probe_test():
    d = Device("5fba8618")
    v = Vision()
    army = ArmyChecker(v)

    log.info("Conectado a dispositivo. Verificando estado en aldea...")
    screen = d.screenshot()

    # Comprobar botón Attack
    tpl_attack = "assets/buttons/attack_btn.png"
    pos = v.find_template(screen, tpl_attack, threshold=0.74, zona="ESQUINA_INF_IZQ")
    if not pos:
        log.error("No se encontró el botón de ataque en la aldea principal.")
        return

    log.info("Pulsando Attack en %s...", pos)
    d.tap(*pos)
    time.sleep(1.0)

    # Buscar Find Match
    tpl_find = "assets/buttons/find_match_btn.png"
    pos_find = v.wait_for_template(d.screenshot, tpl_find, timeout=6, zona="COMPLETA")
    if not pos_find:
        log.error("No se encontró Find Match.")
        return

    log.info("Pulsando Find Match en %s...", pos_find)
    d.tap(*pos_find)
    time.sleep(1.0)

    # Confirm attack si aparece
    tpl_confirms = [
        "assets/buttons/attack_btn_confirm.png",
        "assets/buttons/okay_btn.png",
        "assets/buttons/confirm_attack.png",
    ]
    for _ in range(12):
        s = d.screenshot()
        m = v.find_any(s, tpl_confirms, threshold=0.72, zona="MITAD_INFERIOR")
        if m:
            log.info("Confirmando ataque en (%d, %d)...", m[1], m[2])
            d.tap(m[1], m[2])
            break
        # Si ya aparece el botón de surrender o next, estamos en batalla
        if v.find_template(s, "assets/buttons/next_btn.png", threshold=0.70, zona="ESQUINA_INF_DER"):
            break
        time.sleep(0.3)

    log.info("Esperando llegada a aldea rival...")
    tpl_surrender = "assets/buttons/surrender_battle.png"
    for _ in range(25):
        s = d.screenshot()
        if v.find_template(s, tpl_surrender, threshold=0.70, zona="ESQUINA_INF_IZQ"):
            log.info("Aldea rival cargada.")
            break
        time.sleep(0.5)

    time.sleep(1.0)
    # Zoom out
    log.info("Ejecutando Zoom Out centrado...")
    d.zoom_out(center=(1200, 540))
    time.sleep(1.5)

    # Captura baseline sin toque
    screen_base = d.screenshot()
    cv2.imwrite("logs/red_zone_baseline.png", screen_base)
    log.info("📸 Capturada imagen base sin toque: logs/red_zone_baseline.png")

    # Seleccionar duendes
    goblin_pos = army.find_goblin_in_battle_bar(screen_base)
    if not goblin_pos:
        goblin_pos = (494, 979)
    log.info("Seleccionando duendes en %s...", goblin_pos)
    d.tap(*goblin_pos)
    time.sleep(0.3)

    # Toque de prueba en el centro (1200, 540) - ZONA PROHIBIDA
    log.info("👉 Realizando toque de prueba en zona prohibida (1200, 540)...")
    d.tap(1200, 540)

    # Capturas rápidas para atrapar el destello rojo
    t0 = time.time()
    s1 = d.screenshot()
    dt1 = time.time() - t0
    cv2.imwrite("logs/red_zone_flash_1.png", s1)

    time.sleep(0.15)
    s2 = d.screenshot()
    dt2 = time.time() - t0
    cv2.imwrite("logs/red_zone_flash_2.png", s2)

    log.info("📸 Capturadas imágenes de destello a %.2fs y %.2fs", dt1, dt2)

    # Rendición limpia
    log.info("Iniciando rendición limpia...")
    pos_surr = v.wait_for_template(d.screenshot, tpl_surrender, timeout=4, threshold=0.72, zona="ESQUINA_INF_IZQ")
    if not pos_surr:
        pos_surr = v.find_template(d.screenshot(), "assets/buttons/end_battle_btn.png", threshold=0.70)
    if pos_surr:
        d.tap(*pos_surr)
        time.sleep(0.8)

    # Confirm
    tpl_confirm_end = "assets/buttons/confirm_end.png"
    pos_conf = v.wait_for_template(d.screenshot, tpl_confirm_end, timeout=4, threshold=0.70, zona="CENTRO")
    if not pos_conf:
        pos_conf = v.find_template(d.screenshot(), "assets/buttons/okay_btn.png", threshold=0.70, zona="CENTRO")
    if pos_conf:
        d.tap(*pos_conf)
        time.sleep(0.8)

    # Return Home
    tpl_ret = "assets/buttons/return_home_btn.png"
    pos_ret = v.wait_for_template(d.screenshot, tpl_ret, timeout=6, threshold=0.72, zona="MITAD_INFERIOR")
    if pos_ret:
        d.tap(*pos_ret)
        log.info("Pulsado Return Home. Esperando regreso a aldea...")
        time.sleep(3.5)

    log.info("✅ Prueba de sondeo completada.")


if __name__ == "__main__":
    run_probe_test()
