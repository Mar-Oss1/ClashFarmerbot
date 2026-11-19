"""
Script para prueba en vivo de detección de muros y cruces con U-Net durante un ataque real.
1. Inicia búsqueda de rival desde la aldea principal.
2. Al entrar a la aldea enemiga, toma captura de pantalla completa (2400x1080).
3. Se rinde inmediatamente y regresa a la aldea para no gastar tropas ni tiempo.
4. Aplica WallSegDetector (U-Net) + detección de segmentos y cruces (Hough / Raycast).
5. Genera overlay visual completo y guarda en scratch/live_attack_wall_detection.png.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import cv2
import numpy as np

from core.device import Device
from core.vision import Vision
from core.ui_cache import UICache
from vision.wall_seg_detector import WallSegDetector, build_cyan_overlay
from vision.wall_raycast import detect_global_wall_segments, find_wall_junctions, reset_iso_angles_cache

def main():
    print("=== INICIANDO PRUEBA DE DETECCIÓN DE MUROS EN ATAQUE REAL ===")
    os.makedirs("scratch", exist_ok=True)
    os.makedirs("logs/yolo", exist_ok=True)
    
    device = Device()
    device.connect()
    vision = Vision()
    ui_cache = UICache()
    
    # 1. Verificar estado inicial en la aldea principal
    print("[1/5] Verificando aldea principal...")
    screen = device.screenshot()
    atk_pos = vision.find_any(screen, ["assets/buttons/attack_btn.png", "assets/buttons/attack_btn_event.png", "assets/buttons/attack_btn_chest.png"], threshold=0.72)
    if not atk_pos:
        print("ERROR: No se detecta el botón de ataque en la aldea principal.")
        return False
    
    print(f"Botón de ataque detectado en: {atk_pos[1:]}")
    device.tap(atk_pos[1], atk_pos[2])
    time.sleep(1.0)
    
    # 2. Pulsar Find Match
    print("[2/5] Buscando botón Find Match...")
    find_pos = vision.wait_for_template(device.screenshot, "assets/buttons/find_match_btn.png", timeout=6)
    if not find_pos:
        print("ERROR: No apareció el botón Find Match.")
        return False
    
    print(f"Pulsando Find Match en: {find_pos}")
    device.tap(find_pos[0], find_pos[1])
    time.sleep(0.8)
    
    # Confirmar ataque si aparece ventana de confirmación
    t_start = time.monotonic()
    confirm_pos = None
    while time.monotonic() - t_start < 4.0:
        s_conf = device.screenshot()
        m_conf = vision.find_any(s_conf, ["assets/buttons/confirm_attack_btn.png", "assets/buttons/attack_btn_confirm.png"], threshold=0.72)
        if m_conf:
            confirm_pos = (m_conf[1], m_conf[2])
            break
        time.sleep(0.25)
        
    if confirm_pos:
        print(f"Confirmando ataque en: {confirm_pos}")
        device.tap(confirm_pos[0], confirm_pos[1])
    else:
        # Fallback UI cache
        cached_conf = ui_cache.get_pos("confirm_attack_btn", device.width, device.height)
        print(f"Usando UI cache para confirmar ataque en: {cached_conf}")
        device.tap(cached_conf[0], cached_conf[1])
        
    # 3. Esperar a entrar a la aldea rival (esperar next_btn o surrender_battle/end_battle)
    print("[3/5] Esperando carga de la aldea rival (nubes / matchmaking)...")
    battle_screen = None
    t_match = time.monotonic()
    while time.monotonic() - t_match < 25.0:
        s_battle = device.screenshot()
        has_next = vision.find_template(s_battle, "assets/buttons/next_btn.png", threshold=0.70, zona="ESQUINA_INF_DER")
        has_surr = vision.find_any(s_battle, ["assets/buttons/surrender_battle.png", "assets/buttons/end_battle.png"], threshold=0.70, zona="ESQUINA_INF_IZQ")
        if has_next or has_surr:
            print(f"¡Aldea rival encontrada! (Next: {has_next}, Surrender: {has_surr})")
            # Esperar 1.2 segundos para que se disipen por completo las nubes y el HUD cargue
            time.sleep(1.2)
            battle_screen = device.screenshot()
            break
        time.sleep(0.4)
        
    if battle_screen is None:
        print("ERROR: Timeout esperando aldea rival.")
        return False
        
    # Guardar captura cruda inmediatamente
    raw_path = "scratch/live_attack_village_raw.png"
    cv2.imwrite(raw_path, battle_screen)
    print(f"Captura de aldea rival guardada en: {raw_path}")
    
    # 4. Secuencia de rendición inmediata y segura
    print("[4/5] Ejecutando rendición inmediata para volver a casa...")
    surrender_pos = vision.find_any(battle_screen, ["assets/buttons/end_battle.png", "assets/buttons/surrender_battle.png"], threshold=0.68, zona="ESQUINA_INF_IZQ")
    if surrender_pos:
        surr_pt = (surrender_pos[1], surrender_pos[2])
    else:
        surr_pt = ui_cache.get_pos("surrender_btn", device.width, device.height)
    print(f"Pulsando rendición en: {surr_pt}")
    device.tap(surr_pt[0], surr_pt[1], humanize=False)
    time.sleep(0.4)
    
    # Confirmar diálogo de rendición
    s_dlg = device.screenshot()
    conf_end = vision.find_any(s_dlg, ["assets/buttons/confirm_end.png", "assets/buttons/dialog_confirm_btn.png"], threshold=0.68, zona="CENTRO")
    if conf_end:
        end_pt = (conf_end[1], conf_end[2])
    else:
        end_pt = ui_cache.get_pos("confirm_surrender_btn", device.width, device.height)
    print(f"Confirmando rendición en: {end_pt}")
    device.tap(end_pt[0], end_pt[1], humanize=False)
    time.sleep(1.0)
    
    # Esperar Return Home
    print("Esperando Return Home...")
    t_ret = time.monotonic()
    returned = False
    while time.monotonic() - t_ret < 10.0:
        s_ret = device.screenshot()
        ret_btn = vision.find_template(s_ret, "assets/buttons/return_home.png", threshold=0.70, zona="MITAD_INFERIOR")
        if ret_btn:
            print(f"Pulsando Return Home en: {ret_btn}")
            device.tap(ret_btn[0], ret_btn[1], humanize=False)
            time.sleep(2.5)
            returned = True
            break
        time.sleep(0.3)
        
    if not returned:
        cached_ret = ui_cache.get_pos("return_home_btn", device.width, device.height)
        print(f"Fallback Return Home con UI Cache en: {cached_ret}")
        device.tap(cached_ret[0], cached_ret[1], humanize=False)
        time.sleep(2.5)
        
    print("Verificando regreso a aldea principal...")
    s_home = device.screenshot()
    home_ok = vision.find_any(s_home, ["assets/buttons/attack_btn.png", "assets/buttons/attack_btn_event.png"], threshold=0.72)
    if home_ok:
        print("✓ Regreso seguro confirmado a la aldea principal.")
    else:
        print("Aviso: Comprobación de aldea principal no inmediata, esperando 2s...")
        time.sleep(2.0)
        
    # 5. Procesar la imagen con U-Net y detección de cruces
    print("[5/5] Analizando muros con modelo U-Net y detección de cruces/segmentos...")
    reset_iso_angles_cache()
    
    detector = WallSegDetector(model_path="models/wall_seg.onnx")
    wall_mask = detector.predict(battle_screen)
    wall_px = int(np.count_nonzero(wall_mask))
    total_px = battle_screen.shape[0] * battle_screen.shape[1]
    wall_pct = (wall_px / total_px) * 100.0
    print(f"U-Net: {wall_px:,} píxeles de muro detectados ({wall_pct:.2f}% de la pantalla)")
    
    # Detectar segmentos geométricos
    iso_angles, segments = detect_global_wall_segments(battle_screen)
    print(f"Segmentos de muro detectados: {len(segments)} (Ángulos isométricos dominantes: {iso_angles})")
    
    # Detectar cruces / uniones geométricas
    junctions = find_wall_junctions(segments, iso_angles=iso_angles)
    cross_count = sum(1 for j in junctions if j.get('type') == 'cross')
    tee_count = sum(1 for j in junctions if j.get('type') == 'tee')
    corner_count = sum(1 for j in junctions if j.get('type') == 'corner')
    print(f"Cruces detectados: {len(junctions)} total (Cruces +: {cross_count}, Cruces T: {tee_count}, Esquinas: {corner_count})")
    
    # Construir visualización compuesta de alta fidelidad
    overlay = build_cyan_overlay(battle_screen, wall_mask)
    
    # Dibujar segmentos detectados (líneas naranjas finas)
    for seg in segments:
        x1, y1, x2, y2 = seg
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 140, 255), 2, cv2.LINE_AA)
        
    # Dibujar cruces e intersecciones
    for j in junctions:
        pt = j.get('pt')
        jtype = j.get('type')
        if not pt:
            continue
        cx, cy = int(pt[0]), int(pt[1])
        if jtype == 'cross':
            # Cruz '+': Magenta brillante con borde blanco
            cv2.circle(overlay, (cx, cy), 9, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, "+", (cx - 6, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        elif jtype == 'tee':
            # Cruz en T: Amarillo brillante con borde negro
            cv2.circle(overlay, (cx, cy), 8, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 10, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, "T", (cx - 5, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            # Esquina: Verde lima
            cv2.circle(overlay, (cx, cy), 6, (0, 255, 100), -1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), 8, (0, 0, 0), 1, cv2.LINE_AA)
            
    # Añadir banner informativo en la parte superior
    banner_h = 60
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], banner_h), (20, 20, 20), -1)
    cv2.line(overlay, (0, banner_h), (overlay.shape[1], banner_h), (0, 255, 255), 2)
    
    text_info = f"U-NET WALL SEGMENTATION | Wall Pixels: {wall_px:,} ({wall_pct:.2f}%) | Segments: {len(segments)} | Junctions: {len(junctions)} (Cross +: {cross_count}, Tee T: {tee_count}, Corner: {corner_count})"
    cv2.putText(overlay, text_info, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    
    out_path = "scratch/live_attack_wall_detection.png"
    cv2.imwrite(out_path, overlay)
    cv2.imwrite("logs/yolo/live_attack_wall_detection.png", overlay)
    print(f"✓ Resultado visual guardado en: {out_path}")
    print(f"✓ Copia de respaldo en: logs/yolo/live_attack_wall_detection.png")
    return True

if __name__ == "__main__":
    main()
