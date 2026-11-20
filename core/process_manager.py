"""
core/process_manager.py
Supervisor de proceso, reconexión ante caídas de red y cierre de popups.
"""
import time
from core.device import Device
from core.vision import Vision
from core.logger import get_logger
from humanizer.behavior import HumanBehavior

log = get_logger("process_manager")


class ProcessManager:
    def __init__(self, device: Device, vision: Vision):
        self.device = device
        self.vision = vision

        # Botones de ataque para confirmar aldea lista
        self.tpl_attack_btns = [
            "assets/buttons/attack_btn.png",
            "assets/buttons/attack_btn_event.png",
            "assets/buttons/battle_text.png",
        ]

        # Popups genéricos y avisos
        self.tpl_okay_btn = "assets/buttons/okay_btn.png"
        self.tpl_close_btn = "assets/buttons/close_btn.png"

        # Modales, pestañas y cajones emergentes
        self.tpl_red_x = "assets/buttons/red_x_close.png"
        self.tpl_blue_x = "assets/buttons/blue_x_close.png"
        self.tpl_train_close = "assets/buttons/train_close_btn.png"

        # Star Bonus y Apertura Interactiva de Cofres
        self.tpl_claim_btns = [
            "assets/buttons/claim_btn.png",
            "assets/buttons/claim_text.png",
        ]
        self.tpl_chest_indicators = [
            "assets/buttons/chest_tap_text.png",
            "assets/buttons/chest_handle.png",
        ]
        self.tpl_continue_btns = [
            "assets/buttons/continue_btn.png",
            "assets/buttons/continue_text.png",
        ]
        self.tpl_claim_rewards = [
            "assets/buttons/claim_reward_btn.png",
            "assets/buttons/claim_reward_text.png",
        ]

    def handle_popup(self, screenshot=None) -> bool:
        """
        Detecta y cierra popups genéricos, modales, menús de tienda/ajustes o avisos.
        Retorna True si cerró algún elemento emergente, False si la pantalla estaba limpia.
        """
        screen = screenshot if screenshot is not None else self.device.screenshot()

        # 1. Menús con botón rojo de cerrar (X) en la esquina superior derecha (Tienda, Ajustes, Layout Editor)
        # Nota: Ajustes/Tienda ignoran la tecla BACK nativa, se requiere toque directo en la X.
        red_x_pos = self.vision.find_template(screen, self.tpl_red_x, threshold=0.78, zona="ESQUINA_SUP_DER")
        if not red_x_pos:
            red_x_pos = self.vision.find_template(screen, self.tpl_red_x, threshold=0.78, zona="MITAD_SUPERIOR")
        if red_x_pos:
            log.info("🚨 Botón Red X detectado en %s. Pulsando para cerrar menú...", red_x_pos)
            HumanBehavior.reaction_delay()
            self.device.tap(*red_x_pos)
            time.sleep(1.0)
            return True

        # 2. Modales centrados con botón azul de cerrar (Perfil, Clan, etc.)
        blue_x_pos = self.vision.find_template(screen, self.tpl_blue_x, threshold=0.74, zona="MITAD_SUPERIOR")
        if blue_x_pos:
            log.info("🚨 Modal emergente detectado (Blue X) en %s. Cerrando con tecla BACK...", blue_x_pos)
            HumanBehavior.reaction_delay()
            self.device.d.press("back")
            time.sleep(1.0)
            return True

        # 3. Modal de entrenamiento de tropas (Train Troops)
        train_pos = self.vision.find_template(screen, self.tpl_train_close, threshold=0.75, zona="MITAD_SUPERIOR")
        if train_pos:
            log.info("🚨 Modal Entrenar Tropas detectado en %s. Cerrando con tecla BACK...", train_pos)
            HumanBehavior.reaction_delay()
            self.device.d.press("back")
            time.sleep(1.0)
            return True

        # 4. Pestaña de cierre adicional si existe
        close_btn_pos = self.vision.find_template(screen, self.tpl_close_btn, threshold=0.75, zona="MITAD_SUPERIOR")
        if close_btn_pos:
            log.info("🚨 Botón de cerrar detectado en %s. Cerrando...", close_btn_pos)
            HumanBehavior.reaction_delay()
            self.device.d.press("back")
            time.sleep(1.0)
            return True

        # 5. Avisos informativos con botón Okay (mejoras completadas, noticias)
        okay_pos = self.vision.find_template(screen, self.tpl_okay_btn, threshold=0.74, zona="COMPLETA")
        if okay_pos:
            log.info("🚨 Aviso / Mejora detectada (Okay) en %s. Confirmando...", okay_pos)
            HumanBehavior.reaction_delay()
            self.device.tap(*okay_pos)
            time.sleep(1.0)
            return True

        # 6. Modal de Star Bonus con botón 'Claim'
        claim_match = self.vision.find_any(screen, self.tpl_claim_btns, threshold=0.74, zona="MITAD_INFERIOR")
        if claim_match:
            log.info("⭐ Modal Star Bonus detectado (Claim) en (%d, %d). Reclamando...", claim_match[1], claim_match[2])
            HumanBehavior.reaction_delay()
            self.device.tap(claim_match[1], claim_match[2])
            time.sleep(1.8)
            # Tras pulsar Claim, puede abrirse la pantalla interactiva del cofre
            self.handle_chest_reward_sequence(wait_for_chest=True)
            return True

        # 7. Pantalla interactiva de apertura de cofres (Tap! en el cofre) o recompensa pendiente
        if self.handle_chest_reward_sequence(screen, wait_for_chest=False):
            return True

        # 8. Botón 'Claim Reward' de cofre de evento
        reward_match = self.vision.find_any(screen, self.tpl_claim_rewards, threshold=0.74, zona="MITAD_INFERIOR")
        if reward_match:
            log.info("🎁 Botón 'Claim Reward' de evento detectado en (%d, %d). Reclamando...", reward_match[1], reward_match[2])
            HumanBehavior.reaction_delay()
            self.device.tap(reward_match[1], reward_match[2])
            time.sleep(1.8)
            self.handle_chest_reward_sequence(wait_for_chest=True)
            return True

        return False

    def handle_chest_reward_sequence(self, screen=None, wait_for_chest: bool = False) -> bool:
        """
        Gestiona la secuencia de cofres unificada (Star Bonus / Claim Reward de fin de batalla):
        1. Espera la pantalla interactiva del cofre (Tap! o asa del cofre) según wait_for_chest.
        2. Da 4 toques al centro de la pantalla sobre el cofre.
        3. Espera la animación de apertura (~2.0s).
        4. Usa wait_for_template para detectar el botón verde 'Continue' y lo pulsa.
        5. Repite automáticamente si aparece otro cofre consecutivo (ej. 3 o 2 estrellas).
        """
        cur_screen = screen if screen is not None else self.device.screenshot()
        handled_any = False

        cx = int(self.device.width * 0.50)
        cy = int(self.device.height * 0.48)

        # Si ya estamos en la pantalla de recompensa directamente con botón Continue
        cont_match = self.vision.find_any(cur_screen, self.tpl_continue_btns, threshold=0.74, zona="MITAD_INFERIOR")
        if cont_match:
            log.info("🎁 Botón Continue de recompensa detectado en (%d, %d). Pulsando...", cont_match[1], cont_match[2])
            HumanBehavior.reaction_delay()
            self.device.tap(cont_match[1], cont_match[2])
            time.sleep(1.5)
            return True

        for cofre_idx in range(1, 4):
            # Comprobar primero si cur_screen actual ya tiene cofre (zona COMPLETA)
            chest_match = self.vision.find_any(
                cur_screen, self.tpl_chest_indicators, threshold=0.70, zona="COMPLETA"
            )

            # Si no está en cur_screen:
            if not chest_match:
                # Si wait_for_chest=False y no hemos abierto ningún cofre aún, salir de inmediato sin esperas ni capturas
                if not wait_for_chest and not handled_any:
                    break

                # Tiempo de espera activo: 3.0s si wait_for_chest=True en primera vuelta, o máx 2.5s para cofres encadenados
                wait_timeout = 3.0 if (wait_for_chest and not handled_any) else 2.5
                t_wait_chest = time.monotonic()
                while time.monotonic() - t_wait_chest < wait_timeout:
                    cur_screen = self.device.screenshot()
                    # Verificar si ya saltó a la recompensa (Continue)
                    cont_match = self.vision.find_any(cur_screen, self.tpl_continue_btns, threshold=0.74, zona="MITAD_INFERIOR")
                    if cont_match:
                        log.info("🎁 Botón Continue detectado directamente en (%d, %d). Pulsando...", cont_match[1], cont_match[2])
                        HumanBehavior.reaction_delay()
                        self.device.tap(cont_match[1], cont_match[2])
                        time.sleep(1.8)
                        cur_screen = self.device.screenshot()
                        handled_any = True
                        break

                    chest_match = self.vision.find_any(
                        cur_screen, self.tpl_chest_indicators, threshold=0.70, zona="COMPLETA"
                    )
                    if chest_match:
                        break
                    time.sleep(0.35)

            if not chest_match:
                # Si wait_for_chest=True y tras esperar 3.0s no aparece Continue ni matchean los templates
                # del cofre pero tampoco estamos en la aldea (not is_village_ready()), ejecuta igualmente
                # los 4 taps al centro (cx, cy) y espera Continue como fallback seguro.
                if wait_for_chest and not handled_any and not self.is_village_ready(cur_screen):
                    log.warning(
                        "⚠️ Cofre/Continue no detectado por template tras 3.0s y la aldea no está lista. "
                        "Ejecutando 4 taps al centro (%d, %d) como fallback seguro...",
                        cx, cy
                    )
                else:
                    # No hay más cofres pendientes de abrir
                    break

            t_x = chest_match[1] if chest_match else cx
            t_y = chest_match[2] if chest_match else cy
            log.info("📦 Cofre de recompensa detectado (#%d) en (%d, %d). Dando 4 taps al centro...", cofre_idx, t_x, t_y)
            handled_any = True

            # Dar 4 taps al centro sobre el cofre
            for i in range(4):
                self.device.tap(cx, cy, humanize=False)
                time.sleep(0.35)

            log.info("⏳ Esperando animación de apertura del cofre...")
            time.sleep(2.0)

            # Esperar la aparición del botón verde 'Continue' con wait_for_template
            log.info("🔍 Esperando botón verde 'Continue'...")
            cont_pos = None
            t0 = time.monotonic()
            while time.monotonic() - t0 < 8.0:
                s_gift = self.device.screenshot()
                cont_match = self.vision.find_any(
                    s_gift, self.tpl_continue_btns, threshold=0.74, zona="MITAD_INFERIOR"
                )
                if cont_match:
                    cont_pos = (cont_match[1], cont_match[2])
                    break
                time.sleep(0.30)

            if cont_pos:
                log.info("✨ Botón 'Continue' detectado en %s. Pulsando para continuar...", cont_pos)
                HumanBehavior.reaction_delay()
                self.device.tap(*cont_pos)
                time.sleep(1.8)
            else:
                log.warning("⚠️ Timeout esperando botón Continue; intentando toque central de confirmación...")
                self.device.tap(cx, int(self.device.height * 0.88), humanize=False)
                time.sleep(1.8)

            # Muestrear siguiente pantalla por si encadena otro cofre consecutivo
            try:
                cur_screen = self.device.screenshot()
                if cur_screen is None:
                    break
            except Exception:
                break

        return handled_any

    def is_village_ready(self, screen=None) -> bool:
        """Determina si la aldea está visible y lista reconociendo cualquier variante del botón Attack/Battle."""
        s = screen if screen is not None else self.device.screenshot()
        return bool(self.vision.find_any(s, self.tpl_attack_btns, threshold=0.74, zona="ESQUINA_INF_IZQ"))

    def dismiss_startup_popups(self, max_checks: int = 5):
        """
        Al entrar al juego, busca activamente y cierra mensajes de mejoras, modales
        o avisos antes de que el bot inicie cualquier acción en la aldea.
        """
        log.info("🔍 Comprobando y cerrando modales o avisos de inicio...")
        for _ in range(max_checks):
            screen = self.device.screenshot()
            # Si el botón de ataque ya está visible y libre de popups, salir inmediatamente
            if self.is_village_ready(screen):
                if not self.handle_popup(screen):
                    break
            elif not self.handle_popup(screen):
                break

    def ensure_game_ready(self, timeout: int = 45) -> bool:
        """
        Asegura que el juego esté abierto y listo en la aldea principal:
        1. Inicia Clash of Clans si no está corriendo.
        2. Cierra activamente mensajes de mejoras, modales o pestañas emergentes.
        3. Deselecciona cualquier edificio o muro marcado tocando zona neutral.
        4. Confirma la presencia del botón Attack para continuar.
        """
        if not self.device.is_app_running():
            log.warning("⚠️ Clash of Clans no está activo. Iniciando aplicación...")
            self.device.open_app()
            time.sleep(6)

        # 1. Al entrar al juego antes de hacer cualquier cosa, cerrar mensajes y pestañas
        self.dismiss_startup_popups()

        inicio = time.monotonic()
        while time.monotonic() - inicio < timeout:
            screen = self.device.screenshot()

            # 1) Si el botón de ataque ya está visible y sin popups encima, la aldea está lista:
            if self.is_village_ready(screen):
                # Verificar rápidamente si hay un popup residual bloqueando
                if not self.handle_popup(screen):
                    log.info("🏰 Aldea lista y confirmada (botón Attack/Battle visible).")
                    return True
                time.sleep(1.0)
                continue

            # 2) Si el botón Attack está cubierto, intentar cerrar el popup que lo bloquea
            if self.handle_popup(screen):
                time.sleep(1.0)
                continue

            # Si el botón Attack aún no es visible tras unos segundos:
            elapsed = time.monotonic() - inicio
            if elapsed > 2.0:
                # 1) Descartar overlays o cajones emergentes de pantalla completa con tecla BACK
                log.debug("Intentando tecla BACK para descartar overlay de pantalla completa...")
                self.device.d.press("back")
                time.sleep(0.6)

                # 2) Deseleccionar cualquier edificio o muro marcado tocando césped neutral
                log.debug("Deseleccionando edificio/muro marcado tocando césped neutral...")
                safe_x = int(self.device.width * 0.35)
                safe_y = int(self.device.height * 0.40)
                self.device.tap(safe_x, safe_y)
                time.sleep(0.6)

            time.sleep(0.8)

        log.error("⏱️ Juego bloqueado tras %ds. Ejecutando force-stop y reinicio...", timeout)
        self.device.restart_app()
        time.sleep(8)
        return False
