import time
from enum import Enum, auto

from core.config import SETTINGS
from core.device import Device
from core.vision import Vision, save_debug_image_async
from core.ledger import Ledger
from core.process_manager import ProcessManager
from vision.resource_reader import ResourceReader
from vision.damage_reader import DamageReader
from vision.army_checker import ArmyChecker
from attack.deployment_strategy import DeploymentStrategy
from attack.edrag_strategy import EDragStrategy
from humanizer.behavior import HumanBehavior
from core.logger import get_logger
from core.wall_upgrader import WallUpgrader
from core.ui_cache import UICache
import random
from core import config

log = get_logger("state_machine")


class BotState(Enum):
    ENSURE_GAME_READY = auto()
    MAIN_VILLAGE = auto()
    SEARCHING_MATCH = auto()
    EVALUATING_LOOT = auto()
    ATTACKING = auto()
    WAITING_BATTLE_END = auto()
    COLLECTING_LOOT = auto()
    ERROR_RECOVERY = auto()


class ClashStateMachine:
    def __init__(self, device: Device, vision: Vision):
        self.device = device
        self.vision = vision
        self.ledger = Ledger()
        self.proc_mgr = ProcessManager(device, vision)
        self.reader = ResourceReader(vision)
        self.damage_reader = DamageReader()
        self.army = ArmyChecker(vision)
        self.deployer = DeploymentStrategy(device, vision, self.army)
        self.edrag_deployer = EDragStrategy(device, vision, army_checker=self.army)
        self.active_army_type = "sneaky_farming"
        self.wall_upgrader = WallUpgrader(device, vision)
        self.ui_cache = UICache()
        self._stop_requested = False

        self.state = BotState.ENSURE_GAME_READY

        # Templates
        self.tpl_attack = "assets/buttons/attack_btn.png"
        self.tpl_attack_btns = [
            "assets/buttons/attack_btn.png",
            "assets/buttons/attack_btn_event.png",
            "assets/buttons/battle_text.png",
        ]
        self.tpl_find_match = "assets/buttons/find_match_btn.png"
        self.tpl_attack_confirms = [
            "assets/buttons/attack_btn_confirm.png",
            "assets/buttons/confirm_attack_btn.png"
        ]
        self.tpl_attack_confirm = "assets/buttons/confirm_attack_btn.png"
        self.tpl_next = "assets/buttons/next_btn.png"
        self.tpl_end_battle = "assets/buttons/end_battle.png"
        self.tpl_surrender_battle = "assets/buttons/surrender_battle.png"
        self.tpl_confirm_end = "assets/buttons/confirm_end.png"
        self.tpl_okay_surrender = "assets/buttons/okay_btn.png"
        self.tpl_return_home = "assets/buttons/return_home.png"
        import os
        raw_return_homes = [
            "assets/buttons/return_home.png",
            "assets/buttons/return_home_event.png",
        ]
        self.tpl_return_homes = [
            p for p in raw_return_homes
            if os.path.exists(os.path.join(config.PROJECT_ROOT, p)) or os.path.exists(p)
        ]
        if not self.tpl_return_homes:
            self.tpl_return_homes = [self.tpl_return_home]
        self.tpl_claim_reward = "assets/buttons/claim_reward_btn.png"
        self.tpl_claim_rewards = [
            "assets/buttons/claim_reward_btn.png",
            "assets/buttons/claim_reward_text.png",
        ]

    def ensure_tooltip_cached(self):
        """
        Lee y almacena en caché de sesión la capacidad máxima real de almacenes,
        la producción por hora y los saldos de tesorería (In Treasury) al iniciar
        sesión o arrancar el bot. Si la lectura falla, aplica el fallback oficial TH18.
        """
        if self.ledger.tooltip_cached:
            return

        log.info("🔍 Iniciando lectura de tooltips de capacidad al arranque del bot...")
        try:
            tooltip_data = self.reader.read_all_resource_tooltips(self.device)
            self.ledger.record_tooltip_data(tooltip_data)
        except Exception as e:
            log.warning("⚠️ Error en lectura interactiva de tooltips: %s. Aplicando fallback canónico TH18.", e)
            self.ledger.set_fallback_capacities(th_level=18)

    def _wait_and_tap_return_home(self, timeout: float = 8.0) -> bool:
        """
        Tras interactuar con un modal o cofre de evento al final de la batalla,
        espera y pulsa el botón Return Home para garantizar el regreso a la aldea principal.
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            s = self.device.screenshot()
            match_ret = self.vision.find_any(s, self.tpl_return_homes, threshold=0.72, zona="MITAD_INFERIOR")
            if match_ret:
                log.info("🎯 Botón Return Home detectado en (%d, %d). Volviendo a la aldea...", match_ret[1], match_ret[2])
                self.device.tap(match_ret[1], match_ret[2], humanize=False)
                time.sleep(2.5)
                return True
            time.sleep(0.3)

        cached_ret = self.ui_cache.get_pos("return_home_btn", self.device.width, self.device.height)
        log.info("🎯 Pulsando Return Home usando UI Cache en %s...", cached_ret)
        self.device.tap(cached_ret[0], cached_ret[1], humanize=False)
        time.sleep(2.5)
        return True

    def finish_battle_sequence(self, allow_surrender: bool = False):
        """
        Secuencia completa de cierre de batalla:
        1. Comprobar si 'Return Home' ya está en pantalla.
        2. Si allow_surrender=True:
           - Rendición ultrarrápida sin esperas previas: pulsa botón de fin/rendición,
             confirma diálogo si aparece y pulsa Return Home con muestreo instantáneo.
        3. Si allow_surrender=False:
           - Espera pasiva y segura hasta que Return Home aparezca naturalmente.
        """
        log.info("🏁 Finalizando batalla y esperando Return Home (allow_surrender=%s)...", allow_surrender)
        screen = self.device.screenshot()

        # Si ya estamos en la pantalla de resultados con Return Home o Claim Reward de evento
        match_reward = self.vision.find_any(screen, self.tpl_claim_rewards, threshold=0.72, zona="MITAD_INFERIOR")
        if match_reward:
            log.info("🎁 Botón de evento 'Claim Reward' detectado en (%d, %d). Reclamando cofre...", match_reward[1], match_reward[2])
            self.device.tap(match_reward[1], match_reward[2], humanize=False)
            time.sleep(2.0)
            self.proc_mgr.handle_chest_reward_sequence(wait_for_chest=True)
            self._wait_and_tap_return_home()
            return

        match_ret = self.vision.find_any(screen, self.tpl_return_homes, threshold=0.72, zona="MITAD_INFERIOR")
        if match_ret:
            log.info("🎯 Botón Return Home detectado en (%d, %d). Volviendo a la aldea al instante...", match_ret[1], match_ret[2])
            self.device.tap(match_ret[1], match_ret[2], humanize=False)
            time.sleep(2.5)
            return

        if not allow_surrender:
            log.info("🛡️ Protección activa: Prohibido pulsar Surrender. Esperando aparición natural de Return Home / Claim Reward...")
            t_nat = time.monotonic()
            while time.monotonic() - t_nat < 15.0:
                s_nat = self.device.screenshot()
                match_reward = self.vision.find_any(s_nat, self.tpl_claim_rewards, threshold=0.72, zona="MITAD_INFERIOR")
                if match_reward:
                    log.info("🎁 Botón de evento 'Claim Reward' detectado en (%d, %d). Reclamando cofre...", match_reward[1], match_reward[2])
                    self.device.tap(match_reward[1], match_reward[2], humanize=False)
                    time.sleep(2.0)
                    self.proc_mgr.handle_chest_reward_sequence(wait_for_chest=True)
                    self._wait_and_tap_return_home()
                    return

                match_ret = self.vision.find_any(s_nat, self.tpl_return_homes, threshold=0.72, zona="MITAD_INFERIOR")
                if match_ret:
                    log.info("🎯 Botón Return Home detectado en (%d, %d). Volviendo a la aldea...", match_ret[1], match_ret[2])
                    HumanBehavior.reaction_delay()
                    self.device.tap(match_ret[1], match_ret[2])
                    time.sleep(3.5)
                    return
                time.sleep(0.30)

            log.warning("⚠️ Return Home / Claim Reward no apareció tras espera; intentando tecla BACK suave...")
            self.device.d.press("back")
            time.sleep(2.0)
            return

        # ⚡ MODO RENDICIÓN ULTRARRÁPIDA (allow_surrender=True)
        # Pulsar de inmediato el botón de fin de batalla (End Battle o Surrender) mediante UI Cache
        surrender_pos = self.vision.find_template_cached(
            screen, self.tpl_end_battle, cache_name="surrender_btn", ui_cache=self.ui_cache,
            threshold=0.70, zona="ESQUINA_INF_IZQ"
        )
        if not surrender_pos:
            surrender_pos = self.vision.find_template_cached(
                screen, self.tpl_surrender_battle, cache_name="surrender_btn", ui_cache=self.ui_cache,
                threshold=0.70, zona="ESQUINA_INF_IZQ"
            )
        if not surrender_pos:
            # Coordenada calibrada en UI Cache para botón de rendición
            surrender_pos = self.ui_cache.get_pos("surrender_btn", self.device.width, self.device.height)

        log.info("🎯 Rendición ultrarrápida: Pulsando botón de fin de batalla en %s...", surrender_pos)
        self.device.tap(surrender_pos[0], surrender_pos[1], humanize=False)
        time.sleep(0.30)

        # Confirmar diálogo central si aparece (Okay o Confirm)
        s_dlg = self.device.screenshot()
        confirm_pos = self.vision.find_template_cached(
            s_dlg, self.tpl_okay_surrender, cache_name="confirm_surrender_btn", ui_cache=self.ui_cache,
            threshold=0.70, zona="CENTRO"
        )
        if not confirm_pos:
            confirm_pos = self.vision.find_template_cached(
                s_dlg, self.tpl_confirm_end, cache_name="confirm_surrender_btn", ui_cache=self.ui_cache,
                threshold=0.70, zona="CENTRO"
            )
        if not confirm_pos:
            confirm_pos = self.ui_cache.get_pos("confirm_surrender_btn", self.device.width, self.device.height)

        if confirm_pos:
            log.info("🎯 Confirmando rendición inmediatamente en %s...", confirm_pos)
            self.device.tap(confirm_pos[0], confirm_pos[1], humanize=False)
            time.sleep(0.5)

        # Esperar Return Home o Claim Reward con muestreo rápido y UI Cache
        t_ret = time.monotonic()
        saved_return_event = False
        while time.monotonic() - t_ret < 10.0:
            s_ret = self.device.screenshot()

            # 1. Comprobar botón Claim Reward de cofre de evento
            match_reward = self.vision.find_any(
                s_ret, self.tpl_claim_rewards, threshold=0.72, zona="MITAD_INFERIOR"
            )
            if match_reward:
                log.info("🎁 Botón de evento 'Claim Reward' detectado en (%d, %d). Reclamando cofre...", match_reward[1], match_reward[2])
                save_debug_image_async("logs/runs/return_home_screen_event.png", s_ret, sync=True)
                self.device.tap(match_reward[1], match_reward[2], humanize=False)
                time.sleep(2.0)
                self.proc_mgr.handle_chest_reward_sequence(wait_for_chest=True)
                self._wait_and_tap_return_home()
                return

            # 2. Comprobar botón Return Home estándar o de evento
            match_ret = self.vision.find_any(
                s_ret, self.tpl_return_homes, threshold=0.72, zona="MITAD_INFERIOR"
            )
            if match_ret:
                log.info("🎯 Botón Return Home detectado en (%d, %d). Volviendo a la aldea al instante...", match_ret[1], match_ret[2])
                save_debug_image_async("logs/runs/return_home_screen_event.png", s_ret, sync=True)
                self.device.tap(match_ret[1], match_ret[2], humanize=False)
                time.sleep(2.5)
                return

            if (time.monotonic() - t_ret >= 3.0) and not saved_return_event and s_ret is not None:
                save_debug_image_async("logs/runs/return_home_screen_event.png", s_ret, sync=True)
                saved_return_event = True

            time.sleep(0.25)

        log.warning("⚠️ No se encontró botón Return Home tras rendición por template. Guardando captura de evento y usando UI Cache...")
        s_final = self.device.screenshot()
        save_debug_image_async("logs/runs/return_home_screen_event.png", s_final, sync=True)
        cached_ret = self.ui_cache.get_pos("return_home_btn", self.device.width, self.device.height)
        log.info("🎯 Pulsando Return Home usando UI Cache en %s...", cached_ret)
        self.device.tap(cached_ret[0], cached_ret[1], humanize=False)
        time.sleep(2.5)

    def clean_surrender(self):
        """Secuencia controlada de rendicion sin matar la app."""
        self.finish_battle_sequence(allow_surrender=True)

    def run_forever(self, max_attacks: int | None = None):
        log.info("Iniciando bucle de control CLASH_FARMER_BOT_V2 (máx ataques: %s)...", max_attacks or "infinito")
        self._stop_requested = False
        while True:
            if self._stop_requested:
                log.info("🛑 Detención solicitada (_stop_requested=True). Saliendo del bucle.")
                break
            if max_attacks is not None and self.ledger.attacks_completed >= max_attacks:
                log.info("🎯 Se completaron los %d ataques solicitados. Deteniendo bot de forma limpia.", max_attacks)
                self.ledger.print_summary()
                break
            try:
                # No procesar popups genericos si estamos en pleno ataque
                if self.state not in (BotState.ATTACKING, BotState.WAITING_BATTLE_END):
                    if self.proc_mgr.handle_popup():
                        continue
                self._tick()
                if self._stop_requested:
                    log.info("🛑 Detención solicitada (_stop_requested=True). Saliendo del bucle.")
                    break
            except KeyboardInterrupt:
                log.info("Detencion manual solicitada.")
                self.ledger.print_summary()
                break
            except Exception as e:
                log.error("Excepcion en estado %s: %s", self.state.name, e)
                # Si falla durante batalla, terminar ordenadamente antes de reiniciar
                if self.state in (BotState.ATTACKING, BotState.WAITING_BATTLE_END):
                    self.clean_surrender()
                self.state = BotState.ERROR_RECOVERY

    def _tick(self):
        if self.state == BotState.ENSURE_GAME_READY:
            if self.proc_mgr.ensure_game_ready():
                self.state = BotState.MAIN_VILLAGE
            else:
                self.state = BotState.ERROR_RECOVERY

        elif self.state == BotState.MAIN_VILLAGE:
            # Lectura de tooltips única al inicio de sesión / arranque (en caché para evitar latencias en ataques)
            if not self.ledger.tooltip_cached:
                self.ensure_tooltip_cached()

            screen = self.device.screenshot()
            own_rss = self.reader.read_own_resources(screen)
            self.ledger.record_initial_snapshot(own_rss)
            self.ledger.rss_current = own_rss

            # Evaluar trigger del 90% (y doble tanda secuencial si ambos superan umbral)
            res_sequence = self.ledger.get_wall_upgrade_resources_sequence()
            if res_sequence:
                log.info("🧱 Disparador de mejora de muros activado para secuencia: %s", res_sequence)
                res_results = self.wall_upgrader.upgrade_walls_sequence(res_sequence)
                if any(res_results.values()):
                    return

            if not self.army.can_start_battle(screen):
                log.info("Campamentos no listos. Pausa de 30s...")
                time.sleep(30)
                return

            attack_pos = None
            for tpl in self.tpl_attack_btns:
                attack_pos = self.vision.find_template_cached(
                    screen, tpl, cache_name="attack_btn", ui_cache=self.ui_cache,
                    threshold=0.74, zona="ESQUINA_INF_IZQ"
                )
                if attack_pos:
                    break
            if not attack_pos and self.ui_cache.enabled:
                attack_pos = self.ui_cache.get_pos("attack_btn", self.device.width, self.device.height)
            if attack_pos:
                HumanBehavior.reaction_delay()
                self.device.tap(*attack_pos)

                find_pos = self.vision.wait_for_template(
                    self.device.screenshot, self.tpl_find_match, timeout=6, zona="COMPLETA",
                    cache_name="find_match_btn", ui_cache=self.ui_cache
                )
                if find_pos:
                    HumanBehavior.reaction_delay()
                    self.device.tap(*find_pos)
                    time.sleep(0.8)

                    # Esperar y confirmar ataque (botón que inicia la búsqueda de rivales/recursos)
                    confirm_pos = None
                    t_start = time.monotonic()
                    while time.monotonic() - t_start < 4.5:
                        s_confirm = self.device.screenshot()
                        match = self.vision.find_any(
                            s_confirm, self.tpl_attack_confirms, threshold=0.72, zona="MITAD_INFERIOR"
                        )
                        if match:
                            confirm_pos = (match[1], match[2])
                            self.ui_cache.update_pos("confirm_attack_btn", confirm_pos[0], confirm_pos[1], self.device.width, self.device.height)
                            break
                        time.sleep(0.25)

                    if confirm_pos:
                        log.info("🎯 Botón Confirm Attack detectado en %s. Iniciando búsqueda...", confirm_pos)
                        HumanBehavior.reaction_delay()
                        self.device.tap(*confirm_pos)
                    else:
                        # Clic directo de confirmación usando UI Cache
                        cached_conf = self.ui_cache.get_pos("confirm_attack_btn", self.device.width, self.device.height, jitter=4)
                        log.info("🎯 Ejecutando clic de confirmación de ataque usando UI Cache: %s...", cached_conf)
                        HumanBehavior.reaction_delay()
                        self.device.tap(*cached_conf)

                    self.state = BotState.SEARCHING_MATCH
            else:
                log.warning("⚠️ Botón Attack no detectado en la aldea (umbral 0.74). Revalidando estado del juego...")
                self.state = BotState.ENSURE_GAME_READY


        elif self.state == BotState.SEARCHING_MATCH:
            log.info("Buscando rival...")
            next_pos = self.vision.wait_for_template(
                self.device.screenshot, self.tpl_next, timeout=25, zona="ESQUINA_INF_DER",
                cache_name="next_btn", ui_cache=self.ui_cache
            )
            if next_pos:
                # Espera configurable para que disipen nubes y termine el conteo
                wait_post_next = random.uniform(
                    config.LOOT_WAIT_AFTER_NEXT_MIN,
                    config.LOOT_WAIT_AFTER_NEXT_MAX,
                )
                log.info(f"[LOOT] Espera post-Next: {wait_post_next:.2f}s")
                time.sleep(wait_post_next)
                self.state = BotState.EVALUATING_LOOT
            else:
                log.warning("Timeout esperando boton Next.")
                self.state = BotState.ERROR_RECOVERY
        elif self.state == BotState.EVALUATING_LOOT:
            # Lectura estable con banco de escalas
            screen = self.device.screenshot()
            self.active_army_type = self.army.detect_active_army_type(screen)
            loot = self.reader.read_enemy_loot_stable(self.device.screenshot)

            # Criterios dinámicos según el tipo de ejército activo
            if self.active_army_type == "edrag_rush":
                edrag_crit = SETTINGS.get("attack", {}).get("edrag_rush", {}).get("loot_criteria")
                meets = self.reader.meets_attack_criteria(loot, criteria_override=edrag_crit)
            else:
                meets = self.reader.meets_attack_criteria(loot)

            if loot is not None and meets:
                if getattr(config, "SEARCH_ONLY", False):
                    log.info("🔎 [SEARCH_ONLY] Aldea apta encontrada — deteniendo ataque por modo solo búsqueda.")
                    self._stop_requested = True
                    return
                self.state = BotState.ATTACKING
            else:
                self.ledger.villages_skipped += 1
                # Toque de latencia cero usando la posición en UI Cache
                next_pos = self.ui_cache.get_pos("next_btn", self.device.width, self.device.height, jitter=5)
                HumanBehavior.reaction_delay()
                self.device.tap(*next_pos)
                time.sleep(1.0)
                self.state = BotState.SEARCHING_MATCH

        elif self.state == BotState.ATTACKING:
            if self.active_army_type == "edrag_rush":
                success = self.edrag_deployer.execute_attack()
            else:
                success = self.deployer.execute_attack()

            if success:
                self.state = BotState.WAITING_BATTLE_END
            else:
                self.clean_surrender()
                self.state = BotState.COLLECTING_LOOT

        elif self.state == BotState.WAITING_BATTLE_END:
            if self.active_army_type == "edrag_rush":
                edrag_cfg = SETTINGS.get("attack", {}).get("edrag_rush", {})
                max_sec = edrag_cfg.get("max_battle_duration_sec", 150)
                stagnation_timeout = edrag_cfg.get("damage_stagnation_timeout_sec", 5.0)
                min_combat_sec = edrag_cfg.get("min_combat_duration_before_stagnation_sec", 12.0)
                target_destruction_pct = edrag_cfg.get("target_destruction_percent", 70)

                log.info(
                    "🐉 Esperando combate E-Drag (máx %ds, meta %d%%, timeout estancamiento %.1fs tras %.1fs o Return Home natural)...",
                    max_sec, target_destruction_pct, stagnation_timeout, min_combat_sec
                )
                t0 = time.monotonic()
                cur_atk = self.ledger.attacks_completed + 1
                import os, cv2
                os.makedirs("logs/runs", exist_ok=True)
                last_snap = 0.0

                last_damage: int | None = None
                last_damage_increase_time = t0
                early_exit_triggered = False

                while time.monotonic() - t0 < max_sec:
                    screen = self.device.screenshot()
                    now = time.monotonic()
                    elapsed = now - t0

                    # Tomar capturas periódicas cada 6s para auditoría visual completa de la batalla
                    if elapsed - last_snap >= 6.0 and screen is not None:
                        last_snap = elapsed
                        save_debug_image_async(f"logs/runs/battle_progress_edrag_atk{cur_atk}_{int(elapsed)}s.png", screen)

                    # Si la batalla termina sola (100% destrucción, fin de tropas o tiempo agotado)
                    if self.vision.find_any(screen, self.tpl_return_homes + self.tpl_claim_rewards, threshold=0.72, zona="MITAD_INFERIOR"):
                        log.info("🏆 Batalla terminada naturalmente (Return Home / Claim Reward detectado tras %.1fs).", elapsed)
                        break

                    # Watchdog de daño:
                    if screen is not None:
                        pct = self.damage_reader.read_destruction_percentage(screen)
                        if pct is not None:
                            if last_damage is None or pct > last_damage:
                                log.info("💥 Destrucción actualizada: %d%% (tras %.1fs de batalla)", pct, elapsed)
                                last_damage = pct
                                last_damage_increase_time = now

                            # 🎯 CONDICIÓN 1: Meta del 70% de destrucción alcanzada -> Rendición inmediata al toque
                            if pct >= target_destruction_pct:
                                log.info(
                                    "⚡ META ALCANZADA: %d%% de destrucción (>= %d%%) tras %.1fs — ejecutando rendición ultrarrápida!",
                                    pct, target_destruction_pct, elapsed
                                )
                                early_exit_triggered = True
                                break

                    # 🎯 CONDICIÓN 2: Estancamiento de daño por >= stagnation_timeout segundos
                    if elapsed >= min_combat_sec:
                        time_since_increase = now - last_damage_increase_time
                        if time_since_increase >= stagnation_timeout:
                            log.info(
                                "⚡ Destrucción estancada en %s%% durante %.1fs (>= %.1fs) — finalizando batalla anticipadamente.",
                                f"{last_damage}" if last_damage is not None else "0",
                                time_since_increase,
                                stagnation_timeout
                            )
                            early_exit_triggered = True
                            break

                    time.sleep(0.9)

                s_end = self.device.screenshot()
                if s_end is not None:
                    save_debug_image_async(f"logs/runs/battle_end_edrag_atk{cur_atk}.png", s_end)

                # Rendición controlada ultrarrápida o retorno inmediato
                self.finish_battle_sequence(allow_surrender=early_exit_triggered)
                self.state = BotState.COLLECTING_LOOT
            else:
                wait_sec = SETTINGS.get("attack", {}).get("post_deploy_wait_sec", 10)
                early_exit = False
                natural_end = False
                log.info("Esperando %ds para limpieza de recolectores por las tropas...", wait_sec)
                t0 = time.monotonic()
                last_loot: tuple[int, int, int] | None = None
                consecutive_same_loot = 0

                while time.monotonic() - t0 < wait_sec:
                    screen = self.device.screenshot()
                    # Si termina sola (100% destrucción o tropas agotadas)
                    if self.vision.find_any(screen, self.tpl_return_homes + self.tpl_claim_rewards, threshold=0.72, zona="MITAD_INFERIOR"):
                        log.info("Batalla terminada naturalmente (Return Home / Claim Reward detectado).")
                        natural_end = True
                        break

                    elapsed = time.monotonic() - t0
                    if screen is not None:
                        cur_atk = self.ledger.attacks_completed + 1
                        save_debug_image_async(f"logs/runs/battle_progress_atk{cur_atk}.png", screen)
                        try:
                            loot = self.reader.read_enemy_loot(screen)
                            g = loot.get("gold") or 0
                            e = loot.get("elixir") or 0
                            d = loot.get("dark_elixir") or 0
                            total_loot = g + e + d
                            if total_loot == 0 and elapsed >= 3.0:
                                log.info("⚡ Botín rival vaciado al 100% (0 recursos restantes) — ejecutando rendición inmediata!")
                                early_exit = True
                                break

                            current_loot = (g, e, d)
                            if elapsed >= 5.0:
                                if last_loot is not None and current_loot == last_loot:
                                    consecutive_same_loot += 1
                                    if consecutive_same_loot >= 2:
                                        log.info("⚡ Botín estancado tras saqueo — ejecutando rendición anticipada!")
                                        early_exit = True
                                        break
                                else:
                                    consecutive_same_loot = 0
                                last_loot = current_loot
                        except Exception as e_loot:
                            log.debug("Error leyendo botín en espera post-ataque: %s", e_loot)

                    time.sleep(1.5)

                # Guardar captura final previa a la rendición
                s_end = self.device.screenshot()
                if s_end is not None:
                    cur_atk = self.ledger.attacks_completed + 1
                    save_debug_image_async(f"logs/runs/battle_end_atk{cur_atk}.png", s_end)

                # Ejecutar secuencia de cierre ultrarrápida si se vació el botín o allow_surrender está activo
                allow_surrender_cfg = SETTINGS.get("attack", {}).get("allow_surrender", True)
                should_surrender = not natural_end and (early_exit or allow_surrender_cfg)
                self.finish_battle_sequence(allow_surrender=should_surrender)
                self.state = BotState.COLLECTING_LOOT

        elif self.state == BotState.COLLECTING_LOOT:
            log.info("📊 Evaluando recursos tras el ataque en la aldea principal...")
            # Esperar a que la aldea principal esté completamente cargada (aparición del botón Attack)
            t_wait = time.monotonic()
            while time.monotonic() - t_wait < 10.0:
                screen = self.device.screenshot()
                if self.vision.find_any(screen, self.tpl_attack_btns, threshold=0.74, zona="ESQUINA_INF_IZQ"):
                    break
                time.sleep(1.0)

            # Tomar captura de la aldea principal y leer los recursos propios
            screen = self.device.screenshot()
            post_rss = self.reader.read_own_resources(screen)
            self.ledger.record_post_attack(post_rss)
            self.state = BotState.MAIN_VILLAGE

        elif self.state == BotState.ERROR_RECOVERY:
            log.warning("Entrando en recuperacion de juego...")
            self.device.restart_app()
            time.sleep(6.0)
            self.state = BotState.ENSURE_GAME_READY

