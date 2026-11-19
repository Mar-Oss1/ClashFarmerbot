"""
vision/army_checker.py
Verificación de ejército disponible en aldea y durante la batalla.
"""
import cv2
from core.vision import Vision
from core.config import SETTINGS
from core.logger import get_logger

log = get_logger("army_checker")


class ArmyChecker:
    def __init__(self, vision: Vision):
        self.vision = vision
        self.tpl_goblin_slot = "assets/buttons/goblin_slot.png"
        self.tpl_swb_slot = "assets/buttons/super_wall_breaker_slot.png"
        self.tpl_hero_king = "assets/buttons/hero_king_slot.png"
        self.tpl_hero_queen = "assets/buttons/hero_queen_slot.png"
        self.tpl_hero_warden = "assets/buttons/hero_warden_slot.png"
        self.tpl_hero_champion = "assets/buttons/hero_champion_slot.png"
        self.tpl_hero_minion_prince = "assets/buttons/hero_minion_prince_slot.png"
        self.tpl_hero_duke_dragon = "assets/buttons/hero_duke_dragon_slot.png"
        self.tpl_spell_jump = "assets/buttons/spell_jump.png"
        self.tpl_spell_haste = "assets/buttons/spell_haste.png"
        self.tpl_spell_rage = "assets/buttons/spell_rage.png"
        self.tpl_edrag_slot = "assets/buttons/edrag_slot.png"
        self.tpl_siege_machine = "assets/buttons/siege_machine_slot.png"
        self.tpl_army_ready = "assets/buttons/army_ready_icon.png"

    def can_start_battle(self, screenshot) -> bool:
        # En farming continuo, las tropas siempre están pre-entrenadas instantáneamente
        if SETTINGS.get("army", {}).get("skip_army_ready_check", True):
            return True
        if self.vision._load_template(self.tpl_army_ready) is None:
            return True
        pos = self.vision.find_template(screenshot, self.tpl_army_ready, threshold=0.75, zona="ESQUINA_INF_IZQ")
        if not pos:
            log.warning("⚠️ Icono de ejército listo no detectado.")
            return False
        return True

    def find_goblin_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_goblin_slot, threshold=0.66, zona="MITAD_INFERIOR"
        )

    def find_jump_spell_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_spell_jump, threshold=0.72, zona="MITAD_INFERIOR"
        )

    def find_haste_spell_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_spell_haste, threshold=0.72, zona="MITAD_INFERIOR"
        )

    def find_super_wall_breaker_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_swb_slot, threshold=0.68, zona="MITAD_INFERIOR"
        )

    def find_edrag_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_edrag_slot, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_rage_spell_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_spell_rage, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_siege_machine_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_siege_machine, threshold=0.70, zona="MITAD_INFERIOR"
        )

    def find_minion_prince_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_hero_minion_prince, threshold=0.65, zona="MITAD_INFERIOR"
        )

    def find_hero_king_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_hero_king, threshold=0.68, zona="MITAD_INFERIOR"
        )

    def find_hero_queen_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_hero_queen, threshold=0.68, zona="MITAD_INFERIOR"
        )

    def find_hero_warden_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_hero_warden, threshold=0.68, zona="MITAD_INFERIOR"
        )

    def find_hero_champion_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        # Intenta primero con la plantilla por defecto (Luchadora Real) y luego con aspecto alternativo (Dragon Champion)
        pos = self.vision.find_template(
            screenshot, self.tpl_hero_champion, threshold=0.65, zona="MITAD_INFERIOR"
        )
        if pos:
            return pos
        return self.vision.find_template(
            screenshot, self.tpl_hero_duke_dragon, threshold=0.68, zona="MITAD_INFERIOR"
        )

    def find_hero_duke_dragon_in_battle_bar(self, screenshot) -> tuple[int, int] | None:
        return self.vision.find_template(
            screenshot, self.tpl_hero_duke_dragon, threshold=0.68, zona="MITAD_INFERIOR"
        )

    def find_all_heroes_in_battle_bar(self, screenshot) -> dict[str, tuple[int, int]]:
        """
        Retorna diccionario con las ranuras de TODOS los héroes detectados en la barra de batalla,
        desduplicados espacialmente (para evitar múltiples plantillas sobre la misma carta)
        y ordenados dinámicamente de izquierda a derecha según su coordenada X en la pantalla.
        """
        if screenshot is None:
            return {}

        hero_specs = [
            ("champion", self.tpl_hero_champion, 0.65),
            ("minion_prince", self.tpl_hero_minion_prince, 0.65),
            ("warden", self.tpl_hero_warden, 0.68),
            ("duke_dragon", self.tpl_hero_duke_dragon, 0.68),
            ("king", self.tpl_hero_king, 0.68),
            ("queen", self.tpl_hero_queen, 0.68),
        ]

        screen_gray = self.vision.to_gray(screenshot)
        search_area, offset_x, offset_y = self.vision._crop_zone(screen_gray, "MITAD_INFERIOR")

        candidates = []
        for hero_id, tpl_path, threshold in hero_specs:
            cached = self.vision._load_template(tpl_path)
            if cached is None:
                continue
            tg, (th, tw) = cached
            if search_area.shape[0] < th or search_area.shape[1] < tw:
                continue
            res = cv2.matchTemplate(search_area, tg, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
                cx = max_loc[0] + offset_x + tw // 2
                cy = max_loc[1] + offset_y + th // 2
                candidates.append((hero_id, (cx, cy), float(max_val)))

        # Desduplicación espacial por coordenada horizontal X (distancia entre cartas ~155px, margen 65px)
        # Priorizar la coincidencia con mayor score de confianza
        candidates.sort(key=lambda c: c[2], reverse=True)
        kept_heroes: list[tuple[str, tuple[int, int], float]] = []
        for c in candidates:
            if not any(abs(c[1][0] - k[1][0]) < 65 for k in kept_heroes):
                kept_heroes.append(c)

        # Ordenar estrictamente de izquierda a derecha (coordenada X creciente)
        kept_heroes.sort(key=lambda k: k[1][0])

        heroes = {h_id: pos for h_id, pos, _ in kept_heroes}
        log.info("👑 Héroes detectados en barra (de izq a der): %s", {h: p for h, p in heroes.items()})
        return heroes

    def detect_active_army_type(self, screenshot) -> str:
        """
        Determina la estrategia de ataque activa.
        Respeta la configuración manual si se especificó en settings.json,
        o auto-detecta en base a las tropas visibles en la barra de batalla.
        """
        configured_mode = SETTINGS.get("attack", {}).get("attack_strategy", "auto")
        if configured_mode in {"edrag_rush", "sneaky_farming"}:
            return configured_mode

        if screenshot is None:
            return "sneaky_farming"

        # Detección por firmas de tropas y hechizos clave
        if self.find_edrag_in_battle_bar(screenshot) or self.find_rage_spell_in_battle_bar(screenshot):
            log.info("🐉 Ejército detectado: Dragones Eléctricos + Furias -> Modo 'edrag_rush'")
            return "edrag_rush"

        if self.find_goblin_in_battle_bar(screenshot) or self.find_jump_spell_in_battle_bar(screenshot):
            log.info("👺 Ejército detectado: Duendes Furtivos -> Modo 'sneaky_farming'")
            return "sneaky_farming"

        return "sneaky_farming"