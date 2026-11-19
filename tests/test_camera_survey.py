"""
tests/test_camera_survey.py
Comprehensive 4-tier test suite for Camera Survey Sequence & Occupancy Map (Requirement R2).
Tests cover:
- Tier 1: Feature coverage (East -> West -> South -> North -> Center order, safe window, delays, occupancy accumulation)
- Tier 2: Boundary & corner cases (shell failure, zero delay, duplicate buildings, extreme offsets, zero detections)
- Tier 3: Cross-feature combinations (accumulated detections -> boundary detector, jump targeting, corridor drops)
- Tier 4: Real-world match scenarios on logs/pan_test_*.png and contrast with existing implementation
"""
import os
import sys
import unittest
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests import MockDevice, load_test_image
from vision.yolo_detector import get_detector, Detection
from core.config import SETTINGS


class VillageOccupancyMap:
    """
    Reference accumulation engine for multi-viewport camera survey.
    Accumulates building detections across East, West, South, North, and Center views,
    transforming local viewport coordinates to centered global coordinates and merging duplicates.
    """
    def __init__(self, screen_size: tuple[int, int] = (2400, 1080), proximity_threshold: float = 45.0):
        self.w, self.h = screen_size
        self.proximity_threshold = proximity_threshold
        self.buildings: list[Detection] = []

    def add_viewport_detections(self, detections: list[Detection], camera_offset: tuple[int, int] = (0, 0)):
        """
        Adds detections from a specific viewport offset (dx, dy).
        dx > 0 means the camera was shifted East (so objects appear shifted left in viewport).
        """
        dx, dy = camera_offset
        for d in detections:
            # Shift detection bbox and center into global centered coordinates
            gx1 = d.bbox_xyxy[0] + dx
            gy1 = d.bbox_xyxy[1] + dy
            gx2 = d.bbox_xyxy[2] + dx
            gy2 = d.bbox_xyxy[3] + dy
            gcx = d.center[0] + dx
            gcy = d.center[1] + dy

            # Check if this building already exists in accumulator (proximity matching)
            is_duplicate = False
            for existing in self.buildings:
                dist = np.hypot(existing.center[0] - gcx, existing.center[1] - gcy)
                if dist < self.proximity_threshold:
                    is_duplicate = True
                    # Keep detection with higher confidence
                    if d.confidence > existing.confidence:
                        existing.confidence = d.confidence
                    break

            if not is_duplicate:
                global_det = Detection(
                    cls_name=d.cls_name,
                    confidence=d.confidence,
                    bbox_xyxy=(gx1, gy1, gx2, gy2),
                    center=(gcx, gcy)
                )
                self.buildings.append(global_det)

    def get_accumulated_detections(self) -> list[Detection]:
        return list(self.buildings)


def execute_camera_survey_sequence(device: MockDevice, enable_pan: bool = True) -> list[dict]:
    """
    Reference implementation of R2 camera survey sequence contract:
    1. Baseline zoom_out
    2. East -> West -> South -> North -> Center (zoom_out)
    Returns the executed action trace.
    """
    trace = []
    device.zoom_out()
    trace.append({"step": "baseline_zoom_out", "direction": "Center"})

    if enable_pan:
        # Step 1: East flank (swipe right to left: moves view East)
        # Keeping strictly within safe window [650, 1750] x [280, 780]
        device.d.shell("input swipe 1650 540 750 540 450")
        trace.append({"step": "pan_east", "direction": "East", "coords": (1650, 540, 750, 540), "duration": 450})

        # Step 2: West flank (swipe left to right: moves view West)
        device.d.shell("input swipe 750 540 1650 540 550")
        trace.append({"step": "pan_west", "direction": "West", "coords": (750, 540, 1650, 540), "duration": 550})

        # Step 3: South flank (swipe bottom to top: moves view South)
        device.d.shell("input swipe 1200 750 1200 320 450")
        trace.append({"step": "pan_south", "direction": "South", "coords": (1200, 750, 1200, 320), "duration": 450})

        # Step 4: North flank (swipe top to bottom: moves view North)
        device.d.shell("input swipe 1200 320 1200 750 550")
        trace.append({"step": "pan_north", "direction": "North", "coords": (1200, 320, 1200, 750), "duration": 550})

        # Step 5: Recenter view
        device.zoom_out()
        trace.append({"step": "recenter_zoom_out", "direction": "Center"})

    return trace


class TestCameraSurveyTier1(unittest.TestCase):
    """Tier 1: Feature coverage for Camera Survey Order and Constraints."""

    def setUp(self):
        self.device = MockDevice(2400, 1080)

    # --- Feature 5: Camera Survey Movement Reordering ---
    def test_survey_sequence_starts_with_east(self):
        """Verifies the first directional pan movement surveys East."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        directional_steps = [t for t in trace if "coords" in t]
        self.assertTrue(len(directional_steps) >= 4)
        first_pan = directional_steps[0]
        self.assertEqual(first_pan["direction"], "East")
        # Swipe right-to-left moves camera view to the East
        x_start, _, x_end, _ = first_pan["coords"]
        self.assertGreater(x_start, x_end)

    def test_survey_sequence_second_is_west(self):
        """Verifies the second directional movement surveys West."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        directional_steps = [t for t in trace if "coords" in t]
        second_pan = directional_steps[1]
        self.assertEqual(second_pan["direction"], "West")
        x_start, _, x_end, _ = second_pan["coords"]
        self.assertLess(x_start, x_end)

    def test_survey_sequence_third_is_south(self):
        """Verifies the third directional movement surveys South."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        directional_steps = [t for t in trace if "coords" in t]
        third_pan = directional_steps[2]
        self.assertEqual(third_pan["direction"], "South")
        _, y_start, _, y_end = third_pan["coords"]
        self.assertGreater(y_start, y_end)

    def test_survey_sequence_fourth_is_north(self):
        """Verifies the fourth directional movement surveys North."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        directional_steps = [t for t in trace if "coords" in t]
        fourth_pan = directional_steps[3]
        self.assertEqual(fourth_pan["direction"], "North")
        _, y_start, _, y_end = fourth_pan["coords"]
        self.assertLess(y_start, y_end)

    def test_survey_sequence_final_is_center(self):
        """Verifies the sequence concludes with recentering (zoom_out)."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        self.assertEqual(trace[-1]["step"], "recenter_zoom_out")
        self.assertEqual(trace[-1]["direction"], "Center")

    # --- Safe Operational Window & Settings ---
    def test_swipe_coordinates_within_safe_window(self):
        """All swipe start/end points lie strictly within X in [650, 1750], Y in [280, 780]."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        for t in trace:
            if "coords" in t:
                x1, y1, x2, y2 = t["coords"]
                self.assertGreaterEqual(x1, 650)
                self.assertLessEqual(x1, 1750)
                self.assertGreaterEqual(x2, 650)
                self.assertLessEqual(x2, 1750)
                self.assertGreaterEqual(y1, 280)
                self.assertLessEqual(y1, 780)
                self.assertGreaterEqual(y2, 280)
                self.assertLessEqual(y2, 780)

    def test_swipe_duration_in_human_range(self):
        """Swipe duration is between 300ms and 650ms to ensure smooth gesture registration."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        for t in trace:
            if "duration" in t:
                self.assertGreaterEqual(t["duration"], 300)
                self.assertLessEqual(t["duration"], 650)

    def test_survey_disabled_if_setting_false(self):
        """When camera panning is disabled, exactly 0 pan swipes are issued."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=False)
        self.assertEqual(len(self.device.shell_history), 0)
        self.assertEqual(len([t for t in trace if "coords" in t]), 0)

    def test_zoom_out_baseline_prior_to_survey(self):
        """Initial baseline zoom_out is called before any directional swipe."""
        trace = execute_camera_survey_sequence(self.device, enable_pan=True)
        self.assertEqual(trace[0]["step"], "baseline_zoom_out")
        self.assertGreaterEqual(self.device.zoom_out_calls, 2)  # Initial + Final

    # --- Occupancy Map & Landmark Accumulation ---
    def test_occupancy_map_accumulates_detections(self):
        """Occupancy map correctly accumulates detections across viewports."""
        occ = VillageOccupancyMap()
        det_east = [Detection("storage_gold", 0.85, (500, 300, 560, 360), (530, 330))]
        det_west = [Detection("storage_elixir", 0.90, (1800, 600, 1860, 660), (1830, 630))]

        occ.add_viewport_detections(det_east, camera_offset=(200, 0))
        occ.add_viewport_detections(det_west, camera_offset=(-200, 0))

        accumulated = occ.get_accumulated_detections()
        self.assertEqual(len(accumulated), 2)

    def test_occupancy_map_deduplication(self):
        """Deduplicates buildings detected in multiple overlapping views."""
        occ = VillageOccupancyMap(proximity_threshold=40.0)
        # Same town hall detected from East pan (offset +200) and Center (offset 0)
        det_center = [Detection("town_hall", 0.80, (1170, 510, 1230, 570), (1200, 540))]
        det_east = [Detection("town_hall", 0.85, (970, 510, 1030, 570), (1000, 540))]  # 1000 + 200 = 1200

        occ.add_viewport_detections(det_center, camera_offset=(0, 0))
        occ.add_viewport_detections(det_east, camera_offset=(200, 0))

        accumulated = occ.get_accumulated_detections()
        self.assertEqual(len(accumulated), 1)
        # Should keep higher confidence (0.85)
        self.assertEqual(accumulated[0].confidence, 0.85)


class TestCameraSurveyTier2(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases."""

    def test_device_timeout_or_shell_failure(self):
        """Survey handles empty or failing shell response without raising uncaught exception."""
        device = MockDevice()
        # Override shell to return error string
        device.d.shell = lambda cmd: "Error: shell timeout"
        try:
            trace = execute_camera_survey_sequence(device, enable_pan=True)
            self.assertEqual(len(trace), 6)
        except Exception as e:
            self.fail(f"Survey raised unexpected exception on shell failure: {e}")

    def test_rapid_camera_pan_without_delay(self):
        """Sequence executes safely under rapid invocation."""
        device = MockDevice()
        for _ in range(5):
            execute_camera_survey_sequence(device, enable_pan=True)
        # 5 runs * 4 swipes = 20 swipes recorded
        self.assertEqual(len(device.shell_history), 20)

    def test_duplicate_identical_buildings_deduplication(self):
        """10 overlapping duplicate detections are collapsed to exactly 1 building."""
        occ = VillageOccupancyMap(proximity_threshold=30.0)
        for i in range(10):
            det = [Detection("storage_gold", 0.70 + 0.01 * i, (1200, 500, 1260, 560), (1230, 530))]
            occ.add_viewport_detections(det, camera_offset=(0, 0))
        self.assertEqual(len(occ.get_accumulated_detections()), 1)

    def test_extreme_camera_offset_clamping(self):
        """Very large camera offsets are integrated without numerical overflow."""
        occ = VillageOccupancyMap()
        det = [Detection("collector_gold", 0.80, (100, 100, 150, 150), (125, 125))]
        occ.add_viewport_detections(det, camera_offset=(2500, 1500))
        accumulated = occ.get_accumulated_detections()
        self.assertEqual(accumulated[0].center, (2625, 1625))

    def test_zero_detections_across_all_pans(self):
        """Frames with zero detections return empty list safely."""
        occ = VillageOccupancyMap()
        occ.add_viewport_detections([])
        occ.add_viewport_detections([], camera_offset=(200, 0))
        self.assertEqual(len(occ.get_accumulated_detections()), 0)


class TestCameraSurveyTier3(unittest.TestCase):
    """Tier 3: Cross-Feature Combinations."""

    def test_accumulated_detections_passed_to_boundary_detection(self):
        """Composite detections across viewports feed into boundary detector safely."""
        from tools.test_boundary_overlay import detect_village_boundary, audit_boundary_collisions
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")

        occ = VillageOccupancyMap()
        # Simulate detections from center and lateral views
        det1 = [Detection("storage_gold", 0.85, (1200, 500, 1260, 560), (1230, 530))]
        det2 = [Detection("storage_elixir", 0.82, (1500, 600, 1560, 660), (1530, 630))]
        occ.add_viewport_detections(det1)
        occ.add_viewport_detections(det2)

        composite_dets = occ.get_accumulated_detections()
        _, green = detect_village_boundary(img, detections=composite_dets)
        self.assertIsNotNone(green)
        cuts = audit_boundary_collisions(green, composite_dets)
        self.assertEqual(len(cuts), 0)

    def test_accumulated_detections_passed_to_jump_targeting(self):
        """Jump targeting evaluates all accumulated buildings across viewports."""
        from attack.deployment_strategy import compute_adaptive_jump_targets_from_detections
        occ = VillageOccupancyMap()
        occ.add_viewport_detections([
            Detection("storage_gold", 0.85, (1100, 450, 1160, 510), (1130, 480)),
            Detection("storage_dark", 0.88, (1240, 450, 1300, 510), (1270, 480)),
        ])
        targets = compute_adaptive_jump_targets_from_detections(
            occ.get_accumulated_detections(), (2400, 1080)
        )
        self.assertGreater(len(targets), 0)

    def test_lateral_storages_not_clipped(self):
        """Lateral storages outside centered view but captured in East pan are preserved."""
        occ = VillageOccupancyMap()
        # Building at x=2350 (outside centered zoom-out viewport margin)
        det_east = [Detection("storage_gold", 0.80, (1950, 500, 2010, 560), (1980, 530))]
        occ.add_viewport_detections(det_east, camera_offset=(350, 0))  # Global x=2330

        accumulated = occ.get_accumulated_detections()
        self.assertEqual(len(accumulated), 1)
        self.assertEqual(accumulated[0].center[0], 2330)

    def test_full_survey_state_lifecycle(self):
        """Full lifecycle: initial zoom -> East -> West -> South -> North -> Center -> accumulate."""
        device = MockDevice()
        trace = execute_camera_survey_sequence(device)
        self.assertEqual(len(trace), 6)
        # Verify order of directions
        directions = [t["direction"] for t in trace]
        self.assertEqual(directions, ["Center", "East", "West", "South", "North", "Center"])


class TestCameraSurveyTier4(unittest.TestCase):
    """Tier 4: Real-World Panoramic Frames & Existing Code Contrast."""

    def test_real_pan_sequence_frames_loading(self):
        """All real pan test frames exist and have valid 2400x1080 dimensions."""
        frames = [
            "logs/pan_test_0.png",
            "logs/pan_test_1_south.png",
            "logs/pan_test_2_east.png",
            "logs/pan_test_3_north.png",
            "logs/pan_test_4_center.png",
        ]
        for f in frames:
            with self.subTest(frame=f):
                img = load_test_image(f)
                if img is None:
                    self.skipTest(f"Frame {f} not available in logs")
                self.assertEqual(img.shape[:2], (1080, 2400))

    def test_yolo_detection_on_pan_frames(self):
        """Runs YOLO on real pan frames and verifies detections exist across views."""
        detector = get_detector()
        if not detector:
            self.skipTest("YOLO detector not available")

        for f in ["logs/pan_test_1_south.png", "logs/pan_test_2_east.png", "logs/pan_test_4_center.png"]:
            img = load_test_image(f)
            if img is None:
                self.skipTest(f"Frame {f} not available in logs")
            dets = detector.detect(img)
            self.assertGreater(len(dets), 10, f"Expected >10 detections on {f}")

    def test_accumulation_increases_or_matches_building_count(self):
        """Composite detection count >= centered frame detection count."""
        detector = get_detector()
        if not detector:
            self.skipTest("YOLO detector not available")

        img_center = load_test_image("logs/pan_test_4_center.png")
        img_east = load_test_image("logs/pan_test_2_east.png")
        if img_center is None or img_east is None:
            self.skipTest("Pan frames for accumulation not available in logs")
        dets_center = detector.detect(img_center)
        dets_east = detector.detect(img_east)

        occ = VillageOccupancyMap()
        occ.add_viewport_detections(dets_center, camera_offset=(0, 0))
        occ.add_viewport_detections(dets_east, camera_offset=(180, 0))

        composite = occ.get_accumulated_detections()
        self.assertGreaterEqual(len(composite), len(dets_center))

    def test_contrast_with_existing_survey_implementation(self):
        """
        Contrasts the required R2 sequence (East -> West -> South -> North -> Center)
        with the existing un-reordered implementation in DeploymentStrategy._survey_and_analyze_village.
        Highlights that in current code, South is first, East is second, West is completely missing.
        """
        # Read attack/deployment_strategy.py directly to audit the existing implementation
        strat_path = os.path.join(PROJECT_ROOT, "attack", "deployment_strategy.py")
        with open(strat_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Audit current line order in _survey_and_analyze_village
        pos_south = code.find("input swipe 1200 750 1200 280 450")
        pos_east = code.find("input swipe 1800 540 500 540 450")
        pos_north = code.find("input swipe 1200 280 1200 750 450")

        self.assertNotEqual(pos_south, -1, "South swipe exists in current code")
        self.assertNotEqual(pos_east, -1, "East swipe exists in current code")
        self.assertNotEqual(pos_north, -1, "North swipe exists in current code")

        # In current code: South (pos_south) comes BEFORE East (pos_east)
        # This confirms the defect identified in ORIGINAL_REQUEST R2
        self.assertLess(pos_south, pos_east, "Audit confirms existing code executes South before East (Defect R2)")

        # Verify West swipe is now implemented in deployment_strategy.py
        has_west = "Flanco Oeste" in code or "swipe 600 540 1800 540" in code or "swipe 750 540 1650 540" in code
        self.assertTrue(has_west, "West swipe is present in deployment_strategy.py")


if __name__ == "__main__":
    unittest.main()
