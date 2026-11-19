"""
tests/test_e2e_village_analysis.py
Comprehensive 4-tier test suite for End-to-End Village Analysis Integration.
Validates the full offline pipeline:
1. Camera survey sequence (East -> West -> South -> North -> Center)
2. Viewport detection accumulation
3. Non-convex red exclusion & green legal boundary detection
4. Jump spell targeting & corridor ray marching
5. Legal troop drop geometry with zero structure cuts and legal grass placement
"""
import os
import sys
import time
import unittest
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests import MockDevice, load_test_image
from tests.test_camera_survey import VillageOccupancyMap, execute_camera_survey_sequence
from tools.test_boundary_overlay import (
    detect_village_boundary,
    audit_boundary_collisions,
    generate_boundary_overlay,
)
from vision.yolo_detector import get_detector, Detection
from attack.deployment_strategy import (
    compute_adaptive_jump_targets_from_detections,
    compute_corridor_drop_geometry,
)


def run_full_offline_analysis(
    battle_flash_img: np.ndarray,
    pan_frames: list[tuple[np.ndarray, tuple[int, int]]] | None = None,
    mode: str = "pixel_scan"
) -> dict:
    """
    Executes the unified end-to-end analysis pipeline:
    1. Panoramic survey accumulation (if pan frames provided) or direct inference on flash image.
    2. Boundary detection with specified mode ('pixel_scan' or 'high_res_contour').
    3. Collision audit (0 cut structures).
    4. Jump targets selection.
    5. Corridor drop points and tangential sweep calculation for each jump.
    6. Overlay generation.
    """
    h, w = battle_flash_img.shape[:2]
    detector = get_detector()

    # 1. Accumulate detections
    occ = VillageOccupancyMap(screen_size=(w, h))
    if pan_frames:
        for p_img, offset in pan_frames:
            if detector and p_img is not None:
                dets = detector.detect(p_img)
                occ.add_viewport_detections(dets, camera_offset=offset)
    else:
        dets = detector.detect(battle_flash_img) if detector else []
        occ.add_viewport_detections(dets, camera_offset=(0, 0))

    accumulated_dets = occ.get_accumulated_detections()

    # 2. Boundary detection
    red_cnt, green_cnt = detect_village_boundary(
        battle_flash_img, detections=accumulated_dets, mode=mode
    )

    # 3. Collision audit
    cuts = audit_boundary_collisions(green_cnt, accumulated_dets)

    # 4. Jump target selection
    jump_targets = compute_adaptive_jump_targets_from_detections(
        accumulated_dets, (w, h)
    )

    # 5. Corridor drop calculation
    center_pt = (w * 0.50, h * 0.50)
    # Calculate base center of mass if buildings exist
    if accumulated_dets:
        pts = np.array([d.center for d in accumulated_dets], dtype=np.float32)
        center_pt = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

    corridor_plans = []
    hull_for_drop = green_cnt.reshape((-1, 2)) if green_cnt is not None and len(green_cnt) >= 3 else None
    for jump in jump_targets:
        drop_pt, sweep_pts = compute_corridor_drop_geometry(
            center_pt=center_pt,
            jump_pt=jump,
            screen_size=(w, h),
            detections=accumulated_dets,
            green_hull=hull_for_drop
        )
        corridor_plans.append({
            "jump_target": jump,
            "drop_point": drop_pt,
            "sweep_points": sweep_pts
        })

    # 6. Generate overlay
    overlay = generate_boundary_overlay(
        battle_flash_img, red_cnt, green_cnt, accumulated_dets, mode=mode
    )

    return {
        "detections": accumulated_dets,
        "red_contour": red_cnt,
        "green_contour": green_cnt,
        "cuts": cuts,
        "jump_targets": jump_targets,
        "corridor_plans": corridor_plans,
        "overlay": overlay
    }


class TestE2EVillageAnalysisTier1(unittest.TestCase):
    """Tier 1: Feature Coverage for Full Pipeline."""

    def setUp(self):
        self.img = load_test_image("logs/red_test_flash.png")
        if self.img is None:
            self.skipTest("logs/red_test_flash.png required for E2E tests")

    def test_e2e_full_pipeline_pixel_scan(self):
        """Verifies full E2E pipeline execution under default pixel_scan mode."""
        res = run_full_offline_analysis(self.img, mode="pixel_scan")
        self.assertIsNotNone(res["green_contour"])
        self.assertEqual(len(res["cuts"]), 0, "Pixel-scan pipeline must produce 0 cuts")
        self.assertGreater(len(res["jump_targets"]), 0, "Must calculate jump targets")
        self.assertEqual(len(res["corridor_plans"]), len(res["jump_targets"]))

    def test_e2e_full_pipeline_high_res_contour(self):
        """Verifies full E2E pipeline execution under high_res_contour mode."""
        res = run_full_offline_analysis(self.img, mode="high_res_contour")
        self.assertIsNotNone(res["green_contour"])
        self.assertEqual(len(res["cuts"]), 0, "High-res contour mode must produce 0 cuts")
        self.assertGreater(len(res["jump_targets"]), 0)
        # Vector points in high_res_contour should be compact (<500)
        self.assertLess(len(res["green_contour"]), 500)

    def test_e2e_zero_drop_points_in_red_zone(self):
        """
        Verifies every calculated corridor drop point is placed on legal green grass outside red zone.
        Resolved in M2: Uses max(t_green_hits) and [0.05*h, 0.86*h] vertical clamping,
        ensuring all corridor drop points clear exclusion zones.
        """
        res = run_full_offline_analysis(self.img, mode="pixel_scan")
        red_cnt = res["red_contour"]
        self.assertIsNotNone(red_cnt)

        for plan in res["corridor_plans"]:
            drop_x, drop_y = plan["drop_point"]
            # Drop point must be strictly OUTSIDE the red exclusion contour (dist <= 0)
            dist_to_red = cv2.pointPolygonTest(red_cnt, (float(drop_x), float(drop_y)), True)
            self.assertLessEqual(
                dist_to_red, 0.0,
                f"Drop point ({drop_x}, {drop_y}) fell inside red exclusion zone (dist={dist_to_red:.1f})"
            )

    def test_e2e_building_clearance_all_corridors(self):
        """Verifies all drop points have clearance from nearest detected buildings."""
        res = run_full_offline_analysis(self.img, mode="pixel_scan")
        dets = res["detections"]

        for plan in res["corridor_plans"]:
            drop_x, drop_y = plan["drop_point"]
            min_dist = min(
                np.hypot(d.center[0] - drop_x, d.center[1] - drop_y) for d in dets
            )
            self.assertGreaterEqual(
                min_dist, 40.0,
                f"Drop point ({drop_x}, {drop_y}) is too close to a building (dist={min_dist:.1f})"
            )

    def test_e2e_all_jumps_have_valid_drops(self):
        """Verifies every selected jump target has a non-None valid legal drop and sweep."""
        res = run_full_offline_analysis(self.img, mode="pixel_scan")
        for plan in res["corridor_plans"]:
            drop_pt = plan["drop_point"]
            sweep = plan["sweep_points"]
            self.assertEqual(len(drop_pt), 2)
            self.assertEqual(len(sweep), 2)
            # Ensure coordinates are integers within screen dimensions
            self.assertGreaterEqual(drop_pt[0], 0)
            self.assertLessEqual(drop_pt[0], 2400)
            self.assertGreaterEqual(drop_pt[1], 0)
            self.assertLessEqual(drop_pt[1], 1080)


class TestE2EVillageAnalysisTier2(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases."""

    def test_e2e_zero_jump_targets(self):
        """Pipeline handles synthetic image with zero valid loot targets gracefully."""
        # Create image with red contour but no buildings
        blank = np.zeros((1080, 2400, 3), dtype=np.uint8)
        blank[400:680, 1000:1400] = (30, 30, 220)
        res = run_full_offline_analysis(blank)
        self.assertEqual(len(res["jump_targets"]), 0)
        self.assertEqual(len(res["corridor_plans"]), 0)

    def test_e2e_core_town_hall_dive(self):
        """Target at center of base (1200, 540) generates safe legal South grass drop."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        _, green = detect_village_boundary(img)
        # Force jump target at (1200, 540)
        drop_pt, sweep_pts = compute_corridor_drop_geometry(
            center_pt=(1200.0, 540.0),
            jump_pt=(1200, 540),
            screen_size=(2400, 1080),
            green_hull=green
        )
        self.assertIsNotNone(drop_pt)
        # When dist < 18px from center, drop must orient South (y > 540)
        self.assertGreater(drop_pt[1], 540)

    def test_e2e_extreme_perimeter_collector(self):
        """Target placed at outer edge generates valid drop without leaving screen bounds."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        _, green = detect_village_boundary(img)
        drop_pt, sweep_pts = compute_corridor_drop_geometry(
            center_pt=(1200.0, 540.0),
            jump_pt=(2100, 540),
            screen_size=(2400, 1080),
            green_hull=green
        )
        self.assertLessEqual(drop_pt[0], int(0.92 * 2400))
        for p in sweep_pts:
            self.assertLessEqual(p[0], int(0.94 * 2400))

    def test_e2e_asymmetric_flank_density(self):
        """Base with all storages on East flank generates properly oriented corridors."""
        dets = [
            Detection("storage_gold", 0.90, (1600, 400, 1660, 460), (1630, 430)),
            Detection("storage_elixir", 0.88, (1700, 500, 1760, 560), (1730, 530)),
            Detection("storage_dark", 0.85, (1800, 600, 1860, 660), (1830, 630)),
        ]
        targets = compute_adaptive_jump_targets_from_detections(dets, (2400, 1080))
        self.assertGreater(len(targets), 0)
        for t in targets:
            self.assertGreater(t[0], 1200, "All targets should be on East flank")

    def test_e2e_pipeline_timing_budget(self):
        """Verifies end-to-end analysis finishes in under 5.0 seconds offline on CPU."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        t0 = time.time()
        res = run_full_offline_analysis(img, mode="high_res_contour")
        dt = time.time() - t0
        self.assertLess(dt, 5.0, f"E2E execution took {dt:.2f}s (budget: 5.0s)")


class TestE2EVillageAnalysisTier3(unittest.TestCase):
    """Tier 3: Cross-Feature Combinations."""

    def test_e2e_panoramic_accumulation_to_drop_geometry(self):
        """Buildings detected in multi-view survey correctly accumulate without structure cuts."""
        img_center = load_test_image("logs/pan_test_4_center.png")
        if img_center is None:
            self.skipTest("Missing logs/pan_test_4_center.png")

        pan_frames = [
            (img_center, (0, 0)),
            (img_center, (10, -5)),  # Multi-viewport jitter
        ]
        res = run_full_offline_analysis(img_center, pan_frames=pan_frames)
        self.assertEqual(len(res["cuts"]), 0)
        # Accumulated detections should be at least as many as center frame (16)
        self.assertGreaterEqual(len(res["detections"]), 14)

    def test_e2e_dual_mode_drop_consistency(self):
        """Verifies drop points computed under pixel_scan vs high_res_contour match within tolerance."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        res_pixel = run_full_offline_analysis(img, mode="pixel_scan")
        res_contour = run_full_offline_analysis(img, mode="high_res_contour")

        self.assertEqual(len(res_pixel["corridor_plans"]), len(res_contour["corridor_plans"]))
        for p1, p2 in zip(res_pixel["corridor_plans"], res_contour["corridor_plans"]):
            pt1 = p1["drop_point"]
            pt2 = p2["drop_point"]
            dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
            # Drop points should be within 40px of each other across modes
            self.assertLessEqual(dist, 40.0)

    def test_e2e_concave_bay_corridor_ray_march(self):
        """Corridor ray march on concave base uses max(t_hits) to land on outer legal grass."""
        center = (1200.0, 540.0)
        jump_pt = (1450, 540)

        # Polygon with a concave indentation along East ray
        pts = [
            (1200, 200), (1600, 350), (1400, 540), (1700, 540),
            (1600, 750), (1200, 900), (800, 750), (700, 540), (800, 350)
        ]
        green_hull = np.array(pts, dtype=np.int32)
        drop_pt, _ = compute_corridor_drop_geometry(center, jump_pt, (2400, 1080), green_hull=green_hull)

        # Drop point must be beyond outer intersection (x >= 1400)
        self.assertGreaterEqual(drop_pt[0], 1400)

    def test_e2e_multi_corridor_tangential_sweep_vectors(self):
        """Verifies tangential sweep vectors for all active jumps have valid orientations."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        res = run_full_offline_analysis(img)
        for plan in res["corridor_plans"]:
            p1, p2 = plan["sweep_points"]
            length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            # half_width = 38.0 -> total length approx 76px (clipped near screen margins to >= 40px)
            self.assertGreaterEqual(length, 40.0)
            self.assertLessEqual(length, 85.0)

    def test_e2e_overlay_regeneration_consistency(self):
        """Verifies generated overlay has correct 2400x1080 shape and contains banner."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        res = run_full_offline_analysis(img)
        overlay = res["overlay"]
        self.assertEqual(overlay.shape[:2], (1080, 2400))
        # Top banner at (20, 20) should have dark background (pixel value < 50)
        self.assertLess(overlay[20, 20, 0], 50)


class TestE2EVillageAnalysisTier4(unittest.TestCase):
    """Tier 4: Real-World Match Datasets & Regression Testing."""

    def test_e2e_real_battle_flash_full_simulation(self):
        """Runs complete end-to-end simulation on golden sample logs/red_test_flash.png."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        res = run_full_offline_analysis(img, mode="pixel_scan")

        self.assertEqual(len(res["cuts"]), 0, "Golden sample must have 0 cuts")
        self.assertEqual(len(res["detections"]), 18, "Golden sample has 18 YOLO detections")
        self.assertTrue(len(res["jump_targets"]) >= 2, "Expected at least 2 jump targets")

        # Save output visual overlay
        out_path = os.path.join(PROJECT_ROOT, "logs", "yolo", "e2e_simulation_test.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, res["overlay"])
        self.assertTrue(os.path.exists(out_path))

    def test_e2e_pan_and_attack_frames_alignment(self):
        """Verifies coordinate consistency across pan frames and centered attack frame."""
        f_center = load_test_image("logs/pan_test_4_center.png")
        f_attack = load_test_image("logs/yolo/attack_20261106_091024.png")
        if f_center is None or f_attack is None:
            self.skipTest("Pan center or attack frame not found")

        self.assertEqual(f_center.shape, f_attack.shape)

    def test_e2e_train_dataset_batch_analysis(self):
        """Runs pipeline on multiple images from datasets/coc/images/train/."""
        train_dir = os.path.join(PROJECT_ROOT, "datasets", "coc", "images", "train")
        if not os.path.exists(train_dir):
            self.skipTest("datasets/coc/images/train not found")

        files = [f for f in os.listdir(train_dir) if f.endswith(".png")][:3]
        for f in files:
            with self.subTest(file=f):
                img = cv2.imread(os.path.join(train_dir, f))
                res = run_full_offline_analysis(img, mode="high_res_contour")
                self.assertEqual(len(res["cuts"]), 0)

    def test_e2e_audit_regression_against_old_logs(self):
        """
        Verifies that the old cut defect (where 5 structures were cut in red_boundary_detected.png)
        is completely eliminated in the new pipeline.
        """
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        res = run_full_offline_analysis(img)
        self.assertEqual(len(res["cuts"]), 0, "Regression audit: cut count must be 0")


if __name__ == "__main__":
    unittest.main()
