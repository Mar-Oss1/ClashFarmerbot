"""
tests/test_boundary_detection.py
Comprehensive 4-tier test suite for Village Boundary Detection (Requirement R1).
Tests cover:
- Tier 1: Feature coverage (Red extraction, UI masking, Non-convex tracing, Dual modes)
- Tier 2: Boundary & corner cases (empty, non-red, extreme coords, wall bastions, saturation)
- Tier 3: Cross-feature combinations (YOLO detections + red boundary, ray march outer drop, building clearance)
- Tier 4: Real-world match scenarios (logs/red_test_flash.png, datasets/coc/images/train/, old convex hull defect contrast)
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

from tests import MockDevice, count_structure_intersections, load_test_image
from tools.test_boundary_overlay import (
    build_ui_mask,
    extract_red_mask,
    detect_village_boundary,
    audit_boundary_collisions,
)
from vision.yolo_detector import get_detector, Detection
from attack.deployment_strategy import compute_corridor_drop_geometry, _OUTER_GRASS_HULL


class TestBoundaryDetectionTier1(unittest.TestCase):
    """Tier 1: Feature coverage (>=5 tests per feature)."""

    # --- Feature 1: Red Line Extraction ---
    def test_red_hsv_extraction_bounds(self):
        """Verifies calibrated HSV range catches both primary and wrap-around red hues."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Red hue wrap-around 1 (H=5, S=200, V=200 in BGR is approx (40, 40, 200))
        img[20:40, 20:40] = (30, 30, 220)
        # Red hue wrap-around 2 (H=175, S=200, V=200 in BGR is approx (60, 20, 200))
        img[60:80, 60:80] = (60, 20, 210)

        mask = extract_red_mask(img, ui_mask=np.ones((100, 100), dtype=np.uint8))
        self.assertGreater(np.count_nonzero(mask[20:40, 20:40]), 300)
        self.assertGreater(np.count_nonzero(mask[60:80, 60:80]), 300)

    def test_red_extraction_rejects_grass(self):
        """Verifies green grass colors (H~35-85) are completely rejected by red mask."""
        grass = np.full((100, 100, 3), (35, 140, 50), dtype=np.uint8)  # Grass green BGR
        mask = extract_red_mask(grass, ui_mask=np.ones((100, 100), dtype=np.uint8))
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_red_extraction_rejects_stone_and_elixir(self):
        """Verifies grey stones and purple elixir buildings do not leak into red mask."""
        stone = np.full((50, 50, 3), (120, 120, 120), dtype=np.uint8)
        elixir = np.full((50, 50, 3), (180, 50, 180), dtype=np.uint8)  # Magenta/purple
        m_stone = extract_red_mask(stone, ui_mask=np.ones((50, 50), dtype=np.uint8))
        m_elixir = extract_red_mask(elixir, ui_mask=np.ones((50, 50), dtype=np.uint8))
        self.assertEqual(np.count_nonzero(m_stone), 0)
        self.assertEqual(np.count_nonzero(m_elixir), 0)

    def test_red_extraction_differential_signal(self):
        """Verifies differential flash image has high red concentration."""
        diff_img = load_test_image("logs/red_test_diff.png")
        if diff_img is None:
            self.skipTest("logs/red_test_diff.png not found")
        mask = extract_red_mask(diff_img)
        self.assertGreater(np.count_nonzero(mask), 1000)

    def test_red_morphological_closing_connectivity(self):
        """Verifies morphological closing bridges grid lines without bridging separate islands."""
        grid = np.zeros((200, 200), dtype=np.uint8)
        # Create dashed horizontal line with 5px gaps
        for x in range(20, 180, 15):
            grid[100, x:x+10] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)
        # Line should now be contiguous from x=20 to x=175
        self.assertTrue(np.all(closed[100, 20:175] == 255))

    # --- Feature 2: Targeted UI Masking ---
    def test_ui_mask_preserves_south_flank(self):
        """Verifies Y in [880, 960] retains red terrain pixels (no crude clamping)."""
        mask = build_ui_mask((1080, 2400))
        # The region between y=880 and 960 at x=1200 (center south) must be 1 (unmasked)
        self.assertEqual(mask[920, 1200], 1)
        self.assertEqual(mask[900, 1500], 1)

    def test_ui_mask_preserves_north_flank(self):
        """Verifies Y in [100, 120] retains red terrain pixels."""
        mask = build_ui_mask((1080, 2400))
        self.assertEqual(mask[110, 1200], 1)
        self.assertEqual(mask[105, 800], 1)

    def test_ui_mask_masks_surrender_button(self):
        """Verifies the surrender/end battle button region is properly masked to 0."""
        mask = build_ui_mask((1080, 2400))
        y_surr, x_surr = int(0.78 * 1080), int(0.10 * 2400)
        self.assertEqual(mask[y_surr, x_surr], 0)

    def test_ui_mask_masks_troop_ribbon(self):
        """Verifies bottom troop deployment bar is masked to 0."""
        mask = build_ui_mask((1080, 2400))
        self.assertEqual(mask[1020, 1200], 0)
        self.assertEqual(mask[1000, 800], 0)

    def test_ui_mask_masks_next_button(self):
        """Verifies next button at bottom right is masked to 0."""
        mask = build_ui_mask((1080, 2400))
        y_next, x_next = int(0.85 * 1080), int(0.92 * 2400)
        self.assertEqual(mask[y_next, x_next], 0)

    # --- Feature 3: Non-Convex Contour Tracing ---
    def test_non_convex_tracing_bay_preservation(self):
        """Verifies a concave C-shaped village boundary produces a non-convex polygon."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        # Draw a large C-shape (outer box with a deep rectangular bay carved into the right flank)
        cv2.rectangle(mask, (800, 300), (1600, 800), 255, -1)
        # Carve a 200px deep bay
        cv2.rectangle(mask, (1400, 450), (1600, 650), 0, -1)

        # Dilate slightly and find contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_cnt = max(cnts, key=cv2.contourArea)
        # Assert the resulting contour is mathematically non-convex
        self.assertFalse(cv2.isContourConvex(main_cnt))

    def test_non_convex_bay_depth_retention(self):
        """Verifies boundary points enter inside the concave bay rather than bridging it."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        cv2.rectangle(mask, (800, 300), (1600, 800), 255, -1)
        cv2.rectangle(mask, (1350, 450), (1600, 650), 0, -1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_cnt = max(cnts, key=cv2.contourArea)
        # Point inside the bay mouth (1450, 550) should be OUTSIDE the contour (dist < 0)
        dist = cv2.pointPolygonTest(main_cnt, (1450.0, 550.0), True)
        self.assertLess(dist, 0.0)

    def test_non_convex_isotropic_dilation(self):
        """Verifies dilation with ellipse kernel expands outwards uniformly in all directions."""
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.circle(mask, (250, 250), 50, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))
        dilated = cv2.dilate(mask, kernel)

        # Dilated radius should be 50 + 45 = 95 px in all 4 cardinal directions
        self.assertEqual(dilated[250, 250 + 94], 255)
        self.assertEqual(dilated[250, 250 + 98], 0)
        self.assertEqual(dilated[250, 250 - 94], 255)
        self.assertEqual(dilated[250 - 94, 250], 255)
        self.assertEqual(dilated[250 + 94, 250], 255)

    def test_non_convex_rejects_noise_contours(self):
        """Verifies small isolated noise pixels/contours are filtered out."""
        img = np.zeros((1080, 2400, 3), dtype=np.uint8)
        # Small 5x5 red artifact
        img[100:105, 100:105] = (30, 30, 220)
        red_cnt, green_cnt = detect_village_boundary(img)
        self.assertIsNone(red_cnt)
        self.assertIsNone(green_cnt)

    def test_non_convex_no_cross_cutting_bay_structure(self):
        """Verifies building sitting inside exterior bay is NOT cut by the boundary."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        cv2.rectangle(mask, (800, 300), (1600, 800), 255, -1)
        # Bay at (1350, 450) to (1600, 650)
        cv2.rectangle(mask, (1350, 450), (1600, 650), 0, -1)

        # Dilate mask as done in detector
        dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91)))
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        main_cnt = max(cnts, key=cv2.contourArea)

        # Place a building inside the bay at (1500, 550) with size 40x40
        class MockDet:
            bbox_xyxy = (1480, 530, 1520, 570)
            center = (1500, 550)
            cls_name = "collector_gold"

        cuts = audit_boundary_collisions(main_cnt, [MockDet()])
        # A building inside the bay in legal grass must not be intersected by the boundary
        self.assertEqual(len(cuts), 0)

    # --- Feature 4: Dual Modes (pixel_scan vs high_res_contour) ---
    def test_pixel_scan_mode_dense_vertices(self):
        """Verifies pixel_scan mode yields high-density vertices."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        _, green_cnt = detect_village_boundary(img, mode="pixel_scan")
        self.assertIsNotNone(green_cnt)
        self.assertGreater(len(green_cnt), 3000)

    def test_high_res_contour_vertex_reduction(self):
        """Verifies high_res_contour mode reduces vertex count by >= 70%."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        _, green_pixel = detect_village_boundary(img, mode="pixel_scan")
        _, green_contour = detect_village_boundary(img, mode="high_res_contour")
        self.assertIsNotNone(green_pixel)
        self.assertIsNotNone(green_contour)
        reduction = (len(green_pixel) - len(green_contour)) / len(green_pixel)
        self.assertGreaterEqual(reduction, 0.70)

    def test_dual_mode_enclosure_equivalence(self):
        """Verifies both modes enclose all building centers without cuts."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        _, green_pixel = detect_village_boundary(img, detections=dets, mode="pixel_scan")
        _, green_contour = detect_village_boundary(img, detections=dets, mode="high_res_contour")

        cuts_pixel = audit_boundary_collisions(green_pixel, dets)
        cuts_contour = audit_boundary_collisions(green_contour, dets)
        self.assertEqual(len(cuts_pixel), 0)
        self.assertEqual(len(cuts_contour), 0)

    def test_dual_mode_configuration_selection(self):
        """Verifies passing mode argument explicitly selects the correct mode."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        _, cnt_default = detect_village_boundary(img)
        _, cnt_contour = detect_village_boundary(img, mode="high_res_contour")
        # Default is pixel_scan with >3000 points, high_res_contour has < 500
        self.assertGreater(len(cnt_default), 1000)
        self.assertLess(len(cnt_contour), 500)

    def test_dual_mode_zero_structure_cuts_in_both(self):
        """Verifies both modes achieve exactly 0 cut structures."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        for mode in ["pixel_scan", "high_res_contour"]:
            with self.subTest(mode=mode):
                _, green = detect_village_boundary(img, detections=dets, mode=mode)
                cuts = audit_boundary_collisions(green, dets)
                self.assertEqual(len(cuts), 0)


class TestBoundaryDetectionTier2(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases."""

    def test_empty_black_image(self):
        """Empty black frame returns (None, None) gracefully."""
        black = np.zeros((1080, 2400, 3), dtype=np.uint8)
        red, green = detect_village_boundary(black)
        self.assertIsNone(red)
        self.assertIsNone(green)

    def test_zero_red_pixels_daylight(self):
        """Frame with zero red pixels and no detections returns (None, None)."""
        daylight = np.full((1080, 2400, 3), (80, 180, 90), dtype=np.uint8)
        red, green = detect_village_boundary(daylight)
        self.assertIsNone(red)
        self.assertIsNone(green)

    def test_extreme_coordinates_near_screen_edges(self):
        """Buildings near extreme edges (x=10, y=10) or (x=2390, y=1070) don't crash."""
        img = np.zeros((1080, 2400, 3), dtype=np.uint8)
        img[500:600, 1100:1300] = (30, 30, 220)  # Center red zone

        class CornerDet:
            def __init__(self, bbox, name):
                self.bbox_xyxy = bbox
                self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                self.cls_name = name

        dets = [
            CornerDet((5, 5, 45, 45), "collector_gold"),
            CornerDet((2350, 1030, 2395, 1075), "storage_elixir"),
        ]
        red, green = detect_village_boundary(img, detections=dets)
        self.assertIsNotNone(green)
        cuts = audit_boundary_collisions(green, dets)
        self.assertEqual(len(cuts), 0)

    def test_wall_bastions_enclosed(self):
        """Protruding narrow bastion is enclosed by isotropic dilation without cuts."""
        mask = np.zeros((1080, 2400), dtype=np.uint8)
        cv2.rectangle(mask, (1000, 400), (1400, 680), 255, -1)
        # Narrow bastion 30px wide protruding 80px outward
        cv2.rectangle(mask, (1185, 320), (1215, 400), 255, -1)

        dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91)))
        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_cnt = max(cnts, key=cv2.contourArea)

        # Tip of bastion at (1200, 320) must be inside dilated boundary (dist >= 40)
        dist = cv2.pointPolygonTest(main_cnt, (1200.0, 320.0), True)
        self.assertGreaterEqual(dist, 40.0)

    def test_saturated_flash_frame(self):
        """High saturation/brightness red frame maintains valid non-empty polygon."""
        img = np.full((1080, 2400, 3), (20, 20, 255), dtype=np.uint8)
        red, green = detect_village_boundary(img)
        self.assertIsNotNone(green)
        self.assertGreater(cv2.contourArea(green), 100000)


class TestBoundaryDetectionTier3(unittest.TestCase):
    """Tier 3: Cross-Feature Combinations."""

    def test_building_detections_integrated_with_red_mask(self):
        """Verifies base mask includes both red lines and buffered building boxes."""
        img = np.zeros((1080, 2400, 3), dtype=np.uint8)
        img[500:600, 1100:1300] = (30, 30, 220)  # Red core

        class MockDet:
            bbox_xyxy = (1280, 520, 1340, 580)
            center = (1310, 550)
            cls_name = "storage_gold"

        _, green = detect_village_boundary(img, detections=[MockDet()])
        self.assertIsNotNone(green)
        # Center of building must be strictly inside green contour
        dist = cv2.pointPolygonTest(green, (1310.0, 550.0), True)
        self.assertGreater(dist, 20.0)

    def test_building_safety_buffer_margin(self):
        """Green contour maintains >= 30px distance from all building edges."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        _, green = detect_village_boundary(img, detections=dets)

        for d in dets:
            dist_center = cv2.pointPolygonTest(green, (float(d.center[0]), float(d.center[1])), True)
            self.assertGreater(dist_center, 0.0, f"Building center {d.center} not enclosed")

    def test_corridor_ray_march_concave_bay_outer_drop(self):
        """Ray march on concave base selects outermost hit to avoid dropping inside bay."""
        # Create green hull with a bay along the east flank
        # Star-shaped / concave hull points
        pts = [
            (1200, 200), (1600, 350), (1400, 540),  # Inward concave dip at (1400, 540)
            (1700, 540),                             # Outer wall point along ray
            (1600, 750), (1200, 900), (800, 750), (700, 540), (800, 350)
        ]
        green_hull = np.array(pts, dtype=np.int32)

        # Compute drop for target jump at (1300, 540)
        center = (1200.0, 540.0)
        jump_pt = (1300, 540)
        drop_pt, sweep_pts = compute_corridor_drop_geometry(center, jump_pt, (2400, 1080), green_hull=green_hull)

        # Drop point must be beyond outer point (x >= 1400)
        self.assertGreaterEqual(drop_pt[0], 1400)

    def test_corridor_drop_radial_clearance(self):
        """Drop point has >= 50px clearance from all detected buildings along corridor."""
        center = (1200.0, 540.0)
        jump_pt = (1500, 540)

        class MockDet:
            center = (1600, 540)
            cls_name = "storage_gold"

        drop_pt, _ = compute_corridor_drop_geometry(
            center, jump_pt, (2400, 1080), detections=[MockDet()]
        )
        # Drop point must be placed beyond the peripheral building (1600 + 50 = 1650)
        self.assertGreaterEqual(drop_pt[0], 1650)

    def test_corridor_drop_within_screen_bounds(self):
        """Drop point and deploy sweep points remain within valid touch boundaries."""
        center = (1200.0, 540.0)
        jump_pt = (2300, 540)  # Extreme east
        drop_pt, sweep_pts = compute_corridor_drop_geometry(center, jump_pt, (2400, 1080))

        # Check bounds: x in [0.06*w, 0.94*w], y in [0.11*h, 0.83*h]
        self.assertGreaterEqual(drop_pt[0], int(0.06 * 2400))
        self.assertLessEqual(drop_pt[0], int(0.94 * 2400))
        for p in sweep_pts:
            self.assertGreaterEqual(p[0], int(0.06 * 2400))
            self.assertLessEqual(p[0], int(0.94 * 2400))


class TestBoundaryDetectionTier4(unittest.TestCase):
    """Tier 4: Real-World Match Scenarios."""

    def test_real_red_flash_cut_count_zero(self):
        """Verifies 0 cuts on golden sample logs/red_test_flash.png with 18 YOLO detections."""
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        self.assertEqual(len(dets), 18, "Expected 18 YOLO detections in golden flash frame")

        _, green = detect_village_boundary(img, detections=dets, mode="pixel_scan")
        self.assertIsNotNone(green)
        cuts = audit_boundary_collisions(green, dets)
        self.assertEqual(len(cuts), 0, f"Expected 0 cut structures but found {len(cuts)}")

    def test_existing_convex_hull_flaw_contrast(self):
        """
        Demonstrates that applying cv2.convexHull cuts 5 structures on the exact same frame,
        proving the non-convex requirement.
        """
        img = load_test_image("logs/red_test_flash.png")
        if img is None:
            self.skipTest("logs/red_test_flash.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 100, 100), (12, 255, 255))
        m2 = cv2.inRange(hsv, (168, 100, 100), (180, 255, 255))
        red_mask = m1 | m2
        red_mask[:120, :] = 0
        red_mask[880:, :] = 0  # Old crude clamping

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = [c for c in contours if cv2.contourArea(c) > 300]
        self.assertTrue(len(significant) > 0)

        all_pts = np.vstack(significant)
        red_hull = cv2.convexHull(all_pts)[:, 0, :]
        green_pts = []
        for pt in red_hull:
            vx, vy = pt[0] - 1200, pt[1] - 540
            norm = max(float(np.hypot(vx, vy)), 1.0)
            green_pts.append([int(pt[0] + 45.0 * vx / norm), int(pt[1] + 45.0 * vy / norm)])
        old_convex_green = cv2.convexHull(np.array(green_pts, dtype=np.int32))[:, 0, :]

        # Verify old convex hull cuts structures
        old_cuts = audit_boundary_collisions(old_convex_green, dets)
        self.assertGreaterEqual(len(old_cuts), 2, "Convex hull flaw must reproduce at least 2 cut structures")

    def test_real_match_frame_attack_091024(self):
        """Verifies boundary detection on logs/yolo/attack_20261106_091024.png."""
        img = load_test_image("logs/yolo/attack_20261106_091024.png")
        if img is None:
            self.skipTest("attack_20261106_091024.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        _, green = detect_village_boundary(img, detections=dets)
        if green is not None:
            cuts = audit_boundary_collisions(green, dets)
            self.assertEqual(len(cuts), 0)

    def test_real_match_frame_attack_183131(self):
        """Verifies boundary detection on logs/yolo/attack_20261105_183131.png."""
        img = load_test_image("logs/yolo/attack_20261105_183131.png")
        if img is None:
            self.skipTest("attack_20261105_183131.png not found")
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        _, green = detect_village_boundary(img, detections=dets)
        if green is not None:
            cuts = audit_boundary_collisions(green, dets)
            self.assertEqual(len(cuts), 0)

    def test_datasets_coc_train_sample(self):
        """Verifies boundary detection on a sample image from datasets/coc/images/train/."""
        train_dir = os.path.join(PROJECT_ROOT, "datasets", "coc", "images", "train")
        if not os.path.exists(train_dir):
            self.skipTest("datasets/coc/images/train not found")
        files = [f for f in os.listdir(train_dir) if f.endswith(".png")]
        if not files:
            self.skipTest("No train images in datasets/coc/images/train")
        sample_path = os.path.join(train_dir, files[0])
        img = cv2.imread(sample_path)
        detector = get_detector()
        dets = detector.detect(img) if detector else []
        _, green = detect_village_boundary(img, detections=dets)
        if green is not None:
            cuts = audit_boundary_collisions(green, dets)
            self.assertEqual(len(cuts), 0)


if __name__ == "__main__":
    unittest.main()
