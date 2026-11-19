"""
Package initialization and shared test infrastructure for ClashFarmerbot test suite.
Supports offline execution using standard library unittest without requiring physical ADB hardware.
"""
import os
import sys
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class MockDeviceShell:
    """Mock interface for device.d (uiautomator2 shell and click actions)."""
    def __init__(self, parent_device):
        self.parent = parent_device

    def shell(self, cmd: str) -> str:
        self.parent.shell_history.append(cmd)
        return ""

    def click(self, x: int, y: int):
        self.parent.tap_history.append((int(x), int(y)))


class MockDevice:
    """Offline mock device implementing the Device protocol used by DeploymentStrategy."""
    is_mock: bool = True

    def __init__(self, width: int = 2400, height: int = 1080, screenshot_feed: list[np.ndarray] | None = None):
        self.is_mock = True
        self.width = width
        self.height = height
        self.shell_history: list[str] = []
        self.tap_history: list[tuple[int, int]] = []
        self.hold_tap_history: list[tuple[int, int, float]] = []
        self.swipe_history: list[tuple[int, int, int, int, float]] = []
        self.pan_flank_history: list[tuple[str, bool, int]] = []
        self.zoom_out_calls: int = 0
        self.zoom_half_calls: int = 0
        self.press_back_calls: int = 0
        self.screenshot_feed = list(screenshot_feed) if screenshot_feed else []
        self.default_screenshot: np.ndarray | None = None
        self.d = MockDeviceShell(self)

    def set_default_screenshot(self, img: np.ndarray):
        self.default_screenshot = img.copy()

    def screenshot(self) -> np.ndarray:
        if self.screenshot_feed:
            return self.screenshot_feed.pop(0)
        if self.default_screenshot is not None:
            return self.default_screenshot.copy()
        # Default empty black frame 2400x1080
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def tap(self, x: int | float, y: int | float, humanize: bool = False):
        self.tap_history.append((int(x), int(y)))

    def raw_tap(self, x: int | float, y: int | float):
        self.tap_history.append((int(x), int(y)))

    def press_back(self):
        self.press_back_calls += 1

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        self.swipe_history.append((int(x1), int(y1), int(x2), int(y2), float(duration)))

    def hold_tap(self, x: int | float, y: int | float, duration: float = 0.85):
        self.hold_tap_history.append((int(x), int(y), float(duration)))
        self.tap_history.append((int(x), int(y)))

    def zoom_out(self, center: tuple[int, int] = (1200, 540)):
        self.zoom_out_calls += 1

    def zoom_half(self):
        self.zoom_half_calls += 1

    def pan_camera(self, x1_rel: float, y1_rel: float, x2_rel: float, y2_rel: float, duration: float = 0.9):
        sx1, sy1 = int(x1_rel * self.width), int(y1_rel * self.height)
        sx2, sy2 = int(x2_rel * self.width), int(y2_rel * self.height)
        self.swipe(sx1, sy1, sx2, sy2, duration=duration)

    def pan_flank(self, flank: str, reverse: bool = False, duration_ms: int = 480) -> None:
        self.pan_flank_history.append((flank, reverse, duration_ms))
        flank_coords = {
            "north": ((0.50, 0.222), (0.50, 0.759)),
            "south": ((0.50, 0.722), (0.50, 0.222)),
            "east": ((0.646, 0.50), (0.354, 0.50)),
            "west": ((0.354, 0.50), (0.646, 0.50)),
        }
        flank_key = flank.lower()
        if flank_key in flank_coords:
            p1, p2 = flank_coords[flank_key]
            if reverse:
                p1, p2 = p2, p1
            self.swipe(
                int(round(p1[0] * self.width)),
                int(round(p1[1] * self.height)),
                int(round(p2[0] * self.width)),
                int(round(p2[1] * self.height)),
                duration=duration_ms / 1000.0,
            )

    def wait_camera_settled(self, timeout: float = 2.5, **kwargs) -> bool:
        return True

    def clear_history(self):
        self.shell_history.clear()
        self.tap_history.clear()
        self.hold_tap_history.clear()
        self.swipe_history.clear()
        self.pan_flank_history.clear()
        self.zoom_out_calls = 0
        self.zoom_half_calls = 0
        self.press_back_calls = 0


def count_structure_intersections(contour: np.ndarray, detections: list) -> int:
    """
    Checks if a polygon contour intersects/slices any detected building.
    A building is cut if its bounding box vertices have both interior (dist > 0)
    and exterior (dist < 0) points relative to the contour polygon.
    """
    if contour is None or len(contour) < 3 or not detections:
        return 0

    cnt = np.array(contour, dtype=np.int32)
    if cnt.ndim == 2:
        cnt = cnt.reshape((-1, 1, 2))

    cut_count = 0
    for d in detections:
        if not hasattr(d, "bbox_xyxy"):
            continue
        x1, y1, x2, y2 = d.bbox_xyxy
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        dists = [cv2.pointPolygonTest(cnt, (float(cx), float(cy)), True) for cx, cy in corners]
        # If some corners are strictly inside (>0) and some strictly outside (<0), the polygon edge cuts the building
        if min(dists) < -1.0 and max(dists) > 1.0:
            cut_count += 1
    return cut_count


def load_test_image(path: str) -> np.ndarray | None:
    """Loads an image relative to PROJECT_ROOT safely."""
    full_path = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
    if os.path.exists(full_path):
        return cv2.imread(full_path)
    return None
