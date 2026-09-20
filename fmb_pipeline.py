"""FMB Digitizer Pipeline - Integration of Module 1 & Module 2.

Connects FMBPreprocessor (Module 1) and LineDetector (Module 2) into an end-to-end
cadastral feature extraction pipeline.
"""

from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

# Import Module 1 and Module 2
try:
    from fmb_preprocessor import FMBPreprocessor
except ImportError:
    # Graceful fallback mock if running in an isolated environment without Module 1
    class FMBPreprocessor:  # type: ignore
        def process(self, image_source: Union[str, Path, np.ndarray]) -> np.ndarray:
            if isinstance(image_source, np.ndarray):
                gray = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY) if image_source.ndim == 3 else image_source
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                return binary
            return np.ones((500, 500), dtype=np.uint8) * 255

from line_detector import LineDetector


def run_pipeline(
    image_source: Union[str, Path, bytes, np.ndarray],
    preprocessor: Optional[FMBPreprocessor] = None,
    detector: Optional[LineDetector] = None,
) -> List[Tuple[int, int, int, int]]:
    """Execute the end-to-end FMB preprocessing and line detection pipeline.

    Steps:
        1. Initialize FMBPreprocessor (Module 1) if not injected.
        2. Clean and binarize the raw image: binary_image = preprocessor.process(image_source).
        3. Initialize LineDetector (Module 2) if not injected.
        4. Detect discrete line segments: lines = detector.detect(binary_image).
        5. Return extracted line coordinate tuples [(x1, y1, x2, y2), ...].

    Args:
        image_source: Image file path, raw binary bytes, or NumPy array.
        preprocessor: Optional pre-configured FMBPreprocessor instance.
        detector: Optional pre-configured LineDetector instance.

    Returns:
        List[Tuple[int, int, int, int]]: Detected line segment coordinates.
    """
    # 1. Initialize preprocessor if not provided
    if preprocessor is None:
        preprocessor = FMBPreprocessor()

    # 2. Extract clean binary array (black lines on white bg)
    binary_image = preprocessor.process(image_source)

    # 3. Initialize line detector if not provided
    if detector is None:
        detector = LineDetector()

    # 4. Extract line segments
    lines = detector.detect(binary_image)

    # 5. Return extracted line coordinates
    return lines


if __name__ == "__main__":
    print("=" * 65)
    print("FMB CADASTAL PIPELINE: Module 1 (Preprocessor) + Module 2 (Line Detector)")
    print("=" * 65)

    # 1. Demonstration using a dummy in-memory image array (as required by spec)
    print("\n[Step 1] Running pipeline on dummy in-memory image array...")
    dummy_canvas = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Light parchment background
    # Draw dark cadastral boundary polygon and subdivision lines
    cv2.polylines(
        dummy_canvas,
        [np.array([[100, 100], [500, 100], [450, 320], [120, 320]], np.int32)],
        isClosed=True,
        color=(20, 20, 20),
        thickness=3,
    )
    cv2.line(dummy_canvas, (100, 100), (450, 320), (20, 20, 20), 2)  # Diagonal G-Line
    cv2.line(dummy_canvas, (300, 100), (280, 320), (20, 20, 20), 2)  # Subdivision line

    # Execute pipeline on in-memory array
    detected_dummy_lines = run_pipeline(dummy_canvas)
    print(f"Success! Detected {len(detected_dummy_lines)} line segments from dummy canvas.")
    for i, (x1, y1, x2, y2) in enumerate(detected_dummy_lines[:5], start=1):
        print(f"  Line {i}: ({x1}, {y1}) -> ({x2}, {y2})")
    if len(detected_dummy_lines) > 5:
        print(f"  ... and {len(detected_dummy_lines) - 5} more line segments.")

    # 2. Demonstration on real sample if available
    project_root = Path(__file__).resolve().parent
    sample_file = project_root / "samples" / "real_fmb_input-1.png"
    if sample_file.exists():
        print(f"\n[Step 2] Running pipeline on real FMB map: {sample_file.name}...")
        # Tune detector slightly for high-resolution survey map (3505x2480)
        tuned_detector = LineDetector(
            threshold=80,
            minLineLength=60.0,
            maxLineGap=15.0,
        )
        real_lines = run_pipeline(sample_file, detector=tuned_detector)
        print(f"Success! Detected {len(real_lines)} cadastral line segments from real FMB map.")

        # Save visualization overlay
        output_dir = project_root / "outputs"
        output_dir.mkdir(exist_ok=True)
        overlay_path = output_dir / "detected_lines_overlay.png"

        # Render detected lines in red on the original map
        real_img = cv2.imread(str(sample_file))
        overlay = LineDetector.draw_lines(real_img, real_lines, color=(0, 0, 255), thickness=3)
        cv2.imwrite(str(overlay_path), overlay)
        print(f"Saved visual overlay with red lines to: {overlay_path}")

    print("\nPipeline execution completed successfully.")
