"""FMB Digitizer Pipeline - Unified Integration & Web Service Layer.

Connects:
- Module 1: FMBPreprocessor (Image Cleaning & Binarization)
- Module 2: LineDetector (Cadastral Line Extraction)
- Parcel Extractor: Layout Segmentation & Boundary Corner Detection

Provides both Python API methods and Web GIS serialization interfaces.
"""

import base64
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from core.models import FMBProcessingResult, LineSegment2D, ParcelStructure, Point2D
from line_detector import LineDetector
from parcel_extractor import FMBLayoutSegmenter, ParcelBoundaryDetector, extract_parcel_from_source

try:
    from fmb_preprocessor import FMBPreprocessor
except ImportError:
    class FMBPreprocessor:  # type: ignore
        def process(self, image_source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
            if isinstance(image_source, np.ndarray):
                gray = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY) if image_source.ndim == 3 else image_source
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                return binary
            return np.ones((500, 500), dtype=np.uint8) * 255


def run_pipeline(
    image_source: Union[str, Path, bytes, np.ndarray],
    preprocessor: Optional[FMBPreprocessor] = None,
    detector: Optional[LineDetector] = None,
) -> List[Tuple[int, int, int, int]]:
    """Execute end-to-end line detection pipeline (Module 1 -> Module 2).

    Args:
        image_source: Image file path, raw binary bytes, or NumPy array.
        preprocessor: Optional pre-configured FMBPreprocessor instance.
        detector: Optional pre-configured LineDetector instance.

    Returns:
        List[Tuple[int, int, int, int]]: Detected line segment coordinates [(x1, y1, x2, y2), ...].
    """
    if preprocessor is None:
        preprocessor = FMBPreprocessor()

    binary_image = preprocessor.process(image_source)

    if detector is None:
        detector = LineDetector()

    lines = detector.detect(binary_image)
    return lines


def process_for_web(
    source: Union[str, Path, bytes, np.ndarray],
    survey_no: Optional[str] = None,
) -> FMBProcessingResult:
    """Execute the complete pipeline and return web-ready response with Base64 previews and GeoJSON.

    Ideal for FastAPI / Flask / Streamlit endpoints accepting HTTP multipart/file uploads.

    Args:
        source: Image file path, HTTP uploaded file bytes, or in-memory array.
        survey_no: Optional survey number identifier (e.g., '6/10A').

    Returns:
        FMBProcessingResult: Unified result object containing GeoJSON, metrics, and base64 previews.
    """
    # 1. Module 1: Preprocessing & Binarization
    preprocessor = FMBPreprocessor()
    binary_img = preprocessor.process(source)
    binary_data_uri = FMBPreprocessor.to_base64_data_uri(binary_img)

    # 2. Module 2: Cadastral Line Detection
    detector = LineDetector(threshold=80, minLineLength=60.0, maxLineGap=15.0)
    raw_lines = detector.detect(binary_img)

    # 3. Parcel Outline Isolation & Corner Extraction
    parcel, clean_outline_img = extract_parcel_from_source(source, survey_no=survey_no)

    # 4. Generate Previews
    # Outline preview
    _, outline_buf = cv2.imencode(".png", clean_outline_img)
    outline_b64 = "data:image/png;base64," + base64.b64encode(outline_buf).decode("ascii")

    # Overlay preview
    if isinstance(source, (str, Path)):
        base_img = cv2.imread(str(source))
    elif isinstance(source, np.ndarray):
        base_img = source if source.ndim == 3 else cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    elif isinstance(source, bytes):
        base_img = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
    else:
        base_img = np.ones((parcel.image_height, parcel.image_width, 3), dtype=np.uint8) * 255

    overlay_img = LineDetector.draw_lines(base_img, raw_lines, color=(0, 0, 255), thickness=3)
    _, overlay_buf = cv2.imencode(".png", overlay_img)
    overlay_b64 = "data:image/png;base64," + base64.b64encode(overlay_buf).decode("ascii")

    return FMBProcessingResult(
        success=True,
        parcel=parcel,
        binary_preview_base64=binary_data_uri,
        overlay_preview_base64=overlay_b64,
        outline_preview_base64=outline_b64,
        message=f"Successfully extracted {len(parcel.lines)} parcel boundary lines and {len(parcel.corners)} vertices.",
    )


if __name__ == "__main__":
    print("=" * 65)
    print("FMB PIPELINE: Reusable Cadastral Processing & Web API Engine")
    print("=" * 65)

    sample_file = Path("samples/user_fmb_input-1.png")
    if not sample_file.exists():
        sample_file = Path("samples/real_fmb_input-1.png")

    if sample_file.exists():
        print(f"\nProcessing sample: {sample_file.name}")
        result = process_for_web(sample_file, survey_no="6/10A")
        print(f"Status: {result.message}")
        print(f"GeoJSON Features count: {len(result.to_geojson()['features'])}")
        print(f"Base64 previews generated: Binary ({len(result.binary_preview_base64 or '')} chars), Overlay ({len(result.overlay_preview_base64 or '')} chars), Outline ({len(result.outline_preview_base64 or '')} chars)")

        # Save GeoJSON export to outputs/
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        geojson_path = output_dir / "parcel_features.geojson"
        with open(geojson_path, "w", encoding="utf-8") as f:
            f.write(result.parcel.to_json())
        print(f"Saved GeoJSON export to: {geojson_path}")
