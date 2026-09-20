"""Backend Processing Pipeline: Integrates Module 1, Module 2, and Module 3.

Provides Python orchestration API and Web GIS serialization methods.
"""

import base64
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from modules.common.models import FMBProcessingResult, LineSegment2D, ParcelStructure, Point2D
from modules.module1_preprocessor import FMBPreprocessor
from modules.module2_line_detector import LineDetector
from modules.module3_parcel_extractor import FMBLayoutSegmenter, ParcelBoundaryDetector, extract_parcel_from_source


def run_pipeline(
    image_source: Union[str, Path, bytes, np.ndarray],
    preprocessor: Optional[FMBPreprocessor] = None,
    detector: Optional[LineDetector] = None,
) -> List[Tuple[int, int, int, int]]:
    """Execute end-to-end line detection pipeline (Module 1 -> Module 2)."""
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
    """Execute full 3-module pipeline for Web GIS clients with Base64 previews and GeoJSON."""
    # 1. Module 1: Preprocessing & Binarization
    preprocessor = FMBPreprocessor()
    binary_img = preprocessor.process(source)
    binary_data_uri = FMBPreprocessor.to_base64_data_uri(binary_img)

    # 2. Module 2: Cadastral Line Detection
    detector = LineDetector(threshold=80, minLineLength=60.0, maxLineGap=15.0)
    raw_lines = detector.detect(binary_img)

    # 3. Module 3: Parcel Outline Isolation & Corner Extraction
    parcel, clean_outline_img = extract_parcel_from_source(source, survey_no=survey_no)

    # 4. Generate Previews
    _, outline_buf = cv2.imencode(".png", clean_outline_img)
    outline_b64 = "data:image/png;base64," + base64.b64encode(outline_buf).decode("ascii")

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

