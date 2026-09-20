"""Backend Processing Pipeline: Integrates Modules 1 → 5.

Full pipeline:
  M1: FMBPreprocessor     → enhanced binary image
  M2: LineDetector        → merged boundary line segments + topology
  M3: ParcelBoundaryDetector → clean boundary lines + clustered corners + edges
  M4: FMBTextExtractor    → OCR measurements + scale factor
  M5: CadastralPolygonBuilder → real-world scaled polygon

Web API method:
  process_for_web(source, survey_no) → FMBProcessingResult (JSON-serializable)
"""

import base64
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from modules.common.models import FMBProcessingResult, ParcelStructure
from modules.module1_preprocessor import FMBPreprocessor
from modules.module2_line_detector import LineDetector
from modules.module3_parcel_extractor import extract_parcel_from_source
from modules.module5_polygon import reconstruct_polygon

# OCR is optional — if EasyOCR isn't ready or fails, skip gracefully
try:
    from modules.module4_ocr import FMBTextExtractor
    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core pipeline entry points
# ---------------------------------------------------------------------------

def run_pipeline(
    image_source: Union[str, Path, bytes, np.ndarray],
) -> List[Tuple[int, int, int, int]]:
    """Execute M1 → M2 only: returns raw line segments."""
    binary = FMBPreprocessor().process(image_source)
    return LineDetector().detect(binary)


def process_for_web(
    source: Union[str, Path, bytes, np.ndarray],
    survey_no: Optional[str] = None,
) -> FMBProcessingResult:
    """Execute full M1→M5 pipeline and return JSON-ready result.

    Args:
        source: Image file path, bytes, or numpy array.
        survey_no: Survey number string (e.g. '6/10A').

    Returns:
        FMBProcessingResult with previews, parcel data, polygon, and messages.
    """
    errors = []

    # --- Module 1: Enhanced preprocessing ---
    preprocessor = FMBPreprocessor()
    binary_img = preprocessor.process(source)
    binary_b64 = FMBPreprocessor.to_base64_data_uri(binary_img)

    # --- Module 2: Line detection + topology ---
    detector = LineDetector(threshold=40, minLineLength=40.0, maxLineGap=15.0)
    raw_lines = detector.detect(binary_img)
    topology = detector.count_enclosing_edges(raw_lines, binary_img.shape[:2])

    # --- Load original colour image for overlay ---
    base_img = _load_color_image(source)

    overlay_img = LineDetector.draw_lines(base_img, raw_lines, color=(0, 0, 255), thickness=2)
    _, overlay_buf = cv2.imencode(".png", overlay_img)
    overlay_b64 = "data:image/png;base64," + base64.b64encode(overlay_buf).decode("ascii")

    # --- Module 3: Parcel extraction (precise boundary + corners) ---
    parcel, outline_img = extract_parcel_from_source(source, survey_no=survey_no)

    _, outline_buf = cv2.imencode(".png", outline_img)
    outline_b64 = "data:image/png;base64," + base64.b64encode(outline_buf).decode("ascii")

    # Inject topology data into parcel metadata
    parcel.scale_ratio = None  # will be set after OCR

    # --- Module 4: OCR measurement extraction ---
    measurements = []
    scale_info = None
    if _OCR_AVAILABLE:
        try:
            extractor = FMBTextExtractor(gpu=False, min_confidence=0.35)
            measurements, scale_info = extractor.run_full_extraction(
                source, lines=parcel.lines
            )
            parcel.measurements = measurements
            parcel.scale_info = scale_info
            if scale_info:
                parcel.scale_ratio = float(scale_info.scale_denominator)
        except Exception as e:
            errors.append(f"OCR warning (non-fatal): {str(e)[:120]}")
    else:
        errors.append("EasyOCR not available — skipping measurement extraction.")

    # --- Module 5: Polygon reconstruction ---
    cadastral_polygon = None
    polygon_b64 = None
    try:
        cadastral_polygon = reconstruct_polygon(
            parcel,
            scale_info=scale_info,
            measurements=measurements if measurements else None,
            survey_no=survey_no,
        )
        # Render polygon preview
        polygon_img = _render_polygon_preview(cadastral_polygon, parcel)
        _, poly_buf = cv2.imencode(".png", polygon_img)
        polygon_b64 = "data:image/png;base64," + base64.b64encode(poly_buf).decode("ascii")
    except Exception as e:
        errors.append(f"Polygon reconstruction warning: {str(e)[:120]}")

    msg = (
        f"Extracted {len(parcel.lines)} boundary lines, "
        f"{len(parcel.corners)} vertices, "
        f"{len(measurements)} measurements. "
        f"Topology: {topology.get('outer_boundary_count', 0)} outer boundary points. "
        + (" | " + " | ".join(errors) if errors else "")
    )

    return FMBProcessingResult(
        success=True,
        parcel=parcel,
        binary_preview_base64=binary_b64,
        overlay_preview_base64=overlay_b64,
        outline_preview_base64=outline_b64,
        polygon_preview_base64=polygon_b64,
        cadastral_polygon=cadastral_polygon,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_color_image(
    source: Union[str, Path, bytes, np.ndarray]
) -> np.ndarray:
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is not None:
            return img
    elif isinstance(source, np.ndarray):
        if source.ndim == 3:
            return source.copy()
        return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    elif isinstance(source, bytes):
        arr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


def _render_polygon_preview(cadastral_polygon, parcel: ParcelStructure) -> np.ndarray:
    """Render the reconstructed polygon on a white canvas (pixel space)."""
    from modules.common.models import CadastralPolygon
    h = max(parcel.image_height, 600)
    w = max(parcel.image_width, 800)
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    vertices = cadastral_polygon.vertices
    if len(vertices) < 3:
        cv2.putText(canvas, "Insufficient corners for polygon", (50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 0, 0), 2)
        return canvas

    # Draw polygon outline using pixel coords
    pts = np.array([v.pixel.to_tuple() for v in vertices], dtype=np.int32)
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 120, 255), thickness=3)
    cv2.fillPoly(canvas, [pts], color=(200, 230, 255))
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 80, 200), thickness=3)

    # Label each vertex with edge length
    n = len(vertices)
    for i, v in enumerate(vertices):
        px, py = v.pixel.to_tuple()
        cv2.circle(canvas, (px, py), 6, (200, 0, 0), -1)
        next_v = vertices[(i + 1) % n]
        mx = (px + next_v.pixel.x) // 2
        my = (py + next_v.pixel.y) // 2

        # Find corresponding edge length from polygon edges
        if v.real_world and vertices[(i + 1) % n].real_world:
            import math as _math
            dx = vertices[(i + 1) % n].real_world[0] - v.real_world[0]
            dy = vertices[(i + 1) % n].real_world[1] - v.real_world[1]
            edge_len = round(_math.hypot(dx, dy), 1)
            cv2.putText(canvas, f"{edge_len}m", (int(mx) + 4, int(my) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 0), 1, cv2.LINE_AA)

    # Metadata overlay
    info = (
        f"Survey: {cadastral_polygon.survey_no or 'N/A'} | "
        f"Area: {cadastral_polygon.area_sqm or 0:.1f} sqm | "
        f"Perimeter: {cadastral_polygon.perimeter_m or 0:.1f} m | "
        f"Scale: 1:{cadastral_polygon.scale_denominator}"
    )
    cv2.rectangle(canvas, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(canvas, info, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas
