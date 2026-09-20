"""Root shim for Module 3 (Parcel Extractor)."""

from modules.module3_parcel_extractor import (
    FMBLayoutSegmenter,
    ParcelBoundaryDetector,
    extract_parcel_from_source,
)

__all__ = [
    "FMBLayoutSegmenter",
    "ParcelBoundaryDetector",
    "extract_parcel_from_source",
]

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import cv2
    sample_path = "samples/user_fmb_input-1.png"
    if len(sys.argv) > 1:
        sample_path = sys.argv[1]
    elif not Path(sample_path).exists():
        sample_path = "samples/real_fmb_input-1.png"

    parcel, clean_canvas = extract_parcel_from_source(sample_path, survey_no="6/10A")
    cv2.imwrite("parcel_outline_only.png", clean_canvas)
    print(f"Extracted {len(parcel.lines)} parcel lines and {len(parcel.corners)} corner vertices.")
    print("Saved clean outline to: parcel_outline_only.png")
