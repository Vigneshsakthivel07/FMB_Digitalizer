"""Root shim for Module 1 (FMBPreprocessor)."""

from modules.module1_preprocessor import FMBPreprocessor

__all__ = ["FMBPreprocessor"]

if __name__ == "__main__":
    from pathlib import Path
    import sys
    import cv2
    sample = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples/user_fmb_input-1.png")
    if not sample.exists():
        sample = Path("samples/real_fmb_input-1.png")
    p = FMBPreprocessor()
    res = p.process(sample)
    print(f"Preprocessor result shape: {res.shape}")
