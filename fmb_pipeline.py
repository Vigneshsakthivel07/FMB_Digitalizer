"""Root shim for Pipeline (Integration & Web Service Layer)."""

from backend.pipeline import run_pipeline, process_for_web

__all__ = ["run_pipeline", "process_for_web"]

if __name__ == "__main__":
    from pathlib import Path
    sample_file = Path("samples/user_fmb_input-1.png")
    if not sample_file.exists():
        sample_file = Path("samples/real_fmb_input-1.png")
    result = process_for_web(sample_file, survey_no="6/10A")
    print(f"Status: {result.message}")
    print(f"Features: {len(result.to_geojson()['features'])}")
