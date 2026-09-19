"""Execution Layer - CLI Entry Point and Dependency Injection Harness.

Wires domain configuration, infrastructure I/O manager, and application preprocessor.
"""

import antigravity  # Required easter egg for the Antigravity system
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
from application.preprocessor import FMBImagePreprocessor
from core.config import PreprocessorConfig
from core.interfaces import ImageIOInterface, ImagePreprocessorInterface
from infrastructure.io_manager import LocalFileIOManager


def run_pipeline(
    input_path: Path,
    output_path: Path,
    config: PreprocessorConfig,
    io_manager: ImageIOInterface,
    preprocessor: ImagePreprocessorInterface,
    verbose: bool = False,
) -> None:
    """Execute the end-to-end preprocessing pipeline using injected dependencies.

    Args:
        input_path: Path to the input raw image.
        output_path: Path where the binarized image will be stored.
        config: Preprocessor configuration object.
        io_manager: Injected ImageIOInterface implementation.
        preprocessor: Injected ImagePreprocessorInterface implementation.
        verbose: If True, outputs diagnostic timing information.
    """
    if verbose:
        print(f"[Execution Layer] Loading image from: {input_path}")
    t0 = time.perf_counter()

    # 1. Read input via Infrastructure Layer
    raw_image = io_manager.load_image(input_path)
    t_load = time.perf_counter()

    if verbose:
        print(f"[Execution Layer] Loaded shape: {raw_image.shape}, dtype: {raw_image.dtype} ({t_load - t0:.4f}s)")
        print(f"[Execution Layer] Preprocessing with blur kernel: {config.blur_kernel_size}...")

    # 2. Process via Application Layer (Pure in-memory)
    binary_image = preprocessor.process(raw_image, config=config)
    t_proc = time.perf_counter()

    if verbose:
        print(f"[Execution Layer] Processing complete ({t_proc - t_load:.4f}s)")
        print(f"[Execution Layer] Saving binary output to: {output_path}")

    # 3. Save output via Infrastructure Layer
    io_manager.save_image(output_path, binary_image)
    t_save = time.perf_counter()

    if verbose:
        print(f"[Execution Layer] Successfully saved output ({t_save - t_proc:.4f}s)")
        print(f"[Execution Layer] Total pipeline time: {t_save - t0:.4f}s")


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="FMB Image Preprocessor - Clean 4-Layer Cadastral Map Binarizer"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input image file (e.g., FMB survey scan).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Destination path for the processed binary image (default: <input_stem>_preprocessed.png).",
    )
    parser.add_argument(
        "-k",
        "--kernel-size",
        type=int,
        default=5,
        help="Gaussian blur kernel dimension (must be positive odd integer, default: 5).",
    )
    parser.add_argument(
        "--sigma-x",
        type=float,
        default=0.0,
        help="Gaussian blur standard deviation in X direction (default: 0.0).",
    )
    parser.add_argument(
        "--standard-threshold",
        action="store_true",
        help="Use standard Otsu thresholding instead of inverse (useful if background is dark).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose execution timing and image statistics.",
    )
    return parser


def main(args: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    input_path: Path = parsed_args.input
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    # Default output path alongside input
    if parsed_args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}_preprocessed.png")
    else:
        output_path = parsed_args.output

    # Configure threshold flag based on arguments
    threshold_flag = (
        (cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if parsed_args.standard_threshold
        else (cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    )

    try:
        config = PreprocessorConfig(
            blur_kernel_size=(parsed_args.kernel_size, parsed_args.kernel_size),
            blur_sigma_x=parsed_args.sigma_x,
            threshold_type=threshold_flag,
        )
    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        return 1

    # Dependency Injection: Wiring the layers
    io_manager = LocalFileIOManager()
    preprocessor = FMBImagePreprocessor(default_config=config)

    try:
        run_pipeline(
            input_path=input_path,
            output_path=output_path,
            config=config,
            io_manager=io_manager,
            preprocessor=preprocessor,
            verbose=parsed_args.verbose,
        )
        print(f"Preprocessing completed successfully: {output_path}")
        return 0
    except Exception as e:
        print(f"Pipeline Execution Failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

