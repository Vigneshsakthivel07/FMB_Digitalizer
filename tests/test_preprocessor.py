"""Unit and integration tests for FMB Image Preprocessor module."""

import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

from core.config import PreprocessorConfig
from core.interfaces import ImageIOInterface, ImagePreprocessorInterface
from infrastructure.io_manager import LocalFileIOManager
from application.preprocessor import FMBImagePreprocessor
from main import run_pipeline, build_parser, main


class TestDomainLayer(unittest.TestCase):
    """Test domain layer configuration and interface contracts."""

    def test_default_config_validity(self):
        config = PreprocessorConfig()
        self.assertEqual(config.blur_kernel_size, (5, 5))
        self.assertEqual(config.blur_sigma_x, 0.0)
        self.assertEqual(config.max_value, 255.0)

    def test_custom_config_validity(self):
        config = PreprocessorConfig(blur_kernel_size=(7, 7), blur_sigma_x=1.5)
        self.assertEqual(config.blur_kernel_size, (7, 7))
        self.assertEqual(config.blur_sigma_x, 1.5)

    def test_invalid_kernel_even(self):
        with self.assertRaises(ValueError):
            PreprocessorConfig(blur_kernel_size=(4, 5))
        with self.assertRaises(ValueError):
            PreprocessorConfig(blur_kernel_size=(5, 6))

    def test_invalid_kernel_nonpositive(self):
        with self.assertRaises(ValueError):
            PreprocessorConfig(blur_kernel_size=(-1, 5))
        with self.assertRaises(ValueError):
            PreprocessorConfig(blur_kernel_size=(0, 5))

    def test_invalid_max_value(self):
        with self.assertRaises(ValueError):
            PreprocessorConfig(max_value=300.0)
        with self.assertRaises(ValueError):
            PreprocessorConfig(max_value=-10.0)

    def test_interfaces_are_abstract(self):
        with self.assertRaises(TypeError):
            ImageIOInterface()
        with self.assertRaises(TypeError):
            ImagePreprocessorInterface()


class TestInfrastructureLayer(unittest.TestCase):
    """Test infrastructure layer I/O operations."""

    def setUp(self):
        self.io_manager = LocalFileIOManager()
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_save_and_load_image(self):
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[40:60, 40:60] = [0, 255, 128]

        save_path = self.temp_path / "subdir" / "test_image.png"
        result = self.io_manager.save_image(save_path, test_img)
        self.assertTrue(result)
        self.assertTrue(save_path.exists())

        loaded = self.io_manager.load_image(save_path)
        self.assertEqual(loaded.shape, test_img.shape)
        np.testing.assert_array_equal(loaded, test_img)

    def test_load_nonexistent_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            self.io_manager.load_image(self.temp_path / "non_existent.png")

    def test_save_empty_array_raises_error(self):
        with self.assertRaises(ValueError):
            self.io_manager.save_image(self.temp_path / "out.png", np.array([]))

    def test_save_invalid_type_raises_error(self):
        with self.assertRaises(ValueError):
            self.io_manager.save_image(self.temp_path / "out.png", "not_an_array")  # type: ignore


class TestApplicationLayer(unittest.TestCase):
    """Test core image preprocessing algorithm logic in application layer."""

    def setUp(self):
        self.preprocessor = FMBImagePreprocessor()

    def test_pure_array_processing_bgr(self):
        # Create image with light background and a dark line
        img = np.ones((100, 100, 3), dtype=np.uint8) * 240
        img[45:55, :] = 20  # Dark horizontal line

        binary = self.preprocessor.process(img)

        # Output must be 2D uint8 with only 0 and 255
        self.assertEqual(binary.ndim, 2)
        self.assertEqual(binary.dtype, np.uint8)
        unique_vals = set(np.unique(binary))
        self.assertTrue(unique_vals.issubset({0, 255}))

        # Line should be extracted as foreground (255)
        self.assertEqual(binary[50, 50], 255)
        # Background should be suppressed to 0
        self.assertEqual(binary[10, 10], 0)

    def test_single_channel_grayscale(self):
        img_gray = np.ones((80, 80), dtype=np.uint8) * 230
        img_gray[30:50, 30:50] = 30  # Dark square

        binary = self.preprocessor.process(img_gray)
        self.assertEqual(binary.shape, (80, 80))
        self.assertEqual(binary[40, 40], 255)
        self.assertEqual(binary[5, 5], 0)

    def test_bgra_channel_conversion(self):
        img_bgra = np.ones((60, 60, 4), dtype=np.uint8) * 220
        img_bgra[20:40, 20:40, :3] = 15

        binary = self.preprocessor.process(img_bgra)
        self.assertEqual(binary.shape, (60, 60))
        self.assertEqual(binary[30, 30], 255)

    def test_float_normalized_input(self):
        img_float = np.ones((50, 50, 3), dtype=np.float32) * 0.95
        img_float[20:30, :] = 0.1

        binary = self.preprocessor.process(img_float)
        self.assertEqual(binary.dtype, np.uint8)
        self.assertEqual(binary[25, 25], 255)
        self.assertEqual(binary[5, 5], 0)

    def test_empty_array_raises_error(self):
        with self.assertRaises(ValueError):
            self.preprocessor.process(np.array([]))

    def test_invalid_dimension_raises_error(self):
        with self.assertRaises(ValueError):
            self.preprocessor.process(np.ones((10, 10, 10, 10)))

    def test_invalid_type_raises_error(self):
        with self.assertRaises(ValueError):
            self.preprocessor.process("not_an_image")  # type: ignore


class TestExecutionLayer(unittest.TestCase):
    """Test CLI harness and dependency injection."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.test_dir.name)

        # Create dummy input file
        self.input_file = self.temp_path / "dummy_survey.png"
        dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 230
        dummy_img[48:52, :] = 10
        io = LocalFileIOManager()
        io.save_image(self.input_file, dummy_img)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_run_pipeline_with_mock_io(self):
        """Verify pipeline accepts mock implementations (dependency inversion)."""
        class MockIO(ImageIOInterface):
            def __init__(self):
                self.loaded = False
                self.saved = False
                self.saved_image = None

            def load_image(self, source):
                self.loaded = True
                return np.ones((50, 50, 3), dtype=np.uint8) * 200

            def save_image(self, destination, image):
                self.saved = True
                self.saved_image = image
                return True

        mock_io = MockIO()
        preprocessor = FMBImagePreprocessor()
        config = PreprocessorConfig()

        run_pipeline(
            input_path=Path("dummy_in.png"),
            output_path=Path("dummy_out.png"),
            config=config,
            io_manager=mock_io,
            preprocessor=preprocessor,
        )

        self.assertTrue(mock_io.loaded)
        self.assertTrue(mock_io.saved)
        self.assertIsNotNone(mock_io.saved_image)

    def test_cli_execution_success(self):
        out_file = self.temp_path / "output.png"
        exit_code = main([str(self.input_file), "-o", str(out_file), "-k", "5", "-v"])
        self.assertEqual(exit_code, 0)
        self.assertTrue(out_file.exists())

    def test_cli_execution_nonexistent_file(self):
        exit_code = main([str(self.temp_path / "non_existent.png")])
        self.assertEqual(exit_code, 1)

    def test_cli_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["test_input.png"])
        self.assertEqual(args.input, Path("test_input.png"))
        self.assertIsNone(args.output)
        self.assertEqual(args.kernel_size, 5)
        self.assertEqual(args.sigma_x, 0.0)


class TestFMBPreprocessorModule1(unittest.TestCase):
    """Test Module 1 specification (fmb_preprocessor.FMBPreprocessor)."""

    def setUp(self):
        from fmb_preprocessor import FMBPreprocessor
        self.preprocessor = FMBPreprocessor(blur_kernel_size=(5, 5))
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.test_dir.name)

        # Create sample test image with dark line on light paper
        self.sample_file = self.temp_path / "sample.png"
        img = np.ones((100, 100, 3), dtype=np.uint8) * 235
        img[45:55, :] = 25  # Dark line
        cv2.imwrite(str(self.sample_file), img)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_read_image(self):
        img = self.preprocessor.read_image(self.sample_file)
        self.assertEqual(img.shape, (100, 100, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_to_grayscale(self):
        img = self.preprocessor.read_image(self.sample_file)
        gray = self.preprocessor.to_grayscale(img)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.shape, (100, 100))

    def test_apply_blur(self):
        gray = np.ones((50, 50), dtype=np.uint8) * 200
        blurred = self.preprocessor.apply_blur(gray)
        self.assertEqual(blurred.shape, (50, 50))

    def test_apply_threshold_black_lines_on_white_background(self):
        # Background is light (235), line is dark (25)
        gray = np.ones((60, 60), dtype=np.uint8) * 235
        gray[25:35, :] = 25  # Dark line
        blurred = self.preprocessor.apply_blur(gray)
        binary = self.preprocessor.apply_threshold(blurred)

        # Verify binary output
        self.assertEqual(binary.dtype, np.uint8)
        self.assertTrue(set(np.unique(binary)).issubset({0, 255}))

        # Requirement check: BLACK LINES (0) on WHITE BACKGROUND (255)
        self.assertEqual(binary[30, 30], 0, "Line pixel must be black (0)")
        self.assertEqual(binary[10, 10], 255, "Background pixel must be white (255)")

    def test_full_process_pipeline(self):
        binary = self.preprocessor.process(self.sample_file)
        self.assertIsInstance(binary, np.ndarray)
        self.assertEqual(binary.shape, (100, 100))
        self.assertEqual(binary[50, 50], 0)    # Dark line -> 0
        self.assertEqual(binary[10, 10], 255)  # Background -> 255


if __name__ == "__main__":
    unittest.main()

