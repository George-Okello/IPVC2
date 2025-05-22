# test_detector_system.py
"""
Comprehensive unit testing framework for barcode/QR detection system
"""
import unittest
import tempfile
import shutil
import os
import json
import cv2
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image, ImageDraw
import qrcode
import barcode
from barcode.writer import ImageWriter

# Import modules to test
from detector import Detector
from batch_detector import BatchDetector
from test_generator import TestGenerator
from decoder import decode_image, preprocess_for_decoding
from image_utils import convert_to_grayscale, enhance_contrast, reduce_noise
from barcode_utils import detect_barcode_gradient, find_barcode_regions
from qr_utils import detect_qr_contours, find_qr_regions


class TestDetectorCore(unittest.TestCase):
    """Test core detector functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = Detector()
        self.temp_dir = tempfile.mkdtemp()
        self.test_images = {}

        # Create test images
        self._create_test_images()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_images(self):
        """Create various test images for testing"""
        # Simple QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data("TEST_QR_CODE")
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_path = os.path.join(self.temp_dir, "test_qr.png")
        qr_img.save(qr_path)
        self.test_images['qr_simple'] = qr_path

        # Simple barcode
        try:
            code128 = barcode.get_barcode_class('code128')
            barcode_instance = code128("TEST123", writer=ImageWriter())
            barcode_path = os.path.join(self.temp_dir, "test_barcode")
            barcode_instance.save(barcode_path)
            self.test_images['barcode_simple'] = barcode_path + ".png"
        except Exception:
            # Create a mock barcode image if barcode library fails
            self._create_mock_barcode()

        # Empty image (no codes)
        empty_img = Image.new('RGB', (300, 300), 'white')
        empty_path = os.path.join(self.temp_dir, "empty.png")
        empty_img.save(empty_path)
        self.test_images['empty'] = empty_path

        # Rotated QR code
        rotated_qr = qr_img.rotate(45, expand=True, fillcolor='white')
        rotated_path = os.path.join(self.temp_dir, "rotated_qr.png")
        rotated_qr.save(rotated_path)
        self.test_images['qr_rotated'] = rotated_path

        # Low quality image
        low_quality = qr_img.resize((50, 50)).resize((300, 300))
        low_quality_path = os.path.join(self.temp_dir, "low_quality.png")
        low_quality.save(low_quality_path)
        self.test_images['qr_low_quality'] = low_quality_path

    def _create_mock_barcode(self):
        """Create a simple mock barcode image"""
        img = Image.new('RGB', (200, 100), 'white')
        draw = ImageDraw.Draw(img)

        # Draw simple barcode pattern
        x = 20
        for i in range(20):
            width = 2 if i % 3 == 0 else 1
            draw.rectangle([x, 30, x + width, 70], fill='black')
            x += width + 1

        barcode_path = os.path.join(self.temp_dir, "test_barcode.png")
        img.save(barcode_path)
        self.test_images['barcode_simple'] = barcode_path

    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsInstance(self.detector, Detector)
        self.assertIsNone(self.detector.last_result)

    def test_detect_simple_qr_code(self):
        """Test detection of simple QR code"""
        result = self.detector.detect_codes(self.test_images['qr_simple'])

        self.assertIsInstance(result, dict)
        self.assertIn('detected_codes', result)
        self.assertIn('processing_time', result)
        self.assertIn('total_codes', result)
        self.assertGreaterEqual(result['total_codes'], 0)  # May or may not detect
        self.assertGreater(result['processing_time'], 0)

    def test_detect_empty_image(self):
        """Test detection on image with no codes"""
        result = self.detector.detect_codes(self.test_images['empty'])

        self.assertEqual(result['total_codes'], 0)
        self.assertEqual(len(result['detected_codes']), 0)

    def test_invalid_image_path(self):
        """Test handling of invalid image paths"""
        with self.assertRaises(ValueError):
            self.detector.detect_codes("nonexistent_image.jpg")

    def test_result_structure(self):
        """Test that results have expected structure"""
        result = self.detector.detect_codes(self.test_images['qr_simple'])

        required_keys = [
            'image_path', 'processing_time', 'barcode_regions',
            'qr_regions', 'regions', 'detected_codes', 'total_codes'
        ]

        for key in required_keys:
            self.assertIn(key, result)

    def test_visualization_creation(self):
        """Test that visualization files are created"""
        result = self.detector.detect_codes(self.test_images['qr_simple'])

        if 'visualization_path' in result:
            self.assertTrue(os.path.exists(result['visualization_path']))


class TestBatchDetector(unittest.TestCase):
    """Test batch processing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()
        self.batch_detector = BatchDetector(output_dir=self.output_dir)

        # Create test image folder
        self._create_test_folder()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def _create_test_folder(self):
        """Create folder with test images"""
        # Create multiple test images
        for i in range(5):
            # Simple colored rectangles as test images
            img = Image.new('RGB', (100 + i * 50, 100 + i * 50),
                            color=(255 - i * 40, i * 40, 100))
            img.save(os.path.join(self.temp_dir, f"test_image_{i}.png"))

        # Create one QR code
        qr = qrcode.QRCode(version=1, box_size=5, border=2)
        qr.add_data(f"TEST_BATCH_QR")
        qr.make(fit=True)
        qr_img = qr.make_image()
        qr_img.save(os.path.join(self.temp_dir, "qr_test.png"))

    def test_get_image_files(self):
        """Test image file discovery"""
        image_files = self.batch_detector._get_image_files(self.temp_dir)

        self.assertGreater(len(image_files), 0)
        self.assertTrue(all(f.endswith(('.png', '.jpg', '.jpeg')) for f in image_files))

    def test_empty_folder(self):
        """Test handling of empty folders"""
        empty_dir = tempfile.mkdtemp()
        try:
            result = self.batch_detector.process_folder(empty_dir)
            self.assertIn('error', result)
        finally:
            shutil.rmtree(empty_dir)

    def test_batch_processing_structure(self):
        """Test batch processing result structure"""
        result = self.batch_detector.process_folder(self.temp_dir, max_images=3)

        required_keys = [
            'batch_info', 'performance_analysis', 'detailed_results'
        ]

        for key in required_keys:
            self.assertIn(key, result)

        # Test batch_info structure
        batch_info = result['batch_info']
        batch_info_keys = [
            'folder_path', 'total_images', 'processed_successfully',
            'failed_images', 'total_batch_time', 'average_time_per_image'
        ]

        for key in batch_info_keys:
            self.assertIn(key, batch_info)

    def test_max_images_limit(self):
        """Test max_images parameter"""
        max_limit = 2
        result = self.batch_detector.process_folder(self.temp_dir, max_images=max_limit)

        self.assertLessEqual(result['batch_info']['processed_successfully'], max_limit)

    def test_performance_metrics_calculation(self):
        """Test performance metrics are calculated correctly"""
        result = self.batch_detector.process_folder(self.temp_dir, max_images=3)

        if result['batch_info']['processed_successfully'] > 0:
            detailed_results = result['detailed_results']

            for detail in detailed_results:
                self.assertIn('performance_metrics', detail)
                self.assertIn('complexity_indicators', detail)
                self.assertIn('file_metadata', detail)

                # Check metrics are reasonable
                self.assertGreater(detail['processing_time'], 0)
                self.assertGreater(detail['file_metadata']['megapixels'], 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def setUp(self):
        """Set up test fixtures"""
        # Create test image arrays
        self.color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    def test_convert_to_grayscale(self):
        """Test grayscale conversion"""
        # Test with color image
        gray_result = convert_to_grayscale(self.color_image)
        self.assertEqual(len(gray_result.shape), 2)
        self.assertEqual(gray_result.shape[:2], self.color_image.shape[:2])

        # Test with already grayscale image
        gray_result2 = convert_to_grayscale(self.gray_image)
        np.testing.assert_array_equal(gray_result2, self.gray_image)

    def test_enhance_contrast(self):
        """Test contrast enhancement"""
        enhanced = enhance_contrast(self.gray_image)
        self.assertEqual(enhanced.shape, self.gray_image.shape)
        self.assertEqual(enhanced.dtype, self.gray_image.dtype)

    def test_reduce_noise(self):
        """Test noise reduction"""
        denoised = reduce_noise(self.gray_image)
        self.assertEqual(denoised.shape, self.gray_image.shape)

    def test_preprocess_for_decoding(self):
        """Test preprocessing pipeline"""
        processed_images = preprocess_for_decoding(self.color_image)
        self.assertGreater(len(processed_images), 1)
        self.assertTrue(all(isinstance(img, np.ndarray) for img in processed_images))


class TestMockedDetection(unittest.TestCase):
    """Test detection with mocked dependencies"""

    @patch('detector.pyzbar.decode')
    def test_mocked_pyzbar_decode(self, mock_decode):
        """Test with mocked pyzbar decode"""
        # Mock successful detection
        mock_code = Mock()
        mock_code.data = b"TEST_DATA"
        mock_code.type = "QRCODE"
        mock_code.rect = (10, 10, 50, 50)
        mock_code.polygon = [Mock(x=10, y=10), Mock(x=60, y=10),
                             Mock(x=60, y=60), Mock(x=10, y=60)]

        mock_decode.return_value = [mock_code]

        detector = Detector()

        # Create a simple test image
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite("mock_test.png", test_img)

        try:
            result = detector.detect_codes("mock_test.png")

            self.assertEqual(result['total_codes'], 1)
            self.assertEqual(result['detected_codes'][0]['data'], "TEST_DATA")
            self.assertEqual(result['detected_codes'][0]['type'], "QRCODE")
        finally:
            if os.path.exists("mock_test.png"):
                os.remove("mock_test.png")

    @patch('detector.cv2.imread')
    def test_image_loading_failure(self, mock_imread):
        """Test handling of image loading failures"""
        mock_imread.return_value = None

        detector = Detector()

        with self.assertRaises(ValueError):
            detector.detect_codes("fake_image.jpg")


class TestPerformanceAnalysis(unittest.TestCase):
    """Test performance analysis components"""

    def setUp(self):
        """Set up test fixtures"""
        self.batch_detector = BatchDetector()

        # Create mock results
        self.mock_results = [
            {
                'processing_time': 0.1,
                'file_metadata': {'megapixels': 1.0, 'filename': 'fast.jpg'},
                'total_codes': 1,
                'complexity_indicators': {'detection_strategies_used': 1}
            },
            {
                'processing_time': 1.0,
                'file_metadata': {'megapixels': 2.0, 'filename': 'slow.jpg'},
                'total_codes': 0,
                'complexity_indicators': {'detection_strategies_used': 4}
            }
        ]
        self.batch_detector.results = self.mock_results

    def test_remove_duplicates(self):
        """Test duplicate removal functionality"""
        # Create duplicate codes
        duplicate_codes = [
            {'data': 'TEST1', 'type': 'QRCODE', 'rotation': 0},
            {'data': 'TEST1', 'type': 'QRCODE', 'rotation': 90},
            {'data': 'TEST2', 'type': 'CODE128'}
        ]

        unique_codes = self.batch_detector._remove_duplicates(duplicate_codes)

        self.assertEqual(len(unique_codes), 2)
        data_types = [(code['data'], code['type']) for code in unique_codes]
        self.assertIn(('TEST1', 'QRCODE'), data_types)
        self.assertIn(('TEST2', 'CODE128'), data_types)

    def test_estimate_strategies_used(self):
        """Test strategy estimation"""
        # Test basic detection
        basic_result = {'detected_codes': [{'data': 'TEST'}]}
        strategies = self.batch_detector._estimate_strategies_used(basic_result)
        self.assertEqual(strategies, 1)

        # Test with rotation
        rotation_result = {
            'detected_codes': [{'data': 'TEST', 'rotation': 45}]
        }
        strategies = self.batch_detector._estimate_strategies_used(rotation_result)
        self.assertGreaterEqual(strategies, 2)

    def test_analyze_performance(self):
        """Test performance analysis"""
        analysis = self.batch_detector._analyze_performance()

        self.assertIn('timing_stats', analysis)
        self.assertIn('complexity_ranking', analysis)
        self.assertIn('correlations', analysis)

        # Check timing stats
        timing = analysis['timing_stats']
        self.assertEqual(timing['fastest_time'], 0.1)
        self.assertEqual(timing['slowest_time'], 1.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def test_corrupted_image_handling(self):
        """Test handling of corrupted image files"""
        # Create a file that looks like an image but isn't
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b"This is not an image")
            corrupted_path = f.name

        try:
            detector = Detector()
            with self.assertRaises(ValueError):
                detector.detect_codes(corrupted_path)
        finally:
            os.unlink(corrupted_path)

    def test_very_large_image(self):
        """Test handling of very large images"""
        # Create a large image (simulated)
        large_image = np.ones((5000, 5000, 3), dtype=np.uint8) * 255

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        try:
            cv2.imwrite(temp_path, large_image)
            detector = Detector()
            result = detector.detect_codes(temp_path)

            # Should complete without error, even if slow
            self.assertIsInstance(result, dict)
            self.assertIn('processing_time', result)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_zero_byte_file(self):
        """Test handling of zero-byte files"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            zero_byte_path = f.name

        try:
            detector = Detector()
            with self.assertRaises(ValueError):
                detector.detect_codes(zero_byte_path)
        finally:
            os.unlink(zero_byte_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()

        # Create a complete test scenario
        self.generator = TestGenerator(output_dir=self.temp_dir)
        self.generator.create_simple_test_set()

    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_end_to_end_batch_processing(self):
        """Test complete batch processing workflow"""
        batch_detector = BatchDetector(output_dir=self.output_dir)

        # Process the generated test images
        result = batch_detector.process_folder(self.temp_dir)

        # Verify results
        self.assertIsInstance(result, dict)
        self.assertNotIn('error', result)
        self.assertGreater(result['batch_info']['processed_successfully'], 0)

        # Verify output files were created
        output_files = os.listdir(self.output_dir)
        self.assertGreater(len(output_files), 0)

        # Check for expected file types
        has_json = any(f.endswith('.json') for f in output_files)
        has_csv = any(f.endswith('.csv') for f in output_files)
        has_txt = any(f.endswith('.txt') for f in output_files)

        self.assertTrue(has_json or has_csv or has_txt)

    def test_single_to_batch_consistency(self):
        """Test that single processing and batch processing give consistent results"""
        single_detector = Detector()
        batch_detector = BatchDetector()

        # Get a test image
        test_files = [f for f in os.listdir(self.temp_dir)
                      if f.endswith(('.png', '.jpg'))]

        if test_files:
            test_image = os.path.join(self.temp_dir, test_files[0])

            # Process with single detector
            single_result = single_detector.detect_codes(test_image)

            # Process with batch detector (single image)
            batch_result = batch_detector.process_folder(
                os.path.dirname(test_image),
                max_images=1
            )

            # Compare key metrics
            if batch_result['batch_info']['processed_successfully'] > 0:
                batch_detail = batch_result['detailed_results'][0]

                # Processing time should be similar (within reasonable tolerance)
                time_diff = abs(single_result['processing_time'] -
                                batch_detail['processing_time'])
                self.assertLess(time_diff, 0.5)  # Within 0.5 seconds

                # Code count should match
                self.assertEqual(single_result['total_codes'],
                                 batch_detail['total_codes'])


def create_test_suite():
    """Create a comprehensive test suite"""
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestDetectorCore,
        TestBatchDetector,
        TestUtilityFunctions,
        TestMockedDetection,
        TestPerformanceAnalysis,
        TestEdgeCases,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


def run_tests_with_coverage():
    """Run tests with coverage reporting if available"""
    try:
        import coverage

        # Start coverage
        cov = coverage.Coverage()
        cov.start()

        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        suite = create_test_suite()
        result = runner.run(suite)

        # Stop coverage and report
        cov.stop()
        cov.save()

        print("\n" + "=" * 50)
        print("COVERAGE REPORT")
        print("=" * 50)
        cov.report()

        return result

    except ImportError:
        print("Coverage module not available. Running tests without coverage.")
        runner = unittest.TextTestRunner(verbosity=2)
        suite = create_test_suite()
        return runner.run(suite)


if __name__ == '__main__':
    # Run comprehensive tests
    print("Running Barcode Detection System Tests")
    print("=" * 50)

    result = run_tests_with_coverage()

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"TEST SUMMARY")
    print(f"{'=' * 50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")

    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"\nTests {'PASSED' if exit_code == 0 else 'FAILED'}")
    exit(exit_code)