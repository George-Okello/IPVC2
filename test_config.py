# test_config.py
"""
Testing configuration and utilities for the barcode detection system
"""
import os
import tempfile
import shutil
import pytest
import numpy as np
from PIL import Image, ImageDraw
import qrcode
import cv2
from unittest.mock import Mock


class TestConfig:
    """Configuration for testing"""

    # Test image dimensions
    SMALL_IMAGE_SIZE = (100, 100)
    MEDIUM_IMAGE_SIZE = (300, 300)
    LARGE_IMAGE_SIZE = (1000, 1000)

    # Performance thresholds
    MAX_PROCESSING_TIME = 10.0  # seconds
    MAX_MEMORY_USAGE = 500 * 1024 * 1024  # 500MB

    # Test data
    TEST_QR_DATA = [
        "Hello World!",
        "https://example.com",
        "test@email.com",
        "1234567890",
        "Special chars: !@#$%^&*()",
        "Unicode: 你好世界",
        "Long text: " + "A" * 1000
    ]

    TEST_BARCODE_DATA = [
        "123456789012",  # EAN13
        "12345678",  # EAN8
        "ABC123",  # Code128
        "1234567890123456789",  # Long code
    ]


class TestImageFactory:
    """Factory for creating test images"""

    @staticmethod
    def create_qr_image(data="TEST", size=(200, 200), **kwargs):
        """Create QR code test image"""
        qr = qrcode.QRCode(
            version=kwargs.get('version', 1),
            error_correction=kwargs.get('error_correction', qrcode.constants.ERROR_CORRECT_L),
            box_size=kwargs.get('box_size', 10),
            border=kwargs.get('border', 4),
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        return img.resize(size)

    @staticmethod
    def create_barcode_image(data="123456789012", size=(300, 100)):
        """Create mock barcode image"""
        img = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(img)

        # Simple barcode pattern
        bar_width = size[0] // (len(data) * 3)
        x = 10

        for char in data:
            # Use character value to determine bar pattern
            pattern = [1, 0, 1] if ord(char) % 2 else [0, 1, 0]

            for bar in pattern:
                if bar:
                    draw.rectangle([x, 20, x + bar_width, size[1] - 20], fill='black')
                x += bar_width

        return img

    @staticmethod
    def create_empty_image(size=(200, 200), color='white'):
        """Create empty test image"""
        return Image.new('RGB', size, color)

    @staticmethod
    def create_noisy_image(base_image, noise_level=0.1):
        """Add noise to an image"""
        img_array = np.array(base_image)
        noise = np.random.randint(0, int(255 * noise_level), img_array.shape, dtype=np.uint8)
        noisy = np.clip(img_array.astype(np.int16) + noise - (255 * noise_level // 2), 0, 255)
        return Image.fromarray(noisy.astype(np.uint8))

    @staticmethod
    def create_rotated_image(base_image, angle):
        """Create rotated version of image"""
        return base_image.rotate(angle, expand=True, fillcolor='white')

    @staticmethod
    def create_blurred_image(base_image, blur_radius=2):
        """Create blurred version of image"""
        from PIL import ImageFilter
        return base_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    @staticmethod
    def create_low_contrast_image(base_image, factor=0.5):
        """Create low contrast version of image"""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(base_image)
        return enhancer.enhance(factor)


class TestDataManager:
    """Manage test data and temporary files"""

    def __init__(self):
        self.temp_dirs = []
        self.temp_files = []

    def create_temp_dir(self, prefix="test_"):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def create_temp_file(self, suffix=".png", content=None):
        """Create temporary file"""
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        if content:
            if isinstance(content, Image.Image):
                content.save(temp_path)
            elif isinstance(content, (bytes, str)):
                with open(temp_path, 'wb') as f:
                    if isinstance(content, str):
                        content = content.encode()
                    f.write(content)

        self.temp_files.append(temp_path)
        return temp_path

    def create_test_dataset(self, num_images=10, include_variants=True):
        """Create a comprehensive test dataset"""
        test_dir = self.create_temp_dir("test_dataset_")

        # Basic QR codes
        for i, data in enumerate(TestConfig.TEST_QR_DATA[:num_images // 2]):
            qr_img = TestImageFactory.create_qr_image(data)
            qr_path = os.path.join(test_dir, f"qr_{i:03d}.png")
            qr_img.save(qr_path)

            if include_variants:
                # Create rotated version
                rotated = TestImageFactory.create_rotated_image(qr_img, 45)
                rotated_path = os.path.join(test_dir, f"qr_{i:03d}_rotated.png")
                rotated.save(rotated_path)

                # Create noisy version
                noisy = TestImageFactory.create_noisy_image(qr_img)
                noisy_path = os.path.join(test_dir, f"qr_{i:03d}_noisy.png")
                noisy.save(noisy_path)

        # Basic barcodes
        for i, data in enumerate(TestConfig.TEST_BARCODE_DATA[:num_images // 2]):
            barcode_img = TestImageFactory.create_barcode_image(data)
            barcode_path = os.path.join(test_dir, f"barcode_{i:03d}.png")
            barcode_img.save(barcode_path)

        # Empty images
        for i in range(2):
            empty_img = TestImageFactory.create_empty_image()
            empty_path = os.path.join(test_dir, f"empty_{i:03d}.png")
            empty_img.save(empty_path)

        return test_dir

    def cleanup(self):
        """Clean up all temporary files and directories"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass

        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass

        self.temp_dirs.clear()
        self.temp_files.clear()


class MockObjects:
    """Factory for creating mock objects for testing"""

    @staticmethod
    def create_mock_pyzbar_code(data="TEST", code_type="QRCODE", bbox=(10, 10, 50, 50)):
        """Create mock pyzbar decoded object"""
        mock_code = Mock()
        mock_code.data = data.encode() if isinstance(data, str) else data
        mock_code.type = code_type
        mock_code.rect = bbox

        # Create mock polygon
        x, y, w, h = bbox
        mock_polygon = [
            Mock(x=x, y=y),
            Mock(x=x + w, y=y),
            Mock(x=x + w, y=y + h),
            Mock(x=x, y=y + h)
        ]
        mock_code.polygon = mock_polygon

        return mock_code

    @staticmethod
    def create_mock_detector_result(processing_time=0.1, codes_found=1):
        """Create mock detector result"""
        return {
            'image_path': 'test_image.png',
            'processing_time': processing_time,
            'barcode_regions': [],
            'qr_regions': [(10, 10, 50, 50, 'QRCODE')] if codes_found > 0 else [],
            'regions': [],
            'detected_codes': [
                {
                    'data': 'TEST_DATA',
                    'type': 'QRCODE',
                    'category': 'QR Code',
                    'bbox': (10, 10, 50, 50),
                    'polygon': [(10, 10), (60, 10), (60, 60), (10, 60)]
                }
            ] if codes_found > 0 else [],
            'total_codes': codes_found,
            'visualization_path': 'test_detected.jpg'
        }


class PerformanceProfiler:
    """Profile performance during testing"""

    def __init__(self):
        self.measurements = []

    def __enter__(self):
        import time
        import psutil
        import os

        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.start_memory = self.process.memory_info().rss
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        end_time = time.time()
        end_memory = self.process.memory_info().rss

        measurement = {
            'duration': end_time - self.start_time,
            'memory_used': end_memory - self.start_memory,
            'peak_memory': self.process.memory_info().rss
        }

        self.measurements.append(measurement)

    def get_stats(self):
        """Get performance statistics"""
        if not self.measurements:
            return {}

        durations = [m['duration'] for m in self.measurements]
        memory_usage = [m['memory_used'] for m in self.measurements]

        return {
            'avg_duration': sum(durations) / len(durations),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'avg_memory': sum(memory_usage) / len(memory_usage),
            'max_memory': max(memory_usage),
            'total_measurements': len(self.measurements)
        }


# pytest fixtures
@pytest.fixture
def test_data_manager():
    """Pytest fixture for test data manager"""
    manager = TestDataManager()
    yield manager
    manager.cleanup()


@pytest.fixture
def sample_qr_image():
    """Pytest fixture for sample QR image"""
    return TestImageFactory.create_qr_image("TEST_QR_CODE")


@pytest.fixture
def sample_barcode_image():
    """Pytest fixture for sample barcode image"""
    return TestImageFactory.create_barcode_image("123456789012")


@pytest.fixture
def temp_image_file(sample_qr_image):
    """Pytest fixture for temporary image file"""
    fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    sample_qr_image.save(temp_path)
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def test_dataset(test_data_manager):
    """Pytest fixture for test dataset"""
    return test_data_manager.create_test_dataset(num_images=6)


# test_performance.py
"""
Performance-focused tests for the detection system
"""
import pytest
import time
from test_config import TestConfig, PerformanceProfiler, TestDataManager
from detector import Detector
from batch_detector import BatchDetector


class TestPerformance:
    """Performance tests"""

    def test_single_image_performance(self, temp_image_file):
        """Test single image processing performance"""
        detector = Detector()

        with PerformanceProfiler() as profiler:
            result = detector.detect_codes(temp_image_file)

        stats = profiler.get_stats()

        # Assert performance constraints
        assert stats['max_duration'] < TestConfig.MAX_PROCESSING_TIME
        assert stats['max_memory'] < TestConfig.MAX_MEMORY_USAGE
        assert result['processing_time'] < TestConfig.MAX_PROCESSING_TIME

    def test_batch_processing_performance(self, test_dataset):
        """Test batch processing performance"""
        batch_detector = BatchDetector()

        with PerformanceProfiler() as profiler:
            result = batch_detector.process_folder(test_dataset, max_images=10)

        stats = profiler.get_stats()

        # Performance assertions
        assert stats['max_duration'] < TestConfig.MAX_PROCESSING_TIME * 10  # Allow more time for batch
        assert result['batch_info']['average_time_per_image'] < TestConfig.MAX_PROCESSING_TIME

    def test_memory_usage_scaling(self, test_data_manager):
        """Test memory usage with increasing image sizes"""
        detector = Detector()
        memory_usage = []

        sizes = [TestConfig.SMALL_IMAGE_SIZE, TestConfig.MEDIUM_IMAGE_SIZE, TestConfig.LARGE_IMAGE_SIZE]

        for size in sizes:
            qr_img = TestImageFactory.create_qr_image("TEST", size=size)
            temp_path = test_data_manager.create_temp_file(content=qr_img)

            with PerformanceProfiler() as profiler:
                detector.detect_codes(temp_path)

            stats = profiler.get_stats()
            memory_usage.append(stats['max_memory'])

        # Memory should not grow excessively with image size
        assert all(mem < TestConfig.MAX_MEMORY_USAGE for mem in memory_usage)

    @pytest.mark.parametrize("num_images", [1, 5, 10, 20])
    def test_batch_scaling(self, test_data_manager, num_images):
        """Test batch processing scaling with different numbers of images"""
        test_dir = test_data_manager.create_test_dataset(num_images=num_images)
        batch_detector = BatchDetector()

        start_time = time.time()
        result = batch_detector.process_folder(test_dir)
        end_time = time.time()

        total_time = end_time - start_time

        # Time should scale roughly linearly (with some overhead)
        expected_max_time = num_images * TestConfig.MAX_PROCESSING_TIME + 5  # 5s overhead
        assert total_time < expected_max_time

        # All images should be processed successfully
        assert result['batch_info']['processed_successfully'] == num_images


# test_pytest_examples.py
"""
Examples of pytest-based tests for the detection system
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock
from detector import Detector
from batch_detector import BatchDetector
from test_config import TestImageFactory, MockObjects


class TestDetectorPytest:
    """Pytest-style tests for core detector"""

    def test_detector_initialization(self):
        """Test detector can be initialized"""
        detector = Detector()
        assert detector is not None
        assert detector.last_result is None

    @pytest.mark.parametrize("qr_data", [
        "Simple text",
        "https://example.com",
        "Special chars: !@#$%",
        "Numbers: 1234567890"
    ])
    def test_qr_detection_with_various_data(self, test_data_manager, qr_data):
        """Test QR detection with various data types"""
        qr_img = TestImageFactory.create_qr_image(qr_data)
        temp_path = test_data_manager.create_temp_file(content=qr_img)

        detector = Detector()
        result = detector.detect_codes(temp_path)

        assert isinstance(result, dict)
        assert 'detected_codes' in result
        assert 'processing_time' in result
        assert result['processing_time'] > 0

    @pytest.mark.parametrize("rotation_angle", [0, 45, 90, 135, 180, 225, 270, 315])
    def test_rotated_qr_detection(self, test_data_manager, rotation_angle):
        """Test detection of rotated QR codes"""
        qr_img = TestImageFactory.create_qr_image("ROTATION_TEST")
        rotated_img = TestImageFactory.create_rotated_image(qr_img, rotation_angle)
        temp_path = test_data_manager.create_temp_file(content=rotated_img)

        detector = Detector()
        result = detector.detect_codes(temp_path)

        # Should complete without error regardless of detection success
        assert isinstance(result, dict)
        assert result['total_codes'] >= 0

    def test_empty_image_detection(self, test_data_manager):
        """Test detection on empty image"""
        empty_img = TestImageFactory.create_empty_image()
        temp_path = test_data_manager.create_temp_file(content=empty_img)

        detector = Detector()
        result = detector.detect_codes(temp_path)

        assert result['total_codes'] == 0
        assert len(result['detected_codes']) == 0

    def test_invalid_image_path(self):
        """Test handling of invalid image path"""
        detector = Detector()

        with pytest.raises(ValueError, match="Cannot load image"):
            detector.detect_codes("nonexistent_file.jpg")

    @patch('detector.pyzbar.decode')
    def test_mocked_successful_detection(self, mock_decode, temp_image_file):
        """Test with mocked successful detection"""
        mock_code = MockObjects.create_mock_pyzbar_code("MOCKED_DATA", "QRCODE")
        mock_decode.return_value = [mock_code]

        detector = Detector()
        result = detector.detect_codes(temp_image_file)

        assert result['total_codes'] == 1
        assert result['detected_codes'][0]['data'] == "MOCKED_DATA"
        assert result['detected_codes'][0]['type'] == "QRCODE"

    @patch('detector.cv2.imread')
    def test_image_loading_failure(self, mock_imread):
        """Test handling of image loading failure"""
        mock_imread.return_value = None

        detector = Detector()

        with pytest.raises(ValueError):
            detector.detect_codes("fake_image.jpg")


class TestBatchDetectorPytest:
    """Pytest-style tests for batch detector"""

    def test_batch_detector_initialization(self):
        """Test batch detector initialization"""
        batch_detector = BatchDetector()
        assert batch_detector is not None
        assert hasattr(batch_detector, 'detector')
        assert hasattr(batch_detector, 'results')

    def test_empty_folder_handling(self, test_data_manager):
        """Test handling of empty folder"""
        empty_dir = test_data_manager.create_temp_dir()
        batch_detector = BatchDetector()

        result = batch_detector.process_folder(empty_dir)

        assert 'error' in result
        assert 'No images found' in result['error']

    def test_nonexistent_folder_handling(self):
        """Test handling of nonexistent folder"""
        batch_detector = BatchDetector()

        # This should be handled gracefully or raise appropriate exception
        with pytest.raises((FileNotFoundError, ValueError)):
            batch_detector.process_folder("/nonexistent/folder")

    def test_max_images_parameter(self, test_dataset):
        """Test max_images parameter works correctly"""
        batch_detector = BatchDetector()
        max_limit = 3

        result = batch_detector.process_folder(test_dataset, max_images=max_limit)

        assert result['batch_info']['processed_successfully'] <= max_limit

    def test_result_structure_completeness(self, test_dataset):
        """Test that batch results have complete structure"""
        batch_detector = BatchDetector()
        result = batch_detector.process_folder(test_dataset, max_images=2)

        # Test top-level structure
        required_keys = ['batch_info', 'performance_analysis', 'detailed_results']
        for key in required_keys:
            assert key in result

        # Test batch_info structure
        batch_info_keys = [
            'folder_path', 'total_images', 'processed_successfully',
            'failed_images', 'total_batch_time', 'average_time_per_image'
        ]
        for key in batch_info_keys:
            assert key in result['batch_info']

    @pytest.mark.slow
    def test_large_batch_processing(self, test_data_manager):
        """Test processing larger batch of images"""
        # Create larger test dataset
        large_dataset = test_data_manager.create_test_dataset(num_images=20)
        batch_detector = BatchDetector()

        result = batch_detector.process_folder(large_dataset, max_images=15)

        assert result['batch_info']['processed_successfully'] > 0
        assert result['batch_info']['total_batch_time'] > 0
        assert len(result['detailed_results']) > 0


class TestEdgeCasesPytest:
    """Pytest-style edge case tests"""

    def test_corrupted_image_file(self, test_data_manager):
        """Test handling of corrupted image file"""
        # Create fake image file
        corrupted_path = test_data_manager.create_temp_file(
            suffix=".jpg",
            content="This is not a valid image file"
        )

        detector = Detector()

        with pytest.raises(ValueError):
            detector.detect_codes(corrupted_path)

    def test_zero_byte_image_file(self, test_data_manager):
        """Test handling of zero-byte image file"""
        zero_byte_path = test_data_manager.create_temp_file(suffix=".png")

        detector = Detector()

        with pytest.raises(ValueError):
            detector.detect_codes(zero_byte_path)

    @pytest.mark.parametrize("noise_level", [0.1, 0.3, 0.5, 0.7])
    def test_noisy_image_handling(self, test_data_manager, noise_level):
        """Test detection on noisy images"""
        qr_img = TestImageFactory.create_qr_image("NOISE_TEST")
        noisy_img = TestImageFactory.create_noisy_image(qr_img, noise_level)
        temp_path = test_data_manager.create_temp_file(content=noisy_img)

        detector = Detector()
        result = detector.detect_codes(temp_path)

        # Should complete without error
        assert isinstance(result, dict)
        assert 'processing_time' in result


# conftest.py for pytest configuration
"""
Pytest configuration and shared fixtures
"""
import pytest


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add slow marker to tests that take longer
        if "large_batch" in item.name or "scaling" in item.name:
            item.add_marker(pytest.mark.slow)

        # Add integration marker to end-to-end tests
        if "end_to_end" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.integration)

        # Add performance marker to performance tests
        if "performance" in item.name or "memory" in item.name:
            item.add_marker(pytest.mark.performance)


# Usage examples in documentation format
"""
RUNNING TESTS
=============

1. RUN ALL TESTS
   python -m pytest test_detector_system.py -v

2. RUN SPECIFIC TEST CLASS
   python -m pytest test_detector_system.py::TestDetectorCore -v

3. RUN WITH COVERAGE
   python -m pytest test_detector_system.py --cov=detector --cov=batch_detector

4. RUN PERFORMANCE TESTS ONLY
   python -m pytest -m performance

5. SKIP SLOW TESTS
   python -m pytest -m "not slow"

6. RUN INTEGRATION TESTS
   python -m pytest -m integration

7. GENERATE HTML COVERAGE REPORT
   python -m pytest --cov=detector --cov-report=html

8. RUN TESTS IN PARALLEL
   python -m pytest -n auto  # requires pytest-xdist

TRADITIONAL UNITTEST
====================

1. RUN ALL TESTS
   python test_detector_system.py

2. RUN WITH COVERAGE
   pip install coverage
   python test_detector_system.py

3. RUN SPECIFIC TEST
   python -m unittest test_detector_system.TestDetectorCore.test_detect_simple_qr_code

TEST STRUCTURE
==============

The testing framework provides:

1. UNIT TESTS
   - Individual component testing
   - Mocked dependencies
   - Fast execution

2. INTEGRATION TESTS  
   - End-to-end workflows
   - Real file operations
   - Complete system testing

3. PERFORMANCE TESTS
   - Memory usage monitoring
   - Processing time validation
   - Scaling behavior analysis

4. EDGE CASE TESTS
   - Error condition handling
   - Invalid input processing
   - Boundary value testing

MOCKING STRATEGY
================

1. EXTERNAL DEPENDENCIES
   - pyzbar.decode() for barcode detection
   - cv2.imread() for image loading
   - File system operations

2. PERFORMANCE ISOLATION
   - Mock expensive operations during unit tests
   - Use real implementations for integration tests

3. REPRODUCIBLE RESULTS
   - Mock random elements
   - Fixed test data generation

TEST DATA MANAGEMENT
===================

1. AUTOMATIC CLEANUP
   - Temporary files removed after tests
   - No test pollution between runs

2. PARAMETERIZED TESTS
   - Multiple input variations
   - Comprehensive coverage

3. REALISTIC TEST DATA
   - Various QR code formats
   - Different image qualities
   - Multiple file types
"""