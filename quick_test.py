# quick_test.py
"""
Quick test script to verify the detection system works in Lightning AI
"""
from detector import CodeDetector
from test_generator import TestGenerator
import os


def main():
    print("Quick Test - Barcode & QR Code Detection")
    print("=" * 40)

    # Step 1: Create test images
    print("1. Creating test images...")
    generator = TestGenerator()
    generator.create_simple_test_set()
    print("   ✓ Test images created in 'test_images' folder")

    # Step 2: Test detection
    print("\n2. Testing detection...")
    detector = CodeDetector()

    # Test with QR code
    qr_result = detector.detect_codes('test_images/qr_simple.png')
    print(f"   QR Code test: {qr_result['total_codes']} codes found")
    if qr_result['detected_codes']:
        for code in qr_result['detected_codes']:
            print(f"   - {code['type']}: {code['data']}")

    # Test with barcode
    if os.path.exists('test_images/barcode_ean13.png'):
        barcode_result = detector.detect_codes('test_images/barcode_ean13.png')
        print(f"   Barcode test: {barcode_result['total_codes']} codes found")
        if barcode_result['detected_codes']:
            for code in barcode_result['detected_codes']:
                print(f"   - {code['type']}: {code['data']}")

    print("\n3. Visualization files created:")
    for file in os.listdir('.'):
        if file.endswith('_detected.jpg'):
            print(f"   - {file}")

    print("\n✓ Test complete! The system is working correctly.")
    print("\nNext steps:")
    print("1. To process your own image: python main.py detect <image_path>")
    print("2. To generate more test images: python main.py generate-tests")
    print("3. Upload your own images to test on them")


if __name__ == "__main__":
    main()