# main.py
"""
Enhanced main script with CLI improvements
"""
import sys
import os
import argparse
from detector import Detector
from test_generator import TestGenerator
from simple_gui import main as run_gui


def detect_with_enhanced(image_path, compare=False):
    """Detect codes using the enhanced detector"""
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return

    print(f"Processing image: {image_path}")
    print("-" * 60)

    # Use enhanced detector
    print("Using ENHANCED detector...")
    enhanced_detector = Detector()
    enhanced_result = enhanced_detector.detect_codes(image_path)

    # Print enhanced results
    print(f"Enhanced processing time: {enhanced_result['processing_time']:.3f} seconds")
    print(f"Total codes decoded: {enhanced_result['total_codes']}")
    print()

    if enhanced_result['detected_codes']:
        print("Detected codes:")
        for i, code in enumerate(enhanced_result['detected_codes'], 1):
            code_info = [f"{i}. {code['type']}: '{code['data']}'"]

            if 'rotation' in code:
                code_info.append(f"(Rotation: {code['rotation']}°)")

            if 'preprocess' in code:
                code_info.append(f"(Method: {code['preprocess']})")

            print(" ".join(code_info))
    else:
        print("No codes detected with enhanced detector")

    # Compare with original detector if requested
    if compare:
        print("\nComparing with ORIGINAL detector...")
        original_detector = Detector()
        original_result = original_detector.detect_codes(image_path)

        print(f"Original processing time: {original_result['processing_time']:.3f} seconds")
        print(f"Barcode regions found: {len(original_result['barcode_regions'])}")
        print(f"QR code regions found: {len(original_result['qr_regions'])}")
        print(f"Total codes decoded: {original_result['total_codes']}")
        print()

        if original_result['detected_codes']:
            print("Original detected codes:")
            for i, code in enumerate(original_result['detected_codes'], 1):
                print(f"{i}. {code['type']}: '{code['data']}'")
        else:
            print("No codes detected with original detector")

    # Show visualization paths
    print("\nOutputs:")
    # Use visualization_path if available, otherwise use default path
    vis_path = enhanced_result.get('visualization_path') or os.path.splitext(os.path.basename(image_path))[0] + '_detected.jpg'
    if os.path.exists(vis_path):
        print(f"Visualization saved to: {vis_path}")


def batch_process(folder_path):
    """Process all images in a folder"""
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found")
        return

    print(f"Batch processing images in: {folder_path}")
    print("-" * 60)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []

    for file in os.listdir(folder_path):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            image_files.append(os.path.join(folder_path, file))

    if not image_files:
        print("No image files found in the specified folder")
        return

    # Process each image
    detector = Detector()
    print(f"Found {len(image_files)} images to process\n")

    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        try:
            result = detector.detect_codes(image_path)
            results.append({
                'path': image_path,
                'filename': os.path.basename(image_path),
                'codes': len(result['detected_codes']),
                'time': result['processing_time']
            })

            if result['detected_codes']:
                print(f"  ✓ Found {len(result['detected_codes'])} code(s)")
                for code in result['detected_codes']:
                    print(f"    - {code['type']}: '{code['data']}'")
            else:
                print("  ✗ No codes detected")

            print()
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print()

    # Summary
    print("\nBatch Processing Summary:")
    print("-" * 60)
    print(f"Total images processed: {len(image_files)}")
    successful = sum(1 for r in results if r['codes'] > 0)
    print(f"Images with codes detected: {successful} ({successful / len(image_files) * 100:.1f}%)")
    print(f"Total codes detected: {sum(r['codes'] for r in results)}")
    print(f"Average processing time: {sum(r['time'] for r in results) / len(results):.3f} seconds per image")


def main():
    """Main function with improved CLI interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced Barcode and QR Code Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py detect example.jpg       # Detect codes in a single image
  python main.py detect example.jpg -c    # Compare enhanced vs original detector
  python main.py batch ./images           # Process all images in a folder
  python main.py generate                 # Generate test images
  python main.py gui                      # Run the GUI application
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect codes in an image')
    detect_parser.add_argument('image', help='Path to the image file')
    detect_parser.add_argument('-c', '--compare', action='store_true',
                               help='Compare with original detector')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process all images in a folder')
    batch_parser.add_argument('folder', help='Path to the folder containing images')

    # GUI command
    subparsers.add_parser('gui', help='Run the GUI application')

    # Generate command
    subparsers.add_parser('generate', help='Generate test images')

    # Parse arguments
    args = parser.parse_args()

    # Execute commands
    if args.command == 'detect':
        detect_with_enhanced(args.image, args.compare)
    elif args.command == 'batch':
        batch_process(args.folder)
    elif args.command == 'gui':
        print("Starting GUI application...")
        run_gui()
    elif args.command == 'generate':
        print("Generating test images...")
        generator = TestGenerator()
        generator.create_simple_test_set()
        print("Test images generated successfully!")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()