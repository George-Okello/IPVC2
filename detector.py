# detector.py
"""
Comprehensive barcode and QR code detection system with preprocessing analytics
"""
import cv2
import numpy as np
from pyzbar import pyzbar
from PIL import Image, ImageEnhance
import os
from datetime import datetime


class Detector:
    """Advanced barcode and QR code detector with comprehensive preprocessing tracking"""

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.last_result = None
        os.makedirs(output_dir, exist_ok=True)

    def detect_codes(self, image_path):
        """Detect and decode codes in an image with comprehensive rotation handling"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        start_time = datetime.now()

        # Initialize preprocessing tracking
        preprocessing_analytics = {
            'methods_attempted': [],
            'methods_successful': [],
            'method_success_details': {},
            'total_methods_tried': 0,
            'successful_methods_count': 0
        }

        # Detect using multiple strategies
        all_codes = []
        barcode_regions = []
        qr_regions = []
        other_regions = []

        # Strategy 1: Standard detection
        basic_codes = self._detect_standard(image)
        all_codes.extend(basic_codes)

        # Track if standard detection was successful
        if len(basic_codes) > 0:
            preprocessing_analytics['methods_attempted'].append('standard_detection')
            preprocessing_analytics['methods_successful'].append('standard_detection')
            preprocessing_analytics['method_success_details']['standard_detection'] = {
                'codes_found': len(basic_codes),
                'method_type': 'basic'
            }

        # Strategy 2: Try with comprehensive rotations if needed
        if len(basic_codes) == 0:
            rotated_codes, rot_regions = self._detect_with_comprehensive_rotations(image)
            all_codes.extend(rotated_codes)
            other_regions.extend(rot_regions)

            if len(rotated_codes) > 0:
                preprocessing_analytics['methods_attempted'].append('rotation_detection')
                preprocessing_analytics['methods_successful'].append('rotation_detection')
                preprocessing_analytics['method_success_details']['rotation_detection'] = {
                    'codes_found': len(rotated_codes),
                    'method_type': 'rotation'
                }

        # Strategy 3: Try with enhanced preprocessing + rotations if still needed
        if len(all_codes) == 0:
            enhanced_codes, enh_regions = self._detect_with_enhanced_preprocessing(image)
            all_codes.extend(enhanced_codes)
            other_regions.extend(enh_regions)

            # Extract preprocessing success information from codes
            for code in enhanced_codes:
                if 'preprocessing_success_tracking' in code:
                    tracking = code['preprocessing_success_tracking']
                    for method_name, details in tracking.items():
                        if method_name not in preprocessing_analytics['methods_attempted']:
                            preprocessing_analytics['methods_attempted'].append(method_name)
                        if details['successful'] and method_name not in preprocessing_analytics['methods_successful']:
                            preprocessing_analytics['methods_successful'].append(method_name)
                            preprocessing_analytics['method_success_details'][method_name] = details

        # Strategy 4: Try fine-grained rotation if still no results
        if len(all_codes) == 0:
            fine_codes, fine_regions = self._detect_with_fine_rotations(image)
            all_codes.extend(fine_codes)
            other_regions.extend(fine_regions)

            # Extract preprocessing success information from fine rotation codes
            for code in fine_codes:
                if 'preprocessing_success_tracking' in code:
                    tracking = code['preprocessing_success_tracking']
                    for method_name, details in tracking.items():
                        if method_name not in preprocessing_analytics['methods_attempted']:
                            preprocessing_analytics['methods_attempted'].append(method_name)
                        if details['successful'] and method_name not in preprocessing_analytics['methods_successful']:
                            preprocessing_analytics['methods_successful'].append(method_name)
                            preprocessing_analytics['method_success_details'][method_name] = details

        # Finalize preprocessing analytics
        preprocessing_analytics['total_methods_tried'] = len(preprocessing_analytics['methods_attempted'])
        preprocessing_analytics['successful_methods_count'] = len(preprocessing_analytics['methods_successful'])
        preprocessing_analytics['success_rate'] = (
                                                      preprocessing_analytics['successful_methods_count'] /
                                                      preprocessing_analytics['total_methods_tried']
                                                      if preprocessing_analytics['total_methods_tried'] > 0 else 0
                                                  ) * 100

        # Remove duplicates
        unique_codes = self._remove_duplicates(all_codes)

        # Categorize codes by type
        for code in unique_codes:
            if code['type'].lower() == 'qrcode':
                code['category'] = 'QR Code'
                # Add to qr_regions if there's bounding box information
                if 'bbox' in code:
                    x, y, w, h = code['bbox']
                    qr_regions.append((x, y, w, h, code['type']))
            else:
                code['category'] = 'Barcode'
                # Add to barcode_regions if there's bounding box information
                if 'bbox' in code:
                    x, y, w, h = code['bbox']
                    barcode_regions.append((x, y, w, h, code['type']))

        # Create results
        processing_time = (datetime.now() - start_time).total_seconds()

        result = {
            'image_path': image_path,
            'processing_time': processing_time,
            'barcode_regions': barcode_regions,
            'qr_regions': qr_regions,
            'regions': other_regions,
            'detected_codes': unique_codes,
            'total_codes': len(unique_codes),
            'preprocessing_analytics': preprocessing_analytics
        }

        # Save visualization
        output_path = self._create_visualization(image, result)
        result['visualization_path'] = output_path

        self.last_result = result
        return result

    def _detect_standard(self, image):
        """Standard detection with basic preprocessing"""
        codes = []

        # Try multiple image variants
        variants = [
            ('original_color', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
            ('grayscale', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
            ('otsu_binary', self._otsu_threshold),
            ('inverted_otsu', lambda img: cv2.bitwise_not(self._otsu_threshold(img))),
            ('adaptive_gaussian', lambda img: cv2.adaptiveThreshold(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img,
                255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ('adaptive_mean', lambda img: cv2.adaptiveThreshold(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img,
                255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5))
        ]

        # Add fixed threshold variants
        for threshold in range(80, 181, 20):
            variants.append((f'threshold_{threshold}',
                             lambda img, t=threshold: cv2.threshold(
                                 cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img,
                                 t, 255, cv2.THRESH_BINARY)[1]))

        for variant_name, transform_func in variants:
            try:
                transformed = transform_func(image)
                if transformed is not None:
                    new_codes = self._decode_image_pyzbar(transformed)
                    for code in new_codes:
                        code['image_variant'] = variant_name
                        code['detection_method'] = 'standard'
                    codes.extend(new_codes)
            except Exception as e:
                print(f"Error in variant {variant_name}: {e}")
                continue

        return codes

    def _detect_with_comprehensive_rotations(self, image):
        """Try detection with multiple rotation angles"""
        codes = []
        regions = []

        # Standard rotation angles
        rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        for angle in rotation_angles:
            if angle == 0:
                rotated = image.copy()
            else:
                rotated = self._rotate_image(image, angle)

            # Track region
            h, w = rotated.shape[:2]
            regions.append((0, 0, w, h, f"rotation_{angle}"))

            # Try detection with multiple preprocessing on rotated image
            angle_codes = self._decode_image_pyzbar(rotated)

            # Also try with some basic preprocessing
            if len(angle_codes) == 0:
                # Try grayscale
                gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                gray_codes = self._decode_image_pyzbar(gray)
                for code in gray_codes:
                    code['rotation'] = angle
                    code['detection_method'] = 'rotation_grayscale'
                angle_codes.extend(gray_codes)

                # Try adaptive threshold
                if len(gray_codes) == 0:
                    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                                     2)
                    adaptive_codes = self._decode_image_pyzbar(adaptive)
                    for code in adaptive_codes:
                        code['rotation'] = angle
                        code['detection_method'] = 'rotation_adaptive'
                    angle_codes.extend(adaptive_codes)

            # Add rotation info to codes
            for code in angle_codes:
                if 'rotation' not in code:
                    code['rotation'] = angle
                if 'detection_method' not in code:
                    code['detection_method'] = 'rotation'

            codes.extend(angle_codes)

        return codes, regions

    def _detect_with_enhanced_preprocessing(self, image):
        """Try with enhanced preprocessing methods combined with rotations"""
        codes = []
        regions = []
        preprocessing_success_tracking = {}

        # List of preprocessing methods to try
        preprocess_methods = [
            ('original', lambda img: img),
            ('grayscale', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img),
            ('blur', lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
            ('sharpen', self._sharpen_image),
            ('adaptive_threshold', self._adaptive_threshold),
            ('edge_enhance', self._edge_enhance),
            ('morphology', self._morphological_operations),
            ('contrast_enhance', self._enhance_contrast),
            ('perspective_correction', self._correct_perspective),
            ('barcode_enhancement', self._enhance_for_barcode),
            ('noise_reduction', self._reduce_noise),
            ('gamma_correction', self._gamma_correction)
        ]

        # Rotation angles to try with each preprocessing method
        rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        # Try each preprocessing method
        for name, method in preprocess_methods:
            method_codes = []
            try:
                processed = method(image)
                if processed is not None:
                    # Try each rotation angle with this preprocessing
                    for angle in rotation_angles:
                        if angle == 0:
                            test_image = processed
                        else:
                            test_image = self._rotate_image(processed, angle)

                        # Track region
                        h, w = test_image.shape[:2]
                        regions.append((0, 0, w, h, f"preprocess_{name}_rot_{angle}"))

                        # Try detection
                        new_codes = self._decode_image_pyzbar(test_image)
                        for code in new_codes:
                            code['preprocess'] = name
                            code['rotation'] = angle
                            code['preprocessing_method'] = name
                            code['preprocessing_successful'] = True

                        method_codes.extend(new_codes)

                # Track success for this preprocessing method
                preprocessing_success_tracking[name] = {
                    'attempted': True,
                    'successful': len(method_codes) > 0,
                    'codes_found': len(method_codes),
                    'method_type': self._categorize_preprocessing_method(name)
                }

                codes.extend(method_codes)

            except Exception as e:
                print(f"Error in preprocessing method {name}: {e}")
                preprocessing_success_tracking[name] = {
                    'attempted': True,
                    'successful': False,
                    'codes_found': 0,
                    'error': str(e),
                    'method_type': self._categorize_preprocessing_method(name)
                }

        # Add preprocessing success tracking to codes metadata
        for code in codes:
            if 'preprocessing_success_tracking' not in code:
                code['preprocessing_success_tracking'] = preprocessing_success_tracking

        return codes, regions

    def _categorize_preprocessing_method(self, method_name):
        """Categorize preprocessing methods by type"""
        categories = {
            'basic': ['original', 'grayscale', 'blur'],
            'enhancement': ['sharpen', 'contrast_enhance', 'gamma_correction'],
            'noise_reduction': ['noise_reduction', 'blur'],
            'thresholding': ['adaptive_threshold'],
            'morphological': ['morphology', 'edge_enhance'],
            'advanced': ['perspective_correction', 'barcode_enhancement']
        }

        for category, methods in categories.items():
            if method_name in methods:
                return category
        return 'other'

    def _detect_with_fine_rotations(self, image):
        """Try detection with fine-grained rotation angles (every 5 degrees)"""
        codes = []
        regions = []
        preprocessing_success_tracking = {}

        # Try every 5 degrees for very comprehensive coverage
        rotation_angles = range(0, 360, 5)

        for angle in rotation_angles:
            if angle == 0:
                rotated = image.copy()
            else:
                rotated = self._rotate_image(image, angle)

            # Track region
            h, w = rotated.shape[:2]
            regions.append((0, 0, w, h, f"fine_rotation_{angle}"))

            # Try detection with multiple preprocessing on rotated image
            angle_codes = self._decode_image_pyzbar(rotated)

            # Also try with enhanced preprocessing on this specific rotation
            if len(angle_codes) == 0:
                # Try barcode-specific enhancement
                enhanced = self._enhance_for_barcode(rotated)
                enhanced_codes = self._decode_image_pyzbar(enhanced)

                # Track if barcode enhancement was successful
                if len(enhanced_codes) > 0:
                    preprocessing_success_tracking['barcode_enhancement'] = {
                        'attempted': True,
                        'successful': True,
                        'codes_found': len(enhanced_codes),
                        'method_type': 'advanced',
                        'rotation_angle': angle
                    }

                for code in enhanced_codes:
                    code['rotation'] = angle
                    code['detection_method'] = 'fine_rotation_enhanced'
                    code['preprocessing_method'] = 'barcode_enhancement'
                    code['preprocessing_successful'] = True
                    code['preprocessing_success_tracking'] = preprocessing_success_tracking

                angle_codes.extend(enhanced_codes)

            # Add rotation info to codes
            for code in angle_codes:
                if 'rotation' not in code:
                    code['rotation'] = angle
                if 'detection_method' not in code:
                    code['detection_method'] = 'fine_rotation'

            codes.extend(angle_codes)

        return codes, regions

    def _decode_image_pyzbar(self, image):
        """Decode barcodes and QR codes using pyzbar"""
        codes = []

        try:
            # Convert to PIL Image for pyzbar
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    # BGR to RGB conversion for color images
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    # Grayscale image
                    pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Decode using pyzbar
            decoded_objects = pyzbar.decode(pil_image)

            for obj in decoded_objects:
                # Extract bounding box
                x, y, w, h = obj.rect.left, obj.rect.top, obj.rect.width, obj.rect.height

                code_info = {
                    'type': obj.type,
                    'data': obj.data.decode('utf-8'),
                    'bbox': (x, y, w, h),
                    'polygon': [(point.x, point.y) for point in obj.polygon]
                }
                codes.append(code_info)

        except Exception as e:
            print(f"Error in pyzbar decoding: {e}")

        return codes

    def _rotate_image(self, image, angle):
        """Rotate image by specified angle"""
        if angle == 0:
            return image.copy()

        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new dimensions to fit the entire rotated image
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))

        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_width // 2) - center[0]
        rotation_matrix[1, 2] += (new_height // 2) - center[1]

        # Perform rotation with white background
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        return rotated

    def _remove_duplicates(self, codes):
        """Remove duplicate codes based on data and type"""
        unique_codes = {}

        for code in codes:
            key = (code['data'], code['type'])
            if key not in unique_codes:
                unique_codes[key] = code
            else:
                # Keep the code with more information
                existing = unique_codes[key]
                if self._code_has_more_info(code, existing):
                    unique_codes[key] = code

        return list(unique_codes.values())

    def _code_has_more_info(self, code1, code2):
        """Check if code1 has more information than code2"""
        score1 = len([k for k in code1.keys() if code1[k] is not None])
        score2 = len([k for k in code2.keys() if code2[k] is not None])
        return score1 > score2

    # Preprocessing helper methods
    def _otsu_threshold(self, image):
        """Apply Otsu's thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _adaptive_threshold(self, image):
        """Apply adaptive thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def _sharpen_image(self, image):
        """Sharpen the image using unsharp masking"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

        sharpened = cv2.filter2D(gray, -1, kernel)
        return sharpened

    def _edge_enhance(self, image):
        """Enhance edges using gradient filters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Apply Sobel filters
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Combine gradients
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))

        return magnitude

    def _morphological_operations(self, image):
        """Apply morphological operations"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Apply closing (dilation followed by erosion)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Apply opening (erosion followed by dilation)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        return opened

    def _enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def _correct_perspective(self, image):
        """Attempt basic perspective correction"""
        # This is a simplified perspective correction
        # In practice, you might want more sophisticated methods
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Find contours
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # If we have a quadrilateral, attempt perspective correction
            if len(approx) == 4:
                # Get corner points
                pts = approx.reshape(4, 2)

                # Order points: top-left, top-right, bottom-right, bottom-left
                rect = self._order_points(pts)

                # Calculate dimensions
                width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
                height = int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[3] - rect[0])))

                # Define destination points
                dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

                # Calculate perspective transform matrix
                matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)

                # Apply perspective transformation
                corrected = cv2.warpPerspective(gray, matrix, (width, height))
                return corrected

        return gray

    def _order_points(self, pts):
        """Order points in the order: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum and difference of coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        # Top-left has smallest sum, bottom-right has largest sum
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right has smallest difference, bottom-left has largest difference
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def _enhance_for_barcode(self, image):
        """Apply barcode-specific enhancements"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Apply unsharp masking
        blurred = cv2.GaussianBlur(gray, (0, 0), 2)
        unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        # Apply histogram equalization
        equalized = cv2.equalizeHist(unsharp)

        # Apply vertical morphological closing for horizontal barcodes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        closed = cv2.morphologyEx(equalized, cv2.MORPH_CLOSE, kernel)

        return closed

    def _reduce_noise(self, image):
        """Reduce image noise"""
        if len(image.shape) == 3:
            # For color images, use bilateral filter
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            return cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        else:
            # For grayscale images, use median filter
            return cv2.medianBlur(image, 3)

    def _gamma_correction(self, image):
        """Apply gamma correction"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Apply gamma correction (gamma = 0.5 to brighten)
        gamma = 0.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        corrected = cv2.LUT(gray, table)
        return corrected

    def _create_visualization(self, image, result):
        """Create visualization of detected codes"""
        vis_image = image.copy()

        # Draw bounding boxes for detected codes
        for code in result['detected_codes']:
            if 'bbox' in code:
                x, y, w, h = code['bbox']

                # Choose color based on code type
                if code['type'].lower() == 'qrcode':
                    color = (0, 255, 0)  # Green for QR codes
                else:
                    color = (255, 0, 0)  # Blue for barcodes

                # Draw bounding box
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

                # Add label
                label = f"{code['type']}: {code['data'][:20]}..."
                cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save visualization
        filename = os.path.splitext(os.path.basename(result['image_path']))[0] + '_detected.jpg'
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, vis_image)

        return output_path

    def get_last_result(self):
        """Get the last detection result"""
        return self.last_result

    def save_result_json(self, filename=None):
        """Save the last result as JSON"""
        if self.last_result is None:
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_result_{timestamp}.json"

        import json
        output_path = os.path.join(self.output_dir, filename)

        # Create a JSON-serializable version of the result
        json_result = self.last_result.copy()
        json_result['processing_time'] = float(json_result['processing_time'])

        with open(output_path, 'w') as f:
            json.dump(json_result, f, indent=2, default=str)

        return output_path


def main():
    """Example usage of the Detector class"""
    detector = Detector()

    # Example detection
    try:
        result = detector.detect_codes('test_image.jpg')
        print(f"Detection completed in {result['processing_time']:.3f} seconds")
        print(f"Found {result['total_codes']} codes")

        for code in result['detected_codes']:
            print(f"- {code['type']}: {code['data']}")

        # Save result
        json_path = detector.save_result_json()
        print(f"Result saved to: {json_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()