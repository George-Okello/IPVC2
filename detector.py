# enhanced_detector.py
"""
Enhanced detection system that handles rotated, inverted, and challenging barcodes
with comprehensive rotation angle support
"""
import cv2
import os
from datetime import datetime
import numpy as np
from pyzbar import pyzbar
from PIL import Image


class Detector:
    """Enhanced barcode and QR code detector with comprehensive rotation support"""

    def __init__(self):
        self.last_result = None

    def detect_codes(self, image_path):
        """Detect and decode codes in an image with comprehensive rotation handling"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        start_time = datetime.now()

        # Detect using multiple strategies
        all_codes = []
        barcode_regions = []
        qr_regions = []
        other_regions = []

        # Strategy 1: Standard detection
        basic_codes = self._detect_standard(image)
        all_codes.extend(basic_codes)

        # Strategy 2: Try with comprehensive rotations if needed
        if len(basic_codes) == 0:
            rotated_codes, rot_regions = self._detect_with_comprehensive_rotations(image)
            all_codes.extend(rotated_codes)
            other_regions.extend(rot_regions)

        # Strategy 3: Try with enhanced preprocessing + rotations if still needed
        if len(all_codes) == 0:
            enhanced_codes, enh_regions = self._detect_with_enhanced_preprocessing(image)
            all_codes.extend(enhanced_codes)
            other_regions.extend(enh_regions)

        # Strategy 4: Try fine-grained rotation if still no results
        if len(all_codes) == 0:
            fine_codes, fine_regions = self._detect_with_fine_rotations(image)
            all_codes.extend(fine_codes)
            other_regions.extend(fine_regions)

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
            'total_codes': len(unique_codes)
        }

        # Save visualization
        output_path = self._create_visualization(image, result)
        result['visualization_path'] = output_path

        self.last_result = result
        return result

    def _detect_standard(self, image):
        """Standard detection with pyzbar"""
        return self._decode_image_pyzbar(image)

    def _detect_with_comprehensive_rotations(self, image):
        """Try detection with comprehensive rotation angles"""
        codes = []
        regions = []

        # Try multiple rotation angles including common orientations
        rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        for angle in rotation_angles:
            if angle == 0:
                rotated = image.copy()
            else:
                rotated = self._rotate_image(image, angle)

            # Track region
            h, w = rotated.shape[:2]
            regions.append((0, 0, w, h, f"rotation_{angle}"))

            # Try detection
            angle_codes = self._decode_image_pyzbar(rotated)

            # Add rotation info to codes
            for code in angle_codes:
                code['rotation'] = angle

            codes.extend(angle_codes)

            # Continue checking all angles to find the best detection
            # Don't break early to ensure we get all possible detections

        return codes, regions

    def _detect_with_fine_rotations(self, image):
        """Try detection with fine-grained rotation angles (every 5 degrees)"""
        codes = []
        regions = []

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
                for code in enhanced_codes:
                    code['rotation'] = angle
                    code['detection_method'] = 'fine_rotation_enhanced'
                angle_codes.extend(enhanced_codes)

            # Add rotation info to codes
            for code in angle_codes:
                if 'rotation' not in code:
                    code['rotation'] = angle
                if 'detection_method' not in code:
                    code['detection_method'] = 'fine_rotation'

            codes.extend(angle_codes)

        return codes, regions

    def _detect_with_enhanced_preprocessing(self, image):
        """Try with enhanced preprocessing methods combined with rotations"""
        codes = []
        regions = []

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

                        codes.extend(new_codes)

            except Exception as e:
                print(f"Error in preprocessing method {name}: {e}")

        return codes, regions

    def _decode_image_pyzbar(self, image):
        """Decode using pyzbar with proper image conversion and multiple attempts"""
        results = []

        # Try with different image formats
        image_variants = []

        # Original image
        if len(image.shape) == 3:
            # BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_variants.append(('color', Image.fromarray(image_rgb)))

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_variants.append(('grayscale', Image.fromarray(gray)))
        else:
            # Already grayscale
            image_variants.append(('grayscale', Image.fromarray(image)))

        # Try binary threshold versions
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Different threshold methods
        try:
            _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            image_variants.append(('binary_fixed', Image.fromarray(thresh1)))

            _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_variants.append(('binary_otsu', Image.fromarray(thresh2)))

            # Try inverted binary
            _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            image_variants.append(('binary_inv_otsu', Image.fromarray(thresh_inv)))

            thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            image_variants.append(('adaptive', Image.fromarray(thresh3)))

            # Try adaptive with different parameters
            thresh4 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
            image_variants.append(('adaptive_mean', Image.fromarray(thresh4)))

            # Multiple threshold values for fixed thresholding
            for thresh_val in [80, 100, 120, 140, 160, 180]:
                _, thresh_multi = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                image_variants.append((f'binary_{thresh_val}', Image.fromarray(thresh_multi)))

        except Exception as e:
            print(f"Error creating threshold variants: {e}")

        # Try decoding each variant
        for variant_name, pil_image in image_variants:
            try:
                codes = pyzbar.decode(pil_image)
                for code in codes:
                    try:
                        result = {
                            'data': code.data.decode('utf-8', errors='replace'),
                            'type': code.type,
                            'bbox': code.rect,
                            'polygon': [(p.x, p.y) for p in code.polygon] if code.polygon else [],
                            'image_variant': variant_name
                        }
                        results.append(result)
                    except Exception as e:
                        print(f"Error processing decoded code: {e}")
            except Exception as e:
                print(f"Error in pyzbar decode for {variant_name}: {e}")

        return results

    def _remove_duplicates(self, all_codes):
        """Remove duplicate codes based on data and type"""
        unique_results = {}
        for result in all_codes:
            key = (result['data'], result['type'])
            if key not in unique_results:
                unique_results[key] = result
            else:
                # Keep the one with more detailed information
                existing = unique_results[key]
                if 'rotation' in result and 'rotation' not in existing:
                    unique_results[key] = result

        return list(unique_results.values())

    def _rotate_image(self, image, angle):
        """Rotate an image by given angle with improved handling"""
        if angle == 0:
            return image.copy()

        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding dimensions
        cos_val = np.abs(rotation_matrix[0, 0])
        sin_val = np.abs(rotation_matrix[0, 1])

        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))

        # Adjust the rotation matrix to take into account translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Perform the actual rotation and return the image
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        return rotated

    def _sharpen_image(self, image):
        """Apply sharpening filter"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        return sharpened

    def _adaptive_threshold(self, image):
        """Apply adaptive thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return thresh

    def _edge_enhance(self, image):
        """Enhance edges for barcode detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Edge detection
        edges = cv2.Canny(gray, 100, 200)

        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        return dilated

    def _morphological_operations(self, image):
        """Apply morphological operations to clean up the image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Opening (erosion followed by dilation)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Closing (dilation followed by erosion)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        return closing

    def _enhance_contrast(self, image):
        """Enhance contrast using CLAHE"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def _correct_perspective(self, image):
        """Attempt to correct perspective distortion"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find contours to detect rectangular shapes
        edges = cv2.Canny(gray, 50, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for the largest rectangular contour (potential barcode area)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                # Found a quadrilateral, try perspective correction
                try:
                    # Order points for perspective transform
                    rect = self._order_points(approx.reshape(4, 2))

                    # Calculate dimensions for output
                    width = max(
                        np.linalg.norm(rect[1] - rect[0]),
                        np.linalg.norm(rect[3] - rect[2])
                    )
                    height = max(
                        np.linalg.norm(rect[3] - rect[0]),
                        np.linalg.norm(rect[2] - rect[1])
                    )

                    # Destination points
                    dst = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype=np.float32)

                    # Get perspective transform matrix
                    M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
                    corrected = cv2.warpPerspective(gray, M, (int(width), int(height)))

                    return corrected
                except Exception as e:
                    continue

        return gray

    def _order_points(self, pts):
        """Order points in clockwise order starting from top-left"""
        # Sort by sum - top-left will have smallest sum, bottom-right largest
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        # Sort by difference - top-right will have smallest diff, bottom-left largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def _enhance_for_barcode(self, image):
        """Specific enhancement for barcode detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply multiple enhancement techniques
        # 1. Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 2. Unsharp masking for sharpening
        unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        # 3. Histogram equalization
        equalized = cv2.equalizeHist(unsharp_mask)

        # 4. Morphological operations to enhance bar patterns
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        enhanced = cv2.morphologyEx(equalized, cv2.MORPH_CLOSE, kernel)

        return enhanced

    def _reduce_noise(self, image):
        """Reduce noise while preserving barcode patterns"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Use bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply median blur to remove salt and pepper noise
        denoised = cv2.medianBlur(denoised, 3)

        return denoised

    def _gamma_correction(self, image):
        """Apply gamma correction for brightness adjustment"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Try different gamma values
        gamma_values = [0.5, 0.7, 1.0, 1.5, 2.0]
        best_result = gray

        for gamma in gamma_values:
            # Build lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")

            # Apply gamma correction
            corrected = cv2.LUT(gray, table)

            # Simple metric: try to maximize contrast in potential barcode regions
            contrast = np.std(corrected)
            if contrast > np.std(best_result):
                best_result = corrected

        return best_result

    def _create_visualization(self, image, result):
        """Create detailed visualization of results"""
        vis_image = image.copy()

        # Draw all regions in blue
        for region in result.get('regions', []):
            if len(region) >= 4:  # We need at least x, y, w, h
                x, y, w, h = region[:4]
                label = region[4] if len(region) > 4 else ""

                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(vis_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1)

        # Draw barcode regions in green
        for region in result.get('barcode_regions', []):
            if len(region) >= 4:
                x, y, w, h = region[:4]
                label = region[4] if len(region) > 4 else "Barcode"

                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

        # Draw QR code regions in yellow
        for region in result.get('qr_regions', []):
            if len(region) >= 4:
                x, y, w, h = region[:4]
                label = region[4] if len(region) > 4 else "QR Code"

                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(vis_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1)

        # Draw decoded codes in red
        for code in result['detected_codes']:
            # Draw polygon if available
            if 'polygon' in code and code['polygon']:
                points = np.array([(int(p[0]), int(p[1])) for p in code['polygon']])
                cv2.polylines(vis_image, [points], True, (0, 0, 255), 3)

                # Add label
                x, y = points[0]

            # Fall back to bbox if polygon not available
            elif 'bbox' in code:
                x, y, w, h = code['bbox']
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 3)

            else:
                # Skip if no position info
                continue

            # Create detailed label with code data
            info = []
            info.append(f"{code['type']}")
            info.append(code['data'][:30])  # Show more characters

            if 'rotation' in code:
                info.append(f"Rot: {code['rotation']}°")

            if 'preprocess' in code:
                info.append(f"Proc: {code['preprocess']}")

            if 'image_variant' in code:
                info.append(f"Var: {code['image_variant']}")

            if 'detection_method' in code:
                info.append(f"Method: {code['detection_method']}")

            text = " | ".join(info)

            # Add text with background for better readability
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x, y - 25), (x + text_size[0], y), (0, 0, 0), -1)
            cv2.putText(vis_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        # Save visualization
        basename = os.path.basename(result['image_path'])
        output_path = os.path.splitext(basename)[0] + '_detected.jpg'
        cv2.imwrite(output_path, vis_image)

        return output_path

    def get_summary(self):
        """Get detailed summary of last detection"""
        if not self.last_result:
            return "No detection performed yet"

        result = self.last_result
        summary = f"""
    Enhanced Detection Summary:
    - Image: {os.path.basename(result['image_path'])}
    - Processing time: {result['processing_time']:.3f} seconds
    - Total codes decoded: {result['total_codes']}
    
    Detected codes:
    """
        for i, code in enumerate(result['detected_codes'], 1):
            code_info = [f"{i}. {code['type']}: {code['data']}"]

            if 'rotation' in code:
                code_info.append(f"(Rotation: {code['rotation']}°)")

            if 'preprocess' in code:
                code_info.append(f"(Preprocessing: {code['preprocess']})")

            if 'image_variant' in code:
                code_info.append(f"(Image variant: {code['image_variant']})")

            if 'detection_method' in code:
                code_info.append(f"(Method: {code['detection_method']})")

            summary += " ".join(code_info) + "\n"

        if result['total_codes'] == 0:
            summary += "\nNo codes detected. Consider:\n"
            summary += "- Checking image quality and resolution\n"
            summary += "- Ensuring codes are not damaged or partially obscured\n"
            summary += "- Trying different lighting conditions\n"

        return summary