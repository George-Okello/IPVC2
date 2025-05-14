# enhanced_detector.py
"""
Enhanced detection system that handles rotated, inverted, and challenging barcodes
"""
import cv2
import os
from datetime import datetime
import numpy as np
from pyzbar import pyzbar
from PIL import Image


class Detector:
    """Enhanced barcode and QR code detector with rotation support"""

    def __init__(self):
        self.last_result = None

    def detect_codes(self, image_path):
        """Detect and decode codes in an image with rotation handling"""
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

        # Strategy 2: Try with rotations if needed
        if len(basic_codes) == 0:
            rotated_codes, rot_regions = self._detect_with_rotations(image)
            all_codes.extend(rotated_codes)
            other_regions.extend(rot_regions)

        # Strategy 3: Try with enhanced preprocessing if still needed
        if len(all_codes) == 0:
            enhanced_codes, enh_regions = self._detect_with_enhanced_preprocessing(image)
            all_codes.extend(enhanced_codes)
            other_regions.extend(enh_regions)

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

    def _detect_with_rotations(self, image):
        """Try detection with different rotations"""
        codes = []
        regions = []

        # Try 4 rotations (0, 90, 180, 270 degrees)
        for angle in [0, 90, 180, 270]:
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

            # Stop if we found codes
            if len(angle_codes) > 0:
                break

        return codes, regions

    def _detect_with_enhanced_preprocessing(self, image):
        """Try with enhanced preprocessing methods"""
        codes = []
        regions = []

        # List of preprocessing methods to try
        preprocess_methods = [
            ('original', lambda img: img),
            ('grayscale', lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img),
            ('blur', lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
            ('sharpen', self._sharpen_image),
            ('adaptive_threshold', self._adaptive_threshold),
            ('edge_enhance', self._edge_enhance)
        ]

        # Try each preprocessing method
        for name, method in preprocess_methods:
            try:
                processed = method(image)
                if processed is not None:
                    # Track region
                    h, w = processed.shape[:2]
                    regions.append((0, 0, w, h, f"preprocess_{name}"))

                    # Try normal orientation
                    new_codes = self._decode_image_pyzbar(processed)
                    for code in new_codes:
                        code['preprocess'] = name
                    codes.extend(new_codes)

                    # If we didn't find any codes, try rotated versions
                    if len(new_codes) == 0:
                        for angle in [90, 180, 270]:
                            rotated = self._rotate_image(processed, angle)
                            rot_codes = self._decode_image_pyzbar(rotated)

                            for code in rot_codes:
                                code['preprocess'] = name
                                code['rotation'] = angle

                            codes.extend(rot_codes)

                            if len(rot_codes) > 0:
                                break
            except Exception as e:
                print(f"Error in preprocessing method {name}: {e}")

        return codes, regions

    def _decode_image_pyzbar(self, image):
        """Decode using pyzbar with proper image conversion"""
        # Convert to PIL Image
        if len(image.shape) == 3:
            # BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            # Grayscale
            pil_image = Image.fromarray(image)

        # Use pyzbar for decoding
        try:
            codes = pyzbar.decode(pil_image)
        except Exception as e:
            print(f"Error in pyzbar decode: {e}")
            return []

        results = []
        for code in codes:
            try:
                result = {
                    'data': code.data.decode('utf-8', errors='replace'),
                    'type': code.type,
                    'bbox': code.rect,
                    'polygon': [(p.x, p.y) for p in code.polygon] if code.polygon else []
                }
                results.append(result)
            except Exception as e:
                print(f"Error processing decoded code: {e}")

        return results

    def _remove_duplicates(self, all_codes):
        """Remove duplicate codes"""
        unique_results = {}
        for result in all_codes:
            key = (result['data'], result['type'])
            if key not in unique_results:
                unique_results[key] = result

        return list(unique_results.values())

    def _rotate_image(self, image, angle):
        """Rotate an image by given angle"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Determine new bounds
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])

        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)

        # Adjust rotation matrix
        rotation_matrix[0, 2] += new_width / 2 - center[0]
        rotation_matrix[1, 2] += new_height / 2 - center[1]

        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                 flags=cv2.INTER_LINEAR)
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

    def _create_visualization(self, image, result):
        """Create simple visualization of results"""
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
                cv2.polylines(vis_image, [points], True, (0, 0, 255), 2)

                # Add label
                x, y = points[0]

            # Fall back to bbox if polygon not available
            elif 'bbox' in code:
                x, y, w, h = code['bbox']
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            else:
                # Skip if no position info
                continue

            # Create label with code data
            info = []
            info.append(f"{code['type']}")
            info.append(code['data'][:20])

            if 'rotation' in code:
                info.append(f"Rot: {code['rotation']}°")

            if 'preprocess' in code:
                info.append(f"Proc: {code['preprocess']}")

            text = " | ".join(info)

            # Add text with background for better readability
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x, y - 20), (x + text_size[0], y), (0, 0, 0), -1)
            cv2.putText(vis_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        # Save visualization
        basename = os.path.basename(result['image_path'])
        output_path = os.path.splitext(basename)[0] + '_detected.jpg'
        cv2.imwrite(output_path, vis_image)

        return output_path

    def get_summary(self):
        """Get summary of last detection"""
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
                code_info.append(f"(Method: {code['preprocess']})")

            summary += " ".join(code_info) + "\n"

        return summary