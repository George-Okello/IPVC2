# decoder.py
"""
Simple code decoder using pyzbar
"""
import cv2
from pyzbar import pyzbar
from PIL import Image


def decode_image(image):
    """Decode barcodes and QR codes from an image"""
    # Convert to PIL Image if needed
    if hasattr(image, 'shape'):  # numpy array
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
    else:
        pil_image = image

    # Decode using pyzbar
    codes = pyzbar.decode(pil_image)

    results = []
    for code in codes:
        result = {
            'data': code.data.decode('utf-8'),
            'type': code.type,
            'bbox': code.rect,
            'polygon': [(p.x, p.y) for p in code.polygon]
        }
        results.append(result)

    return results


def preprocess_for_decoding(image):
    """Apply different preprocessing methods for better decoding"""
    preprocessed_images = []

    # Original
    preprocessed_images.append(image)

    # Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed_images.append(gray)

    # OTSU threshold
    if len(image.shape) == 2:
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(otsu)

    return preprocessed_images


def decode_with_preprocessing(image):
    """Try decoding with different preprocessing methods"""
    all_results = []

    # Try with different preprocessing
    processed_images = preprocess_for_decoding(image)

    for processed in processed_images:
        results = decode_image(processed)
        all_results.extend(results)

    # Remove duplicates (same data)
    unique_results = {}
    for result in all_results:
        key = (result['data'], result['type'])
        if key not in unique_results:
            unique_results[key] = result

    return list(unique_results.values())