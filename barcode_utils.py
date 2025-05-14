# barcode_utils.py
"""
Simple barcode detection utilities
"""
import cv2
import numpy as np


def detect_barcode_gradient(image):
    """Detect barcode using gradient analysis"""
    # Calculate gradients
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    # Subtract gradients (barcodes have horizontal patterns)
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)

    # Apply horizontal morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)

    # Clean up
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    return closed


def find_barcode_regions(processed_image):
    """Find potential barcode regions from processed image"""
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    barcode_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = cv2.contourArea(contour)

        # Filter based on barcode characteristics
        if aspect_ratio > 1.5 and area > 500 and h > 20:
            barcode_regions.append((x, y, w, h))

    return barcode_regions