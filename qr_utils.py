# qr_utils.py
"""
Simple QR code detection utilities
"""
import cv2
import numpy as np


def detect_qr_contours(image):
    """Detect QR code using contour analysis"""
    # Apply threshold
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    qr_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = cv2.contourArea(contour)

        # Filter based on QR code characteristics (square-like)
        if 0.7 <= aspect_ratio <= 1.3 and area > 1000:
            # Check if contour is roughly rectangular
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) >= 4:
                qr_regions.append((x, y, w, h))

    return qr_regions


def find_qr_regions(image):
    """Find potential QR code regions"""
    return detect_qr_contours(image)