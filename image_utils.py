# image_utils.py
"""
Simple image utility functions for preprocessing and manipulation
"""
import cv2
import numpy as np


def convert_to_grayscale(image):
    """Convert color image to grayscale"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def enhance_contrast(image):
    """Apply CLAHE to enhance contrast"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def reduce_noise(image):
    """Apply Gaussian blur to reduce noise"""
    return cv2.GaussianBlur(image, (5, 5), 0)


def preprocess_image(image):
    """Apply complete preprocessing pipeline"""
    # Convert to grayscale
    gray = convert_to_grayscale(image)

    # Reduce noise
    denoised = reduce_noise(gray)

    # Enhance contrast
    enhanced = enhance_contrast(denoised)

    return enhanced, gray