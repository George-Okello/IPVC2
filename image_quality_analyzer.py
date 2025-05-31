# image_quality_analyzer.py
"""
Enhanced image quality analysis module for barcode detection system
"""
import cv2
import numpy as np
from PIL import Image, ImageStat
import os
from scipy import ndimage
from skimage import measure, filters
import math


class ImageQualityAnalyzer:
    """Comprehensive image quality analysis for barcode detection evaluation"""

    def __init__(self):
        self.quality_metrics = {}

    def analyze_image_quality(self, image_path):
        """
        Comprehensive image quality analysis

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing various quality metrics
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Convert to different formats for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pil_image = Image.open(image_path)

        # Calculate all quality metrics
        quality_metrics = {
            'filename': os.path.basename(image_path),
            'resolution': self._calculate_resolution(image),
            'file_size_mb': self._calculate_file_size(image_path),
            'contrast_metrics': self._calculate_contrast_metrics(gray),
            'sharpness_metrics': self._calculate_sharpness_metrics(gray),
            'noise_metrics': self._calculate_noise_metrics(gray),
            'lighting_metrics': self._calculate_lighting_metrics(gray),
            'distortion_metrics': self._calculate_distortion_metrics(gray),
            'color_metrics': self._calculate_color_metrics(image, pil_image),
            'structural_metrics': self._calculate_structural_metrics(gray),
            'overall_quality_score': 0  # Will be calculated at the end
        }

        # Calculate overall quality score
        quality_metrics['overall_quality_score'] = self._calculate_overall_quality_score(quality_metrics)

        return quality_metrics

    def _calculate_resolution(self, image):
        """Calculate resolution metrics"""
        height, width = image.shape[:2]
        megapixels = (width * height) / 1_000_000
        aspect_ratio = width / height

        # Categorize resolution
        if megapixels < 0.3:
            resolution_category = "low"
        elif megapixels < 2.0:
            resolution_category = "medium"
        elif megapixels < 8.0:
            resolution_category = "high"
        else:
            resolution_category = "very_high"

        return {
            'width': width,
            'height': height,
            'megapixels': megapixels,
            'aspect_ratio': aspect_ratio,
            'total_pixels': width * height,
            'category': resolution_category
        }

    def _calculate_file_size(self, image_path):
        """Calculate file size in MB"""
        file_size_bytes = os.path.getsize(image_path)
        return file_size_bytes / (1024 * 1024)

    def _calculate_contrast_metrics(self, gray_image):
        """Calculate various contrast metrics"""
        # RMS Contrast
        rms_contrast = np.sqrt(np.mean((gray_image - np.mean(gray_image)) ** 2))

        # Michelson Contrast
        max_luminance = np.max(gray_image)
        min_luminance = np.min(gray_image)
        if max_luminance + min_luminance > 0:
            michelson_contrast = (max_luminance - min_luminance) / (max_luminance + min_luminance)
        else:
            michelson_contrast = 0

        # Weber Contrast
        mean_background = np.mean(gray_image)
        if mean_background > 0:
            weber_contrast = (max_luminance - mean_background) / mean_background
        else:
            weber_contrast = 0

        # Standard deviation contrast
        std_contrast = np.std(gray_image) / 255.0

        # Local contrast using standard deviation of local patches
        patch_size = 16
        local_contrasts = []
        for i in range(0, gray_image.shape[0] - patch_size, patch_size):
            for j in range(0, gray_image.shape[1] - patch_size, patch_size):
                patch = gray_image[i:i + patch_size, j:j + patch_size]
                local_contrasts.append(np.std(patch))

        avg_local_contrast = np.mean(local_contrasts) / 255.0

        # Categorize contrast level
        if rms_contrast < 30:
            contrast_category = "low"
        elif rms_contrast < 70:
            contrast_category = "medium"
        else:
            contrast_category = "high"

        return {
            'rms_contrast': rms_contrast,
            'michelson_contrast': michelson_contrast,
            'weber_contrast': weber_contrast,
            'std_contrast': std_contrast,
            'avg_local_contrast': avg_local_contrast,
            'normalized_contrast': rms_contrast / 255.0,
            'contrast_category': contrast_category,
            'dynamic_range': max_luminance - min_luminance
        }

    def _calculate_sharpness_metrics(self, gray_image):
        """Calculate image sharpness metrics"""
        # Variance of Laplacian (common sharpness measure)
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

        # Gradient magnitude sharpness
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        avg_gradient = np.mean(gradient_magnitude)

        # Tenengrad sharpness (sum of squared gradients)
        tenengrad = np.sum(gradient_magnitude ** 2)

        # Brenner sharpness (sum of squared differences)
        brenner = 0
        for i in range(gray_image.shape[0] - 2):
            brenner += np.sum((gray_image[i + 2, :] - gray_image[i, :]) ** 2)

        # Normalized Graylevel Variance
        normalized_variance = np.var(gray_image) / (np.mean(gray_image) + 1e-6)

        # Categorize sharpness
        if laplacian_var < 100:
            sharpness_category = "blurred"
        elif laplacian_var < 500:
            sharpness_category = "moderate"
        else:
            sharpness_category = "sharp"

        return {
            'laplacian_variance': laplacian_var,
            'avg_gradient_magnitude': avg_gradient,
            'tenengrad': tenengrad,
            'brenner': brenner,
            'normalized_variance': normalized_variance,
            'sharpness_category': sharpness_category
        }

    def _calculate_noise_metrics(self, gray_image):
        """Calculate noise-related metrics"""

        # Estimate noise using median absolute deviation
        def estimate_noise_mad(image):
            """Estimate noise using Median Absolute Deviation"""
            H, W = image.shape
            M = [[1, -2, 1],
                 [-2, 4, -2],
                 [1, -2, 1]]
            M = np.array(M)
            sigma = np.sum(np.sum(np.absolute(ndimage.convolve(image, M))))
            sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))
            return sigma

        noise_level = estimate_noise_mad(gray_image.astype(np.float64))

        # Signal-to-noise ratio
        signal_power = np.mean(gray_image ** 2)
        noise_power = noise_level ** 2
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')

        # Local noise estimation using standard deviation in smooth regions
        # Apply Gaussian blur and compare with original
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        noise_estimate = np.std(gray_image.astype(np.float32) - blurred.astype(np.float32))

        # Categorize noise level
        if noise_level < 5:
            noise_category = "low"
        elif noise_level < 15:
            noise_category = "medium"
        else:
            noise_category = "high"

        return {
            'estimated_noise_level': noise_level,
            'signal_to_noise_ratio': snr,
            'local_noise_estimate': noise_estimate,
            'noise_category': noise_category
        }

    def _calculate_lighting_metrics(self, gray_image):
        """Calculate lighting and exposure metrics"""
        # Basic statistics
        mean_brightness = np.mean(gray_image)
        brightness_std = np.std(gray_image)

        # Histogram analysis
        hist, bins = np.histogram(gray_image, bins=256, range=(0, 256))

        # Check for clipping (over/under exposure)
        underexposed_pixels = np.sum(gray_image < 10) / gray_image.size
        overexposed_pixels = np.sum(gray_image > 245) / gray_image.size

        # Lighting uniformity (coefficient of variation)
        if mean_brightness > 0:
            lighting_uniformity = brightness_std / mean_brightness
        else:
            lighting_uniformity = float('inf')

        # Dynamic range utilization
        actual_range = np.max(gray_image) - np.min(gray_image)
        range_utilization = actual_range / 255.0

        # Categorize lighting condition
        if mean_brightness < 60:
            lighting_category = "underexposed"
        elif mean_brightness > 200:
            lighting_category = "overexposed"
        elif lighting_uniformity > 0.8:
            lighting_category = "uneven"
        else:
            lighting_category = "optimal"

        return {
            'mean_brightness': mean_brightness,
            'brightness_std': brightness_std,
            'lighting_uniformity': lighting_uniformity,
            'underexposed_ratio': underexposed_pixels,
            'overexposed_ratio': overexposed_pixels,
            'dynamic_range_utilization': range_utilization,
            'lighting_category': lighting_category
        }

    def _calculate_distortion_metrics(self, gray_image):
        """Calculate geometric distortion metrics"""
        # Edge detection for line analysis
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

        # Hough line detection to find straight lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        # Calculate line angle deviations
        angle_deviations = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                angle_deg = np.degrees(theta)
                # Check deviation from horizontal/vertical
                deviation = min(abs(angle_deg), abs(angle_deg - 90), abs(angle_deg - 180))
                angle_deviations.append(deviation)

        # Perspective distortion estimation
        avg_angle_deviation = np.mean(angle_deviations) if angle_deviations else 0

        # Corner detection for geometric analysis
        corners = cv2.goodFeaturesToTrack(gray_image, 100, 0.01, 10)
        corner_count = len(corners) if corners is not None else 0

        # Categorize distortion level
        if avg_angle_deviation < 5:
            distortion_category = "minimal"
        elif avg_angle_deviation < 15:
            distortion_category = "moderate"
        else:
            distortion_category = "severe"

        return {
            'avg_angle_deviation': avg_angle_deviation,
            'line_count': len(lines) if lines is not None else 0,
            'corner_count': corner_count,
            'distortion_category': distortion_category
        }

    def _calculate_color_metrics(self, color_image, pil_image):
        """Calculate color-related metrics"""
        # Color saturation
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)

        # Color variance
        color_variance = np.var(color_image.reshape(-1, 3), axis=0)
        avg_color_variance = np.mean(color_variance)

        # Grayscale similarity (how close to grayscale the image is)
        b, g, r = cv2.split(color_image)
        gray_similarity = 1 - (np.std([np.mean(b), np.mean(g), np.mean(r)]) / 255.0)

        # PIL-based statistics
        try:
            stat = ImageStat.Stat(pil_image)
            brightness = sum(stat.mean) / len(stat.mean)
            contrast = sum(stat.stddev) / len(stat.stddev)
        except:
            brightness = np.mean(color_image)
            contrast = np.std(color_image)

        return {
            'avg_saturation': avg_saturation,
            'avg_color_variance': avg_color_variance,
            'grayscale_similarity': gray_similarity,
            'pil_brightness': brightness,
            'pil_contrast': contrast
        }

    def _calculate_structural_metrics(self, gray_image):
        """Calculate structural complexity metrics"""
        # Texture analysis using Local Binary Patterns
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray_image, 8, 1, method='uniform')
            texture_variance = np.var(lbp)
        except ImportError:
            # Fallback if scikit-image not available
            texture_variance = np.var(gray_image)

        # Edge density
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Entropy (measure of information content)
        hist, _ = np.histogram(gray_image, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))

        # Complexity score based on multiple factors
        complexity_score = (edge_density * 0.4 +
                            (texture_variance / 10000) * 0.3 +
                            (entropy / 8) * 0.3)

        return {
            'texture_variance': texture_variance,
            'edge_density': edge_density,
            'entropy': entropy,
            'complexity_score': complexity_score
        }

    def _calculate_overall_quality_score(self, metrics):
        """Calculate an overall quality score (0-100)"""
        weights = {
            'contrast': 0.25,
            'sharpness': 0.25,
            'noise': 0.20,
            'lighting': 0.20,
            'distortion': 0.10
        }

        # Normalize individual scores to 0-100
        contrast_score = min(100, (metrics['contrast_metrics']['normalized_contrast'] * 100))
        sharpness_score = min(100, (metrics['sharpness_metrics']['laplacian_variance'] / 10))
        noise_score = max(0, 100 - (metrics['noise_metrics']['estimated_noise_level'] * 5))
        lighting_score = 100 - abs(metrics['lighting_metrics']['mean_brightness'] - 127.5) / 127.5 * 100
        distortion_score = max(0, 100 - (metrics['distortion_metrics']['avg_angle_deviation'] * 5))

        # Calculate weighted average
        overall_score = (
                contrast_score * weights['contrast'] +
                sharpness_score * weights['sharpness'] +
                noise_score * weights['noise'] +
                lighting_score * weights['lighting'] +
                distortion_score * weights['distortion']
        )

        return min(100, max(0, overall_score))

    def batch_analyze_quality(self, image_folder, output_csv=None):
        """
        Analyze image quality for all images in a folder

        Args:
            image_folder: Path to folder containing images
            output_csv: Optional path to save CSV results

        Returns:
            List of quality analysis results
        """
        import pandas as pd

        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # Find all image files
        image_files = []
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_files.append(os.path.join(root, file))

        print(f"Found {len(image_files)} images to analyze...")

        # Analyze each image
        results = []
        for i, image_path in enumerate(image_files, 1):
            try:
                print(f"Analyzing {i}/{len(image_files)}: {os.path.basename(image_path)}")
                quality_metrics = self.analyze_image_quality(image_path)

                # Flatten the nested dictionary for CSV export
                flattened_metrics = self._flatten_metrics(quality_metrics)
                results.append(flattened_metrics)

            except Exception as e:
                print(f"Error analyzing {image_path}: {e}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")

        return results

    def _flatten_metrics(self, metrics):
        """Flatten nested metrics dictionary for CSV export"""
        flattened = {'filename': metrics['filename']}

        # Resolution metrics
        for key, value in metrics['resolution'].items():
            flattened[f'resolution_{key}'] = value

        flattened['file_size_mb'] = metrics['file_size_mb']

        # Flatten all other metric categories
        for category in ['contrast_metrics', 'sharpness_metrics', 'noise_metrics',
                         'lighting_metrics', 'distortion_metrics', 'color_metrics',
                         'structural_metrics']:
            for key, value in metrics[category].items():
                flattened[f'{category}_{key}'] = value

        flattened['overall_quality_score'] = metrics['overall_quality_score']

        return flattened

    def generate_quality_report(self, results, output_html=None):
        """Generate a comprehensive quality analysis report"""
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(results)

        # Generate summary statistics
        report = f"""
IMAGE QUALITY ANALYSIS REPORT
============================

Dataset Summary:
- Total Images Analyzed: {len(df)}
- Average Quality Score: {df['overall_quality_score'].mean():.2f}
- Quality Score Range: {df['overall_quality_score'].min():.2f} - {df['overall_quality_score'].max():.2f}

Resolution Distribution:
{df['resolution_category'].value_counts()}

Contrast Distribution:
{df['contrast_metrics_contrast_category'].value_counts()}

Sharpness Distribution:
{df['sharpness_metrics_sharpness_category'].value_counts()}

Lighting Distribution:
{df['lighting_metrics_lighting_category'].value_counts()}

Noise Distribution:
{df['noise_metrics_noise_category'].value_counts()}
        """

        print(report)

        if output_html:
            # Create visualizations and save as HTML
            self._create_quality_visualizations(df, output_html)

        return report

    def _create_quality_visualizations(self, df, output_html):
        """Create quality analysis visualizations"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Quality score distribution
        axes[0, 0].hist(df['overall_quality_score'], bins=20, alpha=0.7)
        axes[0, 0].set_title('Overall Quality Score Distribution')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Frequency')

        # Resolution vs Quality
        axes[0, 1].scatter(df['resolution_megapixels'], df['overall_quality_score'], alpha=0.6)
        axes[0, 1].set_title('Resolution vs Quality Score')
        axes[0, 1].set_xlabel('Megapixels')
        axes[0, 1].set_ylabel('Quality Score')

        # Contrast vs Quality
        axes[0, 2].scatter(df['contrast_metrics_normalized_contrast'], df['overall_quality_score'], alpha=0.6)
        axes[0, 2].set_title('Contrast vs Quality Score')
        axes[0, 2].set_xlabel('Normalized Contrast')
        axes[0, 2].set_ylabel('Quality Score')

        # Category distributions
        df['resolution_category'].value_counts().plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Resolution Categories')
        axes[1, 0].tick_params(axis='x', rotation=45)

        df['contrast_metrics_contrast_category'].value_counts().plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Contrast Categories')
        axes[1, 1].tick_params(axis='x', rotation=45)

        df['lighting_metrics_lighting_category'].value_counts().plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Lighting Categories')
        axes[1, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_html.replace('.html', '.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Example usage of the ImageQualityAnalyzer"""
    analyzer = ImageQualityAnalyzer()

    # Analyze a single image
    try:
        quality_metrics = analyzer.analyze_image_quality('test_image.jpg')
        print(f"Quality Score: {quality_metrics['overall_quality_score']:.2f}")
        print(f"Resolution: {quality_metrics['resolution']['category']}")
        print(f"Contrast: {quality_metrics['contrast_metrics']['contrast_category']}")
    except Exception as e:
        print(f"Error: {e}")

    # Batch analyze a folder
    try:
        results = analyzer.batch_analyze_quality('test_images', 'quality_analysis.csv')
        report = analyzer.generate_quality_report(results, 'quality_report.html')
    except Exception as e:
        print(f"Batch analysis error: {e}")


if __name__ == "__main__":
    main()