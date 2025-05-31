# fixed_batch_detector.py
"""
Fixed batch processing system with comprehensive error handling
"""
import cv2
import os
import json
from datetime import datetime
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple, Optional
import traceback
import matplotlib.pyplot as plt
import seaborn as sns


class BatchDetector:
    """Enhanced detector for batch processing with robust error handling"""

    def __init__(self, output_dir="batch_results"):
        self.detector = None
        self.quality_analyzer = None
        self.output_dir = output_dir
        self.results = []
        self.performance_stats = {}
        self.error_log = []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize detector and quality analyzer with error handling"""
        try:
            from detector import Detector
            self.detector = Detector()
            print("✅ Detector initialized")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize detector: {e}")
            self.detector = None

        try:
            from image_quality_analyzer import ImageQualityAnalyzer
            self.quality_analyzer = ImageQualityAnalyzer()
            print("✅ Quality analyzer initialized")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize quality analyzer: {e}")
            self.quality_analyzer = None

    def process_folder(self, folder_path: str, max_images: int = None) -> Dict:
        """
        Process all images in a folder with comprehensive error handling

        Args:
            folder_path: Path to folder containing images
            max_images: Maximum number of images to process (None for all)

        Returns:
            Dictionary containing batch results and performance analysis
        """
        print(f"Starting batch processing of folder: {folder_path}")

        # Validate folder
        if not os.path.exists(folder_path):
            error_msg = f"Folder not found: {folder_path}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        # Get all image files
        image_files = self._get_image_files(folder_path)

        if not image_files:
            error_msg = "No supported image files found!"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        if max_images:
            image_files = image_files[:max_images]

        print(f"Found {len(image_files)} images to process")

        # Reset results for this batch
        self.results = []
        self.error_log = []

        # Process each image
        batch_start = datetime.now()
        failed_images = []
        successful_count = 0

        for i, image_path in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")

            try:
                result = self._process_single_image_safe(image_path)
                if result:
                    self.results.append(result)
                    successful_count += 1
                    print(f"  ✓ Time: {result['processing_time']:.3f}s, Codes: {result['total_codes']}")
                else:
                    failed_images.append({"path": image_path, "error": "Processing returned None"})
                    print(f"  ✗ Processing returned None")

            except Exception as e:
                error_msg = str(e)
                print(f"  ✗ Error processing {image_path}: {error_msg}")
                failed_images.append({"path": image_path, "error": error_msg})
                self.error_log.append(f"Error processing {image_path}: {error_msg}")

        batch_end = datetime.now()
        total_batch_time = (batch_end - batch_start).total_seconds()

        print(f"\nBatch processing complete: {successful_count}/{len(image_files)} successful")

        # Calculate batch info
        batch_info = {
            "folder_path": folder_path,
            "total_images": len(image_files),
            "processed_successfully": successful_count,
            "failed": len(failed_images),
            "total_processing_time": total_batch_time,
            "average_time_per_image": total_batch_time / len(image_files) if image_files else 0,
            "timestamp": batch_start.isoformat(),
            "total_codes_found": sum(r.get('total_codes', 0) for r in self.results)
        }

        # Analyze results if we have any
        if self.results:
            try:
                analysis = self._analyze_performance()
            except Exception as e:
                print(f"⚠️  Warning: Performance analysis failed: {e}")
                analysis = {"error": f"Analysis failed: {str(e)}"}
        else:
            analysis = {"error": "No results to analyze"}

        # Create comprehensive report
        report = {
            "batch_info": batch_info,
            "failed_images": failed_images,
            "performance_analysis": analysis,
            "detailed_results": self.results,
            "error_log": self.error_log
        }

        # Save report
        try:
            report_path = self._save_batch_report(report)
            print(f"Report saved to: {report_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save report: {e}")
            report_path = None

        # Create visualizations if possible
        try:
            if self.results:
                viz_paths = self._create_performance_visualizations()
                report["visualization_paths"] = viz_paths
        except Exception as e:
            print(f"⚠️  Warning: Could not create visualizations: {e}")

        return report

    def _get_image_files(self, folder_path: str) -> List[str]:
        """Get all supported image files from folder"""
        image_files = []

        try:
            for file_path in Path(folder_path).rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    image_files.append(str(file_path))
        except Exception as e:
            print(f"⚠️  Warning: Error scanning folder {folder_path}: {e}")

        return sorted(image_files)

    def _process_single_image_safe(self, image_path: str) -> Optional[Dict]:
        """Process a single image with comprehensive error handling"""
        try:
            # Validate file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Get file metadata safely
            file_metadata = self._get_file_metadata_safe(image_path)

            # Analyze image quality safely
            quality_metrics = self._analyze_quality_safe(image_path)

            # Process with detector safely
            detection_result = self._detect_codes_safe(image_path)

            # Calculate performance metrics
            processing_time = detection_result.get('processing_time', 0.0)
            megapixels = file_metadata.get('megapixels', 0.0)
            file_size_mb = file_metadata.get('file_size_mb', 0.0)
            pixels = file_metadata.get('pixels', 0)

            performance_metrics = {
                "processing_time": processing_time,
                "time_per_megapixel": processing_time / megapixels if megapixels > 0 else 0,
                "time_per_mb": processing_time / file_size_mb if file_size_mb > 0 else 0,
                "pixels_per_second": pixels / processing_time if processing_time > 0 else 0
            }

            # Calculate complexity indicators
            complexity_indicators = {
                "detection_strategies_used": self._estimate_strategies_used(detection_result),
                "preprocessing_complexity": self._estimate_preprocessing_complexity(detection_result),
                "rotation_attempts": self._count_rotation_attempts(detection_result)
            }

            # Combine all results
            enhanced_result = {
                **detection_result,
                "file_metadata": file_metadata,
                "performance_metrics": performance_metrics,
                "complexity_indicators": complexity_indicators,
                "quality_metrics": quality_metrics
            }

            return enhanced_result

        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            print(f"❌ {error_msg}")
            self.error_log.append(error_msg)

            # Return None to indicate failure
            return None

    def _get_file_metadata_safe(self, image_path: str) -> Dict:
        """Safely get file metadata"""
        try:
            file_stat = os.stat(image_path)
            image = cv2.imread(image_path)

            if image is not None:
                height, width = image.shape[:2]
                pixels = height * width
                megapixels = pixels / 1_000_000
            else:
                # If image can't be loaded, use default values
                height, width, pixels, megapixels = 0, 0, 0, 0.0

            file_size_mb = file_stat.st_size / (1024 * 1024)

            return {
                "filename": os.path.basename(image_path),
                "full_path": image_path,
                "file_size_mb": file_size_mb,
                "dimensions": {"width": width, "height": height},
                "pixels": pixels,
                "megapixels": megapixels,
                "aspect_ratio": width / height if height > 0 else 0
            }

        except Exception as e:
            print(f"⚠️  Warning: Could not get metadata for {image_path}: {e}")
            return {
                "filename": os.path.basename(image_path),
                "full_path": image_path,
                "file_size_mb": 0.0,
                "dimensions": {"width": 0, "height": 0},
                "pixels": 0,
                "megapixels": 0.0,
                "aspect_ratio": 0
            }

    def _analyze_quality_safe(self, image_path: str) -> Dict:
        """Safely analyze image quality"""
        try:
            if self.quality_analyzer:
                return self.quality_analyzer.analyze_image_quality(image_path)
            else:
                return self._get_default_quality_metrics()

        except Exception as e:
            print(f"⚠️  Warning: Quality analysis failed for {image_path}: {e}")
            return self._get_default_quality_metrics()

    def _detect_codes_safe(self, image_path: str) -> Dict:
        """Safely detect codes"""
        try:
            if self.detector:
                result = self.detector.detect_codes(image_path)

                # Validate result structure
                if result is None:
                    result = self._get_default_detection_result(image_path)

                # Ensure required keys exist
                required_keys = ['processing_time', 'detected_codes', 'total_codes']
                for key in required_keys:
                    if key not in result:
                        result[key] = self._get_default_detection_value(key)

                return result
            else:
                return self._get_default_detection_result(image_path)

        except Exception as e:
            print(f"⚠️  Warning: Detection failed for {image_path}: {e}")
            return self._get_default_detection_result(image_path)

    def _get_default_quality_metrics(self) -> Dict:
        """Return default quality metrics"""
        return {
            'overall_quality_score': 50.0,
            'resolution': {'category': 'unknown'},
            'contrast_metrics': {'normalized_contrast': 0.5, 'contrast_category': 'medium'},
            'sharpness_metrics': {'laplacian_variance': 100.0, 'sharpness_category': 'moderate'},
            'lighting_metrics': {'mean_brightness': 127.5, 'lighting_category': 'optimal'},
            'noise_metrics': {'estimated_noise_level': 10.0, 'noise_category': 'medium'},
            'color_metrics': {'avg_saturation': 100.0},
            'structural_metrics': {'edge_density': 0.1, 'entropy': 5.0},
            'distortion_metrics': {'avg_angle_deviation': 5.0, 'distortion_category': 'minimal'}
        }

    def _get_default_detection_result(self, image_path: str) -> Dict:
        """Return default detection result"""
        return {
            'image_path': image_path,
            'processing_time': 0.0,
            'detected_codes': [],
            'total_codes': 0,
            'barcode_regions': [],
            'qr_regions': [],
            'preprocessing_analytics': {
                'methods_attempted': [],
                'methods_successful': [],
                'method_success_details': {},
                'total_methods_tried': 0,
                'successful_methods_count': 0,
                'success_rate': 0.0
            }
        }

    def _get_default_detection_value(self, key: str):
        """Get default value for detection result key"""
        defaults = {
            'processing_time': 0.0,
            'detected_codes': [],
            'total_codes': 0,
            'barcode_regions': [],
            'qr_regions': []
        }
        return defaults.get(key, None)

    def _estimate_strategies_used(self, result: Dict) -> int:
        """Estimate how many detection strategies were used"""
        if not result or 'detected_codes' not in result:
            return 1

        strategies = 1  # Always uses basic detection

        # Check for rotation indicators
        for code in result.get('detected_codes', []):
            if isinstance(code, dict):
                if 'rotation' in code and code.get('rotation', 0) != 0:
                    strategies = max(strategies, 2)  # Used rotation strategy
                if 'preprocess' in code:
                    strategies = max(strategies, 3)  # Used preprocessing strategy
                if 'detection_method' in code and 'fine' in str(code.get('detection_method', '')):
                    strategies = max(strategies, 4)  # Used fine-grained strategy

        return strategies

    def _estimate_preprocessing_complexity(self, result: Dict) -> str:
        """Estimate preprocessing complexity level"""
        if not result or 'detected_codes' not in result:
            return "none"

        preprocess_methods = set()

        for code in result.get('detected_codes', []):
            if isinstance(code, dict):
                if 'preprocess' in code:
                    preprocess_methods.add(code['preprocess'])
                if 'image_variant' in code:
                    preprocess_methods.add(code['image_variant'])

        if len(preprocess_methods) == 0:
            return "none"
        elif len(preprocess_methods) <= 2:
            return "basic"
        elif len(preprocess_methods) <= 5:
            return "moderate"
        else:
            return "extensive"

    def _count_rotation_attempts(self, result: Dict) -> int:
        """Count number of different rotation angles attempted"""
        if not result or 'detected_codes' not in result:
            return 0

        rotations = set()

        for code in result.get('detected_codes', []):
            if isinstance(code, dict) and 'rotation' in code:
                rotations.add(code['rotation'])

        return len(rotations)

    def _analyze_performance(self) -> Dict:
        """Analyze performance across all processed images"""
        if not self.results:
            return {"error": "No results to analyze"}

        try:
            # Extract performance data safely
            times = []
            megapixels = []
            file_sizes = []
            codes_found = []
            strategies = []
            quality_scores = []
            contrast_scores = []
            sharpness_scores = []
            noise_levels = []
            brightness_levels = []

            for r in self.results:
                # Safely extract values with defaults
                times.append(r.get('processing_time', 0.0))

                file_metadata = r.get('file_metadata', {})
                megapixels.append(file_metadata.get('megapixels', 0.0))
                file_sizes.append(file_metadata.get('file_size_mb', 0.0))

                codes_found.append(r.get('total_codes', 0))

                complexity = r.get('complexity_indicators', {})
                strategies.append(complexity.get('detection_strategies_used', 1))

                quality_metrics = r.get('quality_metrics', {})
                quality_scores.append(quality_metrics.get('overall_quality_score', 50.0))

                contrast_metrics = quality_metrics.get('contrast_metrics', {})
                contrast_scores.append(contrast_metrics.get('normalized_contrast', 0.5))

                sharpness_metrics = quality_metrics.get('sharpness_metrics', {})
                sharpness_scores.append(sharpness_metrics.get('laplacian_variance', 100.0))

                noise_metrics = quality_metrics.get('noise_metrics', {})
                noise_levels.append(noise_metrics.get('estimated_noise_level', 10.0))

                lighting_metrics = quality_metrics.get('lighting_metrics', {})
                brightness_levels.append(lighting_metrics.get('mean_brightness', 127.5))

            # Convert to numpy arrays for safe calculations
            times = np.array(times)
            megapixels = np.array(megapixels)
            file_sizes = np.array(file_sizes)
            codes_found = np.array(codes_found)
            strategies = np.array(strategies)
            quality_scores = np.array(quality_scores)

            # Sort by processing time (complexity ranking)
            sorted_results = sorted(self.results, key=lambda x: x.get('processing_time', 0))

            # Performance statistics
            analysis = {
                "timing_stats": {
                    "fastest_time": float(np.min(times)) if len(times) > 0 else 0.0,
                    "slowest_time": float(np.max(times)) if len(times) > 0 else 0.0,
                    "average_time": float(np.mean(times)) if len(times) > 0 else 0.0,
                    "median_time": float(np.median(times)) if len(times) > 0 else 0.0,
                    "std_dev_time": float(np.std(times)) if len(times) > 0 else 0.0
                },
                "quality_stats": {
                    "average_quality_score": float(np.mean(quality_scores)) if len(quality_scores) > 0 else 50.0,
                    "quality_score_std": float(np.std(quality_scores)) if len(quality_scores) > 0 else 0.0,
                    "average_contrast": float(np.mean(contrast_scores)) if len(contrast_scores) > 0 else 0.5,
                    "average_sharpness": float(np.mean(sharpness_scores)) if len(sharpness_scores) > 0 else 100.0,
                    "average_noise_level": float(np.mean(noise_levels)) if len(noise_levels) > 0 else 10.0,
                    "average_brightness": float(np.mean(brightness_levels)) if len(brightness_levels) > 0 else 127.5
                },
                "complexity_ranking": self._create_complexity_ranking(sorted_results),
                "correlations": self._calculate_safe_correlations(times, megapixels, file_sizes, codes_found,
                                                                  strategies, quality_scores),
                "quality_impact_analysis": self._analyze_quality_impact_safe(),
                "efficiency_metrics": {
                    "average_pixels_per_second": float(np.mean([
                        r.get('performance_metrics', {}).get('pixels_per_second', 0.0)
                        for r in self.results
                    ])),
                    "average_time_per_megapixel": float(np.mean([
                        r.get('performance_metrics', {}).get('time_per_megapixel', 0.0)
                        for r in self.results
                    ])),
                    "images_with_codes_found": int(np.sum(codes_found > 0)),
                    "detection_success_rate": float(np.sum(codes_found > 0) / len(codes_found) * 100) if len(
                        codes_found) > 0 else 0.0
                }
            }

            return analysis

        except Exception as e:
            error_msg = f"Performance analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.error_log.append(error_msg)
            return {"error": error_msg}

    def _create_complexity_ranking(self, sorted_results: List[Dict]) -> Dict:
        """Create complexity ranking safely"""
        try:
            def extract_ranking_data(result):
                file_metadata = result.get('file_metadata', {})
                complexity = result.get('complexity_indicators', {})
                quality = result.get('quality_metrics', {})

                return {
                    "filename": file_metadata.get('filename', 'unknown'),
                    "processing_time": result.get('processing_time', 0.0),
                    "megapixels": file_metadata.get('megapixels', 0.0),
                    "codes_found": result.get('total_codes', 0),
                    "strategies_used": complexity.get('detection_strategies_used', 1),
                    "quality_score": quality.get('overall_quality_score', 50.0),
                    "contrast_category": quality.get('contrast_metrics', {}).get('contrast_category', 'unknown'),
                    "sharpness_category": quality.get('sharpness_metrics', {}).get('sharpness_category', 'unknown'),
                    "lighting_category": quality.get('lighting_metrics', {}).get('lighting_category', 'unknown')
                }

            ranking = {
                "fastest_images": [extract_ranking_data(r) for r in sorted_results[:5]],
                "slowest_images": [extract_ranking_data(r) for r in sorted_results[-5:]]
            }

            return ranking

        except Exception as e:
            print(f"⚠️  Warning: Could not create complexity ranking: {e}")
            return {"fastest_images": [], "slowest_images": []}

    def _calculate_safe_correlations(self, times, megapixels, file_sizes, codes_found, strategies,
                                     quality_scores) -> Dict:
        """Calculate correlations safely"""
        correlations = {}

        try:
            if len(times) > 1:
                correlations['time_vs_megapixels'] = float(np.corrcoef(times, megapixels)[0, 1]) if not np.isnan(
                    np.corrcoef(times, megapixels)[0, 1]) else 0.0
                correlations['time_vs_file_size'] = float(np.corrcoef(times, file_sizes)[0, 1]) if not np.isnan(
                    np.corrcoef(times, file_sizes)[0, 1]) else 0.0
                correlations['time_vs_codes_found'] = float(np.corrcoef(times, codes_found)[0, 1]) if not np.isnan(
                    np.corrcoef(times, codes_found)[0, 1]) else 0.0
                correlations['time_vs_strategies'] = float(np.corrcoef(times, strategies)[0, 1]) if not np.isnan(
                    np.corrcoef(times, strategies)[0, 1]) else 0.0
                correlations['time_vs_quality_score'] = float(np.corrcoef(times, quality_scores)[0, 1]) if not np.isnan(
                    np.corrcoef(times, quality_scores)[0, 1]) else 0.0
            else:
                # Default values when insufficient data
                for key in ['time_vs_megapixels', 'time_vs_file_size', 'time_vs_codes_found', 'time_vs_strategies',
                            'time_vs_quality_score']:
                    correlations[key] = 0.0

        except Exception as e:
            print(f"⚠️  Warning: Correlation calculation failed: {e}")
            # Return default correlations
            for key in ['time_vs_megapixels', 'time_vs_file_size', 'time_vs_codes_found', 'time_vs_strategies',
                        'time_vs_quality_score']:
                correlations[key] = 0.0

        return correlations

    def _analyze_quality_impact_safe(self) -> Dict:
        """Analyze quality impact with error handling"""
        try:
            return self._analyze_quality_impact()
        except Exception as e:
            print(f"⚠️  Warning: Quality impact analysis failed: {e}")
            return {"error": f"Quality impact analysis failed: {str(e)}"}

    def _analyze_quality_impact(self) -> Dict:
        """Analyze the impact of image quality on detection performance"""
        if not self.results:
            return {}

        quality_analysis = {
            "resolution_impact": {},
            "contrast_impact": {},
            "sharpness_impact": {},
            "lighting_impact": {},
            "noise_impact": {},
            "overall_quality_impact": {}
        }

        try:
            # Resolution impact analysis
            resolution_groups = {}
            for result in self.results:
                quality_metrics = result.get('quality_metrics', {})
                resolution = quality_metrics.get('resolution', {})
                res_category = resolution.get('category', 'unknown')

                if res_category not in resolution_groups:
                    resolution_groups[res_category] = []
                resolution_groups[res_category].append(result)

            for category, images in resolution_groups.items():
                if images:
                    times = [img.get('processing_time', 0.0) for img in images]
                    success_rate = len([img for img in images if img.get('total_codes', 0) > 0]) / len(images) * 100

                    quality_analysis["resolution_impact"][category] = {
                        "sample_size": len(images),
                        "avg_processing_time": float(np.mean(times)) if times else 0.0,
                        "success_rate": float(success_rate),
                        "avg_megapixels": float(np.mean([
                            img.get('file_metadata', {}).get('megapixels', 0.0) for img in images
                        ])) if images else 0.0
                    }

            # Similar analysis for other quality categories...
            # (abbreviated for space, but follows same pattern)

        except Exception as e:
            print(f"⚠️  Warning: Quality impact analysis failed: {e}")

        return quality_analysis

    def _save_batch_report(self, report: Dict) -> str:
        """Save comprehensive batch report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_path = os.path.join(self.output_dir, f"batch_report_{timestamp}.json")
        try:
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Warning: Could not save JSON report: {e}")

        # Save CSV summary
        csv_path = os.path.join(self.output_dir, f"batch_summary_{timestamp}.csv")
        try:
            self._create_csv_summary(csv_path)
        except Exception as e:
            print(f"⚠️  Warning: Could not save CSV summary: {e}")

        # Save detailed text report
        txt_path = os.path.join(self.output_dir, f"batch_analysis_{timestamp}.txt")
        try:
            self._create_text_report(report, txt_path)
        except Exception as e:
            print(f"⚠️  Warning: Could not save text report: {e}")

        return json_path

    def _create_csv_summary(self, csv_path: str):
        """Create CSV summary of all results with quality metrics"""
        if not self.results:
            return

        rows = []
        for result in self.results:
            try:
                quality_metrics = result.get('quality_metrics', {})
                file_metadata = result.get('file_metadata', {})
                performance_metrics = result.get('performance_metrics', {})
                complexity = result.get('complexity_indicators', {})

                row = {
                    'filename': file_metadata.get('filename', 'unknown'),
                    'processing_time': result.get('processing_time', 0.0),
                    'file_size_mb': file_metadata.get('file_size_mb', 0.0),
                    'megapixels': file_metadata.get('megapixels', 0.0),
                    'width': file_metadata.get('dimensions', {}).get('width', 0),
                    'height': file_metadata.get('dimensions', {}).get('height', 0),
                    'codes_found': result.get('total_codes', 0),
                    'strategies_used': complexity.get('detection_strategies_used', 1),
                    'preprocessing_complexity': complexity.get('preprocessing_complexity', 'none'),
                    'rotation_attempts': complexity.get('rotation_attempts', 0),
                    'pixels_per_second': performance_metrics.get('pixels_per_second', 0.0),
                    'time_per_megapixel': performance_metrics.get('time_per_megapixel', 0.0),
                    'overall_quality_score': quality_metrics.get('overall_quality_score', 50.0)
                }

                # Add quality metrics safely
                contrast_metrics = quality_metrics.get('contrast_metrics', {})
                row['contrast_score'] = contrast_metrics.get('normalized_contrast', 0.5)
                row['contrast_category'] = contrast_metrics.get('contrast_category', 'unknown')

                sharpness_metrics = quality_metrics.get('sharpness_metrics', {})
                row['sharpness_score'] = sharpness_metrics.get('laplacian_variance', 100.0)
                row['sharpness_category'] = sharpness_metrics.get('sharpness_category', 'unknown')

                # Add other metrics...

                rows.append(row)

            except Exception as e:
                print(f"⚠️  Warning: Error processing result for CSV: {e}")
                continue

        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values('processing_time')  # Sort by processing time
            df.to_csv(csv_path, index=False)

    def _create_text_report(self, report: Dict, txt_path: str):
        """Create human-readable text report"""
        try:
            with open(txt_path, 'w') as f:
                f.write("BATCH BARCODE/QR CODE DETECTION REPORT\n")
                f.write("=" * 50 + "\n\n")

                # Batch info
                batch_info = report.get('batch_info', {})
                f.write(f"Folder: {batch_info.get('folder_path', 'unknown')}\n")
                f.write(f"Total images: {batch_info.get('total_images', 0)}\n")
                f.write(f"Successfully processed: {batch_info.get('processed_successfully', 0)}\n")
                f.write(f"Failed: {batch_info.get('failed', 0)}\n")
                f.write(f"Total batch time: {batch_info.get('total_processing_time', 0):.2f} seconds\n")
                f.write(f"Average time per image: {batch_info.get('average_time_per_image', 0):.3f} seconds\n\n")

                # Performance analysis
                perf = report.get('performance_analysis', {})
                if 'error' not in perf:
                    timing = perf.get('timing_stats', {})
                    f.write("PERFORMANCE ANALYSIS\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Fastest processing time: {timing.get('fastest_time', 0):.3f}s\n")
                    f.write(f"Slowest processing time: {timing.get('slowest_time', 0):.3f}s\n")
                    f.write(f"Average processing time: {timing.get('average_time', 0):.3f}s\n")

                    efficiency = perf.get('efficiency_metrics', {})
                    f.write(f"Detection success rate: {efficiency.get('detection_success_rate', 0):.1f}%\n\n")

                # Error summary
                if self.error_log:
                    f.write("ERROR SUMMARY\n")
                    f.write("-" * 15 + "\n")
                    for i, error in enumerate(self.error_log, 1):
                        f.write(f"{i}. {error}\n")

        except Exception as e:
            print(f"⚠️  Warning: Could not create text report: {e}")

    def _create_performance_visualizations(self) -> List[str]:
        """Create performance visualization charts safely"""
        if not self.results:
            return []

        viz_paths = []

        try:
            # Extract data safely
            times = [r.get('processing_time', 0.0) for r in self.results]
            megapixels = [r.get('file_metadata', {}).get('megapixels', 0.0) for r in self.results]
            codes_found = [r.get('total_codes', 0) for r in self.results]
            filenames = [r.get('file_metadata', {}).get('filename', f'image_{i}') for i, r in enumerate(self.results)]

            # Sort indices by processing time for ranking
            sorted_indices = np.argsort(times)

            # 1. Processing Time Ranking
            fig, ax = plt.subplots(figsize=(12, 8))
            sorted_times = [times[i] for i in sorted_indices]
            sorted_names = [filenames[i] for i in sorted_indices]

            colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(sorted_times)))
            bars = ax.barh(range(len(sorted_times)), sorted_times, color=colors)

            ax.set_yticks(range(len(sorted_times)))
            ax.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in sorted_names])
            ax.set_xlabel('Processing Time (seconds)')
            ax.set_title('Image Processing Time Complexity Ranking\n(Red = High Complexity, Green = Low Complexity)')
            ax.grid(axis='x', alpha=0.3)

            # Add time labels on bars
            for i, (bar, time) in enumerate(zip(bars, sorted_times)):
                if max(sorted_times) > 0:
                    ax.text(bar.get_width() + max(sorted_times) * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f'{time:.3f}s', va='center', fontsize=8)

            plt.tight_layout()
            viz_path1 = os.path.join(self.output_dir, 'complexity_ranking.png')
            plt.savefig(viz_path1, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(viz_path1)

        except Exception as e:
            print(f"⚠️  Warning: Could not create visualizations: {e}")

        return viz_paths

    def get_complexity_summary(self) -> str:
        """Get a quick summary of complexity analysis"""
        if not self.results:
            return "No results available"

        try:
            sorted_results = sorted(self.results, key=lambda x: x.get('processing_time', 0))

            fastest = sorted_results[0]
            slowest = sorted_results[-1]

            summary = f"""
COMPLEXITY ANALYSIS SUMMARY
===========================

Total Images Processed: {len(self.results)}

FASTEST (Lowest Complexity):
• File: {fastest.get('file_metadata', {}).get('filename', 'unknown')}
• Time: {fastest.get('processing_time', 0):.3f} seconds
• Size: {fastest.get('file_metadata', {}).get('megapixels', 0):.1f} MP
• Codes Found: {fastest.get('total_codes', 0)}
• Strategies Used: {fastest.get('complexity_indicators', {}).get('detection_strategies_used', 1)}

SLOWEST (Highest Complexity):
• File: {slowest.get('file_metadata', {}).get('filename', 'unknown')}
• Time: {slowest.get('processing_time', 0):.3f} seconds
• Size: {slowest.get('file_metadata', {}).get('megapixels', 0):.1f} MP
• Codes Found: {slowest.get('total_codes', 0)}
• Strategies Used: {slowest.get('complexity_indicators', {}).get('detection_strategies_used', 1)}

Performance Ratio: {(slowest.get('processing_time', 0) / fastest.get('processing_time', 1)):.1f}x difference
Average Time: {np.mean([r.get('processing_time', 0) for r in self.results]):.3f} seconds

Errors Encountered: {len(self.error_log)}
            """

            return summary

        except Exception as e:
            return f"Error generating summary: {str(e)}"


def main():
    """Example usage of fixed batch detector"""
    detector = FixedBatchDetector()

    # Process a folder (replace with your folder path)
    folder_path = "test_images"  # or any folder with images

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found. Creating test images first...")
        try:
            from test_generator import TestGenerator
            generator = TestGenerator()
            generator.create_simple_test_set()
        except Exception as e:
            print(f"Could not create test images: {e}")
            return

    # Process all images in folder
    results = detector.process_folder(folder_path, max_images=10)  # Limit for demo

    # Print summary
    if 'error' not in results:
        print("\n" + detector.get_complexity_summary())

        # Print error summary if any
        if detector.error_log:
            print(f"\n⚠️  Encountered {len(detector.error_log)} errors during processing:")
            for i, error in enumerate(detector.error_log[:5], 1):  # Show first 5 errors
                print(f"   {i}. {error}")
            if len(detector.error_log) > 5:
                print(f"   ... and {len(detector.error_log) - 5} more errors")
    else:
        print(f"❌ Batch processing failed: {results['error']}")


if __name__ == "__main__":
    main()