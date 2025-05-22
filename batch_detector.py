# batch_detector.py
"""
Batch processing system for barcode and QR code detection with time complexity ranking
"""
import cv2
import os
import json
from datetime import datetime
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple
from detector import Detector
import matplotlib.pyplot as plt
import seaborn as sns


class BatchDetector:
    """Enhanced detector for batch processing with performance analysis"""

    def __init__(self, output_dir="batch_results"):
        self.detector = Detector()
        self.output_dir = output_dir
        self.results = []
        self.performance_stats = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def process_folder(self, folder_path: str, max_images: int = None) -> Dict:
        """
        Process all images in a folder and analyze performance

        Args:
            folder_path: Path to folder containing images
            max_images: Maximum number of images to process (None for all)

        Returns:
            Dictionary containing batch results and performance analysis
        """
        print(f"Starting batch processing of folder: {folder_path}")

        # Get all image files
        image_files = self._get_image_files(folder_path)

        if not image_files:
            print("No supported image files found!")
            return {"error": "No images found"}

        if max_images:
            image_files = image_files[:max_images]

        print(f"Found {len(image_files)} images to process")

        # Process each image
        batch_start = datetime.now()
        failed_images = []

        for i, image_path in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")

            try:
                result = self._process_single_image(image_path)
                self.results.append(result)

                # Print quick summary
                print(f"  ✓ Time: {result['processing_time']:.3f}s, Codes: {result['total_codes']}")

            except Exception as e:
                print(f"  ✗ Error processing {image_path}: {e}")
                failed_images.append({"path": image_path, "error": str(e)})

        batch_end = datetime.now()
        total_batch_time = (batch_end - batch_start).total_seconds()

        # Analyze results
        analysis = self._analyze_performance()

        # Create comprehensive report
        report = {
            "batch_info": {
                "folder_path": folder_path,
                "total_images": len(image_files),
                "processed_successfully": len(self.results),
                "failed_images": len(failed_images),
                "total_batch_time": total_batch_time,
                "average_time_per_image": total_batch_time / len(image_files) if image_files else 0,
                "timestamp": batch_start.isoformat()
            },
            "failed_images": failed_images,
            "performance_analysis": analysis,
            "detailed_results": self.results
        }

        # Save report
        report_path = self._save_batch_report(report)

        # Create visualizations
        viz_paths = self._create_performance_visualizations()
        report["visualization_paths"] = viz_paths

        print(f"\nBatch processing complete!")
        print(f"Report saved to: {report_path}")

        return report

    def _get_image_files(self, folder_path: str) -> List[str]:
        """Get all supported image files from folder"""
        image_files = []

        for file_path in Path(folder_path).rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                image_files.append(str(file_path))

        return sorted(image_files)

    def _process_single_image(self, image_path: str) -> Dict:
        """Process a single image and collect detailed metrics"""
        # Get file info
        file_stat = os.stat(image_path)
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Collect image metadata
        height, width = image.shape[:2]
        file_size_mb = file_stat.st_size / (1024 * 1024)

        # Process with detector
        start_time = datetime.now()
        result = self.detector.detect_codes(image_path)
        end_time = datetime.now()

        # Calculate detailed timing metrics
        processing_time = (end_time - start_time).total_seconds()
        pixels = height * width
        megapixels = pixels / 1_000_000

        # Enhanced result with metadata
        enhanced_result = {
            **result,
            "file_metadata": {
                "filename": os.path.basename(image_path),
                "full_path": image_path,
                "file_size_mb": file_size_mb,
                "dimensions": {"width": width, "height": height},
                "pixels": pixels,
                "megapixels": megapixels,
                "aspect_ratio": width / height
            },
            "performance_metrics": {
                "processing_time": processing_time,
                "time_per_megapixel": processing_time / megapixels if megapixels > 0 else 0,
                "time_per_mb": processing_time / file_size_mb if file_size_mb > 0 else 0,
                "pixels_per_second": pixels / processing_time if processing_time > 0 else 0
            },
            "complexity_indicators": {
                "detection_strategies_used": self._estimate_strategies_used(result),
                "preprocessing_complexity": self._estimate_preprocessing_complexity(result),
                "rotation_attempts": self._count_rotation_attempts(result)
            }
        }

        return enhanced_result

    def _estimate_strategies_used(self, result: Dict) -> int:
        """Estimate how many detection strategies were used"""
        strategies = 1  # Always uses basic detection

        # Check for rotation indicators
        for code in result.get('detected_codes', []):
            if 'rotation' in code and code['rotation'] != 0:
                strategies = max(strategies, 2)  # Used rotation strategy
            if 'preprocess' in code:
                strategies = max(strategies, 3)  # Used preprocessing strategy
            if 'detection_method' in code and 'fine' in code['detection_method']:
                strategies = max(strategies, 4)  # Used fine-grained strategy

        return strategies

    def _estimate_preprocessing_complexity(self, result: Dict) -> str:
        """Estimate preprocessing complexity level"""
        preprocess_methods = set()

        for code in result.get('detected_codes', []):
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
        rotations = set()

        for code in result.get('detected_codes', []):
            if 'rotation' in code:
                rotations.add(code['rotation'])

        return len(rotations)

    def _analyze_performance(self) -> Dict:
        """Analyze performance across all processed images"""
        if not self.results:
            return {"error": "No results to analyze"}

        # Extract performance data
        times = [r['processing_time'] for r in self.results]
        megapixels = [r['file_metadata']['megapixels'] for r in self.results]
        file_sizes = [r['file_metadata']['file_size_mb'] for r in self.results]
        codes_found = [r['total_codes'] for r in self.results]
        strategies = [r['complexity_indicators']['detection_strategies_used'] for r in self.results]

        # Sort by processing time (complexity ranking)
        sorted_results = sorted(self.results, key=lambda x: x['processing_time'])

        # Performance statistics
        analysis = {
            "timing_stats": {
                "fastest_time": min(times),
                "slowest_time": max(times),
                "average_time": np.mean(times),
                "median_time": np.median(times),
                "std_dev_time": np.std(times)
            },
            "complexity_ranking": {
                "fastest_images": [
                    {
                        "filename": r['file_metadata']['filename'],
                        "processing_time": r['processing_time'],
                        "megapixels": r['file_metadata']['megapixels'],
                        "codes_found": r['total_codes'],
                        "strategies_used": r['complexity_indicators']['detection_strategies_used']
                    }
                    for r in sorted_results[:5]  # Top 5 fastest
                ],
                "slowest_images": [
                    {
                        "filename": r['file_metadata']['filename'],
                        "processing_time": r['processing_time'],
                        "megapixels": r['file_metadata']['megapixels'],
                        "codes_found": r['total_codes'],
                        "strategies_used": r['complexity_indicators']['detection_strategies_used']
                    }
                    for r in sorted_results[-5:]  # Top 5 slowest
                ]
            },
            "correlations": {
                "time_vs_megapixels": np.corrcoef(times, megapixels)[0, 1] if len(times) > 1 else 0,
                "time_vs_file_size": np.corrcoef(times, file_sizes)[0, 1] if len(times) > 1 else 0,
                "time_vs_codes_found": np.corrcoef(times, codes_found)[0, 1] if len(times) > 1 else 0,
                "time_vs_strategies": np.corrcoef(times, strategies)[0, 1] if len(times) > 1 else 0
            },
            "efficiency_metrics": {
                "average_pixels_per_second": np.mean(
                    [r['performance_metrics']['pixels_per_second'] for r in self.results]),
                "average_time_per_megapixel": np.mean(
                    [r['performance_metrics']['time_per_megapixel'] for r in self.results]),
                "images_with_codes_found": len([r for r in self.results if r['total_codes'] > 0]),
                "detection_success_rate": len([r for r in self.results if r['total_codes'] > 0]) / len(
                    self.results) * 100
            }
        }

        return analysis

    def _save_batch_report(self, report: Dict) -> str:
        """Save comprehensive batch report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_path = os.path.join(self.output_dir, f"batch_report_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Save CSV summary
        csv_path = os.path.join(self.output_dir, f"batch_summary_{timestamp}.csv")
        self._create_csv_summary(csv_path)

        # Save detailed text report
        txt_path = os.path.join(self.output_dir, f"batch_analysis_{timestamp}.txt")
        self._create_text_report(report, txt_path)

        return json_path

    def _create_csv_summary(self, csv_path: str):
        """Create CSV summary of all results"""
        rows = []

        for result in self.results:
            row = {
                'filename': result['file_metadata']['filename'],
                'processing_time': result['processing_time'],
                'file_size_mb': result['file_metadata']['file_size_mb'],
                'megapixels': result['file_metadata']['megapixels'],
                'width': result['file_metadata']['dimensions']['width'],
                'height': result['file_metadata']['dimensions']['height'],
                'codes_found': result['total_codes'],
                'strategies_used': result['complexity_indicators']['detection_strategies_used'],
                'preprocessing_complexity': result['complexity_indicators']['preprocessing_complexity'],
                'rotation_attempts': result['complexity_indicators']['rotation_attempts'],
                'pixels_per_second': result['performance_metrics']['pixels_per_second'],
                'time_per_megapixel': result['performance_metrics']['time_per_megapixel']
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('processing_time')  # Sort by processing time
        df.to_csv(csv_path, index=False)

    def _create_text_report(self, report: Dict, txt_path: str):
        """Create human-readable text report"""
        with open(txt_path, 'w') as f:
            f.write("BATCH BARCODE/QR CODE DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Batch info
            batch_info = report['batch_info']
            f.write(f"Folder: {batch_info['folder_path']}\n")
            f.write(f"Total images: {batch_info['total_images']}\n")
            f.write(f"Successfully processed: {batch_info['processed_successfully']}\n")
            f.write(f"Failed: {batch_info['failed_images']}\n")
            f.write(f"Total batch time: {batch_info['total_batch_time']:.2f} seconds\n")
            f.write(f"Average time per image: {batch_info['average_time_per_image']:.3f} seconds\n\n")

            # Performance analysis
            perf = report['performance_analysis']
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Fastest processing time: {perf['timing_stats']['fastest_time']:.3f}s\n")
            f.write(f"Slowest processing time: {perf['timing_stats']['slowest_time']:.3f}s\n")
            f.write(f"Average processing time: {perf['timing_stats']['average_time']:.3f}s\n")
            f.write(f"Detection success rate: {perf['efficiency_metrics']['detection_success_rate']:.1f}%\n\n")

            # Fastest images
            f.write("TOP 5 FASTEST IMAGES (Low Complexity)\n")
            f.write("-" * 40 + "\n")
            for i, img in enumerate(perf['complexity_ranking']['fastest_images'], 1):
                f.write(f"{i}. {img['filename']}\n")
                f.write(f"   Time: {img['processing_time']:.3f}s, ")
                f.write(f"Megapixels: {img['megapixels']:.1f}, ")
                f.write(f"Codes: {img['codes_found']}, ")
                f.write(f"Strategies: {img['strategies_used']}\n")
            f.write("\n")

            # Slowest images
            f.write("TOP 5 SLOWEST IMAGES (High Complexity)\n")
            f.write("-" * 40 + "\n")
            for i, img in enumerate(perf['complexity_ranking']['slowest_images'], 1):
                f.write(f"{i}. {img['filename']}\n")
                f.write(f"   Time: {img['processing_time']:.3f}s, ")
                f.write(f"Megapixels: {img['megapixels']:.1f}, ")
                f.write(f"Codes: {img['codes_found']}, ")
                f.write(f"Strategies: {img['strategies_used']}\n")
            f.write("\n")

            # Correlations
            f.write("COMPLEXITY CORRELATIONS\n")
            f.write("-" * 25 + "\n")
            corr = perf['correlations']
            f.write(f"Time vs Megapixels: {corr['time_vs_megapixels']:.3f}\n")
            f.write(f"Time vs File Size: {corr['time_vs_file_size']:.3f}\n")
            f.write(f"Time vs Codes Found: {corr['time_vs_codes_found']:.3f}\n")
            f.write(f"Time vs Strategies Used: {corr['time_vs_strategies']:.3f}\n")

    def _create_performance_visualizations(self) -> List[str]:
        """Create performance visualization charts"""
        if not self.results:
            return []

        plt.style.use('default')
        viz_paths = []

        # Extract data for plotting
        times = [r['processing_time'] for r in self.results]
        megapixels = [r['file_metadata']['megapixels'] for r in self.results]
        codes_found = [r['total_codes'] for r in self.results]
        strategies = [r['complexity_indicators']['detection_strategies_used'] for r in self.results]
        filenames = [r['file_metadata']['filename'] for r in self.results]

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
            ax.text(bar.get_width() + max(sorted_times) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{time:.3f}s', va='center', fontsize=8)

        plt.tight_layout()
        viz_path1 = os.path.join(self.output_dir, 'complexity_ranking.png')
        plt.savefig(viz_path1, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths.append(viz_path1)

        # 2. Correlation Matrix
        fig, ax = plt.subplots(figsize=(10, 8))

        data_matrix = np.array([times, megapixels, codes_found, strategies]).T
        corr_matrix = np.corrcoef(data_matrix.T)

        labels = ['Processing Time', 'Megapixels', 'Codes Found', 'Strategies Used']
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')

        ax.set_title('Performance Correlation Matrix')
        fig.colorbar(im, ax=ax, label='Correlation Coefficient')
        plt.tight_layout()

        viz_path2 = os.path.join(self.output_dir, 'correlation_matrix.png')
        plt.savefig(viz_path2, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths.append(viz_path2)

        # 3. Time vs Megapixels Scatter
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(megapixels, times, c=strategies, cmap='viridis',
                             s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Image Size (Megapixels)')
        ax.set_ylabel('Processing Time (seconds)')
        ax.set_title('Processing Time vs Image Size\n(Color = Number of Detection Strategies Used)')
        ax.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(megapixels, times, 1)
        p = np.poly1d(z)
        ax.plot(sorted(megapixels), p(sorted(megapixels)), "r--", alpha=0.8, linewidth=2)

        # Color bar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Detection Strategies Used')

        plt.tight_layout()
        viz_path3 = os.path.join(self.output_dir, 'time_vs_size.png')
        plt.savefig(viz_path3, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths.append(viz_path3)

        return viz_paths

    def get_complexity_summary(self) -> str:
        """Get a quick summary of complexity analysis"""
        if not self.results:
            return "No results available"

        sorted_results = sorted(self.results, key=lambda x: x['processing_time'])

        fastest = sorted_results[0]
        slowest = sorted_results[-1]

        summary = f"""
COMPLEXITY ANALYSIS SUMMARY
===========================

Total Images Processed: {len(self.results)}

FASTEST (Lowest Complexity):
• File: {fastest['file_metadata']['filename']}
• Time: {fastest['processing_time']:.3f} seconds
• Size: {fastest['file_metadata']['megapixels']:.1f} MP
• Codes Found: {fastest['total_codes']}
• Strategies Used: {fastest['complexity_indicators']['detection_strategies_used']}

SLOWEST (Highest Complexity):
• File: {slowest['file_metadata']['filename']}
• Time: {slowest['processing_time']:.3f} seconds
• Size: {slowest['file_metadata']['megapixels']:.1f} MP
• Codes Found: {slowest['total_codes']}
• Strategies Used: {slowest['complexity_indicators']['detection_strategies_used']}

Performance Ratio: {slowest['processing_time'] / fastest['processing_time']:.1f}x difference
Average Time: {np.mean([r['processing_time'] for r in self.results]):.3f} seconds
        """

        return summary


def main():
    """Example usage of batch detector"""
    detector = BatchDetector()

    # Process a folder (replace with your folder path)
    folder_path = "test_images"  # or any folder with images

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found. Creating test images first...")
        from test_generator import TestGenerator
        generator = TestGenerator()
        generator.create_simple_test_set()

    # Process all images in folder
    results = detector.process_folder(folder_path, max_images=10)  # Limit for demo

    # Print summary
    print("\n" + detector.get_complexity_summary())


if __name__ == "__main__":
    main()