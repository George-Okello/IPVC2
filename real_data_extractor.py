# real_data_extractor.py
"""
Extract real experimental data from actual image processing results
"""
import pandas as pd
import numpy as np
from batch_detector import BatchDetector
from test_generator import TestGenerator
from preprocessing_analyzer import PreprocessingAnalyzer
import os
import json
from datetime import datetime


class RealDataExtractor:
    """Extract actual experimental values from real image processing"""

    def __init__(self, output_dir="experiment_results"):
        self.batch_detector = BatchDetector(output_dir=output_dir)
        self.preprocessing_analyzer = PreprocessingAnalyzer()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_comprehensive_experiment(self, images_folder="test_images"):
        """Run the complete experiment and extract real data"""
        print("=== RUNNING COMPREHENSIVE BARCODE/QR DETECTION EXPERIMENT ===")

        # Step 1: Ensure we have test images
        if not os.path.exists(images_folder) or len(os.listdir(images_folder)) < 10:
            print(f"Creating test images in {images_folder}...")
            generator = TestGenerator(output_dir=images_folder)
            generator.create_simple_test_set()
            print(f"✅ Test images created in {images_folder}")

        # Step 2: Process all images
        print(f"\n🔄 Processing all images in {images_folder}...")
        results = self.batch_detector.process_folder(images_folder)

        if 'error' in results:
            print(f"❌ Error processing images: {results['error']}")
            return None

        print(f"✅ Successfully processed {results['batch_info']['processed_successfully']} images")

        # Step 3: Extract and analyze real data
        print("\n📊 Extracting experimental data...")
        experimental_data = self.extract_real_experimental_data(results)

        # Step 3.5: Analyze preprocessing effectiveness
        print("\n🔧 Analyzing preprocessing method effectiveness...")
        preprocessing_analysis = self.preprocessing_analyzer.analyze_preprocessing_success_rates(results)
        experimental_data['preprocessing_analysis'] = preprocessing_analysis

        # Step 4: Generate analysis report
        print("\n📋 Generating analysis report...")
        report = self.generate_analysis_report(experimental_data)

        # Step 5: Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw data
        raw_data_file = os.path.join(self.output_dir, f"experimental_results_{timestamp}.json")
        with open(raw_data_file, 'w') as f:
            json.dump(experimental_data, f, indent=2, default=str)

        # Save analysis report
        report_file = os.path.join(self.output_dir, f"analysis_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report)

        # Save CSV for further analysis
        csv_file = os.path.join(self.output_dir, f"detailed_results_{timestamp}.csv")
        self.save_detailed_csv(experimental_data, csv_file)

        # Save preprocessing analysis
        if 'preprocessing_analysis' in experimental_data:
            preprocessing_files = self.preprocessing_analyzer.export_preprocessing_data(
                experimental_data['preprocessing_analysis'],
                os.path.join(self.output_dir, "preprocessing_analysis")
            )

        print(f"\n📁 Results saved:")
        print(f"   📄 Raw data: {raw_data_file}")
        print(f"   📋 Report: {report_file}")
        print(f"   📊 CSV: {csv_file}")

        if 'preprocessing_analysis' in experimental_data:
            print(f"   🔧 Preprocessing analysis: {self.output_dir}/preprocessing_analysis/")

            # Display preprocessing summary
            self.print_preprocessing_summary(experimental_data['preprocessing_analysis'])

        return experimental_data

    def extract_real_experimental_data(self, batch_results):
        """Extract real experimental data from batch processing results"""
        if not batch_results or 'detailed_results' not in batch_results:
            return {}

        detailed_results = batch_results['detailed_results']

        # Convert to DataFrame for easier analysis
        data_rows = []
        for result in detailed_results:
            quality_metrics = result['quality_metrics']

            row = {
                'filename': result['file_metadata']['filename'],
                'processing_time': result['processing_time'],
                'codes_found': result['total_codes'],
                'detection_success': 1 if result['total_codes'] > 0 else 0,
                'megapixels': result['file_metadata']['megapixels'],
                'file_size_mb': result['file_metadata']['file_size_mb'],
                'strategies_used': result['complexity_indicators']['detection_strategies_used'],

                # Image quality metrics
                'overall_quality_score': quality_metrics['overall_quality_score'],
                'resolution_category': quality_metrics['resolution']['category'],
                'contrast_score': quality_metrics['contrast_metrics']['normalized_contrast'],
                'contrast_category': quality_metrics['contrast_metrics']['contrast_category'],
                'sharpness_score': quality_metrics['sharpness_metrics']['laplacian_variance'],
                'sharpness_category': quality_metrics['sharpness_metrics']['sharpness_category'],
                'brightness': quality_metrics['lighting_metrics']['mean_brightness'],
                'lighting_category': quality_metrics['lighting_metrics']['lighting_category'],
                'noise_level': quality_metrics['noise_metrics']['estimated_noise_level'],
                'noise_category': quality_metrics['noise_metrics']['noise_category'],

                # Code type analysis
                'has_qr_code': any(code.get('type', '').upper() == 'QRCODE' for code in result['detected_codes']),
                'has_barcode': any(
                    code.get('type', '').upper() in ['CODE128', 'EAN13', 'EAN8'] for code in result['detected_codes'])
            }
            data_rows.append(row)

        df = pd.DataFrame(data_rows)

        # Extract experimental analyses
        experimental_data = {
            'dataset_summary': self._analyze_dataset_summary(df),
            'performance_by_image_category': self._analyze_performance_by_category(df),
            'resolution_analysis': self._analyze_resolution_impact(df),
            'contrast_lighting_analysis': self._analyze_contrast_lighting(df),
            'quality_correlations': self._analyze_quality_correlations(df),
            'strategy_effectiveness': self._analyze_strategy_effectiveness(df),
            'code_type_performance': self._analyze_code_type_performance(df),
            'raw_dataframe': df.to_dict('records'),  # For CSV export
            'batch_info': batch_results['batch_info'],
            'performance_analysis': batch_results['performance_analysis']
        }

        return experimental_data

    def _analyze_dataset_summary(self, df):
        """Analyze overall dataset summary"""
        total_images = len(df)
        successful_detections = df['detection_success'].sum()
        qr_detections = df['has_qr_code'].sum()
        barcode_detections = df['has_barcode'].sum()

        return {
            'total_images': total_images,
            'successful_detections': successful_detections,
            'overall_detection_rate': (successful_detections / total_images * 100) if total_images > 0 else 0,
            'qr_code_detections': qr_detections,
            'barcode_detections': barcode_detections,
            'qr_detection_rate': (qr_detections / total_images * 100) if total_images > 0 else 0,
            'barcode_detection_rate': (barcode_detections / total_images * 100) if total_images > 0 else 0,
            'average_processing_time': df['processing_time'].mean(),
            'fastest_processing_time': df['processing_time'].min(),
            'slowest_processing_time': df['processing_time'].max(),
            'average_quality_score': df['overall_quality_score'].mean()
        }

    def _analyze_performance_by_category(self, df):
        """Analyze performance by different image categories"""
        categories = {}

        # Group by resolution category
        for category in df['resolution_category'].unique():
            cat_df = df[df['resolution_category'] == category]
            categories[f'resolution_{category}'] = {
                'total_images': len(cat_df),
                'success_rate': (cat_df['detection_success'].sum() / len(cat_df) * 100),
                'avg_processing_time': cat_df['processing_time'].mean()
            }

        # Group by contrast category
        for category in df['contrast_category'].unique():
            cat_df = df[df['contrast_category'] == category]
            categories[f'contrast_{category}'] = {
                'total_images': len(cat_df),
                'success_rate': (cat_df['detection_success'].sum() / len(cat_df) * 100),
                'avg_processing_time': cat_df['processing_time'].mean()
            }

        # Group by lighting category
        for category in df['lighting_category'].unique():
            cat_df = df[df['lighting_category'] == category]
            categories[f'lighting_{category}'] = {
                'total_images': len(cat_df),
                'success_rate': (cat_df['detection_success'].sum() / len(cat_df) * 100),
                'avg_processing_time': cat_df['processing_time'].mean()
            }

        return categories

    def _analyze_resolution_impact(self, df):
        """Analyze the impact of image resolution on detection performance"""

        # Define resolution ranges based on megapixels
        def categorize_resolution(megapixels):
            if megapixels < 0.09:  # < 300x300
                return "< 300x300"
            elif megapixels < 0.48:  # < 800x600
                return "300x300 - 800x600"
            elif megapixels < 2.07:  # < 1920x1080
                return "800x600 - 1920x1080"
            else:
                return "> 1920x1080"

        df['resolution_range'] = df['megapixels'].apply(categorize_resolution)

        resolution_analysis = {}
        for resolution_range in df['resolution_range'].unique():
            range_df = df[df['resolution_range'] == resolution_range]
            resolution_analysis[resolution_range] = {
                'sample_size': len(range_df),
                'success_rate': (range_df['detection_success'].sum() / len(range_df) * 100),
                'avg_processing_time': range_df['processing_time'].mean(),
                'avg_megapixels': range_df['megapixels'].mean()
            }

        return resolution_analysis

    def _analyze_contrast_lighting(self, df):
        """Analyze contrast and lighting impact"""

        # Contrast analysis by score ranges
        def categorize_contrast(score):
            if score < 0.4:
                return "Low (<0.4)"
            elif score < 0.7:
                return "Medium (0.4-0.7)"
            else:
                return "High (>0.7)"

        df['contrast_range'] = df['contrast_score'].apply(categorize_contrast)

        contrast_analysis = {}
        for contrast_range in df['contrast_range'].unique():
            range_df = df[df['contrast_range'] == contrast_range]
            contrast_analysis[contrast_range] = {
                'sample_size': len(range_df),
                'detection_rate': (range_df['detection_success'].sum() / len(range_df) * 100),
                'avg_contrast_score': range_df['contrast_score'].mean()
            }

        # Lighting analysis by category
        lighting_analysis = {}
        for lighting_cat in df['lighting_category'].unique():
            cat_df = df[df['lighting_category'] == lighting_cat]
            lighting_analysis[lighting_cat] = {
                'sample_size': len(cat_df),
                'success_rate': (cat_df['detection_success'].sum() / len(cat_df) * 100),
                'avg_brightness': cat_df['brightness'].mean()
            }

        return {
            'contrast_impact': contrast_analysis,
            'lighting_impact': lighting_analysis
        }

    def _analyze_quality_correlations(self, df):
        """Analyze correlations between quality metrics and performance"""
        correlations = {}

        # Processing time correlations
        correlations['time_vs_megapixels'] = df['processing_time'].corr(df['megapixels'])
        correlations['time_vs_quality_score'] = df['processing_time'].corr(df['overall_quality_score'])
        correlations['time_vs_contrast'] = df['processing_time'].corr(df['contrast_score'])
        correlations['time_vs_sharpness'] = df['processing_time'].corr(df['sharpness_score'])
        correlations['time_vs_noise'] = df['processing_time'].corr(df['noise_level'])
        correlations['time_vs_brightness'] = df['processing_time'].corr(df['brightness'])
        correlations['time_vs_strategies'] = df['processing_time'].corr(df['strategies_used'])

        # Success rate correlations
        correlations['success_vs_quality_score'] = df['detection_success'].corr(df['overall_quality_score'])
        correlations['success_vs_contrast'] = df['detection_success'].corr(df['contrast_score'])
        correlations['success_vs_sharpness'] = df['detection_success'].corr(df['sharpness_score'])

        return correlations

    def _analyze_strategy_effectiveness(self, df):
        """Analyze effectiveness of different detection strategies"""
        strategy_analysis = {}

        for strategy_count in sorted(df['strategies_used'].unique()):
            strategy_df = df[df['strategies_used'] == strategy_count]
            strategy_analysis[f'strategy_{strategy_count}'] = {
                'sample_size': len(strategy_df),
                'success_rate': (strategy_df['detection_success'].sum() / len(strategy_df) * 100),
                'avg_processing_time': strategy_df['processing_time'].mean(),
                'percentage_of_total': (len(strategy_df) / len(df) * 100)
            }

        return strategy_analysis

    def _analyze_code_type_performance(self, df):
        """Analyze performance by code type"""
        qr_df = df[df['has_qr_code'] == True]
        barcode_df = df[df['has_barcode'] == True]

        code_analysis = {
            'qr_codes': {
                'total_detected': len(qr_df),
                'avg_processing_time': qr_df['processing_time'].mean() if len(qr_df) > 0 else 0,
                'avg_quality_score': qr_df['overall_quality_score'].mean() if len(qr_df) > 0 else 0
            },
            'barcodes': {
                'total_detected': len(barcode_df),
                'avg_processing_time': barcode_df['processing_time'].mean() if len(barcode_df) > 0 else 0,
                'avg_quality_score': barcode_df['overall_quality_score'].mean() if len(barcode_df) > 0 else 0
            }
        }

        return code_analysis

    def generate_analysis_report(self, experimental_data):
        """Generate a comprehensive analysis report"""
        report = []
        report.append("EXPERIMENTAL RESULTS ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")

        # Dataset Summary
        summary = experimental_data['dataset_summary']
        report.append("DATASET SUMMARY:")
        report.append(f"Total Images Processed: {summary['total_images']}")
        report.append(f"Overall Detection Rate: {summary['overall_detection_rate']:.1f}%")
        report.append(f"QR Code Detection Rate: {summary['qr_detection_rate']:.1f}%")
        report.append(f"Barcode Detection Rate: {summary['barcode_detection_rate']:.1f}%")
        report.append(f"Average Processing Time: {summary['average_processing_time']:.3f} seconds")
        report.append(f"Fastest Processing: {summary['fastest_processing_time']:.3f} seconds")
        report.append(f"Slowest Processing: {summary['slowest_processing_time']:.3f} seconds")
        report.append("")

        # Resolution Analysis Table
        if 'resolution_analysis' in experimental_data:
            report.append("RESOLUTION IMPACT ANALYSIS:")
            report.append(f"{'Resolution Range':<20} {'Sample Size':<12} {'Success Rate':<15} {'Avg Time (s)'}")
            report.append("-" * 65)
            for res_range, data in experimental_data['resolution_analysis'].items():
                report.append(
                    f"{res_range:<20} {data['sample_size']:<12} {data['success_rate']:<14.1f}% {data['avg_processing_time']:.3f}")
            report.append("")

        # Contrast Analysis
        if 'contrast_lighting_analysis' in experimental_data:
            contrast_data = experimental_data['contrast_lighting_analysis']['contrast_impact']
            report.append("CONTRAST IMPACT ANALYSIS:")
            report.append(f"{'Contrast Level':<20} {'Sample Size':<12} {'Detection Rate':<15} {'Avg Contrast'}")
            report.append("-" * 65)
            for contrast_level, data in contrast_data.items():
                report.append(
                    f"{contrast_level:<20} {data['sample_size']:<12} {data['detection_rate']:<14.1f}% {data['avg_contrast_score']:.3f}")
            report.append("")

        # Quality Correlations
        if 'quality_correlations' in experimental_data:
            correlations = experimental_data['quality_correlations']
            report.append("QUALITY CORRELATIONS:")
            for correlation_name, value in correlations.items():
                if not np.isnan(value):
                    interpretation = self._interpret_correlation(value)
                    report.append(f"{correlation_name.replace('_', ' ').title()}: {value:.3f} ({interpretation})")
            report.append("")

        # Strategy Effectiveness
        if 'strategy_effectiveness' in experimental_data:
            report.append("STRATEGY EFFECTIVENESS:")
            strategy_data = experimental_data['strategy_effectiveness']
            for strategy, data in strategy_data.items():
                report.append(
                    f"{strategy}: {data['percentage_of_total']:.1f}% of images, {data['success_rate']:.1f}% success rate")
            report.append("")

        # Preprocessing Analysis Summary
        if 'preprocessing_analysis' in experimental_data:
            preprocessing = experimental_data['preprocessing_analysis']
            overall_preprocessing = preprocessing.get('overall_statistics', {})

            report.append("PREPROCESSING ANALYSIS SUMMARY:")
            report.append(
                f"Images requiring preprocessing: {overall_preprocessing.get('images_requiring_preprocessing', 0)}")
            report.append(f"Standard detection success: {overall_preprocessing.get('images_solved_by_standard', 0)}")
            report.append(f"Rotation detection success: {overall_preprocessing.get('images_solved_by_rotation', 0)}")

            # Top performing preprocessing methods
            method_effectiveness = preprocessing.get('method_effectiveness', {})
            if method_effectiveness:
                report.append("Top preprocessing methods:")
                sorted_methods = sorted(method_effectiveness.items(),
                                        key=lambda x: x[1]['success_rate'],
                                        reverse=True)[:3]
                for method, stats in sorted_methods:
                    report.append(f"  - {method}: {stats['success_rate']:.1f}% success rate")
            report.append("")

        return "\n".join(report)

    def _interpret_correlation(self, r):
        """Interpret correlation coefficient"""
        abs_r = abs(r)
        if abs_r >= 0.7:
            strength = "Strong"
        elif abs_r >= 0.5:
            strength = "Moderate"
        elif abs_r >= 0.3:
            strength = "Weak"
        else:
            strength = "Very weak"

        direction = "positive" if r > 0 else "negative"
        return f"{strength} {direction}"

    def save_detailed_csv(self, experimental_data, csv_path):
        """Save detailed results to CSV"""
        if 'raw_dataframe' in experimental_data:
            df = pd.DataFrame(experimental_data['raw_dataframe'])
            df.to_csv(csv_path, index=False)
            print(f"📊 Detailed CSV saved: {csv_path}")

    def print_key_values_for_assignment(self, experimental_data):
        """Print key values that can be used directly in the assignment report"""
        print("\n" + "=" * 60)
        print("KEY VALUES FOR ASSIGNMENT REPORT")
        print("=" * 60)

        summary = experimental_data['dataset_summary']

        print("\n📊 OVERALL PERFORMANCE:")
        print(f"   Overall Detection Rate: {summary['overall_detection_rate']:.1f}%")
        print(f"   QR Code Detection Rate: {summary['qr_detection_rate']:.1f}%")
        print(f"   Barcode Detection Rate: {summary['barcode_detection_rate']:.1f}%")
        print(f"   Average Processing Time: {summary['average_processing_time']:.3f}s")

        if 'resolution_analysis' in experimental_data:
            print("\n📏 RESOLUTION ANALYSIS:")
            for res_range, data in experimental_data['resolution_analysis'].items():
                print(
                    f"   {res_range}: {data['success_rate']:.1f}% success, {data['avg_processing_time']:.3f}s avg time")

        if 'quality_correlations' in experimental_data:
            print("\n🔗 KEY CORRELATIONS:")
            correlations = experimental_data['quality_correlations']
            for key, value in correlations.items():
                if not np.isnan(value) and abs(value) > 0.3:  # Only show significant correlations
                    print(f"   {key}: {value:.3f}")

        # Display preprocessing effectiveness
        if 'preprocessing_analysis' in experimental_data:
            print("\n🔧 PREPROCESSING EFFECTIVENESS:")
            preprocessing = experimental_data['preprocessing_analysis']
            if 'method_effectiveness' in preprocessing:
                methods = preprocessing['method_effectiveness']
                # Show top 3 most effective methods
                sorted_methods = sorted(methods.items(),
                                        key=lambda x: x[1]['success_rate'],
                                        reverse=True)[:3]
                for method, stats in sorted_methods:
                    print(
                        f"   {method}: {stats['success_rate']:.1f}% success rate, {stats['images_successful']} images improved")

    def print_preprocessing_summary(self, preprocessing_analysis):
        """Print a summary of preprocessing analysis"""
        if not preprocessing_analysis or 'error' in preprocessing_analysis:
            return

        print("\n" + "=" * 50)
        print("PREPROCESSING METHOD ANALYSIS SUMMARY")
        print("=" * 50)

        overall = preprocessing_analysis.get('overall_statistics', {})
        print(f"📊 Images requiring preprocessing: {overall.get('images_requiring_preprocessing', 0)}")
        print(f"✅ Standard detection success: {overall.get('images_solved_by_standard', 0)}")
        print(f"🔄 Rotation detection success: {overall.get('images_solved_by_rotation', 0)}")

        # Show top performing methods
        method_effectiveness = preprocessing_analysis.get('method_effectiveness', {})
        if method_effectiveness:
            print("\n🏆 TOP PERFORMING PREPROCESSING METHODS:")
            sorted_methods = sorted(method_effectiveness.items(),
                                    key=lambda x: x[1]['success_rate'],
                                    reverse=True)[:5]

            for i, (method, stats) in enumerate(sorted_methods, 1):
                print(f"   {i}. {method}: {stats['success_rate']:.1f}% success rate "
                      f"({stats['images_successful']}/{stats['images_attempted']} images)")

        # Show category effectiveness
        category_analysis = preprocessing_analysis.get('category_analysis', {})
        if category_analysis:
            print("\n📂 PREPROCESSING CATEGORY EFFECTIVENESS:")
            for category, stats in category_analysis.items():
                print(f"   {category}: {stats['success_rate']:.1f}% success rate "
                      f"({stats['methods_count']} methods)")

        print("=" * 50)


def main():
    """Run the complete experimental analysis"""
    extractor = RealDataExtractor()

    # Run comprehensive experiment
    experimental_data = extractor.run_comprehensive_experiment()

    if experimental_data:
        # Print key values for assignment
        extractor.print_key_values_for_assignment(experimental_data)

        print("\n✅ Experiment completed successfully!")
        print("📁 All results saved in 'experiment_results' folder")
        print("📋 Use the values from the analysis report in your assignment")
    else:
        print("❌ Experiment failed")


if __name__ == "__main__":
    main()