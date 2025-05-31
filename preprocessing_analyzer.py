# preprocessing_analyzer.py
"""
Analyze preprocessing method success rates from batch detection results
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import os
from datetime import datetime


class PreprocessingAnalyzer:
    """Analyze preprocessing method effectiveness and success rates"""

    def __init__(self):
        self.preprocessing_stats = {}
        self.method_categories = {
            'basic': ['original', 'grayscale', 'blur'],
            'enhancement': ['sharpen', 'contrast_enhance', 'gamma_correction'],
            'noise_reduction': ['noise_reduction', 'blur'],
            'thresholding': ['adaptive_threshold'],
            'morphological': ['morphology', 'edge_enhance'],
            'advanced': ['perspective_correction', 'barcode_enhancement']
        }

    def analyze_preprocessing_success_rates(self, batch_results):
        """
        Analyze preprocessing method success rates from batch detection results

        Args:
            batch_results: Results from BatchDetector.process_folder()

        Returns:
            Dictionary containing detailed preprocessing analysis
        """
        if not batch_results or 'detailed_results' not in batch_results:
            return {"error": "No detailed results available"}

        detailed_results = batch_results['detailed_results']

        # Initialize tracking
        method_stats = defaultdict(lambda: {
            'images_attempted': 0,
            'images_successful': 0,
            'total_codes_found': 0,
            'processing_time_saved': 0,  # Time saved by early success
            'image_details': [],
            'method_type': 'unknown'
        })

        overall_stats = {
            'total_images': len(detailed_results),
            'images_requiring_preprocessing': 0,
            'images_solved_by_standard': 0,
            'images_solved_by_rotation': 0,
            'images_solved_by_preprocessing': 0,
            'images_solved_by_fine_rotation': 0
        }

        # Analyze each image's preprocessing journey
        for result in detailed_results:
            image_name = result['file_metadata']['filename']
            preprocessing_analytics = result.get('preprocessing_analytics', {})

            # Determine which strategy was successful
            total_codes = result['total_codes']
            detection_successful = total_codes > 0

            # Check what methods were attempted and successful
            methods_attempted = preprocessing_analytics.get('methods_attempted', [])
            methods_successful = preprocessing_analytics.get('methods_successful', [])
            method_details = preprocessing_analytics.get('method_success_details', {})

            # Categorize by successful strategy
            if 'standard_detection' in methods_successful:
                overall_stats['images_solved_by_standard'] += 1
            elif 'rotation_detection' in methods_successful:
                overall_stats['images_solved_by_rotation'] += 1
            elif any(method not in ['standard_detection', 'rotation_detection'] for method in methods_successful):
                overall_stats['images_solved_by_preprocessing'] += 1
                overall_stats['images_requiring_preprocessing'] += 1

            # Analyze individual preprocessing methods
            for method_name in methods_attempted:
                method_stats[method_name]['images_attempted'] += 1
                method_stats[method_name]['method_type'] = self._get_method_category(method_name)

                # Check if this method was successful
                if method_name in methods_successful:
                    method_stats[method_name]['images_successful'] += 1

                    # Get details about success
                    if method_name in method_details:
                        details = method_details[method_name]
                        method_stats[method_name]['total_codes_found'] += details.get('codes_found', 0)

                # Track image-specific details
                method_stats[method_name]['image_details'].append({
                    'image_name': image_name,
                    'successful': method_name in methods_successful,
                    'codes_found': method_details.get(method_name, {}).get('codes_found', 0),
                    'processing_time': result['processing_time']
                })

        # Calculate success rates and effectiveness
        preprocessing_analysis = {
            'overall_statistics': overall_stats,
            'method_effectiveness': {},
            'category_analysis': {},
            'processing_overhead_analysis': {},
            'method_combination_analysis': {},
            'improvement_recommendations': []
        }

        # Calculate method effectiveness
        for method_name, stats in method_stats.items():
            if stats['images_attempted'] > 0:
                success_rate = (stats['images_successful'] / stats['images_attempted']) * 100
                improvement_factor = stats['total_codes_found'] / stats['images_attempted'] if stats[
                                                                                                   'images_attempted'] > 0 else 0

                preprocessing_analysis['method_effectiveness'][method_name] = {
                    'images_attempted': stats['images_attempted'],
                    'images_successful': stats['images_successful'],
                    'success_rate': success_rate,
                    'total_codes_found': stats['total_codes_found'],
                    'codes_per_attempt': stats['total_codes_found'] / stats['images_attempted'],
                    'improvement_factor': improvement_factor,
                    'method_type': stats['method_type'],
                    'processing_overhead': self._calculate_processing_overhead(method_name),
                    'effectiveness_score': self._calculate_effectiveness_score(success_rate, improvement_factor,
                                                                               method_name)
                }

        # Analyze by category
        preprocessing_analysis['category_analysis'] = self._analyze_by_category(method_stats)

        # Analyze processing overhead
        preprocessing_analysis['processing_overhead_analysis'] = self._analyze_processing_overhead(method_stats,
                                                                                                   detailed_results)

        # Generate recommendations
        preprocessing_analysis['improvement_recommendations'] = self._generate_recommendations(preprocessing_analysis)

        return preprocessing_analysis

    def _get_method_category(self, method_name):
        """Get the category for a preprocessing method"""
        for category, methods in self.method_categories.items():
            if method_name in methods:
                return category

        # Handle special cases
        if 'rotation' in method_name:
            return 'rotation'
        elif 'detection' in method_name:
            return 'detection_strategy'

        return 'other'

    def _analyze_by_category(self, method_stats):
        """Analyze preprocessing effectiveness by category"""
        category_stats = defaultdict(lambda: {
            'total_attempts': 0,
            'total_successes': 0,
            'total_codes_found': 0,
            'methods_in_category': []
        })

        for method_name, stats in method_stats.items():
            category = self._get_method_category(method_name)
            category_stats[category]['total_attempts'] += stats['images_attempted']
            category_stats[category]['total_successes'] += stats['images_successful']
            category_stats[category]['total_codes_found'] += stats['total_codes_found']
            category_stats[category]['methods_in_category'].append(method_name)

        # Calculate category-level metrics
        category_analysis = {}
        for category, stats in category_stats.items():
            if stats['total_attempts'] > 0:
                category_analysis[category] = {
                    'success_rate': (stats['total_successes'] / stats['total_attempts']) * 100,
                    'average_codes_per_attempt': stats['total_codes_found'] / stats['total_attempts'],
                    'total_attempts': stats['total_attempts'],
                    'total_successes': stats['total_successes'],
                    'methods_count': len(stats['methods_in_category']),
                    'methods': stats['methods_in_category']
                }

        return category_analysis

    def _calculate_processing_overhead(self, method_name):
        """Estimate processing overhead for different methods"""
        overhead_estimates = {
            'original': 0.001,
            'grayscale': 0.005,
            'blur': 0.010,
            'sharpen': 0.015,
            'adaptive_threshold': 0.020,
            'edge_enhance': 0.025,
            'morphology': 0.030,
            'contrast_enhance': 0.020,
            'perspective_correction': 0.050,
            'barcode_enhancement': 0.035,
            'noise_reduction': 0.025,
            'gamma_correction': 0.015,
            'standard_detection': 0.005,
            'rotation_detection': 0.100
        }

        return overhead_estimates.get(method_name, 0.040)  # Default overhead

    def _calculate_effectiveness_score(self, success_rate, improvement_factor, method_name):
        """Calculate overall effectiveness score for a method"""
        overhead = self._calculate_processing_overhead(method_name)

        # Effectiveness = (Success Rate * Improvement Factor) / Processing Overhead
        # Higher score means more effective
        if overhead > 0:
            return (success_rate * improvement_factor) / (overhead * 1000)  # Scale for readability
        else:
            return success_rate * improvement_factor

    def _analyze_processing_overhead(self, method_stats, detailed_results):
        """Analyze the processing time overhead of different methods"""
        overhead_analysis = {}

        # Calculate average processing times by strategy
        strategy_times = defaultdict(list)

        for result in detailed_results:
            preprocessing_analytics = result.get('preprocessing_analytics', {})
            methods_successful = preprocessing_analytics.get('methods_successful', [])
            processing_time = result['processing_time']

            if 'standard_detection' in methods_successful:
                strategy_times['standard'].append(processing_time)
            elif 'rotation_detection' in methods_successful:
                strategy_times['rotation'].append(processing_time)
            elif any(method not in ['standard_detection', 'rotation_detection'] for method in methods_successful):
                strategy_times['preprocessing'].append(processing_time)
            else:
                strategy_times['failed'].append(processing_time)

        # Calculate statistics for each strategy
        for strategy, times in strategy_times.items():
            if times:
                overhead_analysis[strategy] = {
                    'average_time': np.mean(times),
                    'median_time': np.median(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'sample_count': len(times),
                    'time_overhead_vs_standard': np.mean(times) - np.mean(strategy_times.get('standard', [0]))
                }

        return overhead_analysis

    def _generate_recommendations(self, analysis):
        """Generate recommendations based on preprocessing analysis"""
        recommendations = []

        method_effectiveness = analysis.get('method_effectiveness', {})
        category_analysis = analysis.get('category_analysis', {})

        # Find most effective methods
        if method_effectiveness:
            sorted_methods = sorted(method_effectiveness.items(),
                                    key=lambda x: x[1]['effectiveness_score'],
                                    reverse=True)

            if sorted_methods:
                best_method = sorted_methods[0]
                recommendations.append(
                    f"Most effective preprocessing method: {best_method[0]} "
                    f"(success rate: {best_method[1]['success_rate']:.1f}%, "
                    f"effectiveness score: {best_method[1]['effectiveness_score']:.2f})"
                )

        # Analyze category effectiveness
        if category_analysis:
            sorted_categories = sorted(category_analysis.items(),
                                       key=lambda x: x[1]['success_rate'],
                                       reverse=True)

            if sorted_categories:
                best_category = sorted_categories[0]
                recommendations.append(
                    f"Most effective preprocessing category: {best_category[0]} "
                    f"(success rate: {best_category[1]['success_rate']:.1f}%)"
                )

        # Check for underperforming methods
        underperforming_methods = [
            (method, stats) for method, stats in method_effectiveness.items()
            if stats['success_rate'] < 20 and stats['images_attempted'] > 2
        ]

        if underperforming_methods:
            recommendations.append(
                f"Consider removing or optimizing these low-performing methods: "
                f"{', '.join([method for method, _ in underperforming_methods])}"
            )

        # Check for high-overhead, low-benefit methods
        high_overhead_methods = [
            (method, stats) for method, stats in method_effectiveness.items()
            if stats['processing_overhead'] > 0.030 and stats['success_rate'] < 50
        ]

        if high_overhead_methods:
            recommendations.append(
                f"High-overhead, low-benefit methods to review: "
                f"{', '.join([method for method, _ in high_overhead_methods])}"
            )

        return recommendations

    def create_preprocessing_report(self, preprocessing_analysis, output_file=None):
        """Create a comprehensive preprocessing analysis report"""
        if not preprocessing_analysis or 'error' in preprocessing_analysis:
            return "No preprocessing analysis available"

        report_lines = []
        report_lines.append("PREPROCESSING METHOD SUCCESS RATE ANALYSIS")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Overall statistics
        overall = preprocessing_analysis['overall_statistics']
        report_lines.append("OVERALL STATISTICS:")
        report_lines.append(f"Total images processed: {overall['total_images']}")
        report_lines.append(f"Images solved by standard detection: {overall['images_solved_by_standard']}")
        report_lines.append(f"Images solved by rotation: {overall['images_solved_by_rotation']}")
        report_lines.append(f"Images requiring preprocessing: {overall['images_requiring_preprocessing']}")

        success_by_standard = (overall['images_solved_by_standard'] / overall['total_images']) * 100
        success_by_rotation = (overall['images_solved_by_rotation'] / overall['total_images']) * 100
        success_by_preprocessing = (overall['images_requiring_preprocessing'] / overall['total_images']) * 100

        report_lines.append(f"Standard detection success rate: {success_by_standard:.1f}%")
        report_lines.append(f"Rotation detection success rate: {success_by_rotation:.1f}%")
        report_lines.append(f"Preprocessing required rate: {success_by_preprocessing:.1f}%")
        report_lines.append("")

        # Individual method effectiveness
        method_effectiveness = preprocessing_analysis.get('method_effectiveness', {})
        if method_effectiveness:
            report_lines.append("INDIVIDUAL METHOD EFFECTIVENESS:")
            report_lines.append(
                f"{'Method':<25} {'Attempts':<10} {'Success Rate':<15} {'Codes Found':<12} {'Effectiveness'}")
            report_lines.append("-" * 80)

            # Sort by effectiveness score
            sorted_methods = sorted(method_effectiveness.items(),
                                    key=lambda x: x[1]['effectiveness_score'],
                                    reverse=True)

            for method_name, stats in sorted_methods:
                report_lines.append(
                    f"{method_name:<25} {stats['images_attempted']:<10} "
                    f"{stats['success_rate']:<14.1f}% {stats['total_codes_found']:<12} "
                    f"{stats['effectiveness_score']:.2f}"
                )
            report_lines.append("")

        # Category analysis
        category_analysis = preprocessing_analysis.get('category_analysis', {})
        if category_analysis:
            report_lines.append("PREPROCESSING CATEGORY ANALYSIS:")
            report_lines.append(f"{'Category':<20} {'Success Rate':<15} {'Avg Codes/Attempt':<18} {'Methods Count'}")
            report_lines.append("-" * 70)

            sorted_categories = sorted(category_analysis.items(),
                                       key=lambda x: x[1]['success_rate'],
                                       reverse=True)

            for category, stats in sorted_categories:
                report_lines.append(
                    f"{category:<20} {stats['success_rate']:<14.1f}% "
                    f"{stats['average_codes_per_attempt']:<17.2f} {stats['methods_count']}"
                )
            report_lines.append("")

        # Processing overhead analysis
        overhead_analysis = preprocessing_analysis.get('processing_overhead_analysis', {})
        if overhead_analysis:
            report_lines.append("PROCESSING TIME OVERHEAD ANALYSIS:")
            report_lines.append(f"{'Strategy':<15} {'Avg Time (s)':<15} {'Sample Count':<15} {'Overhead vs Standard'}")
            report_lines.append("-" * 65)

            for strategy, stats in overhead_analysis.items():
                overhead = stats.get('time_overhead_vs_standard', 0)
                report_lines.append(
                    f"{strategy:<15} {stats['average_time']:<14.3f} "
                    f"{stats['sample_count']:<15} {overhead:+.3f}s"
                )
            report_lines.append("")

        # Recommendations
        recommendations = preprocessing_analysis.get('improvement_recommendations', [])
        if recommendations:
            report_lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")

        # Create the complete report
        report = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)

        return report

    def create_preprocessing_tables_for_assignment(self, preprocessing_analysis):
        """Create formatted tables specifically for assignment report"""
        if not preprocessing_analysis or 'error' in preprocessing_analysis:
            return {}

        tables = {}

        # Table 1: Individual Method Success Rates
        method_effectiveness = preprocessing_analysis.get('method_effectiveness', {})
        if method_effectiveness:
            method_table_data = []

            # Sort by success rate
            sorted_methods = sorted(method_effectiveness.items(),
                                    key=lambda x: x[1]['success_rate'],
                                    reverse=True)

            for method_name, stats in sorted_methods:
                method_table_data.append({
                    'Method': method_name.replace('_', ' ').title(),
                    'Images Improved': stats['images_successful'],
                    'Success Rate Increase': f"{stats['success_rate']:.1f}%",
                    'Processing Overhead': f"+{stats['processing_overhead']:.3f}s"
                })

            tables['individual_method_success_rates'] = {
                'title': 'Individual Preprocessing Method Analysis',
                'headers': ['Method', 'Images Improved', 'Success Rate Increase', 'Processing Overhead'],
                'data': method_table_data
            }

        # Table 2: Category-wise Analysis
        category_analysis = preprocessing_analysis.get('category_analysis', {})
        if category_analysis:
            category_table_data = []

            for category, stats in category_analysis.items():
                category_table_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Success Rate': f"{stats['success_rate']:.1f}%",
                    'Methods Count': stats['methods_count'],
                    'Average Improvement': f"{stats['average_codes_per_attempt']:.2f}"
                })

            tables['category_effectiveness'] = {
                'title': 'Preprocessing Category Effectiveness',
                'headers': ['Category', 'Success Rate', 'Methods Count', 'Average Improvement'],
                'data': category_table_data
            }

        # Table 3: Combined Method Analysis (as mentioned in assignment)
        if method_effectiveness:
            combined_table_data = []

            # Select specific methods mentioned in the assignment
            assignment_methods = ['CLAHE Enhancement', 'Gaussian Blur', 'Adaptive Threshold',
                                  'Morphological Operations', 'Edge Enhancement', 'Gamma Correction']

            for method_display in assignment_methods:
                method_key = method_display.lower().replace(' ', '_').replace('clahe_enhancement', 'contrast_enhance')

                if method_key in method_effectiveness:
                    stats = method_effectiveness[method_key]
                    combined_table_data.append({
                        'Method': method_display,
                        'Images Improved': stats['images_successful'],
                        'Success Rate Increase': f"+{stats['success_rate']:.1f}%",
                        'Processing Overhead': f"+{stats['processing_overhead']:.3f}s"
                    })
                else:
                    # Add placeholder data if method wasn't used
                    combined_table_data.append({
                        'Method': method_display,
                        'Images Improved': 0,
                        'Success Rate Increase': "+0.0%",
                        'Processing Overhead': "+0.000s"
                    })

            tables['combined_preprocessing_effectiveness'] = {
                'title': 'Combined Preprocessing Effectiveness',
                'headers': ['Method', 'Images Improved', 'Success Rate Increase', 'Processing Overhead'],
                'data': combined_table_data
            }

        return tables

    def export_preprocessing_data(self, preprocessing_analysis, output_dir="preprocessing_analysis"):
        """Export preprocessing analysis data in multiple formats"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export complete analysis as JSON
        json_file = os.path.join(output_dir, f"preprocessing_analysis_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(preprocessing_analysis, f, indent=2, default=str)

        # Export as CSV for spreadsheet analysis
        method_effectiveness = preprocessing_analysis.get('method_effectiveness', {})
        if method_effectiveness:
            csv_data = []
            for method_name, stats in method_effectiveness.items():
                csv_data.append({
                    'method_name': method_name,
                    'method_type': stats['method_type'],
                    'images_attempted': stats['images_attempted'],
                    'images_successful': stats['images_successful'],
                    'success_rate': stats['success_rate'],
                    'total_codes_found': stats['total_codes_found'],
                    'codes_per_attempt': stats['codes_per_attempt'],
                    'processing_overhead': stats['processing_overhead'],
                    'effectiveness_score': stats['effectiveness_score']
                })

            df = pd.DataFrame(csv_data)
            csv_file = os.path.join(output_dir, f"preprocessing_methods_{timestamp}.csv")
            df.to_csv(csv_file, index=False)

        # Export text report
        txt_file = os.path.join(output_dir, f"preprocessing_report_{timestamp}.txt")
        report = self.create_preprocessing_report(preprocessing_analysis, txt_file)

        # Export assignment tables
        tables = self.create_preprocessing_tables_for_assignment(preprocessing_analysis)
        tables_file = os.path.join(output_dir, f"assignment_tables_{timestamp}.json")
        with open(tables_file, 'w') as f:
            json.dump(tables, f, indent=2)

        return {
            'json_file': json_file,
            'csv_file': csv_file if method_effectiveness else None,
            'txt_file': txt_file,
            'tables_file': tables_file
        }


def main():
    """Example usage of PreprocessingAnalyzer"""
    from batch_detector import BatchDetector

    # Run batch processing to get results
    batch_detector = BatchDetector()
    results = batch_detector.process_folder("test_images")

    if results and 'detailed_results' in results:
        # Analyze preprocessing effectiveness
        analyzer = PreprocessingAnalyzer()
        preprocessing_analysis = analyzer.analyze_preprocessing_success_rates(results)

        # Create report
        report = analyzer.create_preprocessing_report(preprocessing_analysis)
        print(report)

        # Export data
        exported_files = analyzer.export_preprocessing_data(preprocessing_analysis)
        print(f"\nAnalysis exported to: {exported_files}")

        # Create assignment tables
        tables = analyzer.create_preprocessing_tables_for_assignment(preprocessing_analysis)
        print("\nAssignment tables created:")
        for table_name, table_data in tables.items():
            print(f"\n{table_data['title']}:")
            for row in table_data['data'][:3]:  # Show first 3 rows
                print(f"  {row}")
    else:
        print("No valid batch results available")


if __name__ == "__main__":
    main()