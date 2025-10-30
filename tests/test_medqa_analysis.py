#!/usr/bin/env python3
"""
MedQA Gender Analysis - TEST VERSION (10 samples)
"""

from medqa_analysis import MedQAAnalyzer

def main():
    """Test run with 10 samples"""
    print("=" * 60)
    print("MEDQA GENDER ANALYSIS - TEST RUN (10 SAMPLES)")
    print("=" * 60)
    print()

    analyzer = MedQAAnalyzer()

    # Run with 10 samples
    analyzer.run_complete_analysis(sample_size=10)

    print("\n" + "=" * 60)
    print("TEST RUN COMPLETED")
    print("=" * 60)
    print(f"Total cases processed: {len(analyzer.results)}")

    models_used = [name for name in analyzer.get_models_config().keys()
                   if any(f'{name}_answers_match' in result for result in analyzer.results)]
    print(f"Models used: {models_used}")

    print("\nResults saved to: test_medqa_analysis_results.xlsx")
    print("\nPlease validate:")
    print("  1. Gender detection works correctly")
    print("  2. Gender swapping applied properly")
    print("  3. Answer extraction successful (>80%)")
    print("  4. Excel file contains all expected columns")

if __name__ == "__main__":
    main()
