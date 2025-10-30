#!/usr/bin/env python3
"""
Test script for BHCS analysis with new judge prompt
Runs on small sample (5 texts) to verify functionality
"""

from bhcs_analysis import BHCSAnalyzer

def main():
    print("="*80)
    print("BHCS ANALYSIS TEST RUN - 5 SAMPLES")
    print("="*80)
    print("\nThis test will:")
    print("1. Load data")
    print("2. Create gender-modified versions")
    print("3. Process with both models (5 samples each)")
    print("4. Run judge on flipped answers")
    print("5. Generate test Excel output")
    print("\n" + "="*80)

    # Create analyzer
    analyzer = BHCSAnalyzer()

    # Step 1: Load data
    if not analyzer.load_data():
        print("ERROR: Failed to load data")
        return None

    # Step 2: Apply gender mapping
    if 'test' in analyzer.data:
        analyzer.test_original = analyzer.data['test']
        analyzer.test_modified = analyzer.apply_gender_mapping(analyzer.data['test'])
    else:
        print("Warning: 'test' key not found, using first available key")
        key = list(analyzer.data.keys())[0]
        analyzer.test_original = analyzer.data[key]
        analyzer.test_modified = analyzer.apply_gender_mapping(analyzer.data[key])

    # Step 3: Process with small sample
    analyzer.process_texts_with_models(analyzer.test_original, analyzer.test_modified, sample_size=10)

    # Step 4: Save with test filename
    analyzer.save_to_spreadsheet("test_bhcs_analysis_results.xlsx")

    # Print detailed results for inspection
    print("\n" + "="*80)
    print("DETAILED TEST RESULTS")
    print("="*80)

    for i, result in enumerate(analyzer.results):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*80}")

        # Check for each model
        for model_name in ['olmo2_7b', 'deepseek_r1']:
            if f'{model_name}_original_answer' in result:
                print(f"\n{model_name.upper()}:")
                print(f"  Original answer (female): {result.get(f'{model_name}_original_answer', 'N/A')}")
                print(f"  Modified answer (male): {result.get(f'{model_name}_modified_answer', 'N/A')}")
                print(f"  Answers match: {result.get(f'{model_name}_answers_match', 'N/A')}")

                # Show judge results if answers flipped
                if result.get(f'{model_name}_answers_match') == 'no':
                    print(f"\n  JUDGE ASSESSMENT: {result.get(f'{model_name}_judge_answer', 'N/A')}")
                    print(f"  JUDGE EVIDENCE (first 200 chars):")
                    evidence = result.get(f'{model_name}_judge_evidence', 'N/A')
                    print(f"    {evidence[:200] if evidence and evidence != 'N/A' else 'N/A'}...")

                    # Show response snippets
                    print(f"\n  Original response (first 200 chars):")
                    orig_resp = result.get(f'{model_name}_original_response', 'N/A')
                    print(f"    {orig_resp[:200] if orig_resp and orig_resp != 'N/A' else 'N/A'}...")

                    print(f"\n  Modified response (first 200 chars):")
                    mod_resp = result.get(f'{model_name}_modified_response', 'N/A')
                    print(f"    {mod_resp[:200] if mod_resp and mod_resp != 'N/A' else 'N/A'}...")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nPlease review:")
    print("1. Check test_bhcs_analysis_results.xlsx for output format")
    print("2. Verify judge assessments are 'UNFAITHFUL' or 'EXPLICIT BIAS'")
    print("3. Check evidence quality (no 'she | he' type entries)")
    print("4. Confirm no truncation in stored texts")
    print("5. Review summary statistics")

    return analyzer

if __name__ == "__main__":
    analyzer = main()
