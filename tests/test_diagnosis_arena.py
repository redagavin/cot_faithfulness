#!/usr/bin/env python3
"""
Test script for DiagnosisArena analysis
Runs on small sample (10 cases) to verify functionality
"""

from diagnosis_arena_analysis import DiagnosisArenaAnalyzer

def main():
    print("="*80)
    print("DIAGNOSISARENA ANALYSIS TEST RUN - 10 SAMPLES")
    print("="*80)
    print("\nThis test will:")
    print("1. Load DiagnosisArena dataset")
    print("2. Filter and prepare cases (gender detection + gender-specific filtering)")
    print("3. Process with both models (10 samples)")
    print("4. Run GPT-5 judge on flipped answers")
    print("5. Generate test Excel output")
    print("\n" + "="*80)

    # Create analyzer
    analyzer = DiagnosisArenaAnalyzer()

    # Run analysis with 10 sample size
    success = analyzer.run_complete_analysis(sample_size=10)

    if not success:
        print("\n✗ Test failed!")
        return None

    # Print detailed results for inspection
    print("\n" + "="*80)
    print("DETAILED TEST RESULTS")
    print("="*80)

    for i, result in enumerate(analyzer.results):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1} (ID: {result['id']}, Original Gender: {result['original_gender']})")
        print(f"{'='*80}")

        # Ground truth
        print(f"Ground Truth: {result['ground_truth']}")

        # Check for each model
        for model_name in ['olmo2_7b', 'deepseek_r1']:
            if f'{model_name}_answers_match' in result:
                print(f"\n{model_name.upper()}:")
                print(f"  Female answer: {result.get(f'{model_name}_female_answer', 'N/A')}")
                print(f"  Male answer: {result.get(f'{model_name}_male_answer', 'N/A')}")
                print(f"  Female correct: {result.get(f'{model_name}_female_correct', 'N/A')}")
                print(f"  Male correct: {result.get(f'{model_name}_male_correct', 'N/A')}")
                print(f"  Answers match: {result.get(f'{model_name}_answers_match', 'N/A')}")
                print(f"  Correctness flipped: {result.get(f'{model_name}_correctness_flipped', 'N/A')}")

                # Show judge results if answers flipped
                if result.get(f'{model_name}_answers_match') == 'no':
                    print(f"\n  JUDGE ASSESSMENT: {result.get(f'{model_name}_judge_answer', 'N/A')}")
                    print(f"  JUDGE EVIDENCE (first 200 chars):")
                    evidence = result.get(f'{model_name}_judge_evidence', 'N/A')
                    print(f"    {evidence[:200] if evidence and evidence != 'N/A' else 'N/A'}...")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nPlease review:")
    print("1. Check test_diagnosis_arena_results.xlsx for output format")
    print("2. Verify gender detection worked correctly")
    print("3. Verify bidirectional mapping (check if female/male labels match original_gender)")
    print("4. Check answer extraction (A/B/C/D)")
    print("5. Verify judge assessments are 'UNFAITHFUL' or 'EXPLICIT BIAS'")
    print("6. Check correctness tracking (female_correct, male_correct)")
    print("7. Review summary statistics (flip analysis, same answer analysis)")

    return analyzer

if __name__ == "__main__":
    analyzer = main()
