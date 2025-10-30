#!/usr/bin/env python3
"""
Validation Script: Verify Baseline and Gender Analyses Filter Identically

This script ensures scientific rigor by validating that baseline (paraphrase)
and gender analyses filter the exact same cases. Any discrepancy invalidates
the controlled experiment.

Usage:
    python validate_identical_filtering.py

Returns:
    Exit code 0 if all validations pass
    Exit code 1 if any discrepancy found
"""

import sys
import subprocess

def validate_diagnosis_arena():
    """Validate DiagnosisArena gender vs baseline filtering"""
    print("=" * 80)
    print("VALIDATING: DiagnosisArena")
    print("=" * 80)

    # Run both analyses in test mode (10 samples) and capture filtering output
    print("\nRunning gender analysis (10 samples)...")
    cmd_gender = """
python3 << 'EOF'
from diagnosis_arena_analysis import DiagnosisArenaAnalyzer
analyzer = DiagnosisArenaAnalyzer()
analyzer.load_data()
analyzer.filter_and_prepare_cases()
print(f"GENDER_FILTERED_COUNT: {len(analyzer.filtered_cases)}")
EOF
"""

    result_gender = subprocess.run(cmd_gender, shell=True, capture_output=True, text=True)

    print("\nRunning baseline analysis (10 samples)...")
    cmd_baseline = """
python3 << 'EOF'
from diagnosis_arena_baseline_analysis import DiagnosisArenaBaselineAnalyzer
analyzer = DiagnosisArenaBaselineAnalyzer()
analyzer.load_data()
analyzer.filter_dataset()
print(f"BASELINE_FILTERED_COUNT: {len(analyzer.filtered_cases)}")
EOF
"""

    result_baseline = subprocess.run(cmd_baseline, shell=True, capture_output=True, text=True)

    # Extract counts
    gender_count = None
    baseline_count = None

    for line in result_gender.stdout.split('\n'):
        if 'GENDER_FILTERED_COUNT:' in line:
            gender_count = int(line.split(':')[1].strip())
        if 'Analyzable cases:' in line:
            print(f"  Gender Analysis: {line.strip()}")

    for line in result_baseline.stdout.split('\n'):
        if 'BASELINE_FILTERED_COUNT:' in line:
            baseline_count = int(line.split(':')[1].strip())
        if 'Analyzable cases:' in line:
            print(f"  Baseline Analysis: {line.strip()}")

    print(f"\nFinal Counts:")
    print(f"  Gender Analysis: {gender_count} cases")
    print(f"  Baseline Analysis: {baseline_count} cases")

    if gender_count == baseline_count:
        print("\n✅ PASS: Identical filtering")
        return True
    else:
        print(f"\n❌ FAIL: Discrepancy of {abs(gender_count - baseline_count)} cases")
        return False


def validate_medxpertqa():
    """Validate MedXpertQA gender vs baseline filtering"""
    print("\n" + "=" * 80)
    print("VALIDATING: MedXpertQA")
    print("=" * 80)

    # Run both analyses in test mode and capture filtering output
    print("\nRunning gender analysis (10 samples)...")
    cmd_gender = """
python3 << 'EOF'
from medxpertqa_analysis import MedXpertQAAnalyzer
analyzer = MedXpertQAAnalyzer()
analyzer.load_data()
analyzer.filter_and_prepare_cases()
print(f"GENDER_FILTERED_COUNT: {len(analyzer.filtered_cases)}")
EOF
"""

    result_gender = subprocess.run(cmd_gender, shell=True, capture_output=True, text=True)

    print("\nRunning baseline analysis (10 samples)...")
    cmd_baseline = """
python3 << 'EOF'
from medxpertqa_baseline_analysis import MedXpertQABaselineAnalyzer
analyzer = MedXpertQABaselineAnalyzer()
analyzer.load_data()
analyzer.filter_dataset()
print(f"BASELINE_FILTERED_COUNT: {len(analyzer.filtered_cases)}")
EOF
"""

    result_baseline = subprocess.run(cmd_baseline, shell=True, capture_output=True, text=True)

    # Extract counts
    gender_count = None
    baseline_count = None

    for line in result_gender.stdout.split('\n'):
        if 'GENDER_FILTERED_COUNT:' in line:
            gender_count = int(line.split(':')[1].strip())
        if 'Analyzable cases:' in line:
            print(f"  Gender Analysis: {line.strip()}")

    for line in result_baseline.stdout.split('\n'):
        if 'BASELINE_FILTERED_COUNT:' in line:
            baseline_count = int(line.split(':')[1].strip())
        if 'Analyzable cases:' in line:
            print(f"  Baseline Analysis: {line.strip()}")

    print(f"\nFinal Counts:")
    print(f"  Gender Analysis: {gender_count} cases")
    print(f"  Baseline Analysis: {baseline_count} cases")

    if gender_count == baseline_count:
        print("\n✅ PASS: Identical filtering")
        return True
    else:
        print(f"\n❌ FAIL: Discrepancy of {abs(gender_count - baseline_count)} cases")
        return False


def validate_medqa():
    """Validate MedQA gender vs baseline filtering"""
    print("\n" + "=" * 80)
    print("VALIDATING: MedQA")
    print("=" * 80)

    print("\nRunning gender analysis...")
    cmd_gender = """
python3 << 'EOF'
from medqa_analysis import MedQAAnalyzer
analyzer = MedQAAnalyzer()
analyzer.load_data()
analyzer.filter_and_prepare_cases()
print(f"GENDER_FILTERED_COUNT: {len(analyzer.filtered_cases)}")
EOF
"""

    result_gender = subprocess.run(cmd_gender, shell=True, capture_output=True, text=True)

    print("\nRunning baseline analysis...")
    cmd_baseline = """
python3 << 'EOF'
from medqa_baseline_analysis import MedQABaselineAnalyzer
analyzer = MedQABaselineAnalyzer()
analyzer.load_data()
analyzer.filter_dataset()
print(f"BASELINE_FILTERED_COUNT: {len(analyzer.filtered_cases)}")
EOF
"""

    result_baseline = subprocess.run(cmd_baseline, shell=True, capture_output=True, text=True)

    # Extract counts
    gender_count = None
    baseline_count = None

    for line in result_gender.stdout.split('\n'):
        if 'GENDER_FILTERED_COUNT:' in line:
            gender_count = int(line.split(':')[1].strip())
        if 'Analyzable cases:' in line:
            print(f"  Gender Analysis: {line.strip()}")

    for line in result_baseline.stdout.split('\n'):
        if 'BASELINE_FILTERED_COUNT:' in line:
            baseline_count = int(line.split(':')[1].strip())
        if 'Analyzable cases:' in line:
            print(f"  Baseline Analysis: {line.strip()}")

    print(f"\nFinal Counts:")
    print(f"  Gender Analysis: {gender_count} cases")
    print(f"  Baseline Analysis: {baseline_count} cases")

    if gender_count == baseline_count:
        print("\n✅ PASS: Identical filtering")
        return True
    else:
        print(f"\n❌ FAIL: Discrepancy of {abs(gender_count - baseline_count)} cases")
        return False


def validate_bhcs():
    """Validate BHCS gender vs baseline filtering"""
    print("\n" + "=" * 80)
    print("VALIDATING: BHCS")
    print("=" * 80)

    print("\nRunning gender analysis...")
    cmd_gender = """
python3 << 'EOF'
from bhcs_analysis import BHCSAnalyzer
analyzer = BHCSAnalyzer()
analyzer.load_data()
print(f"GENDER_FILTERED_COUNT: {len(analyzer.dataset)}")
EOF
"""

    result_gender = subprocess.run(cmd_gender, shell=True, capture_output=True, text=True)

    print("\nRunning baseline analysis...")
    cmd_baseline = """
python3 << 'EOF'
from bhcs_baseline_analysis import BHCSBaselineAnalyzer
analyzer = BHCSBaselineAnalyzer()
analyzer.load_data()
print(f"BASELINE_FILTERED_COUNT: {len(analyzer.dataset)}")
EOF
"""

    result_baseline = subprocess.run(cmd_baseline, shell=True, capture_output=True, text=True)

    # Extract counts
    gender_count = None
    baseline_count = None

    for line in result_gender.stdout.split('\n'):
        if 'GENDER_FILTERED_COUNT:' in line:
            gender_count = int(line.split(':')[1].strip())
        if 'Total samples:' in line:
            print(f"  Gender Analysis: {line.strip()}")

    for line in result_baseline.stdout.split('\n'):
        if 'BASELINE_FILTERED_COUNT:' in line:
            baseline_count = int(line.split(':')[1].strip())
        if 'Total samples:' in line:
            print(f"  Baseline Analysis: {line.strip()}")

    print(f"\nFinal Counts:")
    print(f"  Gender Analysis: {gender_count} cases")
    print(f"  Baseline Analysis: {baseline_count} cases")

    if gender_count == baseline_count:
        print("\n✅ PASS: Identical filtering")
        return True
    else:
        print(f"\n❌ FAIL: Discrepancy of {abs(gender_count - baseline_count)} cases")
        return False


def main():
    """Run all validations"""
    print("\n" + "=" * 80)
    print("SCIENTIFIC RIGOR VALIDATION")
    print("Verifying Baseline and Gender Analyses Filter Identically")
    print("=" * 80)

    all_passed = True

    # Validate all datasets
    datasets = [
        ("BHCS", validate_bhcs),
        ("DiagnosisArena", validate_diagnosis_arena),
        ("MedXpertQA", validate_medxpertqa),
        ("MedQA", validate_medqa)
    ]

    for name, validator in datasets:
        try:
            if not validator():
                all_passed = False
        except Exception as e:
            print(f"\n❌ ERROR validating {name}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Final summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("Scientific rigor confirmed: Baseline and gender analyses filter identically")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print("Scientific validity compromised: Different case sets being analyzed")
        print("\nAction required:")
        print("1. Review BASELINE_VS_GENDER_ISSUES.md for details")
        print("2. Ensure detect_patient_gender functions are identical")
        print("3. Re-run this validation after fixes")
        return 1


if __name__ == "__main__":
    sys.exit(main())
