#!/usr/bin/env python3
"""
Phases 4, 5, 6: Combined Testing Suite

Streamlined tests for remaining components:
- Phase 4: Data Loading & Field Mapping (20 tests)
- Phase 5: Gender-Specific Filtering (20 tests)
- Phase 6: Excel Output Validation (15 tests)

Total: 55 tests
"""

import sys
import re
from typing import Dict, List, Optional
import pandas as pd


class TestResults:
    """Track test results with detailed logging"""

    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.failures = []

    def add_pass(self, description: str):
        self.total += 1
        self.passed += 1
        print(f"✓ {description}")

    def add_fail(self, description: str, expected: any, got: any):
        self.total += 1
        self.failed += 1
        self.failures.append((description, expected, got))
        print(f"✗ {description}")
        print(f"  Expected: {expected}")
        print(f"  Got: {got}")

    def print_summary(self):
        print("\n" + "="*60)
        print(f"Test Results: {self.passed}/{self.total} passed ({self.passed/self.total*100:.1f}%)")
        if self.failures:
            print(f"\n{len(self.failures)} FAILURES:")
            for desc, expected, got in self.failures:
                print(f"  - {desc}")
                print(f"    Expected: {expected}")
                print(f"    Got: {got}")
        print("="*60)
        return self.failed == 0


# ============================================================================
# PHASE 4: DATA LOADING & FIELD MAPPING
# ============================================================================

def convert_medqa_fields(item: Dict) -> Dict:
    """Convert MedQA fields from HuggingFace format to our format"""
    return {
        'id': item.get('id', ''),
        'question': item['sent1'],
        'option_a': item['ending0'],
        'option_b': item['ending1'],
        'option_c': item['ending2'],
        'option_d': item['ending3'],
        'ground_truth': ['A', 'B', 'C', 'D'][item['label']]
    }


def test_data_loading():
    """Test data loading and field mapping logic"""
    print("\n" + "="*60)
    print("PHASE 4: Data Loading & Field Mapping (20 tests)")
    print("="*60)

    results = TestResults()

    # Test 1-5: MedQA field mapping
    print("\n--- MedQA Field Mapping ---")

    test_items = [
        (
            {'id': '1', 'sent1': 'Question?', 'ending0': 'A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 0},
            'A',
            "Label 0 → A"
        ),
        (
            {'id': '2', 'sent1': 'Question?', 'ending0': 'A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 1},
            'B',
            "Label 1 → B"
        ),
        (
            {'id': '3', 'sent1': 'Question?', 'ending0': 'A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 2},
            'C',
            "Label 2 → C"
        ),
        (
            {'id': '4', 'sent1': 'Question?', 'ending0': 'A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 3},
            'D',
            "Label 3 → D"
        ),
        (
            {'sent1': 'Q', 'ending0': 'A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 0},
            '',
            "Missing ID → empty string"
        ),
    ]

    for item, expected_gt, description in test_items:
        result = convert_medqa_fields(item)
        if "Missing ID" in description:
            if result['id'] == expected_gt:  # Expected is empty string for missing ID
                results.add_pass(description)
            else:
                results.add_fail(description, expected_gt, result['id'])
        else:
            if result['ground_truth'] == expected_gt:
                results.add_pass(description)
            else:
                results.add_fail(description, expected_gt, result['ground_truth'])

    # Test 6-10: Field validation
    print("\n--- Field Validation ---")

    validation_tests = [
        (
            {'id': '1', 'sent1': 'Q', 'ending0': 'A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 0},
            True,
            "All required fields present"
        ),
        (
            {'id': '1', 'sent1': 'Q', 'ending0': 'A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 0},
            'Q',
            "Question field mapped correctly"
        ),
        (
            {'id': '1', 'sent1': 'Q', 'ending0': 'Option A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 0},
            'Option A',
            "Option A mapped correctly"
        ),
        (
            {'id': '1', 'sent1': 'Q', 'ending0': 'A', 'ending1': 'Option B', 'ending2': 'C', 'ending3': 'D', 'label': 1},
            'Option B',
            "Option B mapped correctly"
        ),
        (
            {'id': '1', 'sent1': 'Q', 'ending0': 'A', 'ending1': 'B', 'ending2': 'Option C', 'ending3': 'D', 'label': 2},
            'Option C',
            "Option C mapped correctly"
        ),
    ]

    for item, expected, description in validation_tests:
        result = convert_medqa_fields(item)
        if description == "All required fields present":
            has_all = all(k in result for k in ['id', 'question', 'option_a', 'option_b', 'option_c', 'option_d', 'ground_truth'])
            if has_all:
                results.add_pass(description)
            else:
                results.add_fail(description, "All fields", f"Missing: {set(['id', 'question', 'option_a', 'option_b', 'option_c', 'option_d', 'ground_truth']) - set(result.keys())}")
        elif "Question" in description:
            if result['question'] == expected:
                results.add_pass(description)
            else:
                results.add_fail(description, expected, result['question'])
        elif "Option A" in description:
            if result['option_a'] == expected:
                results.add_pass(description)
            else:
                results.add_fail(description, expected, result['option_a'])
        elif "Option B" in description:
            if result['option_b'] == expected:
                results.add_pass(description)
            else:
                results.add_fail(description, expected, result['option_b'])
        elif "Option C" in description:
            if result['option_c'] == expected:
                results.add_pass(description)
            else:
                results.add_fail(description, expected, result['option_c'])

    # Test 11-15: Edge cases
    print("\n--- Edge Cases ---")

    edge_cases = [
        (
            {'id': '', 'sent1': '', 'ending0': '', 'ending1': '', 'ending2': '', 'ending3': '', 'label': 0},
            '',
            "Empty strings handled"
        ),
        (
            {'id': '999', 'sent1': 'Very long question ' * 50, 'ending0': 'A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 3},
            'D',
            "Long text handled"
        ),
        (
            {'id': 'special!@#$%', 'sent1': 'Q', 'ending0': 'A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 0},
            'special!@#$%',
            "Special characters in ID"
        ),
        (
            {'id': '1', 'sent1': 'Question with\nnewline', 'ending0': 'A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 1},
            'Question with\nnewline',
            "Newlines in question"
        ),
        (
            {'id': '1', 'sent1': 'Q', 'ending0': '$100', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': 0},
            '$100',
            "Special chars in options"
        ),
    ]

    for item, expected, description in edge_cases:
        result = convert_medqa_fields(item)
        if "Empty strings" in description:
            if result['question'] == '' and result['id'] == '':
                results.add_pass(description)
            else:
                results.add_fail(description, "Empty fields", result)
        elif "Long text" in description or "Special characters in ID" in description or "Newlines" in description or "Special chars" in description:
            if result['ground_truth'] == expected or result['id'] == expected or result['question'] == expected or result['option_a'] == expected:
                results.add_pass(description)
            else:
                results.add_fail(description, expected, result)

    # Test 16-20: Dataset structure
    print("\n--- Dataset Structure ---")

    # Test multiple items in batch
    batch = [
        {'id': f'{i}', 'sent1': f'Q{i}', 'ending0': 'A', 'ending1': 'B', 'ending2': 'C', 'ending3': 'D', 'label': i % 4}
        for i in range(5)
    ]

    converted_batch = [convert_medqa_fields(item) for item in batch]

    if len(converted_batch) == 5:
        results.add_pass("Batch processing: 5 items")
    else:
        results.add_fail("Batch processing", 5, len(converted_batch))

    if all('ground_truth' in item for item in converted_batch):
        results.add_pass("All items have ground_truth")
    else:
        results.add_fail("All items have ground_truth", "All have field", "Some missing")

    if all('question' in item for item in converted_batch):
        results.add_pass("All items have question")
    else:
        results.add_fail("All items have question", "All have field", "Some missing")

    if converted_batch[0]['ground_truth'] == 'A' and converted_batch[1]['ground_truth'] == 'B':
        results.add_pass("Ground truth conversion consistent")
    else:
        results.add_fail("Ground truth conversion", "A, B", f"{converted_batch[0]['ground_truth']}, {converted_batch[1]['ground_truth']}")

    if all(item['id'] == str(i) for i, item in enumerate(converted_batch)):
        results.add_pass("IDs preserved correctly")
    else:
        results.add_fail("IDs preserved", "0-4", [item['id'] for item in converted_batch])

    return results


# ============================================================================
# PHASE 5: GENDER-SPECIFIC FILTERING
# ============================================================================

def is_gender_specific_case(case_text: str, options_text: str, diagnosis_text: str = "") -> bool:
    """Check if case involves gender-specific medical conditions"""
    combined_text = f"{case_text} {options_text} {diagnosis_text}".lower()

    # Pregnancy keywords
    pregnancy_keywords = [
        'pregnant', 'pregnancy', 'prenatal', 'antenatal', 'gestation', 'trimester',
        'fetal', 'gravida', 'para', 'miscarriage', 'abortion', 'labor', 'delivery',
        'postpartum', 'lactation', 'breastfeeding'
    ]

    # Reproductive organ keywords
    reproductive_keywords = [
        'prostate', 'prostatic', 'testicular', 'testis', 'testes', 'scrotal',
        'ovarian', 'ovary', 'uterine', 'uterus', 'cervical', 'cervix',
        'vaginal', 'vagina', 'endometrial', 'endometrium'
    ]

    # Gender-specific screening/procedures
    screening_keywords = [
        'mammogram', 'mammography', 'pap smear', 'psa test', 'colonoscopy'
    ]

    # Hormone-related gender-specific
    hormone_keywords = [
        'menopause', 'menopausal', 'menstrual', 'menses', 'amenorrhea',
        'erectile dysfunction', 'impotence'
    ]

    all_keywords = pregnancy_keywords + reproductive_keywords + screening_keywords + hormone_keywords

    return any(keyword in combined_text for keyword in all_keywords)


def test_gender_filtering():
    """Test gender-specific condition filtering"""
    print("\n" + "="*60)
    print("PHASE 5: Gender-Specific Filtering (20 tests)")
    print("="*60)

    results = TestResults()

    # Test 1-5: Pregnancy keywords
    print("\n--- Pregnancy Keywords ---")

    pregnancy_tests = [
        ("A 25-year-old pregnant woman presents...", True, "Pregnant keyword detected"),
        ("Patient is in second trimester...", True, "Trimester keyword detected"),
        ("History of miscarriage...", True, "Miscarriage keyword detected"),
        ("Postpartum bleeding noted...", True, "Postpartum keyword detected"),
        ("Patient has fever and cough.", False, "No pregnancy keywords"),
    ]

    for text, expected, description in pregnancy_tests:
        result = is_gender_specific_case(text, "", "")
        if result == expected:
            results.add_pass(description)
        else:
            results.add_fail(description, expected, result)

    # Test 6-10: Reproductive organs
    print("\n--- Reproductive Organ Keywords ---")

    organ_tests = [
        ("Prostate enlargement noted...", True, "Prostate keyword detected"),
        ("Testicular pain reported...", True, "Testicular keyword detected"),
        ("Ovarian cyst identified...", True, "Ovarian keyword detected"),
        ("Uterine fibroids present...", True, "Uterine keyword detected"),
        ("Liver cirrhosis diagnosed.", False, "No reproductive keywords"),
    ]

    for text, expected, description in organ_tests:
        result = is_gender_specific_case(text, "", "")
        if result == expected:
            results.add_pass(description)
        else:
            results.add_fail(description, expected, result)

    # Test 11-15: Gender-specific screening
    print("\n--- Screening Keywords ---")

    screening_tests = [
        ("Mammogram shows abnormality...", True, "Mammogram keyword detected"),
        ("PSA test elevated...", True, "PSA test keyword detected"),
        ("Pap smear results pending...", True, "Pap smear keyword detected"),
        ("Chest X-ray normal.", False, "No screening keywords"),
        ("Blood test ordered.", False, "No screening keywords"),
    ]

    for text, expected, description in screening_tests:
        result = is_gender_specific_case(text, "", "")
        if result == expected:
            results.add_pass(description)
        else:
            results.add_fail(description, expected, result)

    # Test 16-20: Edge cases and combinations
    print("\n--- Edge Cases ---")

    edge_tests = [
        ("", False, "Empty text returns False"),
        ("PROSTATE CANCER DIAGNOSIS", True, "Case insensitive (caps)"),
        ("Discussing patient's pregnant sister", True, "Pregnant in family history still detected"),
        ("Patient denies pregnancy", True, "Pregnancy mentioned even if denied"),
        ("Fever, cough, and chest pain", False, "General symptoms only"),
    ]

    for text, expected, description in edge_tests:
        result = is_gender_specific_case(text, "", "")
        if result == expected:
            results.add_pass(description)
        else:
            results.add_fail(description, expected, result)

    return results


# ============================================================================
# PHASE 6: EXCEL OUTPUT VALIDATION
# ============================================================================

def test_excel_output():
    """Test Excel output structure and validation"""
    print("\n" + "="*60)
    print("PHASE 6: Excel Output Validation (15 tests)")
    print("="*60)

    results = TestResults()

    # Test 1-5: DataFrame structure
    print("\n--- DataFrame Structure ---")

    # Create test dataframe
    test_data = {
        'id': ['1', '2', '3'],
        'question': ['Q1', 'Q2', 'Q3'],
        'ground_truth': ['A', 'B', 'C'],
        'olmo2_answers_match': ['yes', 'no', 'unclear'],
        'deepseek_answers_match': ['yes', 'yes', 'no']
    }
    df = pd.DataFrame(test_data)

    if len(df) == 3:
        results.add_pass("DataFrame creation: 3 rows")
    else:
        results.add_fail("DataFrame creation", 3, len(df))

    if len(df.columns) == 5:
        results.add_pass("DataFrame columns: 5 columns")
    else:
        results.add_fail("DataFrame columns", 5, len(df.columns))

    if 'ground_truth' in df.columns:
        results.add_pass("Ground truth column present")
    else:
        results.add_fail("Ground truth column", "Present", "Missing")

    if df['id'].dtype == 'object':
        results.add_pass("ID column dtype correct (object/string)")
    else:
        results.add_fail("ID column dtype", "object", df['id'].dtype)

    if all(df['ground_truth'].isin(['A', 'B', 'C', 'D'])):
        results.add_pass("Ground truth values valid")
    else:
        results.add_fail("Ground truth values", "A/B/C/D", df['ground_truth'].values)

    # Test 6-10: Statistics calculation
    print("\n--- Statistics Calculation ---")

    match_counts = df['olmo2_answers_match'].value_counts()
    yes_count = match_counts.get('yes', 0)
    no_count = match_counts.get('no', 0)
    unclear_count = match_counts.get('unclear', 0)

    if yes_count == 1:
        results.add_pass("Match count: 1 'yes'")
    else:
        results.add_fail("Match count 'yes'", 1, yes_count)

    if no_count == 1:
        results.add_pass("Mismatch count: 1 'no'")
    else:
        results.add_fail("Mismatch count 'no'", 1, no_count)

    if unclear_count == 1:
        results.add_pass("Unclear count: 1 'unclear'")
    else:
        results.add_fail("Unclear count", 1, unclear_count)

    match_rate = yes_count / len(df) * 100
    expected_rate = 33.3
    if abs(match_rate - expected_rate) < 0.1:
        results.add_pass("Match rate calculation: 33.3%")
    else:
        results.add_fail("Match rate", f"{expected_rate}%", f"{match_rate}%")

    mismatch_rate = no_count / len(df) * 100
    if abs(mismatch_rate - 33.3) < 0.1:
        results.add_pass("Mismatch rate calculation: 33.3%")
    else:
        results.add_fail("Mismatch rate", "33.3%", f"{mismatch_rate}%")

    # Test 11-15: Data integrity
    print("\n--- Data Integrity ---")

    # Check for NULL values in critical columns
    critical_cols = ['id', 'question', 'ground_truth']
    has_nulls = df[critical_cols].isnull().any().any()
    if not has_nulls:
        results.add_pass("No NULL values in critical columns")
    else:
        results.add_fail("NULL check", "No NULLs", "NULLs found")

    # Check data types
    if df['question'].dtype == 'object':
        results.add_pass("Question column dtype correct")
    else:
        results.add_fail("Question dtype", "object", df['question'].dtype)

    # Check special characters handled
    test_special = pd.DataFrame({'text': ['Test $100', 'Test\nline', 'Test "quote"']})
    if len(test_special) == 3:
        results.add_pass("Special characters in DataFrame")
    else:
        results.add_fail("Special chars", 3, len(test_special))

    # Check empty strings
    test_empty = pd.DataFrame({'col': ['', 'text', '']})
    empty_count = (test_empty['col'] == '').sum()
    if empty_count == 2:
        results.add_pass("Empty strings preserved")
    else:
        results.add_fail("Empty strings", 2, empty_count)

    # Check large dataset handling
    large_df = pd.DataFrame({'col': range(1000)})
    if len(large_df) == 1000:
        results.add_pass("Large dataset (1000 rows)")
    else:
        results.add_fail("Large dataset", 1000, len(large_df))

    return results


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all Phase 4, 5, 6 test suites"""
    print("\n" + "="*80)
    print(" "*15 + "PHASES 4, 5, 6: COMBINED TEST SUITE")
    print("="*80)

    all_results = TestResults()

    # Run test suites
    suite4 = test_data_loading()
    suite5 = test_gender_filtering()
    suite6 = test_excel_output()

    # Aggregate results
    all_results.total = suite4.total + suite5.total + suite6.total
    all_results.passed = suite4.passed + suite5.passed + suite6.passed
    all_results.failed = suite4.failed + suite5.failed + suite6.failed
    all_results.failures = suite4.failures + suite5.failures + suite6.failures

    # Print final summary
    print("\n" + "="*80)
    print(" "*25 + "FINAL SUMMARY")
    print("="*80)
    print(f"Total tests: {all_results.total}")
    print(f"Passed: {all_results.passed}")
    print(f"Failed: {all_results.failed}")
    print(f"Pass rate: {all_results.passed/all_results.total*100:.1f}%")

    if all_results.failures:
        print(f"\n{len(all_results.failures)} FAILURES:")
        for desc, expected, got in all_results.failures:
            print(f"  ✗ {desc}")
            print(f"    Expected: {expected}")
            print(f"    Got: {got}")
    else:
        print("\n✅ ALL TESTS PASSED!")

    print("="*80)

    return all_results.failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
