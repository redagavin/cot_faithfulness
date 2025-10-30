#!/usr/bin/env python3
"""
Comprehensive Test Suite for Medical Analysis Infrastructure
Tests all critical components: extraction, gender detection, swapping, paraphrasing, etc.
"""

import sys
sys.path.insert(0, '/scratch/yang.zih/cot')

from bhcs_analysis import BHCSAnalyzer
from diagnosis_arena_analysis import DiagnosisArenaAnalyzer
from medxpertqa_analysis import MedXpertQAAnalyzer
from medqa_analysis import MedQAAnalyzer


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def add_pass(self, test_name):
        self.passed += 1
        self.tests.append((test_name, True, None))

    def add_fail(self, test_name, expected, got):
        self.failed += 1
        self.tests.append((test_name, False, f"Expected: {expected}, Got: {got}"))

    def print_summary(self):
        print("\n" + "=" * 80)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed out of {self.passed + self.failed} tests")

        if self.failed > 0:
            print("\nFailed tests:")
            for name, passed, msg in self.tests:
                if not passed:
                    print(f"  ✗ {name}: {msg}")

        return self.failed == 0


# ============================================================================
# UNIT TESTS: Answer Extraction
# ============================================================================

def test_answer_extraction_medqa():
    """Test MedQA/DiagnosisArena A-D answer extraction"""
    print("\n" + "=" * 80)
    print("UNIT TEST: MedQA Answer Extraction (A-D format)")
    print("=" * 80)

    analyzer = MedQAAnalyzer()
    results = TestResults()

    test_cases = [
        # Boxed format
        ("\\boxed{A}", "A", "Boxed format A"),
        ("\\boxed{B}", "B", "Boxed format B"),
        ("\\boxed{C}", "C", "Boxed format C"),
        ("\\boxed{D}", "D", "Boxed format D"),
        ("Final answer: \\boxed{B}", "B", "Boxed in sentence"),

        # Explicit answer format
        ("Final answer: A", "A", "Final answer: A"),
        ("The answer is: B", "B", "The answer is: B"),
        ("Answer: C", "C", "Answer: C"),

        # Standalone letter
        ("After analysis, the correct option is C.", "C", "Standalone C"),
        ("Therefore, D.", "D", "Standalone D"),

        # Edge cases
        ("No clear answer", "Unclear", "No answer found"),
        ("", "Unclear", "Empty response"),
    ]

    for response, expected, description in test_cases:
        result = analyzer.extract_diagnosis_answer(response)
        if result == expected:
            results.add_pass(description)
            print(f"  ✓ {description}")
        else:
            results.add_fail(description, expected, result)
            print(f"  ✗ {description} - Expected: {expected}, Got: {result}")

    return results.print_summary()


def test_answer_extraction_medxpertqa():
    """Test MedXpertQA A-J answer extraction"""
    print("\n" + "=" * 80)
    print("UNIT TEST: MedXpertQA Answer Extraction (A-J format)")
    print("=" * 80)

    analyzer = MedXpertQAAnalyzer()
    results = TestResults()

    test_cases = [
        ("\\boxed{A}", "A", "Boxed A"),
        ("\\boxed{E}", "E", "Boxed E"),
        ("\\boxed{J}", "J", "Boxed J"),
        ("Final answer: F", "F", "Final answer: F"),
        ("The answer is: G", "G", "Answer is: G"),
        ("After analysis, H.", "H", "Standalone H"),
        ("No answer here", "Unclear", "No answer"),
    ]

    for response, expected, description in test_cases:
        result = analyzer.extract_answer(response)
        if result == expected:
            results.add_pass(description)
            print(f"  ✓ {description}")
        else:
            results.add_fail(description, expected, result)
            print(f"  ✗ {description} - Expected: {expected}, Got: {result}")

    return results.print_summary()


def test_answer_extraction_bhcs():
    """Test BHCS Yes/No answer extraction"""
    print("\n" + "=" * 80)
    print("UNIT TEST: BHCS Answer Extraction (Yes/No format)")
    print("=" * 80)

    analyzer = BHCSAnalyzer()
    results = TestResults()

    test_cases = [
        ("\\boxed{Yes}", "Yes", "Boxed Yes"),
        ("\\boxed{No}", "No", "Boxed No"),
        ("Final answer: Yes", "Yes", "Final answer: Yes"),
        ("The answer is: No", "No", "Answer is: No"),
        ("Therefore, the patient is at risk: Yes", "Yes", "Statement Yes"),
        ("Patient is not at risk: No", "No", "Statement No"),
        ("</think>\n\n**yes**", "Yes", "Deepseek bold yes"),
        ("</think>\n\n**no**", "No", "Deepseek bold no"),
        ("Unclear situation", "Unclear", "No clear answer"),
    ]

    for response, expected, description in test_cases:
        result = analyzer.extract_depression_risk_answer(response)
        if result == expected:
            results.add_pass(description)
            print(f"  ✓ {description}")
        else:
            results.add_fail(description, expected, result)
            print(f"  ✗ {description} - Expected: {expected}, Got: {result}")

    return results.print_summary()


# ============================================================================
# UNIT TESTS: Gender Detection
# ============================================================================

def test_gender_detection():
    """Test patient gender detection"""
    print("\n" + "=" * 80)
    print("UNIT TEST: Gender Detection")
    print("=" * 80)

    analyzer = MedQAAnalyzer()
    results = TestResults()

    test_cases = [
        # Clear female cases
        ("A 45-year-old woman presents with...", "female", "Woman pattern"),
        ("A 23-year-old girl comes to...", "female", "Girl pattern"),
        ("The female patient shows...", "female", "Female pattern"),
        ("A woman reports she has been experiencing symptoms. She also notes...", "female", "Woman + pronouns (2+ she)"),

        # Clear male cases
        ("A 45-year-old man presents with...", "male", "Man pattern"),
        ("A 15-year-old boy comes to...", "male", "Boy pattern"),
        ("The male patient shows...", "male", "Male pattern"),
        ("A man reports he has been experiencing symptoms. He also notes...", "male", "Man + pronouns (2+ he)"),

        # Unclear cases - insufficient evidence
        ("The patient presents with...", "unclear", "No gender mentioned"),
        ("A 50-year-old presents with...", "unclear", "Age only"),
        ("", "unclear", "Empty text"),
        ("She has been experiencing symptoms", "unclear", "Single pronoun only (insufficient)"),
        ("He has been experiencing symptoms", "unclear", "Single pronoun only (insufficient)"),
    ]

    for text, expected, description in test_cases:
        result = analyzer.detect_patient_gender(text)
        if result == expected:
            results.add_pass(description)
            print(f"  ✓ {description}")
        else:
            results.add_fail(description, expected, result)
            print(f"  ✗ {description} - Expected: {expected}, Got: {result}")

    return results.print_summary()


# ============================================================================
# UNIT TESTS: Gender Swapping
# ============================================================================

def test_gender_swapping():
    """Test gender swapping functionality"""
    print("\n" + "=" * 80)
    print("UNIT TEST: Gender Swapping")
    print("=" * 80)

    analyzer = MedQAAnalyzer()
    results = TestResults()

    test_cases = [
        # Female to Male - PATIENT gender only
        ("A 45-year-old woman presents", "female", "A 45-year-old man presents", "Woman → Man"),
        ("She has symptoms", "female", "He has symptoms", "She → He"),
        ("Her condition worsens", "female", "His condition worsens", "Her → His"),
        ("The woman reports fever", "female", "The man reports fever", "The woman → The man"),

        # Male to Female - PATIENT gender only
        ("A 45-year-old man presents", "male", "A 45-year-old woman presents", "Man → Woman"),
        ("He has symptoms", "male", "She has symptoms", "He → She"),
        ("His condition worsens", "male", "Her condition worsens", "His → Her"),
        ("The man reports fever", "male", "The woman reports fever", "The man → The woman"),

        # Family members should NOT swap (experimental control)
        ("A woman whose mother has diabetes", "female", "A man whose mother has diabetes", "Mother stays mother"),
        ("A man whose father has cancer", "male", "A woman whose father has cancer", "Father stays father"),

        # Medical terms should NOT change
        ("Testicular cancer", "male", "Testicular cancer", "Medical term preserved"),
        ("Prostate examination", "male", "Prostate examination", "Anatomical term preserved"),
    ]

    for original, gender, expected, description in test_cases:
        result = analyzer.apply_gender_swap(original, gender)
        # For family member tests, check that family term is preserved
        if "stays" in description:
            family_term = description.split()[0]  # "Mother" or "Father"
            if family_term.lower() in result.lower() and expected.split()[-2].lower() in result.lower():
                results.add_pass(description)
                print(f"  ✓ {description}")
            else:
                results.add_fail(description, expected, result)
                print(f"  ✗ {description}")
                print(f"      Original: {original}")
                print(f"      Result:   {result}")
        else:
            # Standard check - expected substring in result
            if expected in result:
                results.add_pass(description)
                print(f"  ✓ {description}")
            else:
                results.add_fail(description, expected, result)
                print(f"  ✗ {description}")
                print(f"      Original: {original}")
                print(f"      Result:   {result}")

    return results.print_summary()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_end_to_end_gender_pipeline():
    """Test complete gender analysis pipeline"""
    print("\n" + "=" * 80)
    print("INTEGRATION TEST: Gender Analysis Pipeline")
    print("=" * 80)

    results = TestResults()

    # Test case: Female patient question
    test_question = "A 35-year-old woman presents with fever and cough."

    analyzer = MedQAAnalyzer()

    # Step 1: Detect gender
    detected_gender = analyzer.detect_patient_gender(test_question)
    if detected_gender == "female":
        results.add_pass("Gender detection")
        print("  ✓ Gender detection: female")
    else:
        results.add_fail("Gender detection", "female", detected_gender)
        print(f"  ✗ Gender detection failed: {detected_gender}")

    # Step 2: Apply gender swap
    swapped = analyzer.apply_gender_swap(test_question, detected_gender)
    if "man" in swapped and "woman" not in swapped:
        results.add_pass("Gender swapping")
        print("  ✓ Gender swapping: woman → man")
    else:
        results.add_fail("Gender swapping", "contains 'man'", swapped)
        print(f"  ✗ Gender swapping failed: {swapped}")

    # Step 3: Verify medical terms preserved
    if "fever and cough" in swapped:
        results.add_pass("Medical terms preserved")
        print("  ✓ Medical terms preserved")
    else:
        results.add_fail("Medical terms preserved", "fever and cough", swapped)
        print(f"  ✗ Medical terms not preserved")

    return results.print_summary()


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "COMPREHENSIVE TEST SUITE" + " " * 39 + "║")
    print("╚" + "═" * 78 + "╝")

    all_passed = True

    # Unit tests
    print("\n" + "▶" * 40)
    print("UNIT TESTS")
    print("▶" * 40)

    tests = [
        ("MedQA Answer Extraction", test_answer_extraction_medqa),
        ("MedXpertQA Answer Extraction", test_answer_extraction_medxpertqa),
        ("BHCS Answer Extraction", test_answer_extraction_bhcs),
        ("Gender Detection", test_gender_detection),
        ("Gender Swapping", test_gender_swapping),
    ]

    for name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
                print(f"\n❌ {name} FAILED")
        except Exception as e:
            all_passed = False
            print(f"\n❌ {name} ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Integration tests
    print("\n" + "▶" * 40)
    print("INTEGRATION TESTS")
    print("▶" * 40)

    try:
        if not test_end_to_end_gender_pipeline():
            all_passed = False
    except Exception as e:
        all_passed = False
        print(f"\n❌ Integration test ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("Infrastructure is ready for production use")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Review failures above before production use")
        return 1


if __name__ == "__main__":
    sys.exit(main())
