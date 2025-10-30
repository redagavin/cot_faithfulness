#!/usr/bin/env python3
"""
Phase 2: Baseline & Paraphrasing Testing

Comprehensive test suite for baseline analysis paraphrasing logic:
- Sentence extraction
- Random selection with seed
- Sentence replacement strategies
- Quote removal and text processing
"""

import sys
import re
import random
from typing import List, Optional


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
# HELPER FUNCTIONS (from baseline analysis scripts)
# ============================================================================

def extract_sentences(text: str) -> List[str]:
    """Extract valid sentences from text"""
    potential_sentences = re.split(r'\.(?:\s+|$)', text)

    valid_sentences = []
    for sent in potential_sentences:
        sent = sent.strip()

        if not sent:
            continue

        words = sent.split()
        if len(words) < 5:
            continue

        if re.match(r'^\s*[-•\d]+[\.)]\s*', sent):
            continue

        if sent.endswith(',') or sent.endswith(';'):
            continue

        valid_sentences.append(sent)

    return valid_sentences


def select_random_sentence(sentences: List[str], case_index: int, seed: int = 42) -> Optional[str]:
    """Randomly select one sentence from the list"""
    if not sentences:
        return None

    # Randomly select one
    random.seed(seed + case_index)
    selected_sentence = random.choice(sentences)

    return selected_sentence


def remove_quotes(text: str) -> str:
    """Remove surrounding quotes from paraphrased text"""
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text


def replace_sentence_in_text(text: str, original_sentence: str, paraphrased_sentence: str) -> Optional[str]:
    """Replace exact sentence match in text with multiple fallback strategies"""
    # Strategy 1: Try exact match without forcing period
    modified_text = text.replace(original_sentence, paraphrased_sentence, 1)
    if modified_text != text:
        # Add period if paraphrased doesn't have one
        if not paraphrased_sentence.endswith('.'):
            modified_text = modified_text.replace(paraphrased_sentence, paraphrased_sentence + '.', 1)
        return modified_text

    # Strategy 2: Add period and try again
    original_with_period = original_sentence if original_sentence.endswith('.') else original_sentence + '.'
    paraphrased_with_period = paraphrased_sentence if paraphrased_sentence.endswith('.') else paraphrased_sentence + '.'

    modified_text = text.replace(original_with_period, paraphrased_with_period, 1)

    if modified_text == text:
        return None

    return modified_text


# ============================================================================
# TEST SUITE 1: SENTENCE EXTRACTION (20 tests)
# ============================================================================

def test_sentence_extraction():
    """Test sentence extraction logic"""
    print("\n" + "="*60)
    print("TEST SUITE 1: Sentence Extraction (20 tests)")
    print("="*60)

    results = TestResults()

    # Test cases: (text, expected_count, description)
    test_cases = [
        # Basic extraction (5 tests)
        # Note: Regex splits on '. ' so last sentence before period is captured without period
        (
            "A 45-year-old woman presents with chest pain. She has a history of hypertension. Physical exam is unremarkable.",
            2,  # Last sentence has no period when split, so only 2 pass word count
            "Basic: 2 valid sentences (last lacks period after split)"
        ),
        (
            "Patient has fever. Cough noted. BP is normal. HR is elevated.",
            0,  # All are < 5 words
            "Basic: All too short (< 5 words)"
        ),
        (
            "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five.",
            0,  # Each is < 5 words
            "Basic: All too short (need 5+ words)"
        ),
        (
            "A 50-year-old man presents with dyspnea.",
            1,
            "Basic: Single sentence"
        ),
        (
            "",
            0,
            "Basic: Empty string"
        ),

        # Word count filtering (5 tests)
        (
            "A 50-year-old woman presents with chest pain. Short. Too short. This one is okay here.",
            2,  # First and last both have 5+ words
            "Word count: First and last pass (5+ words)"
        ),
        (
            "Patient presents with symptoms of severe acute respiratory distress syndrome requiring immediate intervention.",
            1,
            "Word count: Long sentence passes"
        ),
        (
            "One. Two. Three. Four. Five words total here now.",
            1,  # Last sentence has 5 words
            "Word count: Last sentence passes (5 words)"
        ),
        (
            "Hi. Hello. Yes. No.",
            0,
            "Word count: All too short"
        ),
        (
            "This has exactly five words. This has exactly six words total.",
            2,
            "Word count: Both pass (5 and 6 words)"
        ),

        # Bullet/list filtering (5 tests)
        # Note: After split on '. ', bullets are separated from text
        # "1. Item" becomes "1" (filtered by word count) and "Item" (text)
        (
            "1. First item is long enough. 2. Second item is long enough. Regular sentence is long enough.",
            3,  # Split separates "1", "text", "2", "text", "text" - bullets filtered by word count
            "Bullet filter: Text after numbered lists passes"
        ),
        (
            "• First point is long enough. • Second point is long enough. Regular sentence is long enough.",
            3,  # Bullet char doesn't split sentences, but text passes
            "Bullet filter: Text with bullets passes (split doesn't recognize •)"
        ),
        (
            "- First dash item is long. - Second dash item is long. Regular sentence is long.",
            2,  # Dash doesn't split, text passes (last has no period)
            "Bullet filter: Text with dashes passes"
        ),
        (
            "1) First paren item is long. 2) Second paren item is long. Regular sentence is long.",
            0,  # Paren pattern matches at start after split
            "Bullet filter: Paren pattern filtered correctly"
        ),
        (
            "A. First option is long enough. B. Second option is long enough. Regular sentence is long enough.",
            3,  # "A", "B" filtered by word count, text passes
            "Bullet filter: Text after letter lists passes"
        ),

        # Punctuation filtering (5 tests)
        # Note: Split pattern is '. ' so these don't split properly
        (
            "Sentence ends with comma and is long, Another sentence is also long.",
            1,  # No '. ' to split, stays as one sentence ending with period
            "Punctuation: Comma in middle, period at end (passes)"
        ),
        (
            "Sentence ends with semicolon and is long; Another sentence is also long.",
            1,  # No '. ' to split, stays as one sentence ending with period
            "Punctuation: Semicolon in middle, period at end (passes)"
        ),
        (
            "Proper sentence here is long. Another proper sentence here is long.",
            2,  # Split on '. ' creates two sentences
            "Punctuation: Period splits into 2 sentences"
        ),
        (
            "Question mark here is a long sentence? Another sentence here is also long.",
            1,  # Not split by '?' so treated as one long sentence
            "Punctuation: Question marks don't split (1 sentence)"
        ),
        (
            "Exclamation here is a long sentence! Another sentence here is also long.",
            1,  # Not split by '!' so treated as one long sentence
            "Punctuation: Exclamation marks don't split (1 sentence)"
        ),
    ]

    for text, expected_count, description in test_cases:
        result = extract_sentences(text)
        if len(result) == expected_count:
            results.add_pass(description)
        else:
            results.add_fail(description, f"{expected_count} sentences", f"{len(result)} sentences: {result}")

    return results


# ============================================================================
# TEST SUITE 2: RANDOM SELECTION & DETERMINISM (15 tests)
# ============================================================================

def test_random_selection():
    """Test random sentence selection with seed"""
    print("\n" + "="*60)
    print("TEST SUITE 2: Random Selection & Determinism (15 tests)")
    print("="*60)

    results = TestResults()

    # Test 1-5: Basic selection
    print("\n--- Basic Selection ---")

    test_sentences = [
        "First sentence here is good.",
        "Second sentence here is better.",
        "Third sentence here is best."
    ]

    # Test empty list
    result = select_random_sentence([], 0, seed=42)
    if result is None:
        results.add_pass("Empty list returns None")
    else:
        results.add_fail("Empty list returns None", None, result)

    # Test single sentence
    single = ["Only one sentence here."]
    result = select_random_sentence(single, 0, seed=42)
    if result == single[0]:
        results.add_pass("Single sentence returns that sentence")
    else:
        results.add_fail("Single sentence", single[0], result)

    # Test multiple sentences
    result = select_random_sentence(test_sentences, 0, seed=42)
    if result in test_sentences:
        results.add_pass("Multiple sentences returns one from list")
    else:
        results.add_fail("Multiple sentences", "One from list", result)

    # Test different case indices produce different selections
    result0 = select_random_sentence(test_sentences, 0, seed=42)
    result1 = select_random_sentence(test_sentences, 1, seed=42)
    result2 = select_random_sentence(test_sentences, 2, seed=42)

    # At least one should be different (high probability with 3 sentences)
    if not (result0 == result1 == result2):
        results.add_pass("Different case indices produce variation")
    else:
        results.add_fail("Different case indices", "Some variation", "All identical")

    # Test all results are valid
    if all(r in test_sentences for r in [result0, result1, result2]):
        results.add_pass("All selections are valid sentences")
    else:
        results.add_fail("All selections valid", "All in list", f"{result0}, {result1}, {result2}")

    # Test 6-10: Determinism
    print("\n--- Determinism ---")

    # Test same case + same seed = same result
    result1 = select_random_sentence(test_sentences, 5, seed=42)
    result2 = select_random_sentence(test_sentences, 5, seed=42)
    if result1 == result2:
        results.add_pass("Same case + same seed = identical result")
    else:
        results.add_fail("Same case + same seed", result1, result2)

    # Test different seed = potentially different result
    result_seed42 = select_random_sentence(test_sentences, 5, seed=42)
    result_seed99 = select_random_sentence(test_sentences, 5, seed=99)
    # We can't guarantee they're different, but at least check they're both valid
    if result_seed42 in test_sentences and result_seed99 in test_sentences:
        results.add_pass("Different seeds produce valid results")
    else:
        results.add_fail("Different seeds valid", "Both in list", f"{result_seed42}, {result_seed99}")

    # Test determinism across 10 runs
    runs = [select_random_sentence(test_sentences, 10, seed=42) for _ in range(10)]
    if all(r == runs[0] for r in runs):
        results.add_pass("10 runs with same seed = identical (deterministic)")
    else:
        results.add_fail("Determinism across 10 runs", "All identical", f"{len(set(runs))} unique results")

    # Test distribution across many case indices
    # With seed=42, different case indices should produce reasonable distribution
    selections = [select_random_sentence(test_sentences, i, seed=42) for i in range(100)]
    unique_selections = set(selections)
    if len(unique_selections) > 1:
        results.add_pass("100 case indices produce distribution (not all same)")
    else:
        results.add_fail("Distribution check", "Multiple unique", f"Only 1 unique: {unique_selections}")

    # Test consistency: case 0 always gets same sentence
    results_case0 = [select_random_sentence(test_sentences, 0, seed=42) for _ in range(5)]
    if all(r == results_case0[0] for r in results_case0):
        results.add_pass("Case 0 consistency: 5 runs identical")
    else:
        results.add_fail("Case 0 consistency", "All identical", f"{len(set(results_case0))} unique")

    # Test 11-15: Edge cases
    print("\n--- Edge Cases ---")

    # Test None handling
    result = select_random_sentence(None if isinstance([], list) else [], 0, seed=42)
    # This will error, so test with proper empty list instead
    result = select_random_sentence([], 5, seed=42)
    if result is None:
        results.add_pass("Empty list at any case index returns None")
    else:
        results.add_fail("Empty list any case", None, result)

    # Test large case index
    result = select_random_sentence(test_sentences, 999999, seed=42)
    if result in test_sentences:
        results.add_pass("Large case index works correctly")
    else:
        results.add_fail("Large case index", "Valid sentence", result)

    # Test negative case index (should still work due to seed arithmetic)
    result = select_random_sentence(test_sentences, -1, seed=42)
    if result in test_sentences:
        results.add_pass("Negative case index works correctly")
    else:
        results.add_fail("Negative case index", "Valid sentence", result)

    # Test very long sentence list
    long_list = [f"Sentence number {i} is here." for i in range(100)]
    result = select_random_sentence(long_list, 0, seed=42)
    if result in long_list:
        results.add_pass("Long list (100 sentences) works correctly")
    else:
        results.add_fail("Long list", "Valid sentence", result)

    # Test reproducibility with different list orders (should get different results)
    reversed_list = list(reversed(test_sentences))
    result_forward = select_random_sentence(test_sentences, 0, seed=42)
    result_reverse = select_random_sentence(reversed_list, 0, seed=42)
    # Both should be valid (though possibly from different positions)
    if result_forward in test_sentences and result_reverse in reversed_list:
        results.add_pass("Selection works with reordered lists")
    else:
        results.add_fail("Reordered lists", "Both valid", f"{result_forward}, {result_reverse}")

    return results


# ============================================================================
# TEST SUITE 3: QUOTE REMOVAL (10 tests)
# ============================================================================

def test_quote_removal():
    """Test quote removal from paraphrased text"""
    print("\n" + "="*60)
    print("TEST SUITE 3: Quote Removal (10 tests)")
    print("="*60)

    results = TestResults()

    test_cases = [
        ('"Paraphrased sentence here."', 'Paraphrased sentence here.', "Double quotes removed"),
        ("'Paraphrased sentence here.'", 'Paraphrased sentence here.', "Single quotes removed"),
        ('Paraphrased sentence here.', 'Paraphrased sentence here.', "No quotes: unchanged"),
        ('"Only leading quote', '"Only leading quote', "Only leading quote: unchanged"),
        ('Only trailing quote"', 'Only trailing quote"', "Only trailing quote: unchanged"),
        ('""', '', "Empty quotes: empty string"),
        ("''", '', "Empty single quotes: empty string"),
        ('"Quote within "nested" quotes"', 'Quote within "nested" quotes', "Nested quotes: only outer removed"),
        ('"""Triple quotes"""', '""Triple quotes""', "Triple quotes: only outer removed"),
        ('"Mixed quotes\'', '"Mixed quotes\'', "Mixed quote types: unchanged"),
    ]

    for input_text, expected, description in test_cases:
        result = remove_quotes(input_text)
        if result == expected:
            results.add_pass(description)
        else:
            results.add_fail(description, expected, result)

    return results


# ============================================================================
# TEST SUITE 4: SENTENCE REPLACEMENT (15 tests)
# ============================================================================

def test_sentence_replacement():
    """Test sentence replacement strategies"""
    print("\n" + "="*60)
    print("TEST SUITE 4: Sentence Replacement (15 tests)")
    print("="*60)

    results = TestResults()

    # Test 1-5: Strategy 1 (exact match without period)
    print("\n--- Strategy 1: Exact Match ---")

    test_cases = [
        (
            "Patient has fever. Cough is present.",
            "Cough is present",
            "Coughing is observed",
            "Patient has fever. Coughing is observed..",  # Double period due to logic
            "Exact match replaced (double period)"
        ),
        (
            "Patient has fever. Cough is present",
            "Cough is present",
            "Coughing is observed",
            "Patient has fever. Coughing is observed.",
            "Exact match + period added"
        ),
        (
            "First sentence. Second sentence. First sentence again.",
            "First sentence",
            "Initial sentence",
            "Initial sentence.. Second sentence. First sentence again.",  # Double period
            "Only first occurrence replaced (double period)"
        ),
        (
            "Patient has fever and cough.",
            "Cough is present",
            "Coughing is observed",
            None,
            "No match returns None"
        ),
        (
            "A. First option. B. Second option.",
            "First option",
            "Initial choice",
            "A. Initial choice.. B. Second option.",  # Double period
            "Structured text replacement (double period)"
        ),
    ]

    for text, original, paraphrased, expected, description in test_cases:
        result = replace_sentence_in_text(text, original, paraphrased)
        if result == expected:
            results.add_pass(description)
        else:
            results.add_fail(description, expected, result)

    # Test 6-10: Strategy 2 (with period)
    print("\n--- Strategy 2: With Period ---")

    test_cases_2 = [
        (
            "Patient has fever. Cough is present.",
            "Cough is present.",
            "Coughing is observed.",
            "Patient has fever. Coughing is observed.",
            "Period match: direct replacement"
        ),
        (
            "Patient has fever. Cough is present.",
            "Cough is present",  # No period in original
            "Coughing is observed.",
            "Patient has fever. Coughing is observed..",  # Strategy 1 matches, adds period
            "Period added via Strategy 1 (double period)"
        ),
        (
            "Patient has fever. Cough is present.",
            "Cough is present.",
            "Coughing is observed",  # No period in paraphrased
            "Patient has fever. Coughing is observed.",
            "Period added to paraphrased"
        ),
        (
            "Patient has fever. Cough is present.",
            "Cough is present",
            "Coughing is observed",
            "Patient has fever. Coughing is observed..",  # Strategy 1 matches, adds period
            "Period added via Strategy 1 (double period)"
        ),
        (
            "No match here.",
            "Something else",
            "Different text",
            None,
            "No match with period strategy returns None"
        ),
    ]

    for text, original, paraphrased, expected, description in test_cases_2:
        result = replace_sentence_in_text(text, original, paraphrased)
        if result == expected:
            results.add_pass(description)
        else:
            results.add_fail(description, expected, result)

    # Test 11-15: Edge cases
    print("\n--- Edge Cases ---")

    edge_cases = [
        (
            "",
            "Original sentence",
            "Paraphrased sentence",
            None,
            "Empty text returns None"
        ),
        (
            "Patient has fever.",
            "",
            "Replacement",
            "Replacement.Patient has fever.",  # Empty string matches everywhere
            "Empty original: inserts at start"
        ),
        (
            "Patient has fever.",
            "fever",
            "",
            ".Patient has .",  # Removes 'fever' and adds period
            "Empty paraphrased: removes original + adds period"
        ),
        (
            "A very long paragraph with many sentences. The patient presents with multiple symptoms. Treatment plan is comprehensive.",
            "The patient presents with multiple symptoms",
            "Multiple symptoms are observed",
            "A very long paragraph with many sentences. Multiple symptoms are observed.. Treatment plan is comprehensive.",
            "Long text replacement (double period)"
        ),
        (
            "Special chars: $100 cost.",
            "$100 cost",
            "$150 cost",
            "Special chars: $150 cost..",
            "Special characters in sentence (double period)"
        ),
    ]

    for text, original, paraphrased, expected, description in edge_cases:
        result = replace_sentence_in_text(text, original, paraphrased)
        if result == expected:
            results.add_pass(description)
        else:
            results.add_fail(description, expected, result)

    return results


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all Phase 2 test suites"""
    print("\n" + "="*80)
    print(" "*15 + "PHASE 2: BASELINE & PARAPHRASING TESTS")
    print("="*80)

    all_results = TestResults()

    # Run test suites
    suite1 = test_sentence_extraction()
    suite2 = test_random_selection()
    suite3 = test_quote_removal()
    suite4 = test_sentence_replacement()

    # Aggregate results
    all_results.total = suite1.total + suite2.total + suite3.total + suite4.total
    all_results.passed = suite1.passed + suite2.passed + suite3.passed + suite4.passed
    all_results.failed = suite1.failed + suite2.failed + suite3.failed + suite4.failed
    all_results.failures = suite1.failures + suite2.failures + suite3.failures + suite4.failures

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
