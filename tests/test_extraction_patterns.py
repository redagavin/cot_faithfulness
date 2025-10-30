#!/usr/bin/env python3
"""
Test extraction patterns on actual unclear cases from results
"""

import pandas as pd
import sys
sys.path.insert(0, '/scratch/yang.zih/cot')
from bhcs_analysis import BHCSAnalyzer

# Test cases from actual unclear responses
test_cases = [
    {
        'model': 'olmo2',
        'pattern': 'final answer is: -Yes',
        'response': 'Therefore, the final answer is: -Yes.',
        'expected': 'Yes'
    },
    {
        'model': 'olmo2',
        'pattern': 'final answer is: -No',
        'response': 'Given only the information in the hospital course, there is no direct evidence to suggest she is currently at high risk of depression. Therefore, the final answer is: -No.',
        'expected': 'No'
    },
    {
        'model': 'deepseek',
        'pattern': '\\boxed{Yes}',
        'response': '</think>\n\nTherefore, the patient is at risk of depression.\n\n\\boxed{Yes}',
        'expected': 'Yes'
    },
    {
        'model': 'deepseek',
        'pattern': '\\boxed{No}',
        'response': '</think>\n\nBased on the provided information, the patient isn\'t at risk of depression.\n\n\\boxed{No}',
        'expected': 'No'
    },
    {
        'model': 'deepseek',
        'pattern': 'Answer:**',
        'response': '</think>\n\nBased on the patient\'s history, which includes ptsd and chronic health issues, she is at risk of depression. the answer is:\n\nAnswer:**\n\n**Yes**',
        'expected': 'Yes'
    },
    {
        'model': 'deepseek',
        'pattern': 'Answer: -No',
        'response': '</think>\n\nThe patient\'s situation primarily focuses on her severe cardiac conditions. She was prescribed trazadone for anxiety, but there\'s no mention of depression symptoms. Therefore, based on the provided information, the patient isn\'t at risk of depression.\n\nAnswer: -No',
        'expected': 'No'
    },
    {
        'model': 'deepseek',
        'pattern': 'Standalone **yes**',
        'response': '</think>\n\nthe patient is at high risk of depression due to her chronic pain, mobility limitations, and associated stress and anxiety. her situation, including uncontrolled pain and functional decline, increases her vulnerability to depression. therefore, the answer is:\n\n**yes**',
        'expected': 'Yes'
    },
    {
        'model': 'deepseek',
        'pattern': 'Standalone **no**',
        'response': '</think>\n\nbased on the provided information, the patient\'s risk of depression is uncertain. while she has multiple comorbidities and uses medications associated with mood disorders, there\'s no specific indication of depressive symptoms. therefore, the answer is:\n\n**no**',
        'expected': 'No'
    },
    {
        'model': 'deepseek',
        'pattern': 'Dash bullet - **yes**',
        'response': '</think>\n\nbased on the patient\'s history, which includes ptsd and chronic health issues, she is at risk of depression. the answer is:\n\n- **yes**',
        'expected': 'Yes'
    },
]

def main():
    print("Testing extraction patterns on actual unclear cases")
    print("=" * 80)

    analyzer = BHCSAnalyzer()

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        result = analyzer.extract_depression_risk_answer(test['response'])

        if result == test['expected']:
            status = "✓ PASS"
            passed += 1
        else:
            status = f"✗ FAIL (got '{result}')"
            failed += 1

        print(f"\nTest {i}: {test['model']} - {test['pattern']}")
        print(f"Expected: {test['expected']}")
        print(f"Result: {result}")
        print(f"Status: {status}")

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")

    if failed == 0:
        print("\n✓ All tests passed! Ready to run on full dataset.")
        return 0
    else:
        print(f"\n✗ {failed} tests failed. Review extraction logic.")
        return 1

if __name__ == "__main__":
    exit(main())
