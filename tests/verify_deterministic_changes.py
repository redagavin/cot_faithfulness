#!/usr/bin/env python3
"""
Verify Deterministic Changes Across All Analysis Scripts
Checks that:
1. All scripts use greedy decoding
2. All scripts have proper random seed initialization
3. Identical functions across file pairs are truly identical
"""

import re
import sys

def check_greedy_decoding(filepath):
    """Check that file uses greedy decoding (do_sample=False)"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check for do_sample=False
    if 'do_sample=False' not in content:
        return False, "Missing do_sample=False"

    # Check that there's no do_sample=True
    if 'do_sample=True' in content:
        return False, "Still has do_sample=True"

    # Check that there's no temperature parameter in generate()
    generate_calls = re.findall(r'model\.generate\([^)]+\)', content, re.DOTALL)
    for call in generate_calls:
        if 'temperature=' in call:
            return False, f"Still has temperature parameter in generate call"

    return True, "Greedy decoding confirmed"

def check_random_seeds(filepath):
    """Check that file has random seed initialization"""
    with open(filepath, 'r') as f:
        content = f.read()

    checks = []

    # Check for random.seed(42)
    if 'random.seed(42)' in content:
        checks.append("✓ random.seed(42)")
    else:
        checks.append("✗ Missing random.seed(42)")

    # Check for torch.manual_seed(42)
    if 'torch.manual_seed(42)' in content:
        checks.append("✓ torch.manual_seed(42)")
    else:
        checks.append("✗ Missing torch.manual_seed(42)")

    # Check for torch.cuda.manual_seed_all(42)
    if 'torch.cuda.manual_seed_all(42)' in content:
        checks.append("✓ torch.cuda.manual_seed_all(42)")
    else:
        checks.append("✗ Missing torch.cuda.manual_seed_all(42)")

    all_present = all('✓' in check for check in checks)
    return all_present, "; ".join(checks)

def extract_function(filepath, function_name):
    """Extract a function's full code from file"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Find function definition
    pattern = rf'^\s*def {function_name}\(.*?\):'
    match = re.search(pattern, content, re.MULTILINE)

    if not match:
        return None

    start_pos = match.start()
    lines = content[:start_pos].count('\n')

    # Extract the function by finding the next function or class definition
    remaining = content[start_pos:]
    function_lines = []
    in_function = False
    base_indent = None

    for line in remaining.split('\n'):
        if not in_function:
            in_function = True
            base_indent = len(line) - len(line.lstrip())
            function_lines.append(line)
        else:
            # Check if we've reached the end of the function
            if line.strip() and not line.strip().startswith('#'):
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent and (line.lstrip().startswith('def ') or line.lstrip().startswith('class ')):
                    break
            function_lines.append(line)

    return '\n'.join(function_lines)

def compare_functions(file1, file2, function_name):
    """Compare a function between two files"""
    func1 = extract_function(file1, function_name)
    func2 = extract_function(file2, function_name)

    if func1 is None and func2 is None:
        return True, f"Function {function_name} not found in either file"

    if func1 is None:
        return False, f"Function {function_name} missing in {file1}"

    if func2 is None:
        return False, f"Function {function_name} missing in {file2}"

    # Normalize whitespace for comparison
    func1_normalized = re.sub(r'\s+', ' ', func1.strip())
    func2_normalized = re.sub(r'\s+', ' ', func2.strip())

    if func1_normalized == func2_normalized:
        return True, f"✓ {function_name} identical"
    else:
        return False, f"✗ {function_name} differs between files"

def main():
    """Run all verification checks"""
    print("="*70)
    print("VERIFICATION: Deterministic Changes in All Analysis Scripts")
    print("="*70)

    files = [
        'bhcs_analysis.py',
        'bhcs_baseline_analysis.py',
        'diagnosis_arena_analysis.py',
        'diagnosis_arena_baseline_analysis.py',
        'medxpertqa_analysis.py',
        'medxpertqa_baseline_analysis.py'
    ]

    all_passed = True

    # Check 1: Greedy decoding in all files
    print("\n## 1. Greedy Decoding Verification")
    print("-" * 70)
    for filepath in files:
        passed, message = check_greedy_decoding(filepath)
        status = "✓" if passed else "✗"
        print(f"{status} {filepath}: {message}")
        if not passed:
            all_passed = False

    # Check 2: Random seeds in all files
    print("\n## 2. Random Seed Initialization")
    print("-" * 70)
    for filepath in files:
        passed, message = check_random_seeds(filepath)
        status = "✓" if passed else "✗"
        print(f"{status} {filepath}")
        print(f"   {message}")
        if not passed:
            all_passed = False

    # Check 3: Identical functions across pairs
    print("\n## 3. Identical Functions Across File Pairs")
    print("-" * 70)

    pairs = [
        ('diagnosis_arena_analysis.py', 'diagnosis_arena_baseline_analysis.py'),
        ('medxpertqa_analysis.py', 'medxpertqa_baseline_analysis.py')
    ]

    functions_to_compare = [
        'detect_patient_gender',
        'load_data',
        'filter_by_gender_and_conditions',  # diagnosis_arena
        'filter_dataset',  # medxpertqa
    ]

    for file1, file2 in pairs:
        print(f"\nComparing: {file1} ↔ {file2}")
        for func_name in functions_to_compare:
            passed, message = compare_functions(file1, file2, func_name)
            print(f"  {message}")
            if not passed and 'not found in either' not in message:
                all_passed = False

    # Check 4: Generation parameters identical across pairs
    print("\n## 4. Generation Parameters Identical Across Pairs")
    print("-" * 70)

    for file1, file2 in pairs:
        print(f"\nComparing: {file1} ↔ {file2}")
        passed, message = compare_functions(file1, file2, 'generate_response')
        print(f"  {message}")
        if not passed:
            all_passed = False

    # Final summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL VERIFICATION CHECKS PASSED")
        print("="*70)
        print("\nAll scripts are properly configured for deterministic generation:")
        print("  - Greedy decoding (do_sample=False)")
        print("  - Random seeds initialized (random, torch, cuda)")
        print("  - Critical functions identical across file pairs")
        print("\nReady for production use!")
        return 0
    else:
        print("❌ VERIFICATION FAILED")
        print("="*70)
        print("\nSome checks did not pass. Review the output above.")
        print("Fix any issues before launching production jobs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
