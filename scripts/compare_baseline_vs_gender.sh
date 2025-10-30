#!/bin/bash

# Systematic comparison of baseline vs gender analysis files
# to ensure apple-to-apple comparison for scientific validity

echo "=========================================================================="
echo "SYSTEMATIC COMPARISON: Baseline vs Gender Analysis"
echo "Checking parts that MUST be identical for valid scientific comparison"
echo "=========================================================================="
echo ""

# Arrays of file pairs to compare
BHCS_GENDER="src/bhcs_analysis.py"
BHCS_BASELINE="src/bhcs_baseline_analysis.py"

DIAG_GENDER="src/diagnosis_arena_analysis.py"
DIAG_BASELINE="src/diagnosis_arena_baseline_analysis.py"

MEDX_GENDER="src/medxpertqa_analysis.py"
MEDX_BASELINE="src/medxpertqa_baseline_analysis.py"

# Function to compare a specific function between two files
compare_function() {
    local file1=$1
    local file2=$2
    local func_name=$3

    echo "=== Comparing: $func_name ==="

    # Extract function from both files
    func1=$(sed -n "/def $func_name/,/^    def \|^class \|^$/p" "$file1" 2>/dev/null)
    func2=$(sed -n "/def $func_name/,/^    def \|^class \|^$/p" "$file2" 2>/dev/null)

    if [ -z "$func1" ] && [ -z "$func2" ]; then
        echo "  ⚠️  Function not found in either file"
    elif [ -z "$func1" ]; then
        echo "  ❌ MISSING in $file1"
    elif [ -z "$func2" ]; then
        echo "  ❌ MISSING in $file2"
    elif [ "$func1" = "$func2" ]; then
        echo "  ✅ IDENTICAL"
    else
        echo "  ❌ DIFFERENT"
        # Show line counts for quick comparison
        lines1=$(echo "$func1" | wc -l)
        lines2=$(echo "$func2" | wc -l)
        echo "     $file1: $lines1 lines"
        echo "     $file2: $lines2 lines"
    fi
    echo ""
}

echo "=========================================================================="
echo "DIAGNOSIS ARENA: Gender vs Baseline"
echo "=========================================================================="
echo ""

echo "--- Critical Functions (MUST be identical) ---"
compare_function "$DIAG_GENDER" "$DIAG_BASELINE" "load_data"
compare_function "$DIAG_GENDER" "$DIAG_BASELINE" "detect_patient_gender"
compare_function "$DIAG_GENDER" "$DIAG_BASELINE" "filter_dataset"

echo "--- CoT Prompt (MUST be identical) ---"
grep -A 20 "self.cot_prompt = " "$DIAG_GENDER" > /tmp/diag_gender_prompt.txt
grep -A 20 "self.cot_prompt = " "$DIAG_BASELINE" > /tmp/diag_baseline_prompt.txt

if diff -q /tmp/diag_gender_prompt.txt /tmp/diag_baseline_prompt.txt > /dev/null 2>&1; then
    echo "  ✅ CoT prompts IDENTICAL"
else
    echo "  ❌ CoT prompts DIFFERENT"
fi
echo ""

echo "=========================================================================="
echo "BHCS: Gender vs Baseline"
echo "=========================================================================="
echo ""

echo "--- Critical Functions (MUST be identical) ---"
compare_function "$BHCS_GENDER" "$BHCS_BASELINE" "load_data"
compare_function "$BHCS_GENDER" "$BHCS_BASELINE" "extract_depression_risk_answer"

echo "--- CoT Prompt (MUST be identical) ---"
grep -A 30 "self.cot_prompt = " "$BHCS_GENDER" > /tmp/bhcs_gender_prompt.txt 2>/dev/null
grep -A 30 "self.cot_prompt = " "$BHCS_BASELINE" > /tmp/bhcs_baseline_prompt.txt 2>/dev/null

if diff -q /tmp/bhcs_gender_prompt.txt /tmp/bhcs_baseline_prompt.txt > /dev/null 2>&1; then
    echo "  ✅ CoT prompts IDENTICAL"
else
    echo "  ❌ CoT prompts DIFFERENT"
fi
echo ""

echo "=========================================================================="
echo "MEDXPERTQA: Gender vs Baseline"
echo "=========================================================================="
echo ""

echo "--- Critical Functions (MUST be identical) ---"
compare_function "$MEDX_GENDER" "$MEDX_BASELINE" "load_data"
compare_function "$MEDX_GENDER" "$MEDX_BASELINE" "detect_patient_gender"
compare_function "$MEDX_GENDER" "$MEDX_BASELINE" "filter_dataset"
compare_function "$MEDX_GENDER" "$MEDX_BASELINE" "extract_answer"

echo "--- CoT Prompt (MUST be identical) ---"
grep -A 20 "self.cot_prompt = " "$MEDX_GENDER" > /tmp/medx_gender_prompt.txt
grep -A 20 "self.cot_prompt = " "$MEDX_BASELINE" > /tmp/medx_baseline_prompt.txt

if diff -q /tmp/medx_gender_prompt.txt /tmp/medx_baseline_prompt.txt > /dev/null 2>&1; then
    echo "  ✅ CoT prompts IDENTICAL"
else
    echo "  ❌ CoT prompts DIFFERENT"
fi
echo ""

echo "=========================================================================="
echo "SUMMARY"
echo "=========================================================================="
echo ""
echo "This comparison checked functions that MUST be identical for valid"
echo "scientific comparison between baseline (paraphrase) and gender analyses."
echo ""
echo "Any differences found above should be investigated and fixed."
