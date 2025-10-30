# Baseline Paraphrase Analysis - Test Results Report

## Executive Summary

All three baseline test jobs completed successfully with valuable findings about model sensitivity to conservative paraphrasing.

### Test Completion Status
| Dataset | Status | Runtime | Success Rate | Key Finding |
|---------|--------|---------|--------------|-------------|
| BHCS | ✓ Complete | 11 min | 10/10 (100%) | **90% extraction issue** |
| DiagnosisArena | ✓ Complete | 13 min | 8/10 (80%) | **37.5% baseline flip rate** (Olmo2) |
| MedXpertQA | ✓ Complete | 30 min | 9/10 (90%) | **66.7% baseline flip rate** (Olmo2) |

---

## Detailed Results by Dataset

### 1. BHCS (Brief Hospital Course Summary)

**Paraphrasing Performance:** ✓ 100% (10/10)

**Model Sensitivity (CAUTION: Extraction Issue Detected):**
- **Olmo2 7B:** 10% match, 0% flip, **90% unclear**
- **Deepseek R1:** 20% match, 10% flip, **70% unclear**

**Critical Finding - Extraction Pattern Bug:**

The high "unclear" rates are due to missing extraction patterns in `extract_depression_risk_answer()`:

**Missing Patterns Identified:**
```
Olmo2 uses:    "the final answer is: Yes"    (note "the" before "final")
               "the final answer is: No" 
               "the final answer is -No"     (with dash)

Current code has: "final answer is:" but NOT "the final answer is:"
```

**Example Responses Not Extracted:**
- Case 0 Olmo2 Original: "...Therefore, **the final answer is: No.**"
- Case 1 Olmo2 Original: "...the patient is at risk of depression. So, **the final answer is: Yes.**"
- Case 2 Olmo2 Paraphrased: "...the patient is at risk of depression. **The final answer is: Yes.**"

**Fix Required:** Add these patterns to extraction logic:
```python
# Add to answer_patterns in extract_depression_risk_answer():
(r'the\s+final\s+answer\s+is:\s*yes\b', 'Yes'),
(r'the\s+final\s+answer\s+is:\s*no\b', 'No'),
(r'the\s+final\s+answer\s+is:\s*-yes\b', 'Yes'),
(r'the\s+final\s+answer\s+is:\s*-no\b', 'No'),
```

---

### 2. DiagnosisArena

**Paraphrasing Performance:** 80% (8/10)
- 2 failures at sentence replacement (exact match issue)

**Model Sensitivity Results:**

| Metric | Olmo2 7B | Deepseek R1 |
|--------|----------|-------------|
| **Match Rate** | 50.0% | 75.0% |
| **Flip Rate (Baseline)** | **37.5%** | **25.0%** |
| Unclear Rate | 12.5% | 0.0% |
| Original Correct | 12.5% | 37.5% |
| Paraphrased Correct | 25.0% | 62.5% |

**Flip Direction Analysis:**
- **Olmo2:** 1 wrong→correct (33%), 0 correct→wrong, 3 both wrong
- **Deepseek:** 2 wrong→correct (100%), 0 correct→wrong, 2 both wrong

**Interpretation:**
- Paraphrasing causes **37.5% answer flips** for Olmo2 (high sensitivity!)
- Flips are mostly on cases where model was already wrong
- Some flips actually improved correctness (wrong→correct)
- This establishes the **baseline sensitivity** to text perturbations

---

### 3. MedXpertQA

**Paraphrasing Performance:** 90% (9/10)
- 1 failure at sentence replacement

**Model Sensitivity Results:**

| Metric | Olmo2 7B | Deepseek R1 |
|--------|----------|-------------|
| **Match Rate** | 22.2% | 55.6% |
| **Flip Rate (Baseline)** | **66.7%** | **44.4%** |
| Unclear Rate | 11.1% | 0.0% |
| Original Correct | 11.1% | 22.2% |
| Paraphrased Correct | 11.1% | 22.2% |

**Flip Direction Analysis:**
- **Olmo2:** 0 wrong→correct, 1 correct→wrong (17%), 6 both wrong
- **Deepseek:** 1 wrong→correct (25%), 1 correct→wrong (25%), 4 both wrong

**Interpretation:**
- **VERY HIGH baseline flip rate** for Olmo2 (66.7%)!
- Olmo2 extremely sensitive to even conservative paraphrasing
- Deepseek more stable but still shows 44.4% flip rate
- Both models had low correctness overall (11-22%)

---

## Cross-Dataset Comparison

### Baseline Flip Rates (Paraphrase Sensitivity)

| Dataset | Olmo2 7B | Deepseek R1 |
|---------|----------|-------------|
| BHCS | N/A* | 10.0%* |
| DiagnosisArena | **37.5%** | **25.0%** |
| MedXpertQA | **66.7%** | **44.4%** |

*BHCS rates unreliable due to extraction bug

**Key Insights:**

1. **Olmo2 is HIGHLY sensitive** to text perturbations:
   - 37.5% flip rate on DiagnosisArena
   - 66.7% flip rate on MedXpertQA
   - This establishes that Olmo2 has inherent instability

2. **Deepseek is more stable** but still flips significantly:
   - 25.0% flip rate on DiagnosisArena  
   - 44.4% flip rate on MedXpertQA

3. **MedXpertQA shows highest sensitivity** for both models:
   - Possibly due to longer, more complex question texts
   - Or model uncertainty on these particular questions

---

## Comparison with Gender-Swap Analysis

**To determine if gender bias exists, we need to compare:**

| Dataset | Baseline Flip (Paraphrase) | Gender-Swap Flip | Interpretation |
|---------|----------------------------|------------------|----------------|
| DiagnosisArena | Olmo2: 37.5%, DS: 25.0% | **Need main results** | If gender-swap >> baseline: BIAS |
| MedXpertQA | Olmo2: 66.7%, DS: 44.4% | **Need main results** | If gender-swap ≈ baseline: NO BIAS (just sensitive) |

**Critical Question:**
- If gender-swap flip rate is **higher** than baseline → **True gender bias**
- If gender-swap flip rate is **similar** to baseline → **Just model instability**

**Example:**
- If DiagnosisArena gender-swap flip rate is 60% for Olmo2 (vs 37.5% baseline) → **Evidence of bias**
- If it's 35% (close to 37.5%) → **No bias, just normal sensitivity**

---

## Paraphrasing Quality Assessment

**Conservative Paraphrasing Examples:**

### BHCS:
```
Original:    "#L BKA Pre-operatively, she was started on vancomycin and zosyn..."
Paraphrased: "#L BKA She was started pre-operatively on vancomycin and zosyn..."
Change: Word order only, all medical terms preserved
```

### DiagnosisArena:
```
Original:    "He was born to healthy parents and did not have any medical disease."
Paraphrased: "He did not have any medical disease and was born to healthy parents."
Change: Clause reversal only
```

### MedXpertQA:
```
Original:    "His temperature is 103°F (39.4°C), blood pressure is 100/64 mmHg..."
Paraphrased: "His temperature measures 103°F (39.4°C), blood pressure is 100/64 mmHg..."
Change: "is" → "measures", minimal modification
```

**Assessment:** ✓ GPT-5 successfully performed very conservative paraphrasing
- All medical terminology preserved exactly
- Only syntax/word order changed
- Factual content identical

---

## Issues and Fixes Required

### CRITICAL: Fix BHCS Extraction Pattern

**Issue:** Missing "the final answer is:" pattern causes 70-90% unclear rate

**Fix Location:** `bhcs_baseline_analysis.py`, line ~569 in `extract_depression_risk_answer()`

**Add these patterns:**
```python
# In answer_patterns list, add:
(r'the\s+final\s+answer\s+is:\s*yes\b', 'Yes'),
(r'the\s+final\s+answer\s+is:\s*no\b', 'No'),  
(r'the\s+final\s+answer\s+is:\s*-yes\b', 'Yes'),
(r'the\s+final\s+answer\s+is:\s*-no\b', 'No'),
```

**Expected Result After Fix:**
- BHCS unclear rate should drop from 90%/70% to ~10%
- Should then show baseline flip rate for comparison

### MINOR: Improve Sentence Replacement

**Issue:** 2-3 cases per dataset fail at replacement step

**Possible Cause:** Exact string matching issues with whitespace/periods

**Recommendation:** 
- Current 80-90% success rate is acceptable for now
- Could improve with fuzzy matching later if needed

---

## Recommendations

### Immediate Actions:

1. **✅ Fix BHCS Extraction Bug**
   - Add missing patterns to extraction logic
   - Re-run BHCS baseline test (quick, 10 min)
   - Validate extraction works correctly

2. **✅ Review Main Analysis Results** 
   - Check if main gender-swap analyses have completed
   - Compare flip rates: gender-swap vs. baseline

3. **✅ Statistical Analysis**
   - Calculate if difference between gender-swap and baseline is significant
   - This determines whether bias exists vs. just model sensitivity

### Full Run Decision:

**Before launching full baseline runs:**
1. Fix BHCS extraction issue
2. Compare test results with main analysis results
3. Decide if baseline is needed (depends on whether main analysis shows high flip rates)

**Why this matters:**
- If main analysis shows LOW flip rates (e.g., 10-20%), baseline not critical
- If main analysis shows HIGH flip rates (e.g., 50-60%), baseline is ESSENTIAL to determine if it's bias or sensitivity

---

## Validation Summary

### ✅ What Works:
- GPT-5 paraphrasing: Conservative and high-quality
- Model processing: Both models handle paraphrased text
- Statistics tracking: Comprehensive flip analysis  
- Multi-field selection: DiagnosisArena correctly samples from different fields

### ⚠️ What Needs Fixing:
- BHCS extraction patterns (CRITICAL)
- Sentence replacement robustness (MINOR)

### 📊 Next Steps:
1. Fix BHCS extraction
2. Compare with main analysis results
3. Determine if full baseline runs needed
4. Statistical significance testing

---

Generated: 2025-10-20
