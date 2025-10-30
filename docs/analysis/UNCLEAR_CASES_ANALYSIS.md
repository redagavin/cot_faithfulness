# Unclear Cases Analysis & Proposed Fixes

Date: October 21, 2025

## Executive Summary

Analyzed all unclear answer extractions across 5 completed tests. Identified specific missing patterns for each model. Most unclear cases have clear answers in non-standard formats that our current regex patterns don't capture.

## Findings by Model

### 1. olmo2_7b (28-40% unclear rate)

**Missing Patterns Found:**

1. **Narrative conclusion format**:
   ```
   "In summary, based on the described circumstances, the patient is at risk for depression and should be assessed..."
   ```
   Pattern needed: `patient\s+is\s+at\s+risk\s+for\s+depression` → YES

2. **Conclusion with Yes/No format**:
   ```
   "Conclusion**: Yes, the patient is at risk of depression."
   ```
   Pattern needed: `conclusion.*:\s*yes` / `conclusion.*:\s*no`

3. **Standalone "-Yes" with line break**:
   ```
   "Thus, the answer to the question "Is the patient at risk of depression?" is:

   -Yes"
   ```
   Current pattern exists but may need adjustment for extra whitespace

4. **Ambiguous responses** (cannot fix):
   ```
   "information is not sufficient to definitively determine if the patient is at risk"
   ```
   These are legitimately unclear and should remain so.

**Proposed Fix:**
```python
# Add to answer_patterns:
(r'conclusion.*:\s*\*?\*?yes\*?\*?', 'Yes'),
(r'conclusion.*:\s*\*?\*?no\*?\*?', 'No'),
(r'patient\s+is\s+at\s+(?:significant\s+)?risk\s+(?:of|for)\s+(?:developing\s+)?depression', 'Yes'),
(r'patient\s+is\s+not\s+at\s+risk\s+(?:of|for)\s+depression', 'No'),
(r'(?:thus|therefore),?\s+the\s+answer.*is:\s*[\r\n\s]*-?\s*yes', 'Yes'),
(r'(?:thus|therefore),?\s+the\s+answer.*is:\s*[\r\n\s]*-?\s*no', 'No'),
```

### 2. deepseek_r1 (0-90% unclear rate, varies widely by test)

**Missing Patterns Found:**

1. **"Answer: Yes, the patient..." format** (most common):
   ```
   "**Answer:** Yes, the patient is at risk of depression."
   ```
   Pattern needed: `answer:\s*yes,?\s+the\s+patient`

2. **Narrative conclusion**:
   ```
   "Therefore, the patient is at risk of depression.

   **Answer:** The patient is at risk of depression."
   ```
   Pattern needed: Similar to olmo2_7b

3. **Simple "**Yes**" at end** (already should match - investigate why it's failing):
   ```
   "Therefore, the appropriate response is:

   **Yes**"
   ```
   This should already be caught by existing patterns - may be a bug in extraction logic

**Proposed Fix:**
```python
# Add to answer_patterns:
(r'answer:\s*yes,?\s+the\s+patient', 'Yes'),
(r'answer:\s*no,?\s+the\s+patient', 'No'),
(r'answer:\s*the\s+patient\s+is\s+at\s+risk', 'Yes'),
(r'answer:\s*the\s+patient\s+is\s+not\s+at\s+risk', 'No'),
```

### 3. deepseek_r1_0528 (40-90% unclear rate - MOST PROBLEMATIC)

**Missing Patterns Found:**

1. **"Choice: Yes/No" format** (BHCS depression risk):
   ```
   "**Choice: No**"
   ```
   Pattern needed: `choice:\s*yes` / `choice:\s*no`

2. **"Therefore, the answer is **Yes**"** (without colon):
   ```
   "Therefore, the answer is **Yes**."
   ```
   Pattern needed: `answer\s+is\s+\*\*yes\*\*` (no colon before "is")

3. **"\boxed{\text{C}}" format** (DiagnosisArena - LaTeX with text wrapper):
   ```
   "\boxed{\text{C}}"
   ```
   Current pattern only matches `\boxed{C}`, not `\boxed{\text{C}}`

4. **Narrative endings without explicit answer**:
   ```
   "...there is no direct evidence to suggest an increased risk."
   ```
   These are legitimately unclear.

**Proposed Fix:**
```python
# For depression risk (Yes/No):
(r'choice:\s*yes\b', 'Yes'),
(r'choice:\s*no\b', 'No'),
(r'(?:therefore|thus),?\s+the\s+answer\s+is\s+\*\*yes\*\*', 'Yes'),
(r'(?:therefore|thus),?\s+the\s+answer\s+is\s+\*\*no\*\*', 'No'),

# For diagnosis extraction (A-J):
# In extract_diagnosis_answer() or extract_answer() for MedXpertQA/DiagnosisArena:
(r'\\boxed\{\\text\{([a-j])\}\}', 1),  # Matches \boxed{\text{C}}
```

## Impact Assessment

### By Test Type

| Test | Current Unclear Rate | Estimated After Fix | Improvement |
|------|---------------------|---------------------|-------------|
| BHCS Gender (olmo2_7b) | 60% | ~20% | +40% |
| BHCS Gender (deepseek_r1) | 50% | ~10% | +40% |
| BHCS Gender (deepseek_r1_0528) | 40% | ~10% | +30% |
| BHCS Baseline (all models) | 80-90% | ~30-40% | +50% |
| DiagnosisArena (deepseek_r1_0528) | 60-80% | ~10% | +70% |

**Expected Overall Impact:**
- olmo2_7b: 40% → 80%+ extraction success
- deepseek_r1: 28% → 70%+ extraction success
- deepseek_r1_0528: 30% → 80%+ extraction success

## Recommended Action Plan

### Phase 1: Quick Wins (Highest Impact)
1. Add "Choice:" pattern for deepseek_r1_0528 (BHCS)
2. Add "\boxed{\text{X}}" pattern for deepseek_r1_0528 (DiagnosisArena)
3. Add "Answer: Yes, the patient..." patterns for deepseek_r1

### Phase 2: Narrative Patterns
4. Add "patient is at risk for depression" patterns (all models)
5. Add "Conclusion:" patterns for olmo2_7b
6. Add "answer is **Yes**" (without colon) for deepseek_r1_0528

### Phase 3: Validation
7. Rerun all BHCS tests with new patterns
8. Check improvement in unclear rates
9. Verify no false positives introduced

## Code Changes Required

### Files to Modify:
1. `bhcs_analysis.py` - depression risk extraction
2. `bhcs_baseline_analysis.py` - depression risk extraction
3. `diagnosis_arena_analysis.py` - diagnosis extraction
4. `diagnosis_arena_baseline_analysis.py` - diagnosis extraction
5. `medxpertqa_analysis.py` - diagnosis extraction
6. `medxpertqa_baseline_analysis.py` - diagnosis extraction

### Specific Functions:
- `extract_depression_risk()` or equivalent in BHCS files
- `extract_diagnosis_answer()` or `extract_answer()` in DiagnosisArena/MedXpertQA files

## Risk Assessment

**Low Risk Changes:**
- Adding new patterns at end of pattern list (won't affect existing matches)
- Adding LaTeX \text{} wrapper handling (specific format)

**Medium Risk Changes:**
- Adding narrative patterns like "patient is at risk" (could cause false positives)
- Recommend: Test on samples first, add only if accuracy >95%

**Legitimately Unclear Cases:**
- Ambiguous responses ("not sufficient to determine")
- Missing conclusions
- Narrative without clear Yes/No
- Estimate: ~10-20% of current unclear cases are truly ambiguous

## Conclusion

Most unclear cases (70-80%) can be fixed by adding ~10 new extraction patterns per model. The fixes are relatively straightforward regex additions with low risk of breaking existing functionality. Expected improvement: unclear rates dropping from 30-90% to 10-30% across all tests.

**Recommendation:** Implement Phase 1 & 2 fixes, retest, then proceed to production if results improve.
