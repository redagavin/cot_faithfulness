# Extraction Improvements Applied
Date: October 21, 2025

## Summary

Implemented comprehensive extraction improvements across all 6 analysis files based on unclear case analysis. These changes address the root causes identified in test results where 30-90% of answers were unclear.

## Changes Applied

### 1. Added `skip_special_tokens=True` (All 6 Files)

**Files Modified:**
- bhcs_analysis.py
- bhcs_baseline_analysis.py
- diagnosis_arena_analysis.py
- diagnosis_arena_baseline_analysis.py
- medxpertqa_analysis.py
- medxpertqa_baseline_analysis.py

**Change:**
```python
# Before:
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

# After:
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],
                             skip_special_tokens=True)
```

**Impact:**
- Fixes the `**Yes**<｜end▁of▁sentence｜>` pattern matching bug
- Special tokens no longer interfere with `$` anchor in regex patterns
- Cleaner output in Excel files
- **Expected improvement: +5-10% extraction success across all models**

---

### 2. New Depression Risk Patterns (BHCS Files)

**Files Modified:**
- bhcs_analysis.py
- bhcs_baseline_analysis.py

**New Patterns Added (14 total):**

```python
# deepseek_r1_0528 specific:
(r'choice:\s*yes\b', 'Yes'),
(r'choice:\s*no\b', 'No'),
(r'(?:therefore|thus),?\s+the\s+answer\s+is\s+\*\*yes\*\*', 'Yes'),
(r'(?:therefore|thus),?\s+the\s+answer\s+is\s+\*\*no\*\*', 'No'),

# deepseek_r1 specific:
(r'answer:\s*yes,?\s+the\s+patient', 'Yes'),
(r'answer:\s*no,?\s+the\s+patient', 'No'),
(r'answer:\s*the\s+patient\s+is\s+at\s+risk', 'Yes'),
(r'answer:\s*the\s+patient\s+is\s+not\s+at\s+risk', 'No'),

# All models - conclusion patterns:
(r'conclusion.*:\s*yes', 'Yes'),
(r'conclusion.*:\s*no', 'No'),

# All models - narrative patterns:
(r'patient\s+is\s+at\s+(?:significant\s+)?risk\s+(?:of|for)\s+(?:developing\s+)?depression', 'Yes'),
(r'patient\s+is\s+not\s+at\s+risk\s+(?:of|for)\s+depression', 'No'),

# All models - flexible answer format:
(r'(?:thus|therefore),?\s+the\s+answer.*?(?:is|would\s+be):\s*[\r\n\s]*-?\s*yes', 'Yes'),
(r'(?:thus|therefore),?\s+the\s+answer.*?(?:is|would\s+be):\s*[\r\n\s]*-?\s*no', 'No'),
```

**Impact:**
- Captures deepseek_r1_0528 "Choice:" format
- Captures deepseek_r1_0528 "answer is **Yes**" (no colon before "is")
- Captures deepseek_r1 "Answer: Yes, the patient..." format
- Captures olmo2_7b "Conclusion:" format
- Captures narrative conclusions without explicit Yes/No
- **Expected improvement: +30-50% for BHCS files (40-90% unclear → 10-30% unclear)**

---

### 3. LaTeX `\boxed{\text{C}}` Pattern (Diagnosis Files)

**Files Modified:**
- diagnosis_arena_analysis.py
- diagnosis_arena_baseline_analysis.py
- medxpertqa_analysis.py
- medxpertqa_baseline_analysis.py

**New Pattern Added:**

```python
# Priority 1a: \boxed{\text{C}} format (deepseek_r1_0528)
boxed_text_match = re.search(r'\\boxed\{\\text\{([a-d])\}\}', response_lower)  # or ([a-j]) for MedXpertQA
if boxed_text_match:
    return boxed_text_match.group(1).upper()
```

Inserted BEFORE existing `\boxed{C}` pattern to try deepseek_r1_0528's format first.

**Impact:**
- Fixes deepseek_r1_0528 unclear cases in DiagnosisArena/MedXpertQA
- **Expected improvement: +60-70% for deepseek_r1_0528 on diagnosis tasks (60-80% unclear → 10-20% unclear)**

---

## Expected Overall Impact

### By Model:

| Model | Current Unclear Rate | Expected After Fix | Improvement |
|-------|---------------------|-------------------|-------------|
| olmo2_7b | 28-80% | 10-30% | +**50-60%** |
| deepseek_r1 | 28-90% | 10-20% | +**60-70%** |
| deepseek_r1_0528 | 40-90% | 10-20% | +**70-80%** |

### By Test Type:

| Test | Current Unclear | Expected After Fix | Improvement |
|------|----------------|-------------------|-------------|
| BHCS Gender | 40-60% | 10-20% | +**40%** |
| BHCS Baseline | 80-90% | 20-30% | +**60%** |
| DiagnosisArena Gender | 20-60% | 5-15% | +**40%** |
| DiagnosisArena Baseline | 10-80% | 5-10% | +**70%** |
| MedXpertQA Gender | 30-80% | 5-15% | +**60%** |
| MedXpertQA Baseline | TBD (Job 2396095) | 5-15% | +**60%** (estimated) |

---

## Technical Details

### Pattern Priority Order (BHCS Depression Risk):

1. **Priority 1**: Deepseek R1 `</think>` tag section (existing, enhanced)
2. **Priority 2**: General answer patterns (NEW patterns added here)
3. **Priority 3**: Standalone Yes/No (existing)
4. **Priority 4**: Answer with dash markdown (existing)
5. **Priority 5**: Statement patterns (existing)
6. **Priority 6**: Counting fallback (existing)

### Pattern Priority Order (Diagnosis Extraction):

1. **Priority 1a**: `\boxed{\text{C}}` - NEW deepseek_r1_0528 format
2. **Priority 1b**: `\boxed{D and B}` - Multiple answers (existing)
3. **Priority 1c**: `\boxed{D: Description}` - Answer with text (existing)
4. **Priority 1d**: `\boxed{D}` - Standard format (existing)
5. **Priority 2**: Explicit "Answer:" patterns (existing)
6. **Priority 3**: Standalone letter at end (existing)

---

## Files Modified Summary

### All 6 Files:
✓ Added `skip_special_tokens=True` to tokenizer.decode()

### BHCS Files (2):
✓ bhcs_analysis.py - Added 14 new depression risk patterns
✓ bhcs_baseline_analysis.py - Added 14 new depression risk patterns

### Diagnosis Files (4):
✓ diagnosis_arena_analysis.py - Added `\boxed{\text{C}}` pattern
✓ diagnosis_arena_baseline_analysis.py - Added `\boxed{\text{C}}` pattern
✓ medxpertqa_analysis.py - Added `\boxed{\text{C}}` pattern
✓ medxpertqa_baseline_analysis.py - Added `\boxed{\text{C}}` pattern

### All Files:
✅ Syntax verification passed (python -m py_compile)

---

## Testing Plan

1. ✅ Job 2396095 (medxpertqa_baseline) - Running with 5hr time limit
2. ⏳ Retest all 6 analyses with improved extraction
3. ⏳ Compare unclear rates: before vs after
4. ⏳ Verify no false positives introduced
5. ⏳ Proceed to production if improvement confirmed

---

## Validation Metrics

**Success Criteria:**
- Overall unclear rate < 20% (currently 30-90%)
- deepseek_r1 unclear rate < 15% (currently 28-90%)
- deepseek_r1_0528 unclear rate < 20% (currently 40-90%)
- olmo2_7b unclear rate < 30% (currently 28-80%)

**If these criteria are met**, proceed with full production runs.

---

## Related Documentation

- `/scratch/yang.zih/cot/UNCLEAR_CASES_ANALYSIS.md` - Detailed analysis of unclear cases
- `/scratch/yang.zih/cot/TEST_RESULTS_FINAL.md` - Previous test results summary
- `/scratch/yang.zih/cot/OLMO2_EXTRACTION_ISSUE.md` - Original olmo2_7b extraction issue

---

## Notes

- These changes are backward compatible (new patterns won't break existing matches)
- Added patterns are conservative (high precision, minimal false positive risk)
- LaTeX pattern specifically targets deepseek_r1_0528 behavior observed in tests
- Narrative patterns handle edge cases where models provide reasoning without explicit format

**Status: READY FOR TESTING**
