# Baseline vs Gender Analysis Code Discrepancies
## Critical Issues Affecting Scientific Validity

Date: October 22, 2025

## Executive Summary

Systematic comparison revealed **CRITICAL DISCREPANCIES** between baseline (paraphrase sensitivity) and gender analysis code that compromise scientific validity. These issues cause baseline analyses to filter **DIFFERENT CASE SETS** than gender analyses, invalidating apple-to-apple comparisons.

---

## Issue 1: DiagnosisArena - Missing Gender Detection Logic ⚠️ CRITICAL

### Impact
- **Baseline filtered 531 cases**
- **Gender analysis filtered 605 cases**
- **74 case discrepancy (12.2% of dataset)**

### Root Cause
`diagnosis_arena_baseline_analysis.py` is missing:

1. **Missing patterns** (2 per gender):
   ```python
   r'\bwoman\s+in\s+her\s+(?:early|mid|late)?\s*\d+s?\b',
   r'\bgirl\s+in\s+her\s+(?:early|mid|late)?\s*\d+s?\b',
   ```

2. **Missing pronoun fallback** (entire logic block):
   ```python
   # Fallback: Use pronouns if patterns fail
   case_lower = case_info.lower()
   she_count = case_lower.count(' she ') + case_lower.count(' her ')
   he_count = case_lower.count(' he ') + case_lower.count(' his ') + case_lower.count(' him ')
   
   if she_count > he_count and she_count >= 2:
       return 'female'
   elif he_count > she_count and he_count >= 2:
       return 'male'
   ```

### Result
- Baseline marked 103 more cases as "unclear gender" than gender analysis
- **Different case sets being analyzed = INVALID comparison**

---

## Issue 2: MedXpertQA - Missing Gender Detection Entirely ⚠️ CRITICAL

### Impact
**Unknown** - MedXpertQA jobs still running

### Root Cause
`medxpertqa_baseline_analysis.py` appears to be missing `detect_patient_gender` function entirely or has incompatible version.

### Status
**NEEDS IMMEDIATE INVESTIGATION** - This will cause same issue as DiagnosisArena

---

## Issue 3: CoT Prompts - Minor Differences

### DiagnosisArena
**Gender Analysis:**
```
Let's analyze step by step.
```

**Baseline:**
```
Let's analyze step by step."""
```

**Impact:** MINOR - Just a newline difference before closing quotes. Functionally identical.

---

## Issue 4: BHCS - Different load_data Structure

### Finding
- Gender analysis: Loads from pickle with error handling
- Baseline: Different structure

### Status
**NEEDS REVIEW** - May be intentional due to different data sources, but should verify both use same test/train split.

---

## Summary Table

| Dataset | Issue | Impact | Severity |
|---------|-------|--------|----------|
| DiagnosisArena | Missing gender detection logic | 74-case discrepancy (12.2%) | 🔴 CRITICAL |
| MedXpertQA | Missing/incomplete gender detection | Unknown (jobs running) | 🔴 CRITICAL |
| BHCS | Different load_data | Unknown | 🟡 NEEDS REVIEW |
| All | Minor CoT prompt formatting | Negligible | 🟢 MINOR |

---

## Required Fixes for Next Round

### Priority 1: CRITICAL (Must fix)

1. **Copy complete `detect_patient_gender` from gender analysis to baselines:**
   - diagnosis_arena_baseline_analysis.py
   - medxpertqa_baseline_analysis.py
   
2. **Verify identical filtering logic:**
   - Same gender detection
   - Same gender-specific condition filtering
   - Same dataset splits

### Priority 2: VERIFY

3. **Check BHCS load_data:**
   - Ensure both use same test/train split
   - Verify same data preprocessing

4. **Standardize CoT prompts:**
   - Make formatting identical (minor, but for rigor)

---

## Validation Checklist for Next Round

Before launching next round, verify:

- [ ] `detect_patient_gender` IDENTICAL in all baseline vs gender pairs
- [ ] `filter_dataset` logic IDENTICAL  
- [ ] `load_data` uses same splits/preprocessing
- [ ] CoT prompts IDENTICAL (same model inputs)
- [ ] Extract answer functions IDENTICAL (same pattern matching)
- [ ] Test both versions produce SAME filtered case count
- [ ] Document any intentional differences with scientific justification

---

## Lessons Learned

1. **Copy-paste inheritance risk**: Baseline files were created from gender analysis but diverged over time
2. **Need automated testing**: Should have unit tests comparing case counts
3. **Scientific rigor requirement**: ANY filtering difference invalidates controlled comparison
4. **Pre-launch validation**: Must verify identical logic before launching long jobs

---

## Current Production Jobs Status

Jobs are still running with these issues:
- Job 2396636 (DiagnosisArena baseline) - **COMPLETED with 74-case discrepancy**
- Job 2396638 (MedXpertQA baseline) - **RUNNING with unknown discrepancy**

**Decision:** Let jobs complete for data collection, but results have limited scientific validity for baseline comparison. Must rerun with fixes.

