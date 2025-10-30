# Test Coverage Analysis & Expansion Plan

**Current Status:** 37.5% script coverage, ~20% component coverage
**Goal:** 100% script coverage, 90%+ component coverage

---

## Current Coverage (test_comprehensive.py)

### Tested Components ✅

| Component | Datasets Covered | Test Count | Status |
|-----------|-----------------|------------|--------|
| Answer Extraction (A-D) | MedQA | 12 | ✅ Complete |
| Answer Extraction (A-J) | MedXpertQA | 7 | ✅ Complete |
| Answer Extraction (Yes/No) | BHCS | 9 | ✅ Complete |
| Gender Detection | MedQA only | 13 | ⚠️ Partial |
| Gender Swapping | MedQA only | 12 | ⚠️ Partial |
| Integration Pipeline | Basic | 3 | ⚠️ Minimal |

**Total:** 56 tests, 3/8 scripts covered (37.5%)

---

## Missing Coverage

### 1. DiagnosisArena Components ❌

**Answer Extraction:**
- DiagnosisArena uses same A-D format as MedQA
- But has different extraction method: `extract_diagnosis_answer()`
- **Test needed:** Verify DiagnosisArena-specific extraction

**Gender Detection:**
- DiagnosisArena has **3-field structure**:
  - case_information
  - physical_examination
  - diagnostic_tests
- Gender detection must work across ALL 3 fields
- **Test needed:** Multi-field gender detection

**Gender Swapping:**
- Must swap gender across ALL 3 fields consistently
- **Test needed:** Multi-field gender swapping

**Estimated tests:** 25-30

---

### 2. Baseline Analysis Components ❌

**Paraphrasing (ALL 4 baseline scripts):**
- Sentence extraction logic
- Random sentence selection (with seed)
- GPT-4 paraphrase generation
- Paraphrase quality validation
- Sentence replacement logic

**Dataset-specific variations:**
- DiagnosisArena: 3-field paraphrasing
- MedXpertQA: Single-field (question)
- MedQA: Single-field (question)
- BHCS: Text paraphrasing

**Estimated tests:** 40-50 per dataset = **160-200 tests**

---

### 3. Judge Components ❌

**Judge Response Parsing:**
- Extract "Question 1" answer (Yes/No)
- Extract "Question 2" assessment (UNFAITHFUL vs EXPLICIT BIAS)
- Extract evidence quotes
- Extract explanation

**Edge cases:**
- Malformed judge responses
- Missing sections
- Unclear classifications
- Multiple evidence quotes

**Estimated tests:** 20-25

---

### 4. Gender-Specific Condition Filtering ❌

**Condition Detection:**
- Pregnancy-related conditions
- Reproductive organ conditions
- Gender-specific cancer screening
- Menopause/andropause
- Erectile dysfunction, etc.

**Test needed:** Verify filtering logic catches all relevant conditions

**Estimated tests:** 15-20

---

### 5. Dataset Loading & Field Mapping ❌

**Dataset-specific loading:**
- DiagnosisArena: 3 fields, specific schema
- MedXpertQA: Dict-based options (A-J keys)
- MedQA: Field renaming (sent1→question, label→ground_truth)
- BHCS: Pickle file loading

**Error handling:**
- Missing fields
- Malformed data
- Type mismatches
- Empty datasets

**Estimated tests:** 20-25

---

### 6. Excel Output Generation ❌

**Sheet Structure:**
- Analysis_Results sheet
- Summary_Statistics sheet (for gender analysis)
- Gender_Mapping sheet (for gender analysis)

**Column Validation:**
- All expected columns present
- Correct data types
- No null values in critical columns
- Formulas working (if any)

**Estimated tests:** 15-20

---

### 7. Edge Cases ❌

**Answer Extraction Edge Cases:**
- Multiple answers: "\\boxed{A} but actually \\boxed{B}"
- Case sensitivity: \\boxed{a} vs \\boxed{A}
- Malformed syntax: \\boxed{A), \\boxed A}, etc.
- Very long responses (>10k chars)
- Special characters: \\boxed{🔥}, \\boxed{♀}
- Whitespace-only responses

**Gender Detection Edge Cases:**
- Mixed case: "WoMaN", "MaN"
- Multiple patients: "A woman and man both present"
- Exactly 2 pronouns (threshold test)
- Very long texts (>100k chars)
- Empty/whitespace-only text

**Gender Swapping Edge Cases:**
- Case preservation: "Woman" → "Man", "WOMAN" → "MAN"
- Word boundaries: "woman" vs "policewoman"
- Multiple swaps in same text
- Swap twice = original?
- Very long texts

**Estimated tests:** 30-40

---

## Comprehensive Test Suite Plan

### Phase 1: Complete Core Components (Priority: HIGH)
**Target: 200 tests**

1. **DiagnosisArena Full Coverage** (30 tests)
   - Answer extraction
   - 3-field gender detection
   - 3-field gender swapping
   - Integration

2. **All Datasets Gender Detection** (40 tests)
   - BHCS: 10 tests
   - DiagnosisArena: 15 tests (3 fields)
   - MedXpertQA: 10 tests
   - MedQA: 13 tests (already done)

3. **All Datasets Gender Swapping** (40 tests)
   - BHCS: 10 tests
   - DiagnosisArena: 15 tests (3 fields)
   - MedXpertQA: 10 tests
   - MedQA: 12 tests (already done)

4. **Judge Components** (25 tests)
   - Response parsing
   - Classification extraction
   - Evidence extraction
   - Edge cases

5. **Edge Cases - Critical** (30 tests)
   - Answer extraction edge cases
   - Gender detection edge cases
   - Gender swapping edge cases

6. **Dataset Loading** (25 tests)
   - All 4 datasets
   - Error handling
   - Field mapping

### Phase 2: Baseline Analysis (Priority: MEDIUM)
**Target: 150 tests**

1. **Paraphrasing - All Datasets** (120 tests)
   - BHCS: 30 tests
   - DiagnosisArena: 40 tests
   - MedXpertQA: 25 tests
   - MedQA: 25 tests

2. **Paraphrase Quality** (30 tests)
   - Semantic equivalence
   - Medical term preservation
   - Sentence structure

### Phase 3: Output & Validation (Priority: LOW)
**Target: 50 tests**

1. **Excel Output** (20 tests)
   - Sheet structure
   - Column validation
   - Data integrity

2. **Gender-Specific Filtering** (20 tests)
   - Condition detection
   - Filter accuracy

3. **Summary Statistics** (10 tests)
   - Calculation accuracy
   - Aggregation logic

---

## Total Comprehensive Coverage

| Phase | Component | Tests | Priority |
|-------|-----------|-------|----------|
| Current | Core (partial) | 56 | ✅ Done |
| 1 | Core completion | 200 | 🔴 HIGH |
| 2 | Baseline analysis | 150 | 🟡 MEDIUM |
| 3 | Output/validation | 50 | 🟢 LOW |
| **TOTAL** | **Full coverage** | **456** | - |

---

## Recommendation

**Immediate action:** Build Phase 1 tests (200 tests) to achieve comprehensive core coverage.

**Priority order:**
1. DiagnosisArena full coverage (unblocks production runs)
2. Judge components (critical for bias detection)
3. All-dataset gender detection/swapping (ensures consistency)
4. Edge cases (robustness)
5. Dataset loading (foundation validation)

**Timeline estimate:**
- Phase 1: 2-3 hours development + 1 hour validation
- Phase 2: 3-4 hours development + 1 hour validation
- Phase 3: 1-2 hours development + 30 min validation

**After Phase 1 completion:**
- Coverage: ~85% of critical components
- Confidence: Production-ready
- Risk: Minimal (all core paths tested)
