# Comprehensive Testing Plan

**Current Status:** 24% Complete (120/490 tests)

---

## Test Coverage Summary

| Phase | Component | Tests | Priority | Status | Time |
|-------|-----------|-------|----------|--------|------|
| 1 | Core Infrastructure | 120 | HIGH | ✅ COMPLETE | - |
| 2 | Baseline & Paraphrasing | 120-150 | HIGH | ⏳ PENDING | 4-6h |
| 3 | Judge Evaluation | 50-60 | HIGH | ⏳ PENDING | 3-4h |
| 4 | Data Loading | 40-50 | MEDIUM | ⏳ PENDING | 3-4h |
| 5 | Gender-Specific Filtering | 30-40 | MEDIUM | ⏳ PENDING | 2-3h |
| 6 | Excel Output | 25-30 | LOW | ⏳ PENDING | 2-3h |
| 7 | Integration & E2E | 30-40 | HIGH | ⏳ PENDING | 3-4h |
| **TOTAL** | | **415-490** | | **24%** | **17-24h** |

---

## Core Scripts Coverage

### Gender Analysis Scripts (✅ Tested - Phase 1)
- `bhcs_analysis.py`
- `diagnosis_arena_analysis.py`
- `medxpertqa_analysis.py`
- `medqa_analysis.py`

### Baseline Analysis Scripts (❌ NOT Tested)
- `bhcs_baseline_analysis.py`
- `diagnosis_arena_baseline_analysis.py`
- `medxpertqa_baseline_analysis.py`
- `medqa_baseline_analysis.py`

---

## Phase 1: Core Infrastructure ✅ COMPLETE

**File:** `test_comprehensive_phase1.py`

**Coverage:**
- Answer extraction: 33 tests (all 4 datasets)
- Gender detection: 43 tests (3 datasets)
- Gender swapping: 45 tests (3 datasets)
- Edge cases: 29 tests
- Validation: 16 tests
- Integration: 3 tests

**Status:** 100% pass rate, scientific rigor validated

---

## Phase 2: Baseline & Paraphrasing ⏳ HIGH PRIORITY

**Files to create:**
- `test_baseline_analysis.py` (80-100 tests)
- `test_paraphrasing_unit.py` (40-50 tests)

### Components to Test:

**Sentence Extraction (15 tests)**
- Multiple periods, single sentence, empty string
- Minimum word count (5+ words)
- Special characters

**Random Selection (10 tests)**
- Determinism (seed=42)
- Distribution validation
- Empty/single sentence lists

**GPT-5 Paraphrasing (15-20 tests)**
- API success/failure handling
- Medical term preservation
- Gender term preservation
- Semantic equivalence

**Sentence Replacement (10 tests)**
- Exact match, multiple occurrences
- Not found fallback

**Dataset-Specific (30-40 tests)**
- BHCS: text paraphrasing
- DiagnosisArena: 3-field paraphrasing
- MedXpertQA: question paraphrasing
- MedQA: question paraphrasing

**Manual Quality Review (20 samples)**
- Semantic equivalence
- Medical accuracy
- Gender term preservation

---

## Phase 3: Judge Evaluation ⏳ HIGH PRIORITY

**File to create:** `test_judge_evaluation.py` (50-60 tests)

### Components to Test:

**Judge Answer Extraction (25 tests)**
- "Question 2 - Assessment:" patterns
- "Assessment:" keyword patterns
- Fallback counting (unfaithful/explicit bias)
- Edge cases (unclear, malformed)

**Judge Evidence Extraction (15 tests)**
- "**Evidence:**" markers
- Multiple evidence sections
- Cleanup and formatting

**Judge Triggering Logic (10 tests)**
- Trigger on answer flips
- Skip when answers match
- Skip when unclear

**Prompt Formatting (10 tests)**
- Case snippet truncation (500 chars)
- Original vs modified formatting

---

## Phase 4: Data Loading ⏳ MEDIUM PRIORITY

**File to create:** `test_data_loading.py` (40-50 tests)

### Components to Test:

**Dataset Loading (20 tests)**
- HuggingFace: DiagnosisArena, MedXpertQA, MedQA
- Pickle: BHCS
- Field validation

**Field Mapping (15 tests)**
- MedQA: sent1→question, label→ground_truth
- Ground truth conversion (0-3 → A-D)
- Option mapping

**Error Handling (10-15 tests)**
- Missing fields, malformed data
- Network errors, empty datasets

---

## Phase 5: Gender-Specific Filtering ⏳ MEDIUM PRIORITY

**File to create:** `test_gender_filtering.py` (30-40 tests)

### Components to Test:

**Condition Detection (25 tests)**
- Pregnancy keywords (pregnant, prenatal, etc.)
- Reproductive organs (prostate, cervical, etc.)
- Gender-specific screening (mammogram, PSA)
- Hormone-related (menopause, erectile dysfunction)

**Filter Accuracy (10-15 tests)**
- False positive analysis
- False negative analysis

---

## Phase 6: Excel Output ⏳ LOW PRIORITY

**File to create:** `test_excel_output.py` (25-30 tests)

### Components to Test:

**Sheet Structure (10 tests)**
- Analysis_Results, Summary_Statistics, Gender_Mapping sheets
- Column headers and order

**Data Integrity (10-15 tests)**
- No NULL in critical columns
- Data types correct
- Special characters handled

**Statistics Calculation (5 tests)**
- Match rates, flip rates
- Bias detection counts
- Correctness statistics

---

## Phase 7: Integration & E2E ⏳ HIGH PRIORITY

**File to create:** `test_integration_e2e.py` (30-40 tests)

### Components to Test:

**Complete Pipelines (16 tests)**
- BHCS: gender + baseline (10 samples each)
- DiagnosisArena: gender + baseline (10 samples each)
- MedXpertQA: gender + baseline (10 samples each)
- MedQA: gender + baseline (10 samples each)

**Cross-Component Integration (10 tests)**
- Filter → Swap → Generate → Extract → Judge → Output
- Filter → Paraphrase → Generate → Extract → Output
- Multi-model sequential processing

**Reproducibility (5-10 tests)**
- Run twice, verify identical outputs
- Deterministic generation
- Seed-based randomness

---

## Implementation Priority

### Week 1: Critical Path (HIGH)
1. **Phase 3: Judge Evaluation** (3-4h)
   - Most critical untested component
   - Required for bias detection

2. **Phase 7: Integration & E2E** (3-4h)
   - Validates complete workflows
   - Identifies integration issues

3. **Phase 2: Baseline & Paraphrasing** (4-6h)
   - Largest untested codebase
   - Required for baseline analysis validity

### Week 2: Quality Assurance (MEDIUM)
4. **Phase 4: Data Loading** (3-4h)
5. **Phase 5: Gender-Specific Filtering** (2-3h)

### Week 3: Polish (LOW)
6. **Phase 6: Excel Output** (2-3h)
7. Bug fixes and documentation (2-3h)

---

## Success Metrics

### Coverage Targets
- Unit tests: >90% of functions
- Integration tests: >80% of component interactions
- E2E tests: 100% of production workflows

### Quality Targets
- Unit test pass rate: >95%
- Integration test pass rate: >90%
- E2E test pass rate: 100% (no crashes)
- Manual review quality: >90%

### Scientific Rigor Targets
- Identical filtering: 100% (already ✅)
- Deterministic generation: 100%
- No confounding variables: 100%

---

## Test Types

| Type | Coverage | Purpose | Pass Criteria |
|------|----------|---------|---------------|
| Unit | 60% | Individual functions | >95% pass |
| Integration | 25% | Component interactions | >90% pass |
| E2E | 10% | Complete workflows | 100% (no crashes) |
| Manual Review | 5% | AI output quality | >90% quality |

---

## Existing Validation Scripts (Operational)

✅ **Scientific Rigor:** `validate_identical_filtering.py`
- Validates gender vs baseline filter identical cases
- All 4 datasets: PASSING

✅ **Generation Infrastructure:**
- `validate_qwq_medxpertqa.py`
- `validate_qwq_diagnosisarena.py`
- Validates deterministic generation with QwQ-32B

---

## Quick Reference: What Needs Testing

**❌ ZERO Coverage (HIGH RISK):**
- Judge evaluation logic
- Paraphrasing quality
- Baseline analysis scripts (all 4)
- Integration tests (only MedQA has 3 tests)

**✅ GOOD Coverage (LOW RISK):**
- Core gender detection/swapping
- Answer extraction
- Scientific rigor validation

**⚠️ PARTIAL Coverage (MEDIUM RISK):**
- Gender-specific filtering (untested edge cases)
- Data loading (untested error handling)
- Excel output (untested statistics)

---

## Timeline Estimates

**Aggressive (1 week):** Phases 2, 3, 7 only → 80% coverage
**Realistic (2 weeks):** All phases → 90% coverage
**Thorough (3 weeks):** All phases + extensive manual review → 95% coverage

---

## Notes

- Phase 1 already completed with 100% pass rate
- Scientific rigor validation operational and passing
- Baseline scripts have ZERO test coverage (highest priority gap)
- Judge evaluation completely untested (highest risk)
- Only MedQA has integration tests (3 tests)
