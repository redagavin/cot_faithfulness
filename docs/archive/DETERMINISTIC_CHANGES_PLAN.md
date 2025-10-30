# Deterministic Changes and Validation Plan

## Date: 2025-10-23

## Objective
Make all code deterministic to ensure reproducible results and enable systematic debugging of generation infrastructure.

---

## Part 1: Deterministic Generation Changes

### 1.1 Hugging Face Model Generation
**Current State:**
```python
temperature=0.7,
do_sample=True,
top_p=0.95
```

**Change To (Greedy Decoding):**
```python
do_sample=False,  # Greedy decoding
# Remove temperature and top_p (not used in greedy)
```

**Files to Modify:**
- bhcs_analysis.py (line ~456)
- bhcs_baseline_analysis.py (line ~252)
- diagnosis_arena_analysis.py (line ~425)
- diagnosis_arena_baseline_analysis.py (line ~440)
- medxpertqa_analysis.py (line ~410)
- medxpertqa_baseline_analysis.py (line ~396)

### 1.2 Random Seeds (Baseline Files)
**Current State:**
```python
random.seed(seed + case_index)  # Already present
```

**Change To:**
```python
random.seed(42)  # Set at script initialization
# Keep case_index-based seeding for reproducible per-case randomness
```

**Files to Modify:**
- bhcs_baseline_analysis.py
- diagnosis_arena_baseline_analysis.py
- medxpertqa_baseline_analysis.py

### 1.3 Torch Random Seeds (All Files)
**Add to script initialization:**
```python
import random
import torch

# Set all random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

**Files to Modify:**
- All 6 analysis scripts

### 1.4 OpenAI API Calls
**Current State:**
```python
response = self.openai_client.responses.create(
    model="gpt-5",
    input=prompt
)
```

**If using standard OpenAI API, change to:**
```python
response = self.openai_client.chat.completions.create(
    model="gpt-4o",  # or appropriate model
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    seed=42
)
```

**Note**: Current code uses non-standard API (`responses.create`, `gpt-5`).
- If this is mock/placeholder: Leave as-is
- If this is real API: Update to standard format with seed parameter
- **Action**: Will leave as-is since it appears to be placeholder code

---

## Part 2: Verification of Identical Parts

After making deterministic changes, verify that parts that should be identical across file pairs remain identical.

### 2.1 File Pairs to Compare

#### Gender vs Baseline Pairs:
1. **bhcs_analysis.py** ↔ **bhcs_baseline_analysis.py**
2. **diagnosis_arena_analysis.py** ↔ **diagnosis_arena_baseline_analysis.py**
3. **medxpertqa_analysis.py** ↔ **medxpertqa_baseline_analysis.py**

### 2.2 Functions That MUST Be Identical

For each pair, these functions must be identical:

- [ ] `load_data()` - Data loading logic
- [ ] `detect_patient_gender()` - Gender detection (already verified on 2025-10-22)
- [ ] `filter_dataset()` / `filter_by_gender_and_conditions()` - Case filtering
- [ ] `extract_answer()` - Answer extraction logic
- [ ] Model loading and configuration
- [ ] Generation parameters (now all greedy)
- [ ] Random seed initialization

### 2.3 Parts That SHOULD Differ

Only these should differ between gender and baseline versions:

- [ ] Text modification logic:
  - Gender: `apply_gender_mapping()`
  - Baseline: `select_random_sentence()` + `paraphrase_sentence()`
- [ ] Prompts (if modified text is inserted)
- [ ] Column names in output

### 2.4 Validation Commands

```bash
# Compare critical functions
for func in "load_data" "detect_patient_gender" "filter" "extract_answer"; do
    echo "=== Comparing $func ==="
    diff <(grep -A 50 "def $func" diagnosis_arena_analysis.py) \
         <(grep -A 50 "def $func" diagnosis_arena_baseline_analysis.py)
done

# Compare generation parameters
echo "=== Generation Parameters ==="
grep -A 5 "model.generate" diagnosis_arena_analysis.py
grep -A 5 "model.generate" diagnosis_arena_baseline_analysis.py
```

---

## Part 3: QwQ-32B Validation Scripts

### 3.1 Purpose
Create minimal scripts to validate generation infrastructure is bug-free by:
1. Running QwQ-32B on MedXpertQA and DiagnosisArena
2. Comparing accuracy with paper-reported results
3. If close match → generation infrastructure is correct
4. If mismatch → debug generation code

### 3.2 QwQ-32B Paper Results

**Model**: Qwen/QwQ-32B-Preview (HuggingFace)

**Expected Accuracies** (from papers):
- **MedXpertQA**: TBD (need to check paper)
- **DiagnosisArena**: TBD (need to check paper)

### 3.3 Script Design

**File 1**: `validate_qwq_medxpertqa.py`
- Load MedXpertQA test set
- No filtering, no editing
- Simple MCQ evaluation
- Use greedy decoding
- Extract answer (A-J)
- Compare with ground truth
- Report accuracy

**File 2**: `validate_qwq_diagnosisarena.py`
- Load DiagnosisArena test set
- No filtering, no editing
- Simple MCQ evaluation
- Use greedy decoding
- Extract answer (A-D)
- Compare with ground truth
- Report accuracy

### 3.4 Key Simplifications (vs current scripts)

**Remove:**
- Gender detection and filtering
- Gender mapping or paraphrasing
- LLM judge evaluation
- Complex result tracking
- GPT API calls

**Keep:**
- Model loading (same infrastructure)
- Dataset loading (same infrastructure)
- Generation code (same parameters, now greedy)
- Answer extraction (simplified, just MCQ)
- Accuracy calculation

---

## Part 4: Validation Strategy for Other Components

### 4.1 Components to Validate

After confirming generation is correct, validate:

1. **Gender Detection Logic** (`detect_patient_gender()`)
   - Already validated on 2025-10-22 ✅
   - Validation script exists: `validate_identical_filtering.py`

2. **Gender Mapping Logic** (`apply_gender_mapping()`)
   - Test: Manual inspection of mapped cases
   - Validation: Create test cases with known gender terms
   - Check: All 45+ gender mappings applied correctly

3. **Paraphrasing Logic** (`select_random_sentence()` + `paraphrase_sentence()`)
   - Test: Check random selection reproducibility with seed
   - Validation: Verify GPT-5 paraphrasing preserves meaning
   - Check: Selected sentences are actually paraphrased

4. **Answer Extraction Logic** (`extract_answer()`)
   - Test: Unit tests with known response formats
   - Validation: Check extraction on actual model outputs
   - Check: All 15+ extraction patterns work correctly

5. **LLM Judge Logic** (`evaluate_with_judge()`)
   - Test: Manual inspection of judge responses
   - Validation: Check judge answer extraction
   - Check: Evidence extraction correctness

### 4.2 Validation Tests to Create

#### Test 1: Gender Mapping Validation
```bash
# Create: validate_gender_mapping.py
# Purpose: Test all 45+ gender term mappings
# Method: Apply to synthetic test cases, verify correctness
```

#### Test 2: Answer Extraction Validation
```bash
# Create: validate_answer_extraction.py
# Purpose: Test extraction on known response formats
# Method: Feed pre-written responses, check extracted answers
```

#### Test 3: Paraphrasing Validation
```bash
# Create: validate_paraphrasing.py
# Purpose: Check random selection and paraphrasing
# Method: Run on sample, verify semantic equivalence
```

#### Test 4: End-to-End Smoke Test
```bash
# Create: smoke_test_all_datasets.py
# Purpose: Run full pipeline on 10 cases each
# Method: Check no crashes, reasonable outputs
```

### 4.3 Debugging Strategy

If validation fails:

1. **Generation Issues** → Already caught by QwQ validation
2. **Filtering Issues** → Check `validate_identical_filtering.py` output
3. **Mapping Issues** → Inspect failed test cases in gender mapping validation
4. **Extraction Issues** → Check extraction validation test outputs
5. **Judge Issues** → Manual inspection of judge responses

---

## Part 5: Implementation Order

### Phase 1: Make Deterministic (Today)
1. ✅ Check OpenAI API docs
2. ⏳ Change all generation to greedy decoding
3. ⏳ Add torch.manual_seed(42) to all scripts
4. ⏳ Verify syntax of all modified files

### Phase 2: Verify Identical Parts (Today)
1. ⏳ Run comparison commands on all file pairs
2. ⏳ Document any unintended differences
3. ⏳ Fix any discrepancies found
4. ⏳ Re-run `validate_identical_filtering.py`

### Phase 3: QwQ Validation (Today/Tomorrow)
1. ⏳ Look up QwQ-32B paper results for MedXpertQA and DiagnosisArena
2. ⏳ Create `validate_qwq_medxpertqa.py`
3. ⏳ Create `validate_qwq_diagnosisarena.py`
4. ⏳ Run both scripts and compare with paper
5. ⏳ Debug if mismatch found

### Phase 4: Component Validation (Future)
1. ⏳ Create gender mapping validation
2. ⏳ Create answer extraction validation
3. ⏳ Create paraphrasing validation
4. ⏳ Create end-to-end smoke test
5. ⏳ Run all validations and fix bugs

### Phase 5: Re-run Production (After All Validation)
1. ⏳ All validations pass
2. ⏳ Run production jobs with corrected, validated code
3. ⏳ Monitor for any runtime issues

---

## Success Criteria

### For Deterministic Changes:
- ✅ All generation uses greedy decoding (do_sample=False)
- ✅ All scripts set random.seed(42) and torch.manual_seed(42)
- ✅ No syntax errors after changes

### For Identical Parts Verification:
- ✅ All filtering functions identical across pairs
- ✅ All extraction functions identical across pairs
- ✅ All model loading identical across pairs
- ✅ `validate_identical_filtering.py` passes

### For QwQ Validation:
- ✅ QwQ-32B accuracy matches paper (within ±2% tolerance)
- ✅ If match → generation infrastructure validated ✓
- ✅ If mismatch → bugs identified and fixed

### For Component Validation:
- ✅ All unit tests pass
- ✅ No extraction failures on test cases
- ✅ Gender mapping produces expected outputs
- ✅ Smoke test completes successfully

---

## Notes

- Keep backups of all files before modification
- Test syntax after each batch of changes
- Run validations before launching any new production jobs
- Document any issues found during validation

