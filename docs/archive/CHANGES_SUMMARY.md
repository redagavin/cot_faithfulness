# Comprehensive Analysis Updates - Summary

## All Changes Completed ✓

### 1. Gender Mapping Analysis ✓
**Finding:** All three datasets (BHCS, DiagnosisArena, MedXpertQA) use the **same** GENDER_MAPPING from `bhcs_analysis.py`.
- DiagnosisArena and MedXpertQA import: `from bhcs_analysis import GENDER_MAPPING`
- No differences in family relationship handling
- **No changes needed**

### 2. Batch Size Verification ✓
**Confirmed:** Batch size is **1** (implicit)
- Each `generate_response()` call processes a single prompt
- No explicit batch_size parameter
- Generation done one-at-a-time

### 3. CoT Prompt Update ✓
**Changed:** "Let's think step by step:" → "Let's analyze step by step."

**Files updated:**
- ✓ bhcs_analysis.py (2 occurrences)
- ✓ diagnosis_arena_analysis.py (2 occurrences)
- ✓ medxpertqa_analysis.py (2 occurrences)
- ✓ bhcs_baseline_analysis.py (1 occurrence)
- ✓ diagnosis_arena_baseline_analysis.py (1 occurrence)
- ✓ medxpertqa_baseline_analysis.py (1 occurrence)

### 4. Generation Code Update ✓
**Changed:** Updated to use `apply_chat_template()` following HuggingFace examples

**Old code:**
```python
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id  # ← REMOVED
)

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)  # ← REMOVED skip_special_tokens
```

**New code:**
```python
messages = [
    {"role": "user", "content": prompt},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
```

**Changes:**
1. ✓ Uses `apply_chat_template()` for proper message formatting
2. ✓ Removed `pad_token_id=tokenizer.eos_token_id`
3. ✓ Removed `skip_special_tokens=True`

**Files updated:**
- ✓ bhcs_analysis.py (2 occurrences - generate_response)
- ✓ diagnosis_arena_analysis.py (2 occurrences)
- ✓ medxpertqa_analysis.py (2 occurrences)
- ✓ bhcs_baseline_analysis.py (2 occurrences)
- ✓ diagnosis_arena_baseline_analysis.py (2 occurrences)
- ✓ medxpertqa_baseline_analysis.py (2 occurrences)

### 5. New Model Added ✓
**Added:** `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` with `_0528` suffix

**Model configuration now includes:**
```python
{
    "olmo2_7b": "allenai/OLMo-2-1124-7B-Instruct",
    "deepseek_r1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek_r1_0528": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # ← NEW
}
```

**Files updated:**
- ✓ bhcs_analysis.py
- ✓ diagnosis_arena_analysis.py
- ✓ medxpertqa_analysis.py
- ✓ bhcs_baseline_analysis.py
- ✓ diagnosis_arena_baseline_analysis.py
- ✓ medxpertqa_baseline_analysis.py

**Note:** All column names and summary entries for this model will automatically include `_0528` suffix (e.g., `deepseek_r1_0528_original_response`, `deepseek_r1_0528_original_answer`, etc.)

### 6. SLURM Batch Files Updated ✓

**Time limits doubled:**
| File | Old Limit | New Limit |
|------|-----------|-----------|
| run_bhcs_analysis.sbatch | 1 day | **2 days** |
| run_diagnosis_arena.sbatch | 1 day | **2 days** |
| run_medxpertqa.sbatch | 2 days | **4 days** |
| run_bhcs_baseline.sbatch | 1 day | **2 days** |
| run_diagnosis_arena_baseline.sbatch | 1 day | **2 days** |
| run_medxpertqa_baseline.sbatch | 2 days | **4 days** |

**Required parameters added/verified (all files):**
```bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang.zih@northeastern.edu
```

**Partition distribution:**
- **frink** (3 jobs):
  1. bhcs_analysis
  2. diagnosis_arena
  3. bhcs_baseline

- **177huntington** (3 jobs):
  1. medxpertqa
  2. diagnosis_arena_baseline
  3. medxpertqa_baseline

## Verification Summary

✓ All 6 analysis scripts have valid Python syntax
✓ All 6 scripts include new `deepseek_r1_0528` model
✓ All 6 scripts have updated CoT prompt ending
✓ All 6 scripts use `apply_chat_template()`
✓ All 6 sbatch files have doubled time limits
✓ All 6 sbatch files have required SLURM parameters
✓ Jobs evenly distributed between frink (3) and 177huntington (3)

## Next Steps

1. **Test all 6 analyses** with small sample sizes:
   - Test gender analyses (3 jobs)
   - Test baseline analyses (3 jobs)

2. **After tests pass:**
   - Launch full production jobs
   - Monitor progress
   - Collect results

## Test Job Submission

### Gender Analysis Tests
```bash
sbatch test_bhcs_analysis.sbatch
sbatch test_diagnosis_arena.sbatch
sbatch test_medxpertqa.sbatch
```

### Baseline Analysis Tests
```bash
sbatch test_bhcs_baseline.sbatch
sbatch test_diagnosis_arena_baseline.sbatch
sbatch test_medxpertqa_baseline.sbatch
```

### Full Production Jobs
```bash
# After tests pass:
sbatch run_bhcs_analysis.sbatch
sbatch run_diagnosis_arena.sbatch
sbatch run_medxpertqa.sbatch
sbatch run_bhcs_baseline.sbatch
sbatch run_diagnosis_arena_baseline.sbatch
sbatch run_medxpertqa_baseline.sbatch
```

## Expected Outputs

Each analysis will now generate results for **3 models**:
1. olmo2_7b
2. deepseek_r1
3. deepseek_r1_0528 (NEW)

Excel output files will contain columns/summaries for all 3 models with appropriate suffixes.
