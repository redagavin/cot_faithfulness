# Analysis Updates Summary

## Findings

### 1. Gender Mapping ✓
**All three datasets use the SAME gender mapping** from `bhcs_analysis.py`:
- DiagnosisArena: `from bhcs_analysis import GENDER_MAPPING`
- MedXpertQA: `from bhcs_analysis import GENDER_MAPPING`
- BHCS: Defines the mapping

**No changes needed** - there's no difference in family relationship handling.

### 2. Batch Size ✓
**Confirmed: Batch size is 1**
- Generation is done one prompt at a time
- No explicit batch_size parameter
- Each `generate_response()` call processes a single prompt

### 3. Current Generation Code
```python
# Line 479-493 in bhcs_analysis.py
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id  # ← REMOVE THIS
)

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)  # ← REMOVE skip_special_tokens
```

## Changes Required

### 1. CoT Prompt Update
Change ending from "Let's think step by step:" → "Let's analyze step by step."

**Files to update:**
- bhcs_analysis.py (line 179)
- diagnosis_arena_analysis.py
- medxpertqa_analysis.py
- bhcs_baseline_analysis.py
- diagnosis_arena_baseline_analysis.py
- medxpertqa_baseline_analysis.py

### 2. Generation Code Update
Follow HuggingFace examples with apply_chat_template():

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

**Files to update:**
- bhcs_analysis.py
- diagnosis_arena_analysis.py
- medxpertqa_analysis.py
- bhcs_baseline_analysis.py
- diagnosis_arena_baseline_analysis.py
- medxpertqa_baseline_analysis.py

### 3. Add New Model
Add `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` with `_0528` suffix in all column names and summary entries.

**Files to update:** All 6 analysis scripts

### 4. Update SLURM Files
Add required parameters and double time limits:
```bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang.zih@northeastern.edu
```

**Job distribution:**
- frink: bhcs_analysis, diagnosis_arena_analysis, bhcs_baseline_analysis
- 177huntington: medxpertqa_analysis, diagnosis_arena_baseline_analysis, medxpertqa_baseline_analysis
