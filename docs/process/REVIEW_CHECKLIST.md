# Code Review Checklist - BHCS Analysis Changes

## Summary of Changes

### 1. Judge Prompt Template (lines 177-255)
**CHANGED:** Complete rewrite to focus on faithfulness detection
- Now asks two questions: (1) Explicit gender statements? (2) Assessment?
- Clear examples of what counts as "explicit" vs "not explicit"
- Outputs "UNFAITHFUL" or "EXPLICIT BIAS" instead of Yes/No

### 2. CoT Prompt (lines 165-175)
**CHANGED:** Unified prompt for both models
- Removed the detailed Olmo2-specific prompt
- Both models now use the same simple prompt
- Reason: Olmo2 Instruct generates good CoT with simple prompt

### 3. Extract Judge Answer (lines 629-668)
**CHANGED:** Extraction logic for new output format
- Looks for "UNFAITHFUL" or "EXPLICIT BIAS"
- Priority-based pattern matching
- More robust fallbacks

### 4. Extract Judge Evidence (lines 670-709)
**CHANGED:** Improved evidence extraction
- Filters out meaningless single-word quotes ("she", "he")
- Better quality control (minimum 10 chars)
- Cleaner output

### 5. Excel Statistics (lines 867-873, 950-959)
**CHANGED:** Updated metric names
- Old: `judge_bias_detected`, `judge_no_bias`
- New: `judge_unfaithful`, `judge_explicit_bias`
- Added rates for both categories

## Critical Safety Checks ✓

### ✓ NO Text Truncation
```bash
grep -n "text\[:" bhcs_analysis.py  # Returns nothing
```
- No slicing like `text[:1500]` or `text[:600]`
- Full texts passed to models and judge

### ✓ NO max_length Parameter
```bash
grep -n "max_length" bhcs_analysis.py  # Returns nothing
```
- Tokenizer calls: `tokenizer(prompt, return_tensors="pt")` (line 484)
- No max_length restriction

### ✓ NO truncation Parameter
```bash
grep -n "truncation" bhcs_analysis.py  # Returns nothing
```
- No `truncation=True` anywhere

### ✓ NO attention_mask Parameter
```bash
grep -n "attention_mask" bhcs_analysis.py  # Returns nothing
```
- model.generate() calls only use: **inputs, max_new_tokens, temperature, do_sample, pad_token_id

### ✓ Correct max_new_tokens
- Depression risk responses: 8192 tokens (line 474)
- Judge responses: 8192 tokens (line 789)

### ✓ Full Text Storage
- Line 705-706: Stores full original_text and modified_text
- Line 781-782: Passes full texts to judge prompt

## What Was NOT Changed

- Model loading/unloading logic
- Gender mapping dictionary
- Depression risk answer extraction
- Data loading from pickle
- Excel export structure
- SLURM job configuration

## Test Plan

### Phase 1: Small Sample Test (5 samples)
- Verify both models load correctly
- Check prompt formatting
- Verify response generation
- Test judge invocation on flipped cases
- Inspect output quality

### Phase 2: Output Validation
- Check Excel columns exist
- Verify judge assessments are "UNFAITHFUL" or "EXPLICIT BIAS"
- Inspect evidence quality
- Review summary statistics

### Phase 3: Full Run (if Phase 1-2 pass)
- Submit SLURM job
- Monitor memory usage
- Review complete results
