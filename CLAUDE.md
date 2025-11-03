# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🔬 SCIENTIFIC RIGOR REQUIREMENT

**THIS IS A SCIENTIFIC RESEARCH PROJECT. ALL CODE CHANGES MUST MAINTAIN SCIENTIFIC VALIDITY.**

### Core Principle: Experimental Control

**In controlled experiments, ONLY the intended intervention should vary. ALL other variables must remain constant.**

This repository conducts controlled experiments to isolate specific effects (e.g., gender bias, paraphrase sensitivity). The validity of these experiments depends critically on ensuring that:

1. **Only the intended change varies** (e.g., patient gender, paraphrase wording)
2. **All control variables remain identical** (e.g., which cases are analyzed, model parameters, extraction logic)
3. **No unintentional changes are introduced** without explicit justification and documentation

As the codebase grows and experiments multiply, there are countless ways control variables can inadvertently change. Common pitfalls include:

- **Data filtering differences**: Different cases analyzed between comparison groups
- **Model parameter drift**: Different hyperparameters, random seeds, or generation settings
- **Preprocessing inconsistencies**: Different tokenization, normalization, or cleaning
- **Extraction logic divergence**: Different answer parsing between versions
- **Prompt variations**: Unintended differences beyond the target intervention
- **Dataset version mismatches**: Different train/test splits or data subsets
- **Dependency version drift**: Different library versions producing different outputs
- **Hardware/environment differences**: Different GPU settings, precision modes, or batch sizes

### Scientific Integrity Checklist

Before ANY code changes to analysis files, verify:

- [ ] **Control Variables Held Constant**: Are you changing ONLY the intended intervention?
  - Same dataset filtering logic across comparison conditions
  - Same model parameters, random seeds, generation settings
  - Same preprocessing, tokenization, and normalization
  - Same answer extraction and evaluation logic
  - Same computational environment (GPU settings, precision, batch size)

- [ ] **Identical Case Selection**: Do all comparison conditions analyze the same cases?
  - Same inclusion/exclusion criteria
  - Same gender detection logic (if applicable)
  - Same condition filtering (if applicable)
  - Verify with validation scripts that case counts match

- [ ] **Documented Intentional Differences**: Is every difference between versions justified?
  - Explain WHY the difference is necessary
  - Document WHAT the difference is
  - Assess HOW it impacts scientific validity
  - Consider whether it introduces confounds

- [ ] **Validation Before Launch**: Have you verified scientific rigor programmatically?
  - Run validation scripts to check for unintended differences
  - Compare outputs between versions on test data
  - Review diffs between parallel analysis files
  - Confirm identical behavior for control variables

### Example: Data Filtering Discrepancy (2025-10-22)

**Issue Discovered**: Baseline and gender analyses filtered **different case sets** (74-case discrepancy in DiagnosisArena), invalidating the controlled experiment.

**Root Cause**: The `detect_patient_gender()` function had incomplete implementations in baseline files:
- Missing regex patterns for detecting gender mentions
- Missing pronoun counting fallback logic
- Result: Baseline version filtered fewer cases (531 vs 605)

**Why This Violated Scientific Rigor**:
We intended to compare model behavior with gender changes vs. paraphrase changes **on the same cases**. By analyzing different case subsets, we introduced a confound - any observed differences could be due to **case selection** rather than the intervention.

**Resolution**:
- Synchronized all `detect_patient_gender()` functions across file pairs
- Created `validate_identical_filtering.py` to prevent recurrence
- See `BASELINE_VS_GENDER_ISSUES.md` for detailed analysis

**Lesson**: This example illustrates how **any** function that affects case selection, preprocessing, or analysis must be identical across comparison conditions. The principle applies to **every aspect** of the codebase, not just filtering.

### General Validation Commands

```bash
# Before launching any production jobs, validate scientific rigor:
conda activate cot

# 1. Validate identical filtering (when applicable)
python validate_identical_filtering.py

# 2. Compare parallel analysis files for unintended differences
diff -u <(grep -v '^#' file1_analysis.py) <(grep -v '^#' file2_baseline_analysis.py) | less

# 3. Test on sample data to verify identical behavior (except intervention)
python file1_analysis.py --test-mode
python file2_baseline_analysis.py --test-mode
# Manually verify same cases analyzed, same processing pipeline

# All validations must pass before production launch
```

---

## 🧪 COMPREHENSIVE TESTING REQUIREMENT

**EVERY CODE CHANGE MUST HAVE CORRESPONDING TESTS. NO EXCEPTIONS.**

### Testing Requirement

- **Mandatory**: All new code requires corresponding tests before committing
- **Coverage**: 100% test coverage required for all new code
- **No exceptions**: Bug fixes, new features, refactoring all require tests

### When to Write Tests

- Adding new functionality
- Modifying existing functions
- Fixing bugs (regression prevention)
- Refactoring code (ensure behavior unchanged)

### What to Test

**Test ALL possible behaviors induced by your changes**, including:

- **Research Rigor**: Ensure scientific validity maintained
- **Edge Cases**: Empty inputs, malformed data, boundary conditions, unusual patterns
- **Consistency**: Components that should remain identical across analyses/scripts are indeed identical
- **Determinism**: Results reproducible where expected
- **All Code Paths**: Every branch, condition, and error handling path

### Running Tests

```bash
# Before ANY commit
pytest tests/ -v                           # Full suite must pass
pytest tests/ --cov=src --cov-report=html  # Verify 100% coverage for new code

# During development
pytest tests/test_<relevant_module>.py -v  # Run relevant subset
```

### Test Quality Standards

- Descriptive test names explaining what is tested
- Docstrings explaining WHY the test matters
- Tests should be self-documenting
- One assertion per test when possible

### Scientific Rigor in Testing

Tests must verify:
- Experimental control maintained (only intended variable changes)
- Consistency maintained where required
- No unintended side effects

### Important Note About Examples

**CRITICAL**: Examples throughout this document (Bug 1 & 2, filtering discrepancies, etc.) are provided to help you **understand principles**, NOT as rigid checklists.

When you encounter new situations:
- **DO**: Apply the underlying principles (experimental control, consistency, validation)
- **DON'T**: Blindly check if specific examples apply
- **THINK**: What consistency matters for THIS specific change?

---

## Project Overview

This repository contains a medical data analysis project focused on Brief Hospital Course Summary (BHCS) data processing. The project includes both the original Jupyter notebook (`bhcs_data.ipynb`) and a complete Python script (`bhcs_analysis.py`) that processes medical text data using open source language models.

## Development Environment

- **Primary file**: `bhcs_analysis.py` - Main Python script for BHCS analysis
- **Legacy file**: `bhcs_data.ipynb` - Original Jupyter notebook
- **Language**: Python 3.8+
- **Conda Environment**: `cot` (REQUIRED - must be activated before running Python)
- **Dependencies**: See `requirements.txt` for complete list
- **Data**: Processes medical text data from `/scratch/yang.zih/mimic/dataset/bhcs_dataset.p`

## Setup and Installation

```bash
# Activate conda environment (REQUIRED)
conda activate cot

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python bhcs_analysis.py
```

## IMPORTANT: Python Environment
**ALWAYS activate the `cot` conda environment before running ANY Python commands:**
```bash
conda activate cot
# or
source activate cot
```

This applies to:
- Running scripts (`python bhcs_analysis.py`)
- Testing code (`python -c "import torch; print(torch.cuda.is_available())"`)
- Installing packages (`pip install package`)
- Any other Python operations

## Key Components

### Gender Mapping System
- Comprehensive gender-specific term mapping (45+ mappings)
- Includes titles, pronouns, family relationships, professional terms
- Excludes medical/anatomical terminology and age-related terms

### Open Source Model Integration
- **Olmo2 7B**: `allenai/OLMo-2-1124-7B`
- **Deepseek R1**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- Optimized for 80GB GPU memory constraint
- Uses torch.float16 and device mapping for memory efficiency

### Chain of Thought (CoT) Prompting
- **Model-specific prompts**:
  - **Olmo2 7B**: Enhanced detailed prompt with analysis factors to elicit better reasoning
  - **Deepseek R1**: Simple prompt (already generates excellent CoT with `</think>` tags)
- Binary classification for depression risk assessment
- Uses structured Yes/No choice format

### Robust Answer Extraction
- **Priority-based extraction logic** with 6 levels:
  1. Deepseek `</think>` tag patterns
  2. Explicit "Answer:" or "Final answer:" markers
  3. Markdown-styled answers (**-Yes**, **Answer: Yes**)
  4. Statement-style ("The patient is at risk of depression")
  5. Contradictory parentheticals handling
  6. Fallback counting in last 100 characters
- Handles 15+ different answer formats
- Fixes misextraction issues (e.g., "**Answer: Yes**" now correctly extracted)

### LLM-as-Judge for Gender Bias Detection
- **Automatic bias detection**: Models judge their own reasoning when answers flip
- **Only runs on flipped answers**: When female→male text change causes Yes↔No flip
- **Evidence extraction**: Quotes specific phrases showing gender influence
- **Separate extraction logic**: Judge answers extracted differently from depression risk answers
- **Self-evaluation**: Olmo2 judges Olmo2 responses, Deepseek judges Deepseek responses

### Output Generation
- Excel spreadsheet output with multiple sheets:
  - **Analysis Results**: All model responses, answers, judge outputs, and evidence
  - **Summary Statistics**: Match rates, flip rates, bias detection rates per model
  - **Gender Mapping Reference**: Complete list of 45+ gender term transformations

## Usage

The main script (`bhcs_analysis.py`) provides a complete comparison pipeline with LLM judge:

1. **Data Loading**: Loads BHCS dataset from pickle file (uses test split)
2. **Gender Mapping**: Creates both original and gender-modified versions
3. **Model-Specific Processing** (sequential, one at a time):
   - **Step 1**: Generate responses for all texts (original + modified) with appropriate prompt
   - **Step 2**: Extract depression risk answers with robust logic
   - **Step 3**: Run LLM judge on cases where answers flip between genders
4. **Results Export**: Comprehensive Excel output with responses, answers, judge evaluations, and evidence

## Important Notes

- Contains sensitive medical data - ensure HIPAA compliance
- Requires GPU with sufficient memory for model inference
- Sample size can be adjusted for testing (default: 50 entries)
- Results saved to `bhcs_analysis_results.xlsx`

## Command Reference

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Full analysis (all data)
python bhcs_analysis.py

# For testing, modify sample_size parameter in main() function
```

### SLURM Cluster Execution
```bash
# Submit job to SLURM scheduler
./submit_job.sh

# Or submit directly
sbatch run_bhcs_analysis.sbatch

# Monitor job status
squeue -u $USER
squeue -j <JOB_ID>

# View logs
tail -f bhcs_analysis_<JOB_ID>.out
tail -f bhcs_analysis_<JOB_ID>.err

# Cancel job if needed
scancel <JOB_ID>
```

## Data Structure

Processes medical records containing:
- Brief Hospital Course summaries with gender-swapped content
- AI-generated analysis from multiple open source models
- Structured CoT reasoning for medical insights