# CoT Faithfulness: Medical AI Bias Detection Infrastructure

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive research infrastructure for detecting gender bias and evaluating chain-of-thought (CoT) faithfulness in large language models (LLMs) applied to medical decision-making tasks.

## 🔬 Project Overview

This repository conducts controlled experiments to evaluate whether LLMs exhibit gender bias in medical diagnoses and whether their chain-of-thought reasoning faithfully represents their decision process. The infrastructure supports:

- **Gender Analysis**: Systematically swaps patient gender mentions and evaluates whether diagnosis changes
- **Baseline Analysis**: Tests paraphrase sensitivity as a control condition
- **LLM-as-Judge**: Automatically detects whether reasoning shows gender bias or unfaithful reasoning
- **Scientific Rigor**: Ensures controlled experiments with identical filtering, processing, and evaluation

### Research Questions

1. **Gender Bias**: Do LLMs change diagnoses when only patient gender changes?
2. **CoT Faithfulness**: Does the reasoning (CoT) actually influence the diagnosis?
3. **Bias Mechanisms**: When bias occurs, what specific patterns appear in the reasoning?

## 📊 Datasets

The infrastructure supports **four medical datasets**:

| Dataset | Size | Task | Source |
|---------|------|------|--------|
| **MedQA** | 10,178 | Multiple-choice medical exam questions | [GBaker/MedQA-USMLE-4-options-hf](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |
| **DiagnosisArena** | 1,000 | Clinical case diagnosis selection | [openlifescienceai/diagnosis_arena](https://huggingface.co/datasets/openlifescienceai/diagnosis_arena) |
| **MedXpertQA** | 5,000+ | Expert-level medical questions | [MedXpertQA](https://huggingface.co/datasets) |
| **BHCS** | Custom | Brief Hospital Course summaries | `/scratch/yang.zih/mimic/dataset/bhcs_dataset.p` |

## 🏗️ Repository Structure

```
cot_faithfulness/
├── CLAUDE.md                  # AI assistant instructions (scientific rigor guidelines)
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── src/                       # All Python source files
│   ├── *_analysis.py          # Gender analysis scripts (4 datasets)
│   ├── *_baseline_analysis.py # Baseline paraphrase analysis (4 datasets)
│   ├── gender_specific_filters.py  # Gender-specific case filtering
│   └── utils/                 # Utility scripts
│
├── tests/                     # Comprehensive test suite
│   ├── test_*.py              # Unit & integration tests (334 tests, 100% pass)
│   ├── validate_*.py          # Scientific rigor validation scripts
│   └── test_results/          # Test output files (gitignored)
│
├── slurm_jobs/                # SLURM job submission scripts
│   ├── run_*.sbatch           # Production job scripts
│   └── test_jobs/             # Test job scripts
│
├── scripts/                   # Shell utilities
│   ├── submit_job.sh          # Job submission helper
│   ├── compare_baseline_vs_gender.sh  # Scientific rigor checker
│   └── monitoring/            # Job monitoring scripts
│
├── results/                   # Production Excel results (tracked in git)
│   ├── *_analysis_results.xlsx      # Gender analysis results
│   └── *_baseline_results.xlsx      # Baseline analysis results
│
├── logs/                      # SLURM job logs (gitignored)
│
├── notebooks/                 # Jupyter notebooks
│   └── bhcs_data.ipynb        # Legacy BHCS exploration
│
└── docs/                      # Comprehensive documentation
    ├── architecture/          # System design docs
    ├── testing/               # Test strategy & results
    ├── issues/                # Bug reports & fixes
    ├── research/              # Research findings
    ├── analysis/              # Data analysis outputs
    ├── process/               # Development processes
    └── archive/               # Historical documentation
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **Conda**: For environment management
- **GPU**: NVIDIA GPU with 40GB+ VRAM (for model inference)
- **SLURM**: For cluster job submission (optional for local testing)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/redagavin/cot_faithfulness.git
cd cot_faithfulness

# 2. Activate conda environment (REQUIRED)
conda activate cot

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Running Analyses

#### Local Testing (Small Sample)

```bash
# Run MedQA gender analysis on 50 cases
python src/medqa_analysis.py

# Run MedQA baseline analysis on 50 cases
python src/medqa_baseline_analysis.py

# Results saved to:
# - results/medqa_analysis_results.xlsx
# - results/medqa_baseline_results.xlsx
```

#### SLURM Cluster Execution (Full Dataset)

```bash
# Submit production job
sbatch slurm_jobs/run_medqa_analysis.sbatch

# Monitor job
squeue -u $USER
tail -f logs/medqa_gender_<JOB_ID>.out

# Check results
ls -lh results/medqa_analysis_results.xlsx
```

## 📝 Scientific Rigor

**This is a controlled experiment infrastructure.** All code changes must maintain scientific validity.

### Core Principle

In controlled experiments, **ONLY the intended intervention should vary**. All other variables must remain constant.

### Key Requirements

- ✅ **Identical Case Selection**: Gender and baseline analyses must filter the same cases
- ✅ **Identical Processing**: Same model parameters, prompts (except intervention), extraction logic
- ✅ **Validated Changes**: All differences between versions must be justified and documented

### Validation Commands

```bash
# Verify identical filtering between gender/baseline pairs
python tests/validate_identical_filtering.py

# Compare parallel analysis files
bash scripts/compare_baseline_vs_gender.sh

# Run comprehensive test suite (334 tests)
pytest tests/ -v
```

**See [CLAUDE.md](CLAUDE.md) for detailed scientific rigor guidelines.**

## 🧪 Testing

The repository includes a comprehensive test suite with **334 tests** covering:

- **Phase 1**: Core infrastructure (answer extraction, gender detection, swapping)
- **Phase 2**: Baseline paraphrasing (sentence extraction, GPT-5 paraphrasing)
- **Phase 3**: Judge evaluation (bias detection, evidence extraction)
- **Phase 4-6**: Data loading, gender filtering, Excel output
- **Phase 7**: End-to-end integration & cross-component validation

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_core_infrastructure.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Current Status**: 334/334 tests passing (100% pass rate)

## 📊 Output Format

Each analysis produces an Excel spreadsheet with multiple sheets:

### 1. **Analysis Results** Sheet
- Case details (ID, question, options, ground truth)
- Original and modified text versions
- Model responses (Olmo2-7B, DeepSeek-R1)
- Extracted answers (Yes/No for BHCS, A/B/C/D for MCQ)
- Answer comparison (Match/Flip)
- Judge evaluation (for flipped answers)
- Evidence quotes (bias indicators)

### 2. **Summary Statistics** Sheet
- Match rates per model
- Flip rates per model
- Bias detection rates
- Case counts

### 3. **Gender Mapping Reference** Sheet (gender analysis only)
- Complete list of gender term transformations (45+ mappings)

## 🔧 Key Components

### Gender Detection & Swapping

- **Pattern Matching**: Detects gender mentions using regex patterns (titles, pronouns, family terms)
- **Pronoun Counting**: Falls back to pronoun frequency when patterns don't match
- **Comprehensive Mapping**: 45+ gender term pairs (Ms./Mr., woman/man, maternal/paternal, etc.)
- **Medical Exclusions**: Preserves anatomical terms (pregnancy, prostate, etc.)

### Chain-of-Thought Prompting

- **Model-Specific**: Optimized prompts for Olmo2-7B (detailed) and DeepSeek-R1 (simple)
- **Structured Format**: Elicits step-by-step reasoning before final answer
- **Binary Classification**: Clear Yes/No or A/B/C/D answer extraction

### Robust Answer Extraction

- **6-Level Priority Logic**: Handles 15+ answer formats
- **Deepseek `</think>` Tags**: Special handling for DeepSeek-R1 reasoning tags
- **Fallback Strategies**: Multiple extraction patterns to handle edge cases

### LLM-as-Judge

- **Automatic Bias Detection**: GPT-5 judges model responses when answers flip
- **Evidence Extraction**: Quotes specific phrases showing gender influence
- **Self-Evaluation**: Each model judges its own reasoning for objectivity

## 💻 Models Used

| Model | Size | Purpose | Source |
|-------|------|---------|--------|
| **Olmo2-7B** | 7B | Target model (diagnosis) | `allenai/OLMo-2-1124-7B` |
| **DeepSeek-R1** | 8B | Target model (diagnosis) | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| **GPT-5** | - | Judge (bias detection) | OpenAI API |
| **GPT-5** | - | Paraphrasing (baseline) | OpenAI API |

## 📚 Documentation

Comprehensive documentation is organized in `docs/`:

- **[Architecture](docs/architecture/)**: System design and infrastructure walkthrough
- **[Testing](docs/testing/)**: Test strategy, results, and coverage reports
- **[Issues](docs/issues/)**: Bug reports, fixes, and lessons learned
- **[Research](docs/research/)**: Research findings (e.g., GPT-5 determinism)
- **[Analysis](docs/analysis/)**: Data analysis outputs and investigations
- **[Process](docs/process/)**: Development processes and checklists

## 🔄 Workflow

### Typical Analysis Workflow

1. **Data Loading**: Load dataset (HuggingFace or local pickle)
2. **Gender Detection**: Identify patient gender in each case
3. **Gender Filtering**: Exclude gender-specific conditions (pregnancy, prostate, etc.)
4. **Text Modification**:
   - Gender analysis: Swap gender terms
   - Baseline analysis: Paraphrase one sentence per case
5. **Model Inference**: Generate responses for original and modified texts
6. **Answer Extraction**: Parse model responses to extract diagnoses
7. **Answer Comparison**: Identify cases where answers flip
8. **Judge Evaluation**: Run GPT-5 judge on flipped cases only
9. **Evidence Extraction**: Quote specific bias indicators
10. **Results Export**: Save to Excel with statistics

### Scientific Rigor Checks

Before each production run:

```bash
# 1. Validate identical filtering
python tests/validate_identical_filtering.py

# 2. Run test suite
pytest tests/ -v

# 3. Compare baseline vs gender implementations
bash scripts/compare_baseline_vs_gender.sh

# 4. Test on small sample
python src/medqa_analysis.py  # Sample size=50 by default
python src/medqa_baseline_analysis.py  # Sample size=50
```

## 🤝 Contributing

### Development Guidelines

1. **Scientific Validity First**: All changes must maintain experimental control
2. **Test Coverage**: Add tests for new functionality
3. **Documentation**: Update docs for significant changes
4. **Validation**: Run validation scripts before committing

### Pull Request Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Scientific rigor validated (`validate_identical_filtering.py`)
- [ ] Documentation updated
- [ ] CLAUDE.md guidelines followed

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaboration:
- **Email**: yang.zih@northeastern.edu
- **GitHub**: [@redagavin](https://github.com/redagavin)

## 🙏 Acknowledgments

- **Datasets**: MedQA, DiagnosisArena, MedXpertQA, MIMIC-III
- **Models**: Olmo2 (Allen Institute), DeepSeek-R1, GPT-5 (OpenAI)
- **Compute**: Northeastern University Discovery Cluster

## 📊 Project Status

- ✅ **Infrastructure**: Complete (8 analysis scripts, 4 datasets)
- ✅ **Testing**: 334 tests, 100% pass rate
- ✅ **Documentation**: Comprehensive
- ✅ **Validation**: Scientific rigor verified
- 🔄 **Data Collection**: Production runs ongoing
- 🔜 **Analysis**: Results analysis in progress
- 🔜 **Publication**: Manuscript in preparation

---

**Last Updated**: 2025-10-30
