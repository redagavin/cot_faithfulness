# Infrastructure Walkthrough & Test Explanation

**Purpose:** Detailed explanation of every test and infrastructure component for manual verification

---

## Part 1: Current Test Suite Walkthrough (56 Tests)

### TEST GROUP 1: MedQA Answer Extraction (12 tests)

**What it tests:** Extracting A-D answers from model responses

**Method tested:** `MedQAAnalyzer.extract_diagnosis_answer(response)`

**Test expectations explained:**

```python
Test 1-4: Boxed format
  Input: "\\boxed{A}", "\\boxed{B}", "\\boxed{C}", "\\boxed{D}"
  Expected: "A", "B", "C", "D"
  Why: Models often use LaTeX boxed format for final answers
  Manual check: Look for pattern r'\\boxed\{([a-d])\}' in code

Test 5: Boxed in sentence
  Input: "Final answer: \\boxed{B}"
  Expected: "B"
  Why: Answer can be embedded in sentence
  Manual check: Pattern should work anywhere in response

Test 6-8: Explicit answer patterns
  Input: "Final answer: A", "The answer is: B", "Answer: C"
  Expected: "A", "B", "C"
  Why: Common explicit answer formats
  Manual check: Look for patterns like r'final\s+answer:\s*([a-d])\b'

Test 9-10: Standalone letters
  Input: "After analysis, the correct option is C.", "Therefore, D."
  Expected: "C", "D"
  Why: Model might just state letter at end
  Manual check: Pattern r'\b([a-d])\.?\s*$' should match last 20 chars

Test 11-12: Edge cases
  Input: "No clear answer", ""
  Expected: "Unclear", "Unclear"
  Why: Graceful handling when no answer found
  Manual check: Function should return "Unclear" when no pattern matches
```

**Manual verification steps:**
1. Open `medqa_analysis.py`
2. Find `extract_diagnosis_answer()` method
3. Verify it has patterns for:
   - \\boxed{} format
   - "Final answer:", "Answer is:", "Answer:"
   - Standalone letter at end
   - Returns "Unclear" if no match

---

### TEST GROUP 2: MedXpertQA Answer Extraction (7 tests)

**What it tests:** Extracting A-J answers (10 options instead of 4)

**Method tested:** `MedXpertQAAnalyzer.extract_answer(response)`

**Test expectations explained:**

```python
Test 1-3: Boxed format (wider range)
  Input: "\\boxed{A}", "\\boxed{E}", "\\boxed{J}"
  Expected: "A", "E", "J"
  Why: MedXpertQA has 10 options (A-J)
  Manual check: Pattern should be r'\\boxed\{([a-j])\}' not just [a-d]

Test 4-6: Explicit patterns
  Input: "Final answer: F", "The answer is: G", "After analysis, H."
  Expected: "F", "G", "H"
  Why: Same logic as MedQA but wider range
  Manual check: Patterns should use [a-j] not [a-d]

Test 7: No answer
  Input: "No answer here"
  Expected: "Unclear"
  Why: Graceful handling
  Manual check: Returns "Unclear" when no match
```

**Manual verification:**
1. Open `medxpertqa_analysis.py`
2. Find `extract_answer()` method
3. **KEY CHECK:** Patterns must use `[a-j]` NOT `[a-d]`
4. Verify same structure as MedQA but wider range

---

### TEST GROUP 3: BHCS Answer Extraction (9 tests)

**What it tests:** Extracting Yes/No depression risk answers

**Method tested:** `BHCSAnalyzer.extract_depression_risk_answer(response)`

**Test expectations explained:**

```python
Test 1-2: Boxed format
  Input: "\\boxed{Yes}", "\\boxed{No}"
  Expected: "Yes", "No"
  Why: Same boxed format, different answer type
  Manual check: Pattern r'\\boxed\{(yes|no)\}' case-insensitive

Test 3-4: Explicit patterns
  Input: "Final answer: Yes", "The answer is: No"
  Expected: "Yes", "No"
  Why: Common formats
  Manual check: Patterns for "final answer:", "answer is:", etc.

Test 5-6: Statement style
  Input: "Therefore, the patient is at risk: Yes"
        "Patient is not at risk: No"
  Expected: "Yes", "No"
  Why: Models might embed answer in statement
  Manual check: Pattern should extract from longer sentences

Test 7-8: Deepseek-specific format
  Input: "</think>\n\n**yes**", "</think>\n\n**no**"
  Expected: "Yes", "No"
  Why: Deepseek R1 uses </think> tag + bold answer
  Manual check: Pattern for r'\*\*(yes|no)\*\*' after </think>

Test 9: No answer
  Input: "Unclear situation"
  Expected: "Unclear"
  Why: Graceful handling
  Manual check: Returns "Unclear" when no match
```

**Manual verification:**
1. Open `bhcs_analysis.py`
2. Find `extract_depression_risk_answer()` method
3. Verify patterns for:
   - \\boxed{yes/no}
   - Explicit "answer:" formats
   - Statement-style extraction
   - Deepseek bold format (**yes**/**no**)
   - </think> tag handling

---

### TEST GROUP 4: Gender Detection (13 tests)

**What it tests:** Detecting patient gender from medical text

**Method tested:** `MedQAAnalyzer.detect_patient_gender(text)`

**Test expectations explained:**

```python
Test 1-4: Clear female cases
  Input: "A 45-year-old woman presents with..."
         "A 23-year-old girl comes to..."
         "The female patient shows..."
         "A woman reports she has been... She also notes..."
  Expected: "female"
  Why: Pattern-based detection with clear female indicators
  Manual check:
    - Patterns for r'\bwoman\b', r'\bgirl\b', r'\bfemale\b'
    - Multiple pronouns (2+ "she"/"her") as fallback

Test 5-8: Clear male cases
  Input: "A 45-year-old man presents..."
         "A 15-year-old boy comes..."
         "The male patient shows..."
         "A man reports he has been... He also notes..."
  Expected: "male"
  Why: Same logic for male patterns
  Manual check:
    - Patterns for r'\bman\b', r'\bboy\b', r'\bmale\b'
    - Multiple pronouns (2+ "he"/"his"/"him") as fallback

Test 9-13: Unclear cases (SAFETY THRESHOLD)
  Input: "The patient presents..."
         "A 50-year-old presents..."
         "" (empty)
         "She has been experiencing symptoms"  (single pronoun)
         "He has been experiencing symptoms"   (single pronoun)
  Expected: "unclear"
  Why: Defensive programming - need clear evidence
  Manual check:
    - No pattern match → "unclear"
    - Less than 2 pronouns → "unclear" (INTENTIONAL THRESHOLD)
    - Empty text → "unclear"
```

**Manual verification:**
1. Open `medqa_analysis.py`
2. Find `detect_patient_gender()` method
3. **CRITICAL CHECK - Two-stage detection:**
   ```python
   Stage 1: Pattern matching
   - Check for female patterns (woman, girl, female)
   - Check for male patterns (man, boy, male)
   - If found and unambiguous → return gender

   Stage 2: Pronoun counting (FALLBACK)
   - Count "she"/"her" vs "he"/"his"/"him"
   - **THRESHOLD: Requires 2+ pronouns** (not just 1)
   - If she_count > he_count AND she_count >= 2 → "female"
   - If he_count > she_count AND he_count >= 2 → "male"
   - Otherwise → "unclear"
   ```

4. **KEY DESIGN DECISION:**
   - Why 2+ threshold? Real medical texts use "A 45-year-old woman"
   - Single pronoun = insufficient evidence (could be fragment)
   - Conservative approach prevents misclassification

---

### TEST GROUP 5: Gender Swapping (12 tests)

**What it tests:** Swapping patient gender while preserving control variables

**Method tested:** `MedQAAnalyzer.apply_gender_swap(text, original_gender)`

**Test expectations explained:**

```python
Test 1-4: Female → Male (PATIENT TERMS ONLY)
  Input: "A 45-year-old woman presents", gender="female"
  Expected: "A 45-year-old man presents"
  Why: Swap patient-referring terms
  Manual check: woman→man in GENDER_MAPPING

  Input: "She has symptoms", gender="female"
  Expected: "He has symptoms"
  Why: Pronoun swapping
  Manual check: She→He in GENDER_MAPPING

  Input: "Her condition worsens", gender="female"
  Expected: "His condition worsens"
  Why: Possessive pronoun swapping
  Manual check: Her→His in GENDER_MAPPING

  Input: "The woman reports fever", gender="female"
  Expected: "The man reports fever"
  Why: Determiner + gender term
  Manual check: woman→man with word boundaries

Test 5-8: Male → Female (REVERSE MAPPING)
  Same logic, opposite direction
  Manual check: REVERSED_GENDER_MAPPING created correctly

Test 9-10: Family members should NOT swap (EXPERIMENTAL CONTROL)
  Input: "A woman whose mother has diabetes", gender="female"
  Expected: "A man whose mother has diabetes"
  ^^^ CRITICAL: "mother" stays "mother" ^^^

  Why: Changing family member gender would introduce 2nd variable
  Manual check: "mother" NOT in GENDER_MAPPING

  Input: "A man whose father has cancer", gender="male"
  Expected: "A woman whose father has cancer"
  ^^^ CRITICAL: "father" stays "father" ^^^

  Why: Same reasoning - experimental control
  Manual check: "father" NOT in GENDER_MAPPING

Test 11-12: Medical terms preserved
  Input: "Testicular cancer", "Prostate examination"
  Expected: Unchanged
  Why: Medical/anatomical terms should never swap
  Manual check: These terms NOT in GENDER_MAPPING
```

**Manual verification:**
1. Open `bhcs_analysis.py`
2. Find `GENDER_MAPPING` dictionary (around line 88)
3. **VERIFY THE FOLLOWING:**

```python
# Should be IN the mapping (patient terms):
"Woman": "Man",
"woman": "man",
"She": "He",
"she": "he",
"Her": "His",
"her": "his",
# ...all patient-referring pronouns and terms

# Should NOT be in mapping (control variables):
# NO "mother": "father"
# NO "father": "mother"
# NO "grandmother": "grandfather"
# NO medical/anatomical terms
```

4. Open `medqa_analysis.py`
5. Find where REVERSED_GENDER_MAPPING is created:
```python
REVERSED_GENDER_MAPPING = {}
for female_term, male_term in GENDER_MAPPING.items():
    if male_term not in REVERSED_GENDER_MAPPING:
        REVERSED_GENDER_MAPPING[male_term] = female_term
```

6. Verify `apply_gender_swap()` uses correct mapping based on gender:
```python
if original_gender == 'female':
    mapping = GENDER_MAPPING  # female→male
elif original_gender == 'male':
    mapping = REVERSED_GENDER_MAPPING  # male→female
```

---

### TEST GROUP 6: Integration Pipeline (3 tests)

**What it tests:** End-to-end flow from raw text to swapped version

**Test expectations:**

```python
Test 1: Gender detection in pipeline
  Input: "A 35-year-old woman presents with fever and cough."
  Expected: Detects as "female"
  Why: Ensure detection works on realistic medical text
  Manual check: Full sentence with pattern should work

Test 2: Gender swapping in pipeline
  Input: Same text, gender="female"
  Expected: "A 35-year-old man presents with fever and cough."
  Why: Swap applied correctly
  Manual check: woman→man, rest unchanged

Test 3: Medical terms preserved in pipeline
  Expected: "fever and cough" unchanged
  Why: Only gender swaps, medical content identical
  Manual check: No medical terms in GENDER_MAPPING
```

---

## Part 2: Infrastructure Architecture

Let me explain the complete infrastructure so you can manually verify:

### Architecture Diagram:

```
[Dataset] → [Analyzer] → [Filter] → [Process] → [Judge] → [Output]
```

Let's walk through each component:

### Component 1: Dataset Loading

**Location:** Each `*_analysis.py` has `load_data()` method

**DiagnosisArena:**
```python
def load_data(self):
    dataset = load_dataset("shzyk/DiagnosisArena", split="test")
    # 915 total cases
    # Fields: case_information, physical_examination, diagnostic_tests,
    #         option_a, option_b, option_c, option_d, ground_truth
```

**MedXpertQA:**
```python
def load_data(self):
    dataset = load_dataset("TsinghuaC3I/MedXpertQA", "Text", split="test")
    # 2450 total cases
    # Fields: question, options (dict with A-J keys), label (A-J)
```

**MedQA:**
```python
def load_data(self):
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
    # 1273 total cases
    # Fields: sent1, ending0-3, label (0-3)
    # MAPS TO: question, option_a-d, ground_truth (A-D)
```

**BHCS:**
```python
def load_data(self):
    data = pickle.load(open(dataset_path, 'rb'))
    self.test_original = data['test_original']  # Original texts
    # Uses test split from pickle file
```

**Manual check:**
1. Verify each dataset path is correct
2. Check split used (all use "test" except BHCS)
3. Verify field mappings are correct

---

### Component 2: Filtering (Gender Analysis Only)

**Method:** `filter_and_prepare_cases()` or `filter_dataset()`

**Three-stage filtering:**

```python
Stage 1: Gender Detection
  - Call detect_patient_gender() on question text
  - Count unclear cases

Stage 2: Gender-Specific Condition Filtering
  - Call is_gender_specific_case()
  - Remove pregnancy, reproductive conditions, etc.

Stage 3: Prepare Swapped Versions
  - For each valid case:
    - Create original version
    - Create swapped version (apply_gender_swap)
    - Store both with metadata
```

**Manual check:**
1. Verify unclear cases are excluded
2. Verify gender-specific conditions are excluded
3. Verify swapping creates true counterfactual pairs

---

### Component 3: Model Processing

**Sequential processing:**
```python
for model_name in models:
    1. Load model
    2. For each case:
       - Generate response for original version
       - Generate response for swapped version
    3. Extract answers from both responses
    4. Compare answers (match/flip/unclear)
    5. If answers flip → mark for judging
    6. Unload model
```

**Manual check:**
1. Verify greedy decoding (do_sample=False)
2. Verify random seed = 42
3. Verify model unloading between models

---

### Component 4: Judge Evaluation

**When triggered:** Only when answers flip (match='no')

**Process:**
```python
for flipped_case in flipped_cases:
    1. Format judge prompt with:
       - Original case + response
       - Swapped case + response
       - Both answers
    2. Call GPT-5 API
    3. Parse judge response:
       - Extract Question 1 answer
       - Extract assessment (UNFAITHFUL or EXPLICIT BIAS)
       - Extract evidence quotes
    4. Store in results
```

**Manual check:**
1. Verify only flipped cases are judged
2. Verify prompt template is correct
3. Verify parsing extracts all required fields

---

### Component 5: Output Generation

**Excel structure:**
```
Sheet 1: Analysis_Results
  - All cases with columns:
    - Original question/case
    - Swapped question/case
    - Model responses (original + swapped)
    - Extracted answers
    - Match status
    - Correctness
    - Judge output (if applicable)

Sheet 2: Summary_Statistics (gender analysis only)
  - Match rates per model
  - Flip rates per model
  - Bias detection counts

Sheet 3: Gender_Mapping (gender analysis only)
  - Complete list of gender term mappings
```

**Manual check:**
1. Open Excel file after run
2. Verify all columns present
3. Spot-check formulas and data

---

## Part 3: Manual Verification Checklist

Use this to verify each component:

**☐ Gender Detection (all datasets):**
1. Open each *_analysis.py
2. Find detect_patient_gender()
3. Verify patterns for woman/girl/female
4. Verify patterns for man/boy/male
5. Verify pronoun counting has threshold >= 2
6. Verify returns 'unclear' for insufficient evidence

**☐ Gender Swapping (all datasets):**
1. Open bhcs_analysis.py
2. Find GENDER_MAPPING
3. Verify patient terms included (she, he, woman, man, her, his)
4. Verify family terms NOT included (mother, father, etc.)
5. Verify medical terms NOT included
6. Check each *_analysis.py imports GENDER_MAPPING
7. Verify REVERSED_GENDER_MAPPING created correctly
8. Verify apply_gender_swap() uses correct mapping

**☐ Answer Extraction (all datasets):**
1. MedQA: extract_diagnosis_answer() uses [a-d]
2. DiagnosisArena: extract_diagnosis_answer() uses [a-d]
3. MedXpertQA: extract_answer() uses [a-j]
4. BHCS: extract_depression_risk_answer() uses yes/no
5. All have multiple patterns (boxed, explicit, standalone)
6. All return "Unclear" when no match

**☐ Filtering:**
1. Check filter logic excludes unclear gender
2. Check gender-specific conditions filtered
3. Verify identical filtering between gender/baseline pairs

**☐ Model Processing:**
1. Greedy decoding (do_sample=False)
2. Random seed = 42
3. Sequential model processing (not parallel)
4. Model unloading between runs

**☐ Judge:**
1. Only triggered for flipped cases
2. Prompt template correct
3. Parsing extracts all fields

---

This is the complete walkthrough of the current infrastructure. Would you like me to now expand the test suite to achieve comprehensive coverage (456 tests)?
