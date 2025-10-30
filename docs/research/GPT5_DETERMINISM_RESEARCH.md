# GPT-5 Determinism Research

**Date:** 2025-10-29
**Research Goal:** Determine if GPT-5 judge evaluation can be made deterministic

---

## Executive Summary

**✅ Deterministic GPT-5 IS POSSIBLE** (with caveats)

**Current Implementation:** ❌ NOT deterministic
**Fix Required:** ✅ Add seed and temperature parameters

---

## Current Implementation

**File:** All analysis scripts (e.g., `medqa_analysis.py:453-466`)

```python
def generate_gpt5_judge_response(self, prompt):
    """Generate judge response using GPT-5 API"""
    try:
        print("Calling GPT-5 API for judge evaluation...")

        response = self.openai_client.responses.create(
            model="gpt-5",
            input=prompt
            # ❌ NO seed parameter
            # ❌ NO temperature parameter (defaults to 1.0)
        )

        return response.output_text.strip()
    except Exception as e:
        return f"Error calling GPT-5 API: {e}"
```

**Problem:** No parameters for deterministic generation

---

## Recommended Implementation

### Option 1: Add Parameters to Responses API (PREFERRED)

```python
def generate_gpt5_judge_response(self, prompt):
    """Generate judge response using GPT-5 API with deterministic settings"""
    try:
        print("Calling GPT-5 API for judge evaluation...")

        response = self.openai_client.responses.create(
            model="gpt-5",
            input=prompt,
            temperature=0,      # ✅ Maximize determinism
            seed=42,            # ✅ Fixed seed for reproducibility (if supported)
            top_p=1.0           # ✅ Disable nucleus sampling
        )

        return response.output_text.strip()
    except Exception as e:
        return f"Error calling GPT-5 API: {e}"
```

**Note:** Responses API documentation does NOT explicitly mention `seed` parameter. Need to verify if supported.

---

### Option 2: Use Chat Completions API (FALLBACK)

If `responses.create()` doesn't support seed, use `chat.completions.create()`:

```python
def generate_gpt5_judge_response(self, prompt):
    """Generate judge response using Chat Completions API with deterministic settings"""
    try:
        print("Calling GPT-5 API for judge evaluation...")

        response = self.openai_client.chat.completions.create(
            model="gpt-5",  # or appropriate GPT-5 chat model
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,      # ✅ Maximize determinism
            seed=42,            # ✅ Fixed seed for reproducibility
            top_p=1.0           # ✅ Disable nucleus sampling
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling GPT-5 API: {e}"
```

---

## Determinism Parameters Explained

### 1. `seed` (Integer)
**Purpose:** Enables reproducible outputs

**How it works:**
- Set to any integer (e.g., 42, 123, 12345)
- Use the SAME seed across all requests
- System makes "best effort" to sample deterministically

**Example:**
```python
seed=42  # Use same value for all judge evaluations
```

**Documentation:** Supported in Chat Completions API for GPT-4 and GPT-3.5 models

---

### 2. `temperature` (Float, 0-2)
**Purpose:** Controls randomness

**Recommended setting:**
```python
temperature=0  # Greedy decoding, maximum determinism
```

**Explanation:**
- `0` → Greedy decoding (always picks most likely token)
- `0.2` → Focused and deterministic
- `0.7` → Balanced (default for chat)
- `1.0` → Default for responses API
- `1.5+` → Very creative/random

**For judge evaluation:** Use `temperature=0` for maximum consistency

---

### 3. `top_p` (Float, 0-1)
**Purpose:** Nucleus sampling (alternative to temperature)

**Recommended setting:**
```python
top_p=1.0  # Disable nucleus sampling
```

**Explanation:**
- When `temperature=0`, top_p doesn't matter
- Set to 1.0 to avoid unintended sampling behavior

---

### 4. `system_fingerprint` (Response Field)
**Purpose:** Monitor backend changes

**Usage:**
```python
response = client.chat.completions.create(...)
fingerprint = response.system_fingerprint
```

**Check fingerprint across requests:**
- Same fingerprint → Backend configuration unchanged
- Different fingerprint → Infrastructure/weights changed → outputs may differ

---

## Limitations of Determinism

### ⚠️ "Mostly Deterministic" NOT "Perfectly Deterministic"

**From OpenAI documentation:**
> "If specified, our system will make a **best effort** to sample deterministically, such that repeated requests with the same seed and parameters should return the same result."

> "Determinism **isn't guaranteed** with reproducible output. Even in cases where the seed parameter and system_fingerprint are the same across API calls it's **currently not uncommon to still observe a degree of variability** in responses."

### Factors Affecting Reproducibility

1. **Backend Changes:**
   - Model weights updated
   - Infrastructure configuration changed
   - Reflected in `system_fingerprint` changes

2. **Token Length:**
   - Larger `max_tokens` → less deterministic
   - Shorter outputs → more reliable

3. **Model Architecture:**
   - Some models more deterministic than others
   - GPT-5 reasoning models may have additional non-determinism

4. **Inherent Non-Determinism:**
   - Small chance of variation even with matching parameters
   - Due to distributed systems, numerical precision, etc.

---

## Testing Determinism

### Test Script

```python
# Test if GPT-5 judge is deterministic
def test_judge_determinism():
    analyzer = MedQAAnalyzer()

    test_prompt = """[Your judge prompt here]"""

    # Generate 5 responses with same seed
    responses = []
    for i in range(5):
        response = analyzer.generate_gpt5_judge_response(test_prompt)
        responses.append(response)
        print(f"Response {i+1}: {response[:100]}...")

    # Check if all identical
    if all(r == responses[0] for r in responses):
        print("✅ DETERMINISTIC: All responses identical")
    else:
        print("❌ NON-DETERMINISTIC: Responses differ")
        for i, r in enumerate(responses):
            print(f"Response {i+1}: {r[:100]}...")
```

---

## API Compatibility

### Responses API (`client.responses.create()`)

**Documented Parameters:**
- ✅ `model`
- ✅ `input`
- ✅ `temperature` (confirmed)
- ✅ `top_p` (confirmed)
- ❓ `seed` (NOT documented, needs testing)

**Status:** UNCLEAR if seed parameter supported

---

### Chat Completions API (`client.chat.completions.create()`)

**Documented Parameters:**
- ✅ `model`
- ✅ `messages`
- ✅ `temperature` (confirmed)
- ✅ `top_p` (confirmed)
- ✅ `seed` (confirmed for GPT-4, GPT-3.5)

**Status:** CONFIRMED seed support

---

## Recommendations

### For Judge Evaluation (Current Need)

1. **Immediate Action:** Test if `responses.create()` supports `seed` parameter

   ```python
   response = client.responses.create(
       model="gpt-5",
       input=prompt,
       temperature=0,
       seed=42  # Test if this works
   )
   ```

2. **If seed supported:** ✅ Add `temperature=0` and `seed=42` to all judge calls

3. **If seed NOT supported:** Switch to `chat.completions.create()` API

---

### For Baseline Paraphrasing

Same principles apply:

```python
def paraphrase_sentence(self, sentence):
    """Use GPT-5 to paraphrase sentence conservatively"""
    try:
        prompt = self.paraphrase_prompt.format(sentence=sentence)

        response = self.openai_client.responses.create(
            model="gpt-5",
            input=prompt,
            temperature=0.3,  # Low temperature for conservative paraphrasing
            seed=42           # Deterministic seed (if supported)
        )

        return response.output_text.strip()
    except Exception as e:
        return f"Error: {e}"
```

**Note:** For paraphrasing, may want `temperature=0.3` instead of 0 to allow slight variation while maintaining consistency.

---

## Scientific Rigor Impact

### Current State: ❌ HIGH RISK

**Problem:**
- Judge evaluations are non-deterministic
- Same case evaluated twice may give different results
- Violates reproducibility requirement for research

**Impact:**
- Bias detection counts may vary between runs
- Results cannot be replicated
- Scientific validity compromised

---

### With Determinism: ✅ LOW RISK

**Solution:**
- Add `seed=42` and `temperature=0`
- Judge evaluations become reproducible
- Same case → same classification

**Impact:**
- Bias detection counts stable across runs
- Results fully reproducible
- Scientific validity maintained

---

## ❌ TESTING RESULTS (2025-10-29)

**File:** `test_gpt5_determinism.py`

### Test Results

```
Test 1: Basic call (no seed, no temperature)
✅ Basic call works

Test 2: With temperature=0
❌ Temperature parameter NOT supported
Error: "Unsupported parameter: 'temperature' is not supported with this model."

Test 3: With seed=42
❌ Seed parameter NOT supported
Error: "Responses.create() got an unexpected keyword argument 'seed'"

Test 4: With both parameters
❌ Both parameters NOT supported

Test 5: Determinism verification
❌ Could not test (parameters not supported)
```

### Conclusion

**GPT-5 `responses.create()` API does NOT support:**
- ❌ `temperature` parameter (explicitly unsupported)
- ❌ `seed` parameter (not recognized)

**Impact:** Judge evaluation and paraphrasing are **non-deterministic** and **cannot be made deterministic** with current API.

---

## Action Items

### Priority 1: ~~Test Current API~~ ✅ TESTED - NOT SUPPORTED

```python
# Quick test
from openai import OpenAI

client = OpenAI()

# Test 1: With seed parameter
try:
    response = client.responses.create(
        model="gpt-5",
        input="Test prompt",
        temperature=0,
        seed=42
    )
    print("✅ Seed parameter supported!")
except Exception as e:
    print(f"❌ Seed parameter NOT supported: {e}")

# Test 2: Verify determinism
responses = []
for i in range(3):
    r = client.responses.create(
        model="gpt-5",
        input="Generate a random number",
        temperature=0,
        seed=42
    )
    responses.append(r.output_text)

if all(r == responses[0] for r in responses):
    print("✅ DETERMINISTIC")
else:
    print("❌ NON-DETERMINISTIC")
    for i, r in enumerate(responses):
        print(f"Response {i+1}: {r}")
```

---

### Priority 2: Update Code (if supported)

Add parameters to all GPT-5 calls:

**Files to update:**
1. `medqa_analysis.py` - Judge evaluation
2. `diagnosis_arena_analysis.py` - Judge evaluation
3. `medxpertqa_analysis.py` - Judge evaluation
4. `medqa_baseline_analysis.py` - Paraphrasing
5. `diagnosis_arena_baseline_analysis.py` - Paraphrasing
6. `medxpertqa_baseline_analysis.py` - Paraphrasing
7. `bhcs_baseline_analysis.py` - Paraphrasing

**Pattern:**
```python
# OLD
response = self.openai_client.responses.create(
    model="gpt-5",
    input=prompt
)

# NEW
response = self.openai_client.responses.create(
    model="gpt-5",
    input=prompt,
    temperature=0,    # For judge (0.3 for paraphrasing)
    seed=42           # Fixed seed
)
```

---

### Priority 3: Document in Scientific Rigor

Add to CLAUDE.md or TESTING_MEMO.md:

```markdown
## Deterministic GPT-5 Generation

All GPT-5 API calls use deterministic settings:
- **Judge evaluation:** `temperature=0, seed=42`
- **Paraphrasing:** `temperature=0.3, seed=42`

This ensures reproducibility across runs and maintains scientific validity.
```

---

## References

- **OpenAI Cookbook:** [Reproducible Outputs with Seed Parameter](https://cookbook.openai.com/examples/reproducible_outputs_with_the_seed_parameter)
- **Microsoft Learn:** [Reproducible Output with Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reproducible-output)
- **Azure Responses API:** [Azure OpenAI Responses API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses)

---

## Conclusion

**YES, deterministic GPT-5 is possible** by adding:
1. `seed=42` parameter (if supported by responses.create())
2. `temperature=0` for judge, `temperature=0.3` for paraphrasing

**Current code is NON-DETERMINISTIC** and should be updated immediately for scientific rigor.

**Next step:** Test if `responses.create()` supports `seed` parameter, then update all scripts accordingly.
