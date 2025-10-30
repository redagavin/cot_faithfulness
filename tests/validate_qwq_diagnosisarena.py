#!/usr/bin/env python3
"""
QwQ-32B Validation on DiagnosisArena
Simple MCQ evaluation to validate generation infrastructure
"""

import re
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Expected accuracy from QwQ-32B-Preview paper
EXPECTED_ACCURACY = 0.5139  # 51.39% on DiagnosisArena
TOLERANCE = 0.02  # ±2% tolerance

class QwQDiagnosisArenaValidator:
    def __init__(self):
        """Initialize validator"""
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.results = []

    def load_data(self):
        """Load DiagnosisArena test dataset"""
        print("Loading DiagnosisArena dataset...")
        try:
            dataset = load_dataset("shzyk/DiagnosisArena")
            self.dataset = dataset['test']
            print(f"Loaded {len(self.dataset)} test cases")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def load_model(self):
        """Load QwQ-32B model"""
        model_path = "Qwen/QwQ-32B"
        print(f"\nLoading {model_path}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            self.model.eval()
            print(f"✓ Model loaded successfully")
            print(f"  Device map: {self.model.hf_device_map}")
            return True

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False

    def generate_response(self, prompt, max_new_tokens=2048):
        """Generate response with greedy decoding"""
        try:
            messages = [{"role": "user", "content": prompt}]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            # Greedy decoding for determinism
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False  # Greedy decoding
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            return f"Error: {e}"

    def extract_answer(self, response):
        """
        Extract A-D answer from model response
        Priority: \\boxed{} > Answer: > standalone letter
        """
        response_lower = response.lower()
        last_200 = response[-200:]
        last_200_lower = last_200.lower()

        # Priority 1: \boxed{} format variations
        boxed_patterns = [
            r'\\boxed\{\\text\{([a-d])\}\}',  # \boxed{\text{C}}
            r'\\boxed\{([a-d])\}',             # \boxed{C}
        ]

        for pattern in boxed_patterns:
            match = re.search(pattern, response_lower)
            if match:
                return match.group(1).upper()

        # Priority 2: Explicit "Answer:" patterns
        answer_patterns = [
            (r'final\s+answer:\s*([a-d])\b', 1),
            (r'answer:\s*([a-d])\b', 1),
            (r'final\s+answer\s+is:\s*([a-d])\b', 1),
            (r'answer\s+is:\s*([a-d])\b', 1),
            (r'the\s+answer:\s*([a-d])\b', 1),
            (r'the\s+answer\s+is:\s*([a-d])\b', 1),
        ]

        for pattern, group in answer_patterns:
            match = re.search(pattern, last_200_lower)
            if match:
                return match.group(group).upper()

        # Priority 3: Standalone letter at end
        last_20_lower = response_lower[-20:]
        standalone_match = re.search(r'\b([a-d])\.?\s*$', last_20_lower)
        if standalone_match:
            return standalone_match.group(1).upper()

        return 'Unclear'

    def run_evaluation(self, sample_size=None):
        """Run MCQ evaluation on DiagnosisArena"""
        if not self.load_data():
            return False

        if not self.load_model():
            return False

        # Sample for testing
        dataset_to_eval = self.dataset[:sample_size] if sample_size else self.dataset
        print(f"\nEvaluating on {len(dataset_to_eval)} cases...")

        correct = 0
        total = 0
        unclear = 0

        for i, case in enumerate(dataset_to_eval):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(dataset_to_eval)} " +
                      f"(Accuracy so far: {correct/total*100:.1f}% on {total} cases)")

            # Format prompt
            case_info = case['Case Information']
            physical_exam = case['Physical Examination']
            diagnostic_tests = case['Diagnostic Tests']

            options_str = f"""A. {case['Options']['A']}
B. {case['Options']['B']}
C. {case['Options']['C']}
D. {case['Options']['D']}"""

            prompt = f"""According to the provided medical case, select the most appropriate diagnosis from the following four options. Put your final answer within \\boxed{{}}.

Case Information:
{case_info}

Physical Examination:
{physical_exam}

Diagnostic Tests:
{diagnostic_tests}

Options:
{options_str}

Let's think step by step."""

            # Generate response
            response = self.generate_response(prompt)

            # Extract answer
            predicted_answer = self.extract_answer(response)
            ground_truth = case['Right Option']  # Should be A-D

            # Check correctness
            if predicted_answer == 'Unclear':
                unclear += 1
            else:
                total += 1
                if predicted_answer == ground_truth:
                    correct += 1

            # Store result
            self.results.append({
                'index': i,
                'case_id': case.get('id', i),
                'specialty': case.get('specialty', 'Unknown'),
                'predicted_answer': predicted_answer,
                'ground_truth': ground_truth,
                'correct': (predicted_answer == ground_truth) if predicted_answer != 'Unclear' else None,
                'response_length': len(response)
            })

        # Calculate final accuracy
        if total == 0:
            print("\n⚠️  WARNING: No valid predictions made!")
            return False

        accuracy = correct / total

        print("\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)
        print(f"Total cases evaluated: {len(dataset_to_eval)}")
        print(f"Valid predictions: {total}")
        print(f"Unclear answers: {unclear}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy*100:.2f}%")

        if EXPECTED_ACCURACY is not None:
            diff = abs(accuracy - EXPECTED_ACCURACY)
            print(f"\nExpected accuracy: {EXPECTED_ACCURACY*100:.2f}%")
            print(f"Difference: {diff*100:.2f}%")

            if diff <= TOLERANCE:
                print(f"✅ PASS: Within ±{TOLERANCE*100}% tolerance")
                print("   Generation infrastructure validated!")
                return True
            else:
                print(f"❌ FAIL: Exceeds ±{TOLERANCE*100}% tolerance")
                print("   Possible issues:")
                print("   - Bug in generation code")
                print("   - Bug in answer extraction")
                print("   - Dataset/model version mismatch")
                print("   - Prompt format difference")
                return False
        else:
            print("\n⚠️  No expected accuracy set (EXPECTED_ACCURACY = None)")
            print("   Please check QwQ-32B paper results for DiagnosisArena")
            print("   Update EXPECTED_ACCURACY at top of script")
            print(f"   Current result: {accuracy*100:.2f}%")
            return None

def main():
    """Main function"""
    validator = QwQDiagnosisArenaValidator()

    # Run on full test set
    # For quick testing, use sample_size=50
    result = validator.run_evaluation(sample_size=None)

    if result is True:
        print("\n🎉 Validation successful!")
        exit(0)
    elif result is False:
        print("\n⚠️  Validation failed or incomplete")
        exit(1)
    else:
        print("\n📋 Results recorded, awaiting expected accuracy")
        exit(0)

if __name__ == "__main__":
    main()
