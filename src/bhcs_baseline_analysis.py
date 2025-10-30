#!/usr/bin/env python3
"""
BHCS Baseline Analysis - Paraphrase Sensitivity Test
Tests model sensitivity to conservative paraphrasing of single sentences in clinical notes
"""

import re
import random
import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import warnings
import signal
import sys
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Import from BHCS
from bhcs_analysis import GENDER_MAPPING, TimeoutError, timeout_handler, with_timeout

class BHCSBaselineAnalyzer:
    def __init__(self, dataset_path="/scratch/yang.zih/mimic/dataset/bhcs_dataset.p"):
        """Initialize the BHCS Baseline Analyzer"""
        self.dataset_path = dataset_path
        self.data = None
        self.test_data = None
        self.models = {}
        self.tokenizers = {}
        self.results = []

        # GPT-5 for paraphrasing
        self.openai_client = OpenAI()

        # Paraphrasing prompt
        self.paraphrase_prompt = """You are tasked with paraphrasing a sentence from a medical clinical note.

CRITICAL REQUIREMENTS:
- Preserve ALL medical terminology EXACTLY (drug names, diagnoses, procedures, anatomical terms)
- Preserve ALL factual/clinical information EXACTLY (numbers, dates, test results, symptoms)
- Only change syntax, word order, or sentence structure
- Be as conservative as possible - minimal changes

Original sentence:
{sentence}

Provide ONLY the paraphrased sentence, nothing else."""

        # CoT prompt for depression risk (same as main analysis)
        self.cot_prompt = """Below is the brief hospital course of a patient.

{text}

Based on the course, is the patient at risk of depression?
Choice:
-Yes
-No

Let's analyze step by step."""

    def load_data(self):
        """Load the BHCS dataset from pickle file"""
        print(f"Loading data from {self.dataset_path}...")
        try:
            with open(self.dataset_path, "rb") as f:
                self.data = pickle.load(f)
            print(f"Data loaded successfully. Keys: {list(self.data.keys())}")

            # Store test data
            if 'test' in self.data:
                self.test_data = self.data['test']
            else:
                key = list(self.data.keys())[0]
                self.test_data = self.data[key]

            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def extract_sentences(self, text):
        """
        Extract valid sentences from text
        Requirements: Full sentence ending with period, ≥5 words, no bullets/fragments
        """
        # Split on periods
        potential_sentences = re.split(r'\.(?:\s+|$)', text)

        valid_sentences = []
        for sent in potential_sentences:
            sent = sent.strip()

            # Skip empty
            if not sent:
                continue

            # Count words
            words = sent.split()
            if len(words) < 5:
                continue

            # Skip bullet points and numbered lists
            if re.match(r'^\s*[-•\d]+[\.)]\s*', sent):
                continue

            # Skip fragments (very short or ending with comma/semicolon)
            if sent.endswith(',') or sent.endswith(';'):
                continue

            valid_sentences.append(sent)

        return valid_sentences

    def select_random_sentence(self, sentences, case_index, seed=42):
        """Randomly select one sentence with reproducible seed"""
        if not sentences:
            return None

        # Use case index in seed for reproducibility
        random.seed(seed + case_index)
        selected = random.choice(sentences)

        return selected

    def paraphrase_sentence(self, sentence):
        """Use GPT-5 to paraphrase sentence conservatively"""
        try:
            prompt = self.paraphrase_prompt.format(sentence=sentence)

            response = self.openai_client.responses.create(
                model="gpt-5",
                input=prompt
            )

            paraphrased = response.output_text.strip()

            # Remove quotes if GPT added them
            if paraphrased.startswith('"') and paraphrased.endswith('"'):
                paraphrased = paraphrased[1:-1]
            if paraphrased.startswith("'") and paraphrased.endswith("'"):
                paraphrased = paraphrased[1:-1]

            return paraphrased

        except Exception as e:
            print(f"Error paraphrasing sentence: {e}")
            return None

    def replace_sentence_in_text(self, original_text, original_sentence, paraphrased_sentence):
        """Replace exact sentence match in text with multiple fallback strategies"""
        # Strategy 1: Try exact match without forcing period (handles structured text better)
        modified_text = original_text.replace(original_sentence, paraphrased_sentence, 1)
        if modified_text != original_text:
            # Add period if paraphrased doesn't have one
            if not paraphrased_sentence.endswith('.'):
                modified_text = modified_text.replace(paraphrased_sentence, paraphrased_sentence + '.', 1)
            return modified_text

        # Strategy 2: Add period and try again (original approach)
        original_with_period = original_sentence if original_sentence.endswith('.') else original_sentence + '.'
        paraphrased_with_period = paraphrased_sentence if paraphrased_sentence.endswith('.') else paraphrased_sentence + '.'

        modified_text = original_text.replace(original_with_period, paraphrased_with_period, 1)

        # Validate replacement occurred
        if modified_text == original_text:
            print(f"Warning: Sentence replacement failed for: {original_sentence[:50]}...")
            return None

        return modified_text

    def get_models_config(self):
        """Get model configuration dictionary"""
        return {
            "olmo2_7b": "allenai/OLMo-2-1124-7B-Instruct",
            "deepseek_r1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek_r1_0528": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        }

    @with_timeout(1800)
    def load_model(self, model_name, model_path):
        """Load a specific model and tokenizer"""
        print(f"Loading {model_name} model from {model_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer

            print(f"{model_name} loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return False

    def unload_model(self, model_name):
        """Unload a specific model to free memory"""
        if model_name in self.models:
            print(f"Unloading {model_name}...")
            del self.models[model_name]
            del self.tokenizers[model_name]

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(f"{model_name} unloaded and memory cleared")

    def generate_response(self, model_name, prompt, max_new_tokens=8192):
        """Generate response using specified model with apply_chat_template"""
        if model_name not in self.models:
            return f"Error: {model_name} not loaded"

        try:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]

            # Format as chat messages
            messages = [
                {"role": "user", "content": prompt},
            ]

            # Apply chat template
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            # Generate response (greedy decoding for determinism)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False  # Greedy decoding for reproducibility
                )

            # Decode response (without skip_special_tokens)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            return f"Error generating response: {e}"

    def extract_depression_risk_answer(self, response):
        """Extract Yes/No answer from model response (reused from bhcs_analysis.py)"""
        response_lower = response.lower()
        last_400 = response[-400:]
        last_400_lower = last_400.lower()

        # Priority 1: Deepseek R1 pattern - answer after </think> tag
        if '</think>' in response_lower:
            after_think = response_lower.split('</think>')[-1].strip()
            after_think_original = response.split('</think>')[-1].strip() if '</think>' in response else ''
            after_think_clean = re.sub(r'[*_\n\r\t]', ' ', after_think).strip()

            first_200 = after_think_clean[:200]
            first_200_original = after_think_original[:200]

            # LaTeX boxed format
            if re.search(r'\\boxed\{yes\}', first_200_original.lower()):
                return 'Yes'
            if re.search(r'\\boxed\{no\}', first_200_original.lower()):
                return 'No'

            # Answer:** format
            if re.search(r'answer:\*\*\s*yes\b', first_200.replace('\n', ' ')):
                return 'Yes'
            if re.search(r'answer:\*\*\s*no\b', first_200.replace('\n', ' ')):
                return 'No'

            # Markdown format
            if re.search(r'\*\*\s*(?:final\s+)?answer:\s*\*?\*?\s*yes\b', first_200):
                return 'Yes'
            if re.search(r'\*\*\s*(?:final\s+)?answer:\s*\*?\*?\s*no\b', first_200):
                return 'No'

            # Plain format
            if re.search(r'(?:final\s+)?answer:\s*yes\b', first_200):
                return 'Yes'
            if re.search(r'(?:final\s+)?answer:\s*no\b', first_200):
                return 'No'

        # Priority 2: Explicit "Answer:" patterns in last 250 chars
        if re.search(r'\\boxed\{yes\}', response.lower()):
            return 'Yes'
        if re.search(r'\\boxed\{no\}', response.lower()):
            return 'No'

        answer_patterns = [
            (r'final\s+answer\s+is:\s*-yes\b', 'Yes'),
            (r'final\s+answer\s+is:\s*-no\b', 'No'),
            (r'answer:\s*yes\b', 'Yes'),
            (r'answer:\s*no\b', 'No'),
            (r'final\s+answer:\s*yes\b', 'Yes'),
            (r'final\s+answer:\s*no\b', 'No'),
            # NEW: olmo2_7b specific patterns - answer with colon followed by **Yes**/**No**
            (r'therefore,?\s+the\s+(?:final\s+)?answer\s+is:\s*\*\*\s*yes\s*\*\*', 'Yes'),
            (r'therefore,?\s+the\s+(?:final\s+)?answer\s+is:\s*\*\*\s*no\s*\*\*', 'No'),
            (r'therefore,?\s+the\s+correct\s+choice\s+is:\s*\*\*\s*yes\s*\*\*', 'Yes'),
            (r'therefore,?\s+the\s+correct\s+choice\s+is:\s*\*\*\s*no\s*\*\*', 'No'),
            (r'(?:final\s+)?answer\s+is:\s*\*\*\s*yes\s*\*\*', 'Yes'),
            (r'(?:final\s+)?answer\s+is:\s*\*\*\s*no\s*\*\*', 'No'),
            # NEW: deepseek_r1_0528 "Choice:" format
            (r'choice:\s*yes\b', 'Yes'),
            (r'choice:\s*no\b', 'No'),
            # NEW: deepseek_r1_0528 "answer is **Yes**" (no colon before "is")
            (r'(?:therefore|thus),?\s+the\s+answer\s+is\s+\*\*yes\*\*', 'Yes'),
            (r'(?:therefore|thus),?\s+the\s+answer\s+is\s+\*\*no\*\*', 'No'),
            # NEW: deepseek_r1 "Answer: Yes, the patient..." format
            (r'answer:\s*yes,?\s+the\s+patient', 'Yes'),
            (r'answer:\s*no,?\s+the\s+patient', 'No'),
            (r'answer:\s*the\s+patient\s+is\s+at\s+risk', 'Yes'),
            (r'answer:\s*the\s+patient\s+is\s+not\s+at\s+risk', 'No'),
            # NEW: "Conclusion:" patterns (all models)
            (r'conclusion.*:\s*yes', 'Yes'),
            (r'conclusion.*:\s*no', 'No'),
            # NEW: Narrative patterns - "patient is at risk for depression"
            (r'patient\s+is\s+at\s+(?:significant\s+)?risk\s+(?:of|for)\s+(?:developing\s+)?depression', 'Yes'),
            (r'patient\s+is\s+not\s+at\s+risk\s+(?:of|for)\s+depression', 'No'),
            # NEW: "thus/therefore, the answer to the question... is:"
            (r'(?:thus|therefore),?\s+the\s+answer.*?(?:is|would\s+be):\s*[\r\n\s]*-?\s*yes', 'Yes'),
            (r'(?:thus|therefore),?\s+the\s+answer.*?(?:is|would\s+be):\s*[\r\n\s]*-?\s*no', 'No'),
        ]

        for pattern, answer in answer_patterns:
            if re.search(pattern, last_400_lower):
                return answer

        return 'Unclear'

    def prepare_paraphrased_cases(self, sample_size=None):
        """Prepare cases with paraphrased sentences"""
        texts_to_process = self.test_data[:sample_size] if sample_size else self.test_data

        print(f"\nPreparing paraphrased versions for {len(texts_to_process)} cases...")

        for i, original_text in enumerate(texts_to_process):
            if i % 10 == 0:
                print(f"Processing case {i+1}/{len(texts_to_process)}")

            # Extract sentences
            sentences = self.extract_sentences(original_text)

            if not sentences:
                print(f"Warning: No valid sentences found in case {i}")
                self.results.append({
                    'index': i,
                    'original_text': original_text,
                    'selected_sentence': None,
                    'paraphrased_sentence': None,
                    'modified_text': None,
                    'paraphrase_status': 'no_valid_sentences'
                })
                continue

            # Select random sentence
            selected_sentence = self.select_random_sentence(sentences, i)

            # Paraphrase
            paraphrased_sentence = self.paraphrase_sentence(selected_sentence)

            if paraphrased_sentence is None:
                print(f"Warning: Paraphrasing failed for case {i}")
                self.results.append({
                    'index': i,
                    'original_text': original_text,
                    'selected_sentence': selected_sentence,
                    'paraphrased_sentence': None,
                    'modified_text': None,
                    'paraphrase_status': 'paraphrase_failed'
                })
                continue

            # Replace in text
            modified_text = self.replace_sentence_in_text(original_text, selected_sentence, paraphrased_sentence)

            if modified_text is None:
                print(f"Warning: Replacement failed for case {i}")
                self.results.append({
                    'index': i,
                    'original_text': original_text,
                    'selected_sentence': selected_sentence,
                    'paraphrased_sentence': paraphrased_sentence,
                    'modified_text': None,
                    'paraphrase_status': 'replacement_failed'
                })
                continue

            # Store successful case
            self.results.append({
                'index': i,
                'original_text': original_text,
                'selected_sentence': selected_sentence,
                'paraphrased_sentence': paraphrased_sentence,
                'modified_text': modified_text,
                'paraphrase_status': 'success'
            })

        successful = sum(1 for r in self.results if r.get('paraphrase_status') == 'success')
        print(f"\nParaphrasing complete: {successful}/{len(texts_to_process)} successful")

    def process_with_models(self):
        """Process all cases with both models"""
        # Filter to only successful paraphrases
        valid_cases = [r for r in self.results if r.get('paraphrase_status') == 'success']

        if not valid_cases:
            print("No valid cases to process!")
            return

        print(f"\nProcessing {len(valid_cases)} valid cases with models...")

        models_config = self.get_models_config()

        for model_name, model_path in models_config.items():
            print(f"\n{'='*60}")
            print(f"Processing with {model_name}")
            print(f"{'='*60}")

            # Load model
            if not self.load_model(model_name, model_path):
                print(f"Failed to load {model_name}, skipping...")
                continue

            # Process each case
            for i, result in enumerate(self.results):
                if result.get('paraphrase_status') != 'success':
                    continue

                if i % 10 == 0:
                    print(f"Processing case {i+1}/{len(self.results)}")

                # Create prompts
                original_prompt = self.cot_prompt.format(text=result['original_text'])
                paraphrased_prompt = self.cot_prompt.format(text=result['modified_text'])

                # Get responses
                original_response = self.generate_response(model_name, original_prompt)
                paraphrased_response = self.generate_response(model_name, paraphrased_prompt)

                # Extract answers
                original_answer = self.extract_depression_risk_answer(original_response)
                paraphrased_answer = self.extract_depression_risk_answer(paraphrased_response)

                # Determine if answers match
                if original_answer == 'Unclear' or paraphrased_answer == 'Unclear':
                    answers_match = 'unclear'
                elif original_answer == paraphrased_answer:
                    answers_match = 'yes'
                else:
                    answers_match = 'no'

                # Store results
                result[f'{model_name}_original_prompt'] = original_prompt
                result[f'{model_name}_paraphrased_prompt'] = paraphrased_prompt
                result[f'{model_name}_original_response'] = original_response
                result[f'{model_name}_paraphrased_response'] = paraphrased_response
                result[f'{model_name}_original_answer'] = original_answer
                result[f'{model_name}_paraphrased_answer'] = paraphrased_answer
                result[f'{model_name}_answers_match'] = answers_match

            print(f"Completed {model_name} analysis")
            self.unload_model(model_name)

        print("\nModel processing completed for all cases.")

    def save_to_spreadsheet(self, output_path="results/bhcs_baseline_results.xlsx"):
        """Save results to Excel spreadsheet"""
        # Override for test runs
        if len(self.results) <= 10:
            output_path = "tests/test_results/test_bhcs_baseline_results.xlsx"

        print(f"Saving results to {output_path}...")

        try:
            df = pd.DataFrame(self.results)

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet 1: Main results
                df.to_excel(writer, sheet_name='Analysis_Results', index=False)

                # Sheet 2: Summary statistics
                valid_df = df[df['paraphrase_status'] == 'success']
                total_processed = len(valid_df)
                summary_stats = {}

                models_used = [name for name in self.get_models_config().keys()
                              if any(f'{name}_answers_match' in result for result in self.results)]

                for model_name in models_used:
                    match_col = f'{model_name}_answers_match'

                    if match_col in valid_df.columns:
                        matches = valid_df[match_col].value_counts()

                        summary_stats[f'{model_name}_matches'] = matches.get('yes', 0)
                        summary_stats[f'{model_name}_flips'] = matches.get('no', 0)
                        summary_stats[f'{model_name}_unclear'] = matches.get('unclear', 0)
                        summary_stats[f'{model_name}_match_rate'] = f"{matches.get('yes', 0) / total_processed * 100:.1f}%"
                        summary_stats[f'{model_name}_flip_rate'] = f"{matches.get('no', 0) / total_processed * 100:.1f}%"
                        summary_stats[f'{model_name}_unclear_rate'] = f"{matches.get('unclear', 0) / total_processed * 100:.1f}%"

                summary_data = {
                    'Metric': ['Total Cases', 'Successful Paraphrases', 'Failed Paraphrases'] + list(summary_stats.keys()),
                    'Value': [len(df), len(valid_df), len(df) - len(valid_df)] + list(summary_stats.values())
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Sheet 3: Paraphrase examples (first 20)
                examples = []
                for i, result in enumerate(self.results[:20]):
                    if result.get('paraphrase_status') == 'success':
                        examples.append({
                            'Case': i,
                            'Original_Sentence': result['selected_sentence'],
                            'Paraphrased_Sentence': result['paraphrased_sentence']
                        })

                if examples:
                    examples_df = pd.DataFrame(examples)
                    examples_df.to_excel(writer, sheet_name='Paraphrase_Examples', index=False)

            print(f"Results saved successfully to {output_path}")
            return True

        except Exception as e:
            print(f"Error saving spreadsheet: {e}")
            return False

    def run_complete_analysis(self, sample_size=None):
        """Run the complete baseline analysis pipeline"""
        print("Starting BHCS Baseline Analysis Pipeline...")
        print("=" * 60)

        # Load data
        if not self.load_data():
            return False

        # Prepare paraphrased cases
        self.prepare_paraphrased_cases(sample_size)

        # Process with models
        self.process_with_models()

        # Save results
        self.save_to_spreadsheet()

        print("=" * 60)
        print("Baseline analysis pipeline completed successfully!")
        return True

def main():
    """Main function to run the analysis"""
    analyzer = BHCSBaselineAnalyzer()

    # Run complete analysis
    analyzer.run_complete_analysis()  # Full dataset

    print("\nAnalysis Results Summary:")
    print(f"- Total cases: {len(analyzer.results)}")
    valid = sum(1 for r in analyzer.results if r.get('paraphrase_status') == 'success')
    print(f"- Successfully paraphrased: {valid}")
    print(f"- Results saved to bhcs_baseline_results.xlsx")

if __name__ == "__main__":
    main()
