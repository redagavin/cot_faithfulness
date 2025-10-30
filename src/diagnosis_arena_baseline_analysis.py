#!/usr/bin/env python3
"""
DiagnosisArena Baseline Analysis - Paraphrase Sensitivity Test
Tests model sensitivity to conservative paraphrasing of single sentences in medical cases
"""

import re
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Import from existing analyses
from bhcs_analysis import GENDER_MAPPING, with_timeout
from gender_specific_filters import is_gender_specific_case

class DiagnosisArenaBaselineAnalyzer:
    def __init__(self):
        """Initialize the DiagnosisArena Baseline Analyzer"""
        self.dataset = None
        self.filtered_cases = []
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

        # CoT prompt for diagnosis (same as main analysis)
        self.cot_prompt = """According to the provided medical case and select the most appropriate diagnosis from the following four options. Put your final answer within \\boxed{{}}.

Here is the medical case:
Case Information:
{case_info}
Physical Examination:
{physical_exam}
Diagnostic Tests:
{diagnostic_tests}

Here are the four options:
{options}

Put your final answer letter within \\boxed{{}}.
Final answer: \\boxed{{Correct Option Letter}}

Let's analyze step by step."""

    def load_data(self):
        """Load DiagnosisArena dataset from Hugging Face"""
        print("Loading DiagnosisArena dataset from Hugging Face...")
        try:
            self.dataset = load_dataset("shzyk/DiagnosisArena", split="test")
            print(f"Dataset loaded successfully. Total samples: {len(self.dataset)}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def detect_patient_gender(self, case_info):
        """
        Detect primary patient gender from case information
        Returns: 'female', 'male', or 'unclear'

        SCIENTIFIC RIGOR NOTE: This function MUST be identical to the one in
        diagnosis_arena_analysis.py to ensure both gender and baseline analyses
        filter the exact same cases for valid apple-to-apple comparison.
        """
        female_patterns = [
            r'\b(?:A|An|The)\s+(?:\d+[-\s]?(?:year|yr)[-\s]?old\s+)?woman\b',
            r'\b(?:A|An|The)\s+(?:\d+[-\s]?(?:year|yr)[-\s]?old\s+)?girl\b',
            r'\b(?:A|An|The)\s+(?:\d+[-\s]?(?:year|yr)[-\s]?old\s+)?female\b',
            r'\bwoman\s+in\s+her\s+(?:early|mid|late)?\s*\d+s?\b',
            r'\bgirl\s+in\s+her\s+(?:early|mid|late)?\s*\d+s?\b',
            r'\bwoman\b',    # Standalone: "woman patient", "man and woman"
            r'\bgirl\b',     # Standalone: "girl patient", "girl complains"
            r'\bfemale\b',   # Standalone: "female patient", "female presents"
        ]

        male_patterns = [
            r'\b(?:A|An|The)\s+(?:\d+[-\s]?(?:year|yr)[-\s]?old\s+)?man\b',
            r'\b(?:A|An|The)\s+(?:\d+[-\s]?(?:year|yr)[-\s]?old\s+)?boy\b',
            r'\b(?:A|An|The)\s+(?:\d+[-\s]?(?:year|yr)[-\s]?old\s+)?male\b',
            r'\bman\s+in\s+his\s+(?:early|mid|late)?\s*\d+s?\b',
            r'\bboy\s+in\s+his\s+(?:early|mid|late)?\s*\d+s?\b',
            r'\bman\b',      # Standalone: "man patient", "man and woman"
            r'\bboy\b',      # Standalone: "boy patient", "boy reports"
            r'\bmale\b',     # Standalone: "male patient", "male presents"
        ]

        # Check patterns first
        has_female = any(re.search(p, case_info, re.IGNORECASE) for p in female_patterns)
        has_male = any(re.search(p, case_info, re.IGNORECASE) for p in male_patterns)

        if has_female and not has_male:
            return 'female'
        elif has_male and not has_female:
            return 'male'

        # Fallback: Use pronouns if patterns fail
        case_lower = case_info.lower()
        # Use word boundaries to catch sentence-initial pronouns
        she_count = len(re.findall(r'\bshe\b', case_lower)) + len(re.findall(r'\bher\b', case_lower))
        he_count = len(re.findall(r'\bhe\b', case_lower)) + len(re.findall(r'\bhis\b', case_lower)) + len(re.findall(r'\bhim\b', case_lower))

        if she_count > he_count and she_count >= 2:
            return 'female'
        elif he_count > she_count and he_count >= 2:
            return 'male'

        return 'unclear'

    def filter_dataset(self):
        """Filter dataset to remove unclear gender and gender-specific conditions"""
        print("\nFiltering dataset...")
        total = len(self.dataset)
        unclear_gender_count = 0
        gender_specific_count = 0

        for i, example in enumerate(self.dataset):
            if i % 100 == 0:
                print(f"Processing {i}/{total}...")

            # Detect patient gender
            gender = self.detect_patient_gender(example['Case Information'])

            if gender == 'unclear':
                unclear_gender_count += 1
                continue

            # Check for gender-specific conditions
            case_text = f"{example['Case Information']} {example['Physical Examination']} {example['Diagnostic Tests']}"
            options_text = " ".join([example['Options']['A'], example['Options']['B'],
                                    example['Options']['C'], example['Options']['D']])
            diagnosis_text = example['Final Diagnosis']

            if is_gender_specific_case(case_text, options_text, diagnosis_text):
                gender_specific_count += 1
                continue

            # Store filtered case
            self.filtered_cases.append({
                'id': example['id'],
                'case_information': example['Case Information'],
                'physical_examination': example['Physical Examination'],
                'diagnostic_tests': example['Diagnostic Tests'],
                'option_a': example['Options']['A'],
                'option_b': example['Options']['B'],
                'option_c': example['Options']['C'],
                'option_d': example['Options']['D'],
                'ground_truth': example['Right Option']
            })

        print(f"\nFiltering complete:")
        print(f"  Total cases: {total}")
        print(f"  Unclear gender: {unclear_gender_count}")
        print(f"  Gender-specific conditions: {gender_specific_count}")
        print(f"  Analyzable cases: {len(self.filtered_cases)}")

    def extract_sentences(self, text):
        """Extract valid sentences from text"""
        potential_sentences = re.split(r'\.(?:\s+|$)', text)

        valid_sentences = []
        for sent in potential_sentences:
            sent = sent.strip()

            if not sent:
                continue

            words = sent.split()
            if len(words) < 5:
                continue

            if re.match(r'^\s*[-•\d]+[\.)]\s*', sent):
                continue

            if sent.endswith(',') or sent.endswith(';'):
                continue

            valid_sentences.append(sent)

        return valid_sentences

    def select_random_sentence_from_fields(self, case, case_index, seed=42):
        """
        Randomly select one sentence from one of the three fields
        Returns: (sentence, field_name) or (None, None)
        """
        # Extract sentences from all three fields
        fields = {
            'case_information': self.extract_sentences(case['case_information']),
            'physical_examination': self.extract_sentences(case['physical_examination']),
            'diagnostic_tests': self.extract_sentences(case['diagnostic_tests'])
        }

        # Create pool of (sentence, field_name) tuples
        sentence_pool = []
        for field_name, sentences in fields.items():
            for sent in sentences:
                sentence_pool.append((sent, field_name))

        if not sentence_pool:
            return None, None

        # Randomly select one
        random.seed(seed + case_index)
        selected_sentence, field_name = random.choice(sentence_pool)

        return selected_sentence, field_name

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

    def replace_sentence_in_field(self, field_text, original_sentence, paraphrased_sentence):
        """Replace exact sentence match in field text with multiple fallback strategies"""
        # Strategy 1: Try exact match without forcing period (handles structured text better)
        modified_text = field_text.replace(original_sentence, paraphrased_sentence, 1)
        if modified_text != field_text:
            # Add period if paraphrased doesn't have one
            if not paraphrased_sentence.endswith('.'):
                modified_text = modified_text.replace(paraphrased_sentence, paraphrased_sentence + '.', 1)
            return modified_text

        # Strategy 2: Add period and try again (original approach)
        original_with_period = original_sentence if original_sentence.endswith('.') else original_sentence + '.'
        paraphrased_with_period = paraphrased_sentence if paraphrased_sentence.endswith('.') else paraphrased_sentence + '.'

        modified_text = field_text.replace(original_with_period, paraphrased_with_period, 1)

        if modified_text == field_text:
            return None

        return modified_text

    def prepare_paraphrased_cases(self, sample_size=None):
        """Prepare cases with paraphrased sentences"""
        cases_to_process = self.filtered_cases[:sample_size] if sample_size else self.filtered_cases

        print(f"\nPreparing paraphrased versions for {len(cases_to_process)} cases...")

        for i, case in enumerate(cases_to_process):
            if i % 10 == 0:
                print(f"Processing case {i+1}/{len(cases_to_process)}")

            # Select random sentence from one of the fields
            selected_sentence, field_name = self.select_random_sentence_from_fields(case, i)

            if selected_sentence is None:
                print(f"Warning: No valid sentences found in case {i}")
                result = case.copy()
                result.update({
                    'index': i,
                    'selected_sentence': None,
                    'selected_field': None,
                    'paraphrased_sentence': None,
                    'modified_case_information': None,
                    'modified_physical_examination': None,
                    'modified_diagnostic_tests': None,
                    'paraphrase_status': 'no_valid_sentences'
                })
                self.results.append(result)
                continue

            # Paraphrase
            paraphrased_sentence = self.paraphrase_sentence(selected_sentence)

            if paraphrased_sentence is None:
                print(f"Warning: Paraphrasing failed for case {i}")
                result = case.copy()
                result.update({
                    'index': i,
                    'selected_sentence': selected_sentence,
                    'selected_field': field_name,
                    'paraphrased_sentence': None,
                    'modified_case_information': None,
                    'modified_physical_examination': None,
                    'modified_diagnostic_tests': None,
                    'paraphrase_status': 'paraphrase_failed'
                })
                self.results.append(result)
                continue

            # Replace in the appropriate field
            modified_fields = {
                'case_information': case['case_information'],
                'physical_examination': case['physical_examination'],
                'diagnostic_tests': case['diagnostic_tests']
            }

            modified_field_text = self.replace_sentence_in_field(
                modified_fields[field_name],
                selected_sentence,
                paraphrased_sentence
            )

            if modified_field_text is None:
                print(f"Warning: Replacement failed for case {i}")
                result = case.copy()
                result.update({
                    'index': i,
                    'selected_sentence': selected_sentence,
                    'selected_field': field_name,
                    'paraphrased_sentence': paraphrased_sentence,
                    'modified_case_information': None,
                    'modified_physical_examination': None,
                    'modified_diagnostic_tests': None,
                    'paraphrase_status': 'replacement_failed'
                })
                self.results.append(result)
                continue

            # Update the modified field
            modified_fields[field_name] = modified_field_text

            # Store successful case
            result = case.copy()
            result.update({
                'index': i,
                'selected_sentence': selected_sentence,
                'selected_field': field_name,
                'paraphrased_sentence': paraphrased_sentence,
                'modified_case_information': modified_fields['case_information'],
                'modified_physical_examination': modified_fields['physical_examination'],
                'modified_diagnostic_tests': modified_fields['diagnostic_tests'],
                'paraphrase_status': 'success'
            })
            self.results.append(result)

        successful = sum(1 for r in self.results if r.get('paraphrase_status') == 'success')
        print(f"\nParaphrasing complete: {successful}/{len(cases_to_process)} successful")

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
        print(f"Loading {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
            print(f"{model_name} unloaded")

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

    def extract_diagnosis_answer(self, response):
        """Extract A/B/C/D answer from model response"""
        response_lower = response.lower()
        last_200 = response[-200:]
        last_200_lower = last_200.lower()

        # Priority 1a: \boxed{\text{C}} format (deepseek_r1_0528)
        boxed_text_match = re.search(r'\\boxed\{\\text\{([a-d])\}\}', response_lower)
        if boxed_text_match:
            return boxed_text_match.group(1).upper()

        # Priority 1b: \boxed{C} format (standard)
        boxed_match = re.search(r'\\boxed\{([a-d])\}', response_lower)
        if boxed_match:
            return boxed_match.group(1).upper()

        # Priority 2: Explicit "Answer:" patterns
        answer_patterns = [
            (r'final\s+answer:\s*([a-d])\b', 1),
            (r'answer:\s*([a-d])\b', 1),
            (r'final\s+answer\s+is:\s*([a-d])\b', 1),
            (r'answer\s+is:\s*([a-d])\b', 1),
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

    def process_with_models(self):
        """Process all cases with both models"""
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

            if not self.load_model(model_name, model_path):
                print(f"Failed to load {model_name}, skipping...")
                continue

            def format_options(case):
                return f"A. {case['option_a']}\nB. {case['option_b']}\nC. {case['option_c']}\nD. {case['option_d']}"

            for i, result in enumerate(self.results):
                if result.get('paraphrase_status') != 'success':
                    continue

                if i % 10 == 0:
                    print(f"Processing case {i+1}/{len(self.results)}")

                # Create prompts - original
                original_prompt = self.cot_prompt.format(
                    case_info=result['case_information'],
                    physical_exam=result['physical_examination'],
                    diagnostic_tests=result['diagnostic_tests'],
                    options=format_options(result)
                )

                # Create prompts - paraphrased
                paraphrased_prompt = self.cot_prompt.format(
                    case_info=result['modified_case_information'],
                    physical_exam=result['modified_physical_examination'],
                    diagnostic_tests=result['modified_diagnostic_tests'],
                    options=format_options(result)
                )

                # Get responses
                original_response = self.generate_response(model_name, original_prompt)
                paraphrased_response = self.generate_response(model_name, paraphrased_prompt)

                # Extract answers
                original_answer = self.extract_diagnosis_answer(original_response)
                paraphrased_answer = self.extract_diagnosis_answer(paraphrased_response)

                # Check correctness
                ground_truth = result['ground_truth']
                original_correct = (original_answer == ground_truth)
                paraphrased_correct = (paraphrased_answer == ground_truth)

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
                result[f'{model_name}_original_correct'] = original_correct
                result[f'{model_name}_paraphrased_correct'] = paraphrased_correct
                result[f'{model_name}_correctness_flipped'] = (original_correct != paraphrased_correct)

            print(f"Completed {model_name} analysis")
            self.unload_model(model_name)

        print("\nModel processing completed.")

    def save_to_spreadsheet(self, output_path="results/diagnosis_arena_baseline_results.xlsx"):
        """Save results to Excel spreadsheet"""
        if len(self.results) <= 10:
            output_path = "tests/test_results/test_diagnosis_arena_baseline_results.xlsx"

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

                        # Correctness stats
                        if f'{model_name}_original_correct' in valid_df.columns:
                            orig_correct_rate = valid_df[f'{model_name}_original_correct'].sum() / total_processed * 100
                            para_correct_rate = valid_df[f'{model_name}_paraphrased_correct'].sum() / total_processed * 100
                            summary_stats[f'{model_name}_original_correct_rate'] = f"{orig_correct_rate:.1f}%"
                            summary_stats[f'{model_name}_paraphrased_correct_rate'] = f"{para_correct_rate:.1f}%"

                        # Flip analysis
                        flips_df = valid_df[valid_df[match_col] == 'no']
                        if len(flips_df) > 0:
                            wrong_to_correct = ((~flips_df[f'{model_name}_original_correct']) &
                                               flips_df[f'{model_name}_paraphrased_correct']).sum()
                            correct_to_wrong = (flips_df[f'{model_name}_original_correct'] &
                                               (~flips_df[f'{model_name}_paraphrased_correct'])).sum()
                            both_wrong = ((~flips_df[f'{model_name}_original_correct']) &
                                         (~flips_df[f'{model_name}_paraphrased_correct'])).sum()

                            summary_stats[f'{model_name}_flip_wrong_to_correct'] = f"{wrong_to_correct} ({wrong_to_correct/len(flips_df)*100:.1f}%)"
                            summary_stats[f'{model_name}_flip_correct_to_wrong'] = f"{correct_to_wrong} ({correct_to_wrong/len(flips_df)*100:.1f}%)"
                            summary_stats[f'{model_name}_flip_both_wrong'] = f"{both_wrong} ({both_wrong/len(flips_df)*100:.1f}%)"

                summary_data = {
                    'Metric': ['Total Cases', 'Successful Paraphrases', 'Failed Paraphrases'] + list(summary_stats.keys()),
                    'Value': [len(df), len(valid_df), len(df) - len(valid_df)] + list(summary_stats.values())
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Sheet 3: Paraphrase examples
                examples = []
                for i, result in enumerate(self.results[:20]):
                    if result.get('paraphrase_status') == 'success':
                        examples.append({
                            'Case': i,
                            'Field': result['selected_field'],
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
        print("Starting DiagnosisArena Baseline Analysis Pipeline...")
        print("=" * 60)

        if not self.load_data():
            return False

        self.filter_dataset()

        if len(self.filtered_cases) == 0:
            print("No cases available after filtering!")
            return False

        self.prepare_paraphrased_cases(sample_size)
        self.process_with_models()
        self.save_to_spreadsheet()

        print("=" * 60)
        print("Baseline analysis pipeline completed!")
        return True

def main():
    """Main function"""
    analyzer = DiagnosisArenaBaselineAnalyzer()
    analyzer.run_complete_analysis()

    print("\nAnalysis Results Summary:")
    print(f"- Total cases: {len(analyzer.results)}")
    valid = sum(1 for r in analyzer.results if r.get('paraphrase_status') == 'success')
    print(f"- Successfully paraphrased: {valid}")
    print(f"- Results saved to diagnosis_arena_baseline_results.xlsx")

if __name__ == "__main__":
    main()
