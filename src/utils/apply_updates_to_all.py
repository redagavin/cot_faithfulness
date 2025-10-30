"""
Apply all updates to all 6 analysis scripts
"""
import re

# List of all analysis files
files = [
    'diagnosis_arena_analysis.py',
    'medxpertqa_analysis.py',
    'bhcs_baseline_analysis.py',
    'diagnosis_arena_baseline_analysis.py',
    'medxpertqa_baseline_analysis.py'
]

for filename in files:
    print(f"\n{'='*80}")
    print(f"Processing {filename}")
    print(f"{'='*80}")

    with open(filename, 'r') as f:
        content = f.read()

    # Change 1: Update CoT prompt ending
    if 'Let\'s think step by step:' in content:
        content = content.replace('Let\'s think step by step:', 'Let\'s analyze step by step.')
        print("✓ Updated CoT prompt ending")
    elif 'Let\'s analyze step by step.' in content:
        print("✓ CoT prompt already updated")
    else:
        print("⚠ CoT prompt pattern not found")

    # Change 2: Update get_models_config to add new DeepSeek model
    old_models_config = '''    def get_models_config(self):
        """Get model configuration dictionary"""
        return {
            "olmo2_7b": "allenai/OLMo-2-1124-7B-Instruct",
            "deepseek_r1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        }'''

    new_models_config = '''    def get_models_config(self):
        """Get model configuration dictionary"""
        return {
            "olmo2_7b": "allenai/OLMo-2-1124-7B-Instruct",
            "deepseek_r1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek_r1_0528": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        }'''

    if old_models_config in content:
        content = content.replace(old_models_config, new_models_config)
        print("✓ Added new DeepSeek-R1-0528 model")
    elif 'deepseek_r1_0528' in content:
        print("✓ DeepSeek-R1-0528 model already added")
    else:
        print("⚠ Model config pattern not found")

    # Change 3: Update generate_response function
    # This is complex, so we'll use a regex pattern
    old_generate_pattern = r'def generate_response\(self, model_name, prompt, max_new_tokens=\d+\):[\s\S]*?# Tokenize input\s*inputs = tokenizer\(prompt, return_tensors="pt"\)[\s\S]*?pad_token_id=tokenizer\.eos_token_id[\s\S]*?skip_special_tokens=True\)'

    new_generate_code = '''def generate_response(self, model_name, prompt, max_new_tokens=8192):
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

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True
                )

            # Decode response (without skip_special_tokens)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
            return response.strip()

        except Exception as e:
            return f"Error generating response: {e}"'''

    # Find and replace the generate_response function
    if 'apply_chat_template' not in content:
        # Try to find the function more carefully
        match = re.search(r'(    def generate_response\(self, model_name, prompt.*?\n(?:        .*\n)*?        except Exception as e:\n            return f"Error generating response: \{e\}")', content, re.DOTALL)
        if match:
            content = content.replace(match.group(0), '    ' + new_generate_code)
            print("✓ Updated generate_response function")
        else:
            print("⚠ Could not find generate_response function pattern")
    else:
        print("✓ generate_response already uses apply_chat_template")

    # Write back
    with open(filename, 'w') as f:
        f.write(content)

    print(f"✓ Completed {filename}")

print("\n" + "="*80)
print("ALL FILES UPDATED")
print("="*80)
