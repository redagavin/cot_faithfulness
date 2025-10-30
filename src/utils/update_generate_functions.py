"""
Update generate_response functions in remaining files
"""

files = [
    "medxpertqa_analysis.py",
    "bhcs_baseline_analysis.py",
    "diagnosis_arena_baseline_analysis.py",
    "medxpertqa_baseline_analysis.py"
]

old_function_template = """    def generate_response(self, model_name, prompt, max_new_tokens=8192):
        \"\"\"Generate response using specified model\"\"\"
        if model_name not in self.models:
            return f"Error: {model_name} not loaded"

        try:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            return f"Error generating response: {e}\""""

new_function = """    def generate_response(self, model_name, prompt, max_new_tokens=8192):
        \"\"\"Generate response using specified model with apply_chat_template\"\"\"
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
            return f"Error generating response: {e}\""""

for filename in files:
    print(f"Processing {filename}...")

    with open(filename, 'r') as f:
        content = f.read()

    if old_function_template in content:
        content = content.replace(old_function_template, new_function)
        print(f"  ✓ Updated generate_response function")

        with open(filename, 'w') as f:
            f.write(content)
    else:
        print(f"  ⚠ Pattern not found (may already be updated)")

print("\nAll files processed!")
