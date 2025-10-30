#!/bin/bash

# Quick updates for remaining files

files=(
    "src/medxpertqa_analysis.py"
    "src/bhcs_baseline_analysis.py"
    "src/diagnosis_arena_baseline_analysis.py"
    "src/medxpertqa_baseline_analysis.py"
)

for file in "${files[@]}"; do
    echo "Updating $file..."

    # Update CoT prompt
    sed -i "s/Let's think step by step:/Let's analyze step by step./g" "$file"

    # Add new model (find the deepseek_r1 line and add new model after it)
    sed -i '/\"deepseek_r1\": \"deepseek-ai\/DeepSeek-R1-Distill-Llama-8B\"/a\            \"deepseek_r1_0528\": \"deepseek-ai\/DeepSeek-R1-0528-Qwen3-8B\"' "$file"

    echo "✓ $file updated (CoT prompt and model added)"
done

echo "Done! Note: generate_response functions need manual update."
