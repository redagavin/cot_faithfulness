#!/usr/bin/env python3
"""
Test script to diagnose model loading issues
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import time

def test_environment():
    """Test the environment setup"""
    print("="*50)
    print("ENVIRONMENT TEST")
    print("="*50)

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Total memory: {props.total_memory / 1024**3:.1f}GB")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.1f}GB")
            print(f"  Memory cached: {torch.cuda.memory_reserved(i) / 1024**3:.1f}GB")
    else:
        print("No CUDA GPUs available")

def test_model_loading(model_name, model_path):
    """Test loading a specific model"""
    print(f"\n{'='*50}")
    print(f"TESTING {model_name}")
    print(f"{'='*50}")

    try:
        print(f"Step 1: Loading tokenizer for {model_name}...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"Tokenizer loaded in {time.time() - start_time:.1f} seconds")

        print(f"Step 2: Loading model for {model_name}...")
        start_time = time.time()

        # Try minimal loading first
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        print(f"Model loaded in {time.time() - start_time:.1f} seconds")
        print(f"Model device: {next(model.parameters()).device}")

        if torch.cuda.is_available():
            print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")

        # Test a simple generation
        print("Step 3: Testing simple generation...")
        test_prompt = "The patient is"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Test generation successful: '{response}'")

        # Clean up
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"{model_name} test PASSED")
        return True

    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    test_environment()

    models_to_test = {
        "olmo2_7b": "allenai/OLMo-2-1124-7B",
        "deepseek_r1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    }

    results = {}
    for model_name, model_path in models_to_test.items():
        results[model_name] = test_model_loading(model_name, model_path)

    print(f"\n{'='*50}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*50}")

    for model_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{model_name}: {status}")

    if all(results.values()):
        print("\nAll tests passed! Models should work in main script.")
    else:
        print("\nSome tests failed. Check the errors above.")

if __name__ == "__main__":
    main()