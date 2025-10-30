#!/usr/bin/env python3
"""
Quick test to verify if GPT-5 responses.create() supports seed and temperature
"""

from openai import OpenAI

def test_gpt5_parameters():
    """Test if GPT-5 supports seed and temperature parameters"""
    client = OpenAI()

    test_prompt = "What is 2+2? Answer with just the number."

    print("Testing GPT-5 parameter support...\n")

    # Test 1: Basic call (no parameters)
    print("Test 1: Basic call (no seed, no temperature)")
    try:
        response = client.responses.create(
            model="gpt-5",
            input=test_prompt
        )
        print(f"✅ Basic call works")
        print(f"Response: {response.output_text.strip()}\n")
    except Exception as e:
        print(f"❌ Basic call failed: {e}\n")
        return

    # Test 2: With temperature only
    print("Test 2: With temperature=0")
    try:
        response = client.responses.create(
            model="gpt-5",
            input=test_prompt,
            temperature=0
        )
        print(f"✅ Temperature parameter supported")
        print(f"Response: {response.output_text.strip()}\n")
    except Exception as e:
        print(f"❌ Temperature parameter NOT supported: {e}\n")

    # Test 3: With seed only
    print("Test 3: With seed=42")
    try:
        response = client.responses.create(
            model="gpt-5",
            input=test_prompt,
            seed=42
        )
        print(f"✅ Seed parameter supported")
        print(f"Response: {response.output_text.strip()}\n")
    except Exception as e:
        print(f"❌ Seed parameter NOT supported: {e}\n")

    # Test 4: With both seed and temperature
    print("Test 4: With seed=42 and temperature=0")
    try:
        response = client.responses.create(
            model="gpt-5",
            input=test_prompt,
            seed=42,
            temperature=0
        )
        print(f"✅ Both parameters supported")
        print(f"Response: {response.output_text.strip()}\n")
    except Exception as e:
        print(f"❌ Both parameters NOT supported: {e}\n")

    # Test 5: Determinism test (if seed works)
    print("Test 5: Determinism verification (3 runs with seed=42, temperature=0)")
    try:
        responses = []
        for i in range(3):
            response = client.responses.create(
                model="gpt-5",
                input=test_prompt,
                seed=42,
                temperature=0
            )
            responses.append(response.output_text.strip())
            print(f"Run {i+1}: {responses[-1]}")

        if all(r == responses[0] for r in responses):
            print("✅ DETERMINISTIC: All 3 responses identical")
        else:
            print("❌ NON-DETERMINISTIC: Responses differ")
            print(f"Unique responses: {len(set(responses))}")
    except Exception as e:
        print(f"❌ Determinism test failed: {e}")

if __name__ == "__main__":
    test_gpt5_parameters()
