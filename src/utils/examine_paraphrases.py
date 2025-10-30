"""
Examine all paraphrased sentences and full notes from baseline test results
"""
import pandas as pd
import os

def examine_baseline_results(file_path, dataset_name):
    """
    Examine paraphrasing quality for a baseline test result file
    """
    if not os.path.exists(file_path):
        print(f"\n❌ File not found: {file_path}")
        return

    print(f"\n{'='*80}")
    print(f"{dataset_name} - Paraphrasing Quality Assessment")
    print(f"{'='*80}")

    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name='Analysis_Results')

    print(f"\nTotal cases: {len(df)}")

    # Examine each case
    for idx, row in df.iterrows():
        case_num = row.get('Case', idx)
        paraphrase_success = row.get('Paraphrase Success', 'Unknown')

        print(f"\n{'-'*80}")
        print(f"CASE {case_num} - Paraphrase Success: {paraphrase_success}")
        print(f"{'-'*80}")

        if paraphrase_success != True and paraphrase_success != 'True':
            print("⚠️  PARAPHRASE FAILED - Skipping detailed examination")
            continue

        # Get the selected sentence and paraphrased version
        original_sentence = row.get('Selected Sentence', 'N/A')
        paraphrased_sentence = row.get('Paraphrased Sentence', 'N/A')

        print(f"\n📝 ORIGINAL SENTENCE:")
        print(f"   {original_sentence}")
        print(f"\n📝 PARAPHRASED SENTENCE:")
        print(f"   {paraphrased_sentence}")

        # Check conservativeness
        if original_sentence == 'N/A' or paraphrased_sentence == 'N/A':
            print("\n⚠️  Missing sentence data")
            continue

        # Simple conservativeness checks
        orig_words = set(original_sentence.lower().split())
        para_words = set(paraphrased_sentence.lower().split())

        added_words = para_words - orig_words
        removed_words = orig_words - para_words

        print(f"\n📊 WORD-LEVEL CHANGES:")
        print(f"   Original words: {len(orig_words)}")
        print(f"   Paraphrased words: {len(para_words)}")
        print(f"   Words added: {len(added_words)} → {sorted(added_words) if added_words else 'None'}")
        print(f"   Words removed: {len(removed_words)} → {sorted(removed_words) if removed_words else 'None'}")

        # Check for substantial length changes (red flag for non-conservative paraphrasing)
        length_change_pct = abs(len(paraphrased_sentence) - len(original_sentence)) / len(original_sentence) * 100
        print(f"   Length change: {length_change_pct:.1f}%")

        if length_change_pct > 30:
            print("   ⚠️  WARNING: Large length change (>30%) - may not be conservative")
        elif length_change_pct > 15:
            print("   ⚡ CAUTION: Moderate length change (>15%)")
        else:
            print("   ✓ Length change acceptable")

        # Examine the full note after paraphrasing
        print(f"\n📄 FULL NOTE AFTER PARAPHRASING:")

        if dataset_name == "BHCS":
            original_note = row.get('Original Note', 'N/A')
            modified_note = row.get('Modified Note', 'N/A')
        elif dataset_name == "DiagnosisArena":
            # DiagnosisArena has field-specific notes
            field_name = row.get('Selected Field', 'Unknown')
            original_note = row.get(f'Original {field_name}', 'N/A')
            modified_note = row.get(f'Modified {field_name}', 'N/A')
        else:  # MedXpertQA
            original_note = row.get('Original Question', 'N/A')
            modified_note = row.get('Modified Question', 'N/A')

        if original_note == 'N/A' or modified_note == 'N/A':
            print("   ⚠️  Missing note data")
            continue

        # Check if the sentence replacement worked correctly
        if paraphrased_sentence not in modified_note:
            print(f"   ❌ ERROR: Paraphrased sentence NOT found in modified note!")
            print(f"   This indicates a replacement failure.")
        else:
            print(f"   ✓ Paraphrased sentence found in modified note")

        if original_sentence in modified_note:
            print(f"   ❌ ERROR: Original sentence STILL in modified note!")
            print(f"   This indicates replacement did not work properly.")
        else:
            print(f"   ✓ Original sentence successfully replaced")

        # Show a snippet of the modified note around the paraphrased sentence
        if paraphrased_sentence in modified_note:
            start_idx = modified_note.find(paraphrased_sentence)
            context_start = max(0, start_idx - 100)
            context_end = min(len(modified_note), start_idx + len(paraphrased_sentence) + 100)

            context = modified_note[context_start:context_end]
            if context_start > 0:
                context = "..." + context
            if context_end < len(modified_note):
                context = context + "..."

            print(f"\n   📌 CONTEXT IN MODIFIED NOTE:")
            print(f"   {context}")

def main():
    """
    Examine all three baseline test results
    """
    print("="*80)
    print("BASELINE PARAPHRASING QUALITY EXAMINATION")
    print("="*80)

    test_files = [
        ('/scratch/yang.zih/cot/test_bhcs_baseline_results.xlsx', 'BHCS'),
        ('/scratch/yang.zih/cot/test_diagnosis_arena_baseline_results.xlsx', 'DiagnosisArena'),
        ('/scratch/yang.zih/cot/test_medxpertqa_baseline_results.xlsx', 'MedXpertQA'),
    ]

    for file_path, dataset_name in test_files:
        try:
            examine_baseline_results(file_path, dataset_name)
        except Exception as e:
            print(f"\n❌ Error examining {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("EXAMINATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
