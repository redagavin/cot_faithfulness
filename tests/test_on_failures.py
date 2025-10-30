"""
Test improved replacement on actual failed cases
"""
import pandas as pd
from improved_replacement import replace_sentence_robust

def test_on_actual_failures():
    """Test improved replacement on actual failed cases"""

    print("="*100)
    print("TESTING IMPROVED REPLACEMENT ON ACTUAL FAILURES")
    print("="*100)

    # Test DiagnosisArena failures
    print("\n" + "="*100)
    print("DIAGNOSISARENA FAILURES")
    print("="*100)

    df_da = pd.read_excel('test_diagnosis_arena_baseline_results.xlsx', sheet_name='Analysis_Results')
    failed_da = df_da[df_da['paraphrase_status'] == 'replacement_failed']

    for idx, row in failed_da.iterrows():
        case_num = row['index']
        field_name = row['selected_field']
        original_sent = row['selected_sentence']
        paraphrased_sent = row['paraphrased_sentence']
        original_text = row[field_name]

        print(f"\n{'-'*100}")
        print(f"Case {case_num} - Field: {field_name}")
        print(f"{'-'*100}")

        print(f"\nOriginal sentence (first 200 chars):")
        print(f"  {str(original_sent)[:200]}...")

        print(f"\nParaphrased sentence (first 200 chars):")
        print(f"  {str(paraphrased_sent)[:200]}...")

        # Test improved replacement
        result, strategy = replace_sentence_robust(original_text, original_sent, paraphrased_sent)

        if result is not None:
            print(f"\n✓ SUCCESS with strategy: {strategy}")
            print(f"\nVerification:")
            print(f"  Paraphrased sentence in result: {paraphrased_sent in result}")
            print(f"  Original sentence in result: {original_sent in result}")
        else:
            print(f"\n❌ STILL FAILED with strategy: {strategy}")
            print(f"\nDiagnosis:")
            print(f"  Original sentence in original text: {str(original_sent) in str(original_text)}")
            print(f"  Original sentence type: {type(original_sent)}")

    # Test MedXpertQA failures
    print("\n" + "="*100)
    print("MEDXPERTQA FAILURES")
    print("="*100)

    df_mx = pd.read_excel('test_medxpertqa_baseline_results.xlsx', sheet_name='Analysis_Results')
    failed_mx = df_mx[df_mx['paraphrase_status'] == 'replacement_failed']

    for idx, row in failed_mx.iterrows():
        case_num = row['index']
        original_sent = row['selected_sentence']
        paraphrased_sent = row['paraphrased_sentence']
        original_text = row['question']

        print(f"\n{'-'*100}")
        print(f"Case {case_num}")
        print(f"{'-'*100}")

        print(f"\nOriginal sentence (first 200 chars):")
        print(f"  {str(original_sent)[:200]}...")

        print(f"\nParaphrased sentence (first 200 chars):")
        print(f"  {str(paraphrased_sent)[:200]}...")

        # Test improved replacement
        result, strategy = replace_sentence_robust(original_text, original_sent, paraphrased_sent)

        if result is not None:
            print(f"\n✓ SUCCESS with strategy: {strategy}")
            print(f"\nVerification:")
            print(f"  Paraphrased sentence in result: {str(paraphrased_sent)[:100] in result}")
        else:
            print(f"\n❌ STILL FAILED with strategy: {strategy}")

if __name__ == '__main__':
    test_on_actual_failures()
