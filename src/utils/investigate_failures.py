"""
Investigate paraphrasing failures to understand the root cause
"""
import pandas as pd

def investigate_failures(file_path, dataset_name):
    """Examine failed paraphrasing cases"""
    print(f"\n{'='*100}")
    print(f"{dataset_name} - FAILURE INVESTIGATION")
    print(f"{'='*100}")

    df = pd.read_excel(file_path, sheet_name='Analysis_Results')

    failed_cases = df[df['paraphrase_status'] != 'success']

    if len(failed_cases) == 0:
        print(f"\n✓ No failures! Success rate: 100%")
        return

    print(f"\nTotal failures: {len(failed_cases)}/{len(df)} ({len(failed_cases)/len(df)*100:.1f}%)")

    for idx, row in failed_cases.iterrows():
        case_num = row['index']
        print(f"\n{'-'*100}")
        print(f"FAILED CASE {case_num}")
        print(f"{'-'*100}")

        print(f"\nStatus: {row['paraphrase_status']}")

        selected_sentence = row.get('selected_sentence', 'N/A')
        paraphrased_sentence = row.get('paraphrased_sentence', 'N/A')

        print(f"\n📝 Selected Sentence:")
        if pd.notna(selected_sentence) and selected_sentence != 'N/A':
            print(f"   {selected_sentence[:300]}{'...' if len(str(selected_sentence)) > 300 else ''}")
        else:
            print(f"   {selected_sentence}")

        print(f"\n📝 Paraphrased Sentence:")
        if pd.notna(paraphrased_sentence) and paraphrased_sentence != 'N/A':
            print(f"   {paraphrased_sentence[:300]}{'...' if len(str(paraphrased_sentence)) > 300 else ''}")
        else:
            print(f"   {paraphrased_sentence}")

        # Check if the sentence is actually in the original text
        if dataset_name == 'BHCS':
            original_text = row.get('original_text', '')
        elif dataset_name == 'DiagnosisArena':
            field = row.get('selected_field', 'case_information')
            original_text = row.get(field, '')
        else:  # MedXpertQA
            original_text = row.get('question', '')

        print(f"\n🔍 Diagnosis:")
        if pd.notna(selected_sentence) and selected_sentence != 'N/A':
            if str(selected_sentence) in str(original_text):
                print(f"   ✓ Selected sentence IS in original text")
                print(f"   ❌ But replacement failed - likely GPT-5 paraphrasing issue")
            else:
                print(f"   ❌ Selected sentence NOT in original text!")
                print(f"   This is a sentence extraction/selection bug")

                # Try to find similar sentences
                sentences_in_text = str(original_text).split('.')
                print(f"\n   Original text has {len(sentences_in_text)} sentences")

                # Check if it's a substring issue
                if str(selected_sentence).strip() in str(original_text):
                    print(f"   ⚠️  Sentence found without exact match (whitespace/punctuation issue)")
        else:
            print(f"   ❌ No sentence was selected - extraction failed")

def main():
    files = [
        ('test_bhcs_baseline_results.xlsx', 'BHCS'),
        ('test_diagnosis_arena_baseline_results.xlsx', 'DiagnosisArena'),
        ('test_medxpertqa_baseline_results.xlsx', 'MedXpertQA'),
    ]

    for file_path, dataset_name in files:
        try:
            investigate_failures(file_path, dataset_name)
        except Exception as e:
            print(f"\n❌ Error investigating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
