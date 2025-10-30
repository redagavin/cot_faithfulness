"""Quick check of success rates in test results"""
import pandas as pd
import os

files = [
    ('test_bhcs_baseline_results.xlsx', 'BHCS'),
    ('test_diagnosis_arena_baseline_results.xlsx', 'DiagnosisArena'),
    ('test_medxpertqa_baseline_results.xlsx', 'MedXpertQA'),
]

for file_path, dataset_name in files:
    if not os.path.exists(file_path):
        print(f"{dataset_name}: File not found yet")
        continue

    df = pd.read_excel(file_path, sheet_name='Analysis_Results')

    total = len(df)
    success = len(df[df['paraphrase_status'] == 'success'])
    failed = total - success

    print(f"{dataset_name}:")
    print(f"  Total: {total}")
    print(f"  Success: {success} ({success/total*100:.1f}%)")
    print(f"  Failed: {failed} ({failed/total*100:.1f}%)")

    if failed > 0:
        print(f"  Failure reasons:")
        failure_reasons = df[df['paraphrase_status'] != 'success']['paraphrase_status'].value_counts()
        for reason, count in failure_reasons.items():
            print(f"    {reason}: {count}")

    print()
