"""Check column names in Excel files"""
import pandas as pd

files = [
    ('test_bhcs_baseline_results.xlsx', 'BHCS'),
    ('test_diagnosis_arena_baseline_results.xlsx', 'DiagnosisArena'),
    ('test_medxpertqa_baseline_results.xlsx', 'MedXpertQA')
]

for f, name in files:
    try:
        df = pd.read_excel(f, sheet_name='Analysis_Results')
        print(f"\n{name} ({f}):")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        print(f"\n  First row data:")
        for col in df.columns:
            print(f"    {col}: {df[col].iloc[0]}")
    except Exception as e:
        print(f"\n{name}: Error - {e}")
