"""Check sheet names in Excel files"""
import pandas as pd

files = [
    'test_bhcs_baseline_results.xlsx',
    'test_diagnosis_arena_baseline_results.xlsx',
    'test_medxpertqa_baseline_results.xlsx'
]

for f in files:
    try:
        xls = pd.ExcelFile(f)
        print(f"\n{f}:")
        print(f"  Sheet names: {xls.sheet_names}")
    except Exception as e:
        print(f"\n{f}: Error - {e}")
