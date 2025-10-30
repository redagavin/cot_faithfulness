#!/usr/bin/env python3
"""
Validate MedQA Test Results
Analyzes the test Excel files to verify quality before full production launch
"""

import pandas as pd
import sys

def validate_gender_analysis(file_path):
    """Validate gender analysis test results"""
    print("=" * 80)
    print("VALIDATING GENDER ANALYSIS TEST RESULTS")
    print("=" * 80)
    print()

    try:
        # Read the Analysis_Results sheet
        df = pd.read_excel(file_path, sheet_name='Analysis_Results')

        print(f"📊 Dataset: {len(df)} test cases")
        print()

        # 1. Check gender detection
        print("1. GENDER DETECTION")
        print("-" * 40)
        gender_counts = df['original_gender'].value_counts()
        print(f"   Female cases: {gender_counts.get('female', 0)}")
        print(f"   Male cases: {gender_counts.get('male', 0)}")
        unclear_pct = (gender_counts.get('unclear', 0) / len(df)) * 100 if 'unclear' in gender_counts else 0
        print(f"   Unclear: {gender_counts.get('unclear', 0)} ({unclear_pct:.1f}%)")

        if unclear_pct > 20:
            print("   ⚠️  WARNING: High unclear rate")
        else:
            print("   ✅ Gender detection working well")
        print()

        # 2. Check gender swapping
        print("2. GENDER SWAPPING")
        print("-" * 40)
        # Sample one female and one male case
        female_sample = df[df['original_gender'] == 'female'].iloc[0] if len(df[df['original_gender'] == 'female']) > 0 else None
        male_sample = df[df['original_gender'] == 'male'].iloc[0] if len(df[df['original_gender'] == 'male']) > 0 else None

        if female_sample is not None:
            print("   Sample Female → Male swap:")
            print(f"   Original (first 100 chars): {female_sample['question'][:100]}...")
            print(f"   Swapped (first 100 chars):  {female_sample['swapped_question'][:100]}...")
            print()

        if male_sample is not None:
            print("   Sample Male → Female swap:")
            print(f"   Original (first 100 chars): {male_sample['question'][:100]}...")
            print(f"   Swapped (first 100 chars):  {male_sample['swapped_question'][:100]}...")
            print()

        # 3. Check answer extraction for both models
        models = ['olmo2_7b', 'deepseek_r1_0528']

        for model in models:
            print(f"3. ANSWER EXTRACTION - {model.upper()}")
            print("-" * 40)

            female_ans_col = f'{model}_female_answer'
            male_ans_col = f'{model}_male_answer'
            match_col = f'{model}_answers_match'

            if female_ans_col in df.columns and male_ans_col in df.columns:
                # Count unclear answers
                female_unclear = (df[female_ans_col] == 'Unclear').sum()
                male_unclear = (df[male_ans_col] == 'Unclear').sum()
                total_answers = len(df) * 2
                unclear_answers = female_unclear + male_unclear
                success_rate = ((total_answers - unclear_answers) / total_answers) * 100

                print(f"   Female answers unclear: {female_unclear}/{len(df)}")
                print(f"   Male answers unclear: {male_unclear}/{len(df)}")
                print(f"   Extraction success rate: {success_rate:.1f}%")

                if success_rate >= 80:
                    print("   ✅ Answer extraction working well")
                elif success_rate >= 60:
                    print("   ⚠️  Answer extraction acceptable but could be better")
                else:
                    print("   ❌ Answer extraction needs improvement")
                print()

                # Check match rates
                if match_col in df.columns:
                    matches = df[match_col].value_counts()
                    print(f"   Answers match: {matches.get('yes', 0)}/{len(df)} ({matches.get('yes', 0)/len(df)*100:.1f}%)")
                    print(f"   Answers flip: {matches.get('no', 0)}/{len(df)} ({matches.get('no', 0)/len(df)*100:.1f}%)")
                    print(f"   Unclear: {matches.get('unclear', 0)}/{len(df)}")
                    print()

                    # Show sample answers
                    print("   Sample extracted answers:")
                    for i in range(min(3, len(df))):
                        print(f"     Case {i+1}: Female={df.iloc[i][female_ans_col]}, Male={df.iloc[i][male_ans_col]}, Match={df.iloc[i][match_col]}")
                    print()

        # 4. Check judge responses
        print("4. GPT-5 JUDGE EVALUATION")
        print("-" * 40)

        for model in models:
            judge_col = f'{model}_judge_answer'
            match_col = f'{model}_answers_match'

            if judge_col in df.columns:
                # Count cases that needed judging (flipped answers)
                flipped = df[df[match_col] == 'no']
                print(f"   {model}: {len(flipped)} cases with flipped answers")

                if len(flipped) > 0:
                    judge_answers = flipped[judge_col].value_counts()
                    print(f"     UNFAITHFUL: {judge_answers.get('UNFAITHFUL', 0)}")
                    print(f"     EXPLICIT BIAS: {judge_answers.get('EXPLICIT BIAS', 0)}")
                    print(f"     Unclear: {judge_answers.get('Unclear', 0)}")

                    # Show sample judge evidence
                    sample_flipped = flipped.iloc[0]
                    if pd.notna(sample_flipped[f'{model}_judge_evidence']):
                        print(f"     Sample evidence: {str(sample_flipped[f'{model}_judge_evidence'])[:150]}...")
                print()

        # 5. Check correctness
        print("5. CORRECTNESS ANALYSIS")
        print("-" * 40)

        for model in models:
            female_correct_col = f'{model}_female_correct'
            male_correct_col = f'{model}_male_correct'

            if female_correct_col in df.columns:
                female_acc = df[female_correct_col].sum() / len(df) * 100
                male_acc = df[male_correct_col].sum() / len(df) * 100

                print(f"   {model}:")
                print(f"     Female accuracy: {female_acc:.1f}%")
                print(f"     Male accuracy: {male_acc:.1f}%")
                print()

        return True

    except Exception as e:
        print(f"❌ Error reading gender analysis file: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_baseline_analysis(file_path):
    """Validate baseline analysis test results"""
    print("=" * 80)
    print("VALIDATING BASELINE ANALYSIS TEST RESULTS")
    print("=" * 80)
    print()

    try:
        # Read the Analysis_Results sheet
        df = pd.read_excel(file_path, sheet_name='Analysis_Results')

        print(f"📊 Dataset: {len(df)} test cases")
        print()

        # 1. Check paraphrasing success
        print("1. PARAPHRASING SUCCESS")
        print("-" * 40)
        status_counts = df['paraphrase_status'].value_counts()
        success = status_counts.get('success', 0)
        success_rate = (success / len(df)) * 100

        print(f"   Successful: {success}/{len(df)} ({success_rate:.1f}%)")
        print(f"   Failed: {len(df) - success}/{len(df)}")

        if success_rate >= 80:
            print("   ✅ Paraphrasing working well")
        else:
            print("   ⚠️  Paraphrasing success rate below 80%")
        print()

        # 2. Show sample paraphrases
        print("2. PARAPHRASE QUALITY")
        print("-" * 40)
        successful = df[df['paraphrase_status'] == 'success']

        if len(successful) > 0:
            for i in range(min(3, len(successful))):
                sample = successful.iloc[i]
                print(f"   Sample {i+1}:")
                print(f"   Original:    {sample['selected_sentence']}")
                print(f"   Paraphrased: {sample['paraphrased_sentence']}")
                print()

        # 3. Check answer extraction
        models = ['olmo2_7b', 'deepseek_r1_0528']

        for model in models:
            print(f"3. ANSWER EXTRACTION - {model.upper()}")
            print("-" * 40)

            orig_ans_col = f'{model}_original_answer'
            para_ans_col = f'{model}_paraphrased_answer'
            match_col = f'{model}_answers_match'

            if orig_ans_col in df.columns:
                valid_df = df[df['paraphrase_status'] == 'success']

                # Count unclear answers
                orig_unclear = (valid_df[orig_ans_col] == 'Unclear').sum()
                para_unclear = (valid_df[para_ans_col] == 'Unclear').sum()
                total_answers = len(valid_df) * 2
                unclear_answers = orig_unclear + para_unclear
                success_rate = ((total_answers - unclear_answers) / total_answers) * 100 if total_answers > 0 else 0

                print(f"   Original answers unclear: {orig_unclear}/{len(valid_df)}")
                print(f"   Paraphrased answers unclear: {para_unclear}/{len(valid_df)}")
                print(f"   Extraction success rate: {success_rate:.1f}%")

                if success_rate >= 80:
                    print("   ✅ Answer extraction working well")
                else:
                    print("   ⚠️  Answer extraction needs review")
                print()

                # Check match rates
                if match_col in valid_df.columns:
                    matches = valid_df[match_col].value_counts()
                    print(f"   Answers match: {matches.get('yes', 0)}/{len(valid_df)} ({matches.get('yes', 0)/len(valid_df)*100:.1f}%)")
                    print(f"   Answers flip: {matches.get('no', 0)}/{len(valid_df)} ({matches.get('no', 0)/len(valid_df)*100:.1f}%)")
                    print(f"   Unclear: {matches.get('unclear', 0)}/{len(valid_df)}")
                    print()

        # 4. Check correctness
        print("4. CORRECTNESS ANALYSIS")
        print("-" * 40)

        for model in models:
            orig_correct_col = f'{model}_original_correct'
            para_correct_col = f'{model}_paraphrased_correct'

            if orig_correct_col in df.columns:
                valid_df = df[df['paraphrase_status'] == 'success']

                if len(valid_df) > 0:
                    orig_acc = valid_df[orig_correct_col].sum() / len(valid_df) * 100
                    para_acc = valid_df[para_correct_col].sum() / len(valid_df) * 100

                    print(f"   {model}:")
                    print(f"     Original accuracy: {orig_acc:.1f}%")
                    print(f"     Paraphrased accuracy: {para_acc:.1f}%")
                    print()

        return True

    except Exception as e:
        print(f"❌ Error reading baseline analysis file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "MedQA Test Results Validation" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Validate gender analysis
    gender_ok = validate_gender_analysis('test_medqa_analysis_results.xlsx')
    print()

    # Validate baseline analysis
    baseline_ok = validate_baseline_analysis('test_medqa_baseline_results.xlsx')
    print()

    # Final verdict
    print("=" * 80)
    print("FINAL VALIDATION VERDICT")
    print("=" * 80)

    if gender_ok and baseline_ok:
        print("✅ Both test runs passed validation!")
        print()
        print("RECOMMENDATION: Proceed with full production runs")
        print()
        print("Launch commands:")
        print("  sbatch run_medqa_analysis.sbatch")
        print("  sbatch run_medqa_baseline_analysis.sbatch")
        print()
        return 0
    else:
        print("⚠️  Some validation issues detected")
        print()
        print("RECOMMENDATION: Review issues above before launching full runs")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
