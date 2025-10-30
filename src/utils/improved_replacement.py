"""
Improved sentence replacement with fuzzy matching to maximize success rate
"""
import re

def normalize_whitespace(text):
    """Normalize whitespace for better matching"""
    # Replace multiple spaces/newlines/tabs with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def replace_sentence_robust(field_text, original_sentence, paraphrased_sentence):
    """
    Replace sentence with multiple fallback strategies

    Strategy 1: Exact match (current approach)
    Strategy 2: Normalized whitespace match
    Strategy 3: Regex-based match with flexible whitespace
    Strategy 4: Find closest match and replace
    """

    # Ensure sentences end with period
    if not original_sentence.endswith('.'):
        original_sentence_with_period = original_sentence + '.'
    else:
        original_sentence_with_period = original_sentence

    if not paraphrased_sentence.endswith('.'):
        paraphrased_sentence_with_period = paraphrased_sentence + '.'
    else:
        paraphrased_sentence_with_period = paraphrased_sentence

    # Strategy 1: Try exact match first (fastest)
    modified_text = field_text.replace(original_sentence_with_period, paraphrased_sentence_with_period, 1)
    if modified_text != field_text:
        return modified_text, 'exact_match'

    # Strategy 2: Try without period (in case it was included already)
    modified_text = field_text.replace(original_sentence, paraphrased_sentence, 1)
    if modified_text != field_text:
        # Add period if needed
        if not paraphrased_sentence.endswith('.'):
            modified_text = modified_text.replace(paraphrased_sentence, paraphrased_sentence_with_period, 1)
        return modified_text, 'no_period_match'

    # Strategy 3: Normalize whitespace and try again
    original_normalized = normalize_whitespace(original_sentence_with_period)

    # Create a regex pattern that allows flexible whitespace
    # Escape special regex characters except whitespace
    pattern_parts = []
    for char in original_normalized:
        if char == ' ':
            # Allow any whitespace (including newlines)
            pattern_parts.append(r'\s+')
        elif char in r'\.^$*+?{}[]()|\-':
            # Escape special regex characters
            pattern_parts.append('\\' + char)
        else:
            pattern_parts.append(char)

    pattern = ''.join(pattern_parts)

    # Try to find and replace using regex
    match = re.search(pattern, field_text, re.DOTALL)
    if match:
        modified_text = field_text[:match.start()] + paraphrased_sentence_with_period + field_text[match.end():]
        return modified_text, 'whitespace_normalized'

    # Strategy 4: Try to find the sentence with just the core content (first 50 chars)
    # This handles cases where sentence extraction might have captured different boundaries
    core_content = original_sentence[:min(50, len(original_sentence))]
    core_normalized = normalize_whitespace(core_content)

    if core_normalized in normalize_whitespace(field_text):
        # Found the core content - try to extract full sentence and replace
        # This is a last resort and might not be perfect
        # For now, return None to indicate failure but log it as partial_match
        return None, 'partial_match_found_but_not_replaced'

    return None, 'no_match_found'

# Test the improved function
if __name__ == '__main__':
    # Test case 1: Normal sentence
    text1 = "This is a test. The patient has symptoms. End of text."
    orig1 = "The patient has symptoms"
    para1 = "The patient exhibits symptoms"
    result1, strategy1 = replace_sentence_robust(text1, orig1, para1)
    print(f"Test 1: {strategy1}")
    print(f"Result: {result1}\n")

    # Test case 2: Sentence with newlines
    text2 = "This is a test.\nThe patient has\nsymptoms.\nEnd of text."
    orig2 = "The patient has symptoms"
    para2 = "The patient exhibits symptoms"
    result2, strategy2 = replace_sentence_robust(text2, orig2, para2)
    print(f"Test 2: {strategy2}")
    print(f"Result: {result2}\n")

    # Test case 3: Sentence with extra spaces
    text3 = "This is a test.  The  patient   has   symptoms.  End of text."
    orig3 = "The patient has symptoms"
    para3 = "The patient exhibits symptoms"
    result3, strategy3 = replace_sentence_robust(text3, orig3, para3)
    print(f"Test 3: {strategy3}")
    print(f"Result: {result3}\n")

    # Test case 4: Sentence with special characters
    text4 = "- Imaging Studies: Brain MRI showed abnormalities. End."
    orig4 = "- Imaging Studies: Brain MRI showed abnormalities"
    para4 = "- Imaging Studies: Brain MRI demonstrated abnormalities"
    result4, strategy4 = replace_sentence_robust(text4, orig4, para4)
    print(f"Test 4: {strategy4}")
    print(f"Result: {result4}\n")
