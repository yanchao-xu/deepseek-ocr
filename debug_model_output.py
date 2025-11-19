#!/usr/bin/env python3
"""Debug actual model output to understand grounding issues"""

import sys
import re
sys.path.append('/Users/ookumaneko/Documents/deepseek-ocr/deepseek_ocr_vllm')

def analyze_model_output(raw_output: str):
    """Comprehensive analysis of model output"""
    
    print("=== Model Output Analysis ===\n")
    print(f"Raw output length: {len(raw_output)} characters")
    print(f"Raw output (first 500 chars): {repr(raw_output[:500])}")
    print()
    
    # Check for various grounding-related patterns
    patterns_to_check = [
        (r'<\|grounding\|>', "Grounding tag"),
        (r'<\|ref\|>', "Reference start tag"),
        (r'<\|/ref\|>', "Reference end tag"),
        (r'<\|det\|>', "Detection start tag"),
        (r'<\|/det\|>', "Detection end tag"),
        (r'\[[\d\s,]+\]', "Coordinate arrays"),
        (r'<\|ref\|>.*?<\|/ref\|>', "Complete reference blocks"),
        (r'<\|det\|>.*?<\|/det\|>', "Complete detection blocks"),
        (r'<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>', "Complete grounding blocks"),
    ]
    
    print("Pattern Analysis:")
    for pattern, description in patterns_to_check:
        matches = re.findall(pattern, raw_output, re.DOTALL)
        print(f"  {description}: {len(matches)} matches")
        if matches and len(matches) <= 5:  # Show first few matches
            for i, match in enumerate(matches[:3]):
                print(f"    Match {i+1}: {repr(match[:100])}")
        elif matches:
            print(f"    (showing first 3 of {len(matches)} matches)")
            for i, match in enumerate(matches[:3]):
                print(f"    Match {i+1}: {repr(match[:100])}")
    
    print()
    
    # Check for alternative formats that might be used
    alternative_patterns = [
        (r'<ref>.*?</ref>', "HTML-style ref tags"),
        (r'<det>.*?</det>', "HTML-style det tags"),
        (r'\{[^}]*"box"[^}]*\}', "JSON-style boxes"),
        (r'\{[^}]*"coordinates"[^}]*\}', "JSON-style coordinates"),
        (r'bbox:\s*\[[\d\s,]+\]', "bbox format"),
        (r'location:\s*\[[\d\s,]+\]', "location format"),
    ]
    
    print("Alternative Format Analysis:")
    for pattern, description in alternative_patterns:
        matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)
        if matches:
            print(f"  {description}: {len(matches)} matches")
            for i, match in enumerate(matches[:2]):
                print(f"    Match {i+1}: {repr(match[:100])}")
    
    print()
    
    # Look for any coordinate-like patterns
    coord_patterns = [
        r'\b\d{1,4}[,\s]+\d{1,4}[,\s]+\d{1,4}[,\s]+\d{1,4}\b',  # Four numbers
        r'\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)',  # Parentheses format
    ]
    
    print("Coordinate Pattern Analysis:")
    for pattern in coord_patterns:
        matches = re.findall(pattern, raw_output)
        if matches:
            print(f"  Pattern '{pattern}': {len(matches)} matches")
            for i, match in enumerate(matches[:3]):
                print(f"    Match {i+1}: {match}")

def test_with_sample_outputs():
    """Test with various sample outputs that might be generated"""
    
    sample_outputs = [
        # Standard expected format
        "This is some text <|ref|>Hello World<|/ref|><|det|>[100, 200, 300, 400]<|/det|> more text.",
        
        # Without grounding tags but with content
        "This is just plain OCR text without any grounding information.",
        
        # Alternative format possibilities
        "Text with <ref>Hello World</ref><det>[100, 200, 300, 400]</det> alternative tags.",
        
        # JSON-like format
        'Text with {"text": "Hello World", "box": [100, 200, 300, 400]} JSON format.',
        
        # Multiple grounding blocks
        "<|grounding|>First <|ref|>text<|/ref|><|det|>[10, 20, 30, 40]<|/det|> and <|ref|>second<|/ref|><|det|>[50, 60, 70, 80]<|/det|> text.",
    ]
    
    print("\n=== Testing Sample Outputs ===\n")
    
    for i, output in enumerate(sample_outputs, 1):
        print(f"Sample {i}:")
        analyze_model_output(output)
        print("=" * 80)

if __name__ == "__main__":
    # If you have actual model output, paste it here:
    actual_output = """
    # Paste your actual model output here to debug
    # For example, if your API returns raw_text, copy that content here
    """
    
    if actual_output.strip() and not actual_output.strip().startswith("#"):
        print("Analyzing actual model output:")
        analyze_model_output(actual_output)
    else:
        print("No actual output provided. Testing with sample outputs:")
        test_with_sample_outputs()
        
        print("\n" + "="*80)
        print("TO DEBUG YOUR ACTUAL OUTPUT:")
        print("1. Make an API call with grounding=True")
        print("2. Copy the 'raw_text' field from the response")
        print("3. Paste it in the 'actual_output' variable above")
        print("4. Run this script again")