#!/usr/bin/env python3
"""Debug script to test grounding functionality"""

import sys
import os
sys.path.append('/Users/ookumaneko/Documents/deepseek-ocr/deepseek_ocr_vllm')

from utils.grounding_parser import parse_detections, clean_grounding_text, DET_BLOCK
import re

def test_grounding_parser():
    """Test the grounding parser with sample data"""
    
    # Test cases with different formats
    test_cases = [
        # Standard format
        "<|ref|>Hello World<|/ref|><|det|>[100, 200, 300, 400]<|/det|>",
        
        # With spaces
        "<|ref|>Test Text<|/ref|> <|det|> [50, 60, 150, 160] <|/det|>",
        
        # Multiple detections
        "<|ref|>First<|/ref|><|det|>[10, 20, 30, 40]<|/det|> Some text <|ref|>Second<|/ref|><|det|>[50, 60, 70, 80]<|/det|>",
        
        # With grounding tag
        "<|grounding|><|ref|>Grounded Text<|/ref|><|det|>[100, 100, 200, 200]<|/det|>",
        
        # Real example format that might be generated
        "This is some text <|ref|>important text<|/ref|><|det|>[123, 456, 789, 012]<|/det|> and more text.",
    ]
    
    print("=== Testing Grounding Parser ===\n")
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Input: {test_text}")
        
        # Test regex matching
        matches = list(DET_BLOCK.finditer(test_text))
        print(f"Regex matches: {len(matches)}")
        for match in matches:
            print(f"  Label: '{match.group('label')}'")
            print(f"  Coords: '{match.group('coords')}'")
        
        # Test parsing
        boxes = parse_detections(test_text, 1000, 1000)  # 1000x1000 test image
        print(f"Parsed boxes: {boxes}")
        
        # Test cleaning
        cleaned = clean_grounding_text(test_text)
        print(f"Cleaned text: '{cleaned}'")
        print("-" * 50)

def debug_actual_output(raw_output: str):
    """Debug actual model output"""
    print("=== Debugging Actual Output ===\n")
    print(f"Raw output: {repr(raw_output)}")
    print(f"Contains <|ref|>: {'<|ref|>' in raw_output}")
    print(f"Contains <|det|>: {'<|det|>' in raw_output}")
    print(f"Contains <|grounding|>: {'<|grounding|>' in raw_output}")
    
    # Check for different possible formats
    patterns = [
        r"<\|ref\|>.*?<\|/ref\|>",
        r"<\|det\|>.*?<\|/det\|>",
        r"\[[\d\s,]+\]",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, raw_output, re.DOTALL)
        print(f"Pattern '{pattern}' matches: {matches}")
    
    boxes = parse_detections(raw_output, 1000, 1000)
    print(f"Final parsed boxes: {boxes}")

if __name__ == "__main__":
    test_grounding_parser()
    
    # If you have actual output to debug, uncomment and modify:
    # actual_output = "paste your actual model output here"
    # debug_actual_output(actual_output)