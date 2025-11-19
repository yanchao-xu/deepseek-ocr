#!/usr/bin/env python3
"""Test prompt building for grounding functionality"""

import sys
sys.path.append('/Users/ookumaneko/Documents/deepseek-ocr/deepseek_ocr_vllm')

from utils.prompt import build_image_prompt, OCRMode

def test_prompt_building():
    """Test different prompt configurations"""
    
    test_cases = [
        {
            "name": "Plain OCR with grounding=True",
            "mode": OCRMode.plain_ocr,
            "grounding": True,
            "user_prompt": "",
            "find_term": None,
            "schema": None,
            "include_caption": False
        },
        {
            "name": "Plain OCR with grounding=False",
            "mode": OCRMode.plain_ocr,
            "grounding": False,
            "user_prompt": "",
            "find_term": None,
            "schema": None,
            "include_caption": False
        },
        {
            "name": "Find ref mode (auto grounding)",
            "mode": OCRMode.find_ref,
            "grounding": False,
            "user_prompt": "",
            "find_term": "Total",
            "schema": None,
            "include_caption": False
        },
        {
            "name": "HTML mode (auto grounding)",
            "mode": OCRMode.html,
            "grounding": False,
            "user_prompt": "",
            "find_term": None,
            "schema": None,
            "include_caption": False
        },
        {
            "name": "Custom prompt with grounding",
            "mode": OCRMode.freeform,
            "grounding": True,
            "user_prompt": "Extract all text with precise locations",
            "find_term": None,
            "schema": None,
            "include_caption": False
        }
    ]
    
    print("=== Testing Prompt Building ===\n")
    
    for test_case in test_cases:
        print(f"Test: {test_case['name']}")
        
        prompt = build_image_prompt(
            mode=test_case['mode'],
            user_prompt=test_case['user_prompt'],
            grounding=test_case['grounding'],
            find_term=test_case['find_term'],
            schema=test_case['schema'],
            include_caption=test_case['include_caption']
        )
        
        print(f"Generated prompt:")
        print(f"'{prompt}'")
        print(f"Contains <|grounding|>: {'<|grounding|>' in prompt}")
        print("-" * 60)

if __name__ == "__main__":
    test_prompt_building()