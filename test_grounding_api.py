#!/usr/bin/env python3
"""Test script to verify grounding functionality with actual API"""

import requests
import json

def test_grounding_api():
    """Test the grounding functionality with a real API call"""
    
    # API endpoint
    url = "http://localhost:9003/ocr"
    
    # Test with a sample image URL (you'll need to replace this with an actual image)
    test_data = {
        "url": "https://example.com/test-image.jpg",  # Replace with actual image URL
        "mode": "plain_ocr",
        "grounding": True,
        "prompt": "Extract all text with location information"
    }
    
    try:
        print("Testing grounding API...")
        print(f"Request data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success', False)}")
            print(f"Text: {result.get('text', '')}")
            print(f"Raw text: {result.get('raw_text', '')}")
            print(f"Boxes: {result.get('boxes', [])}")
            print(f"Grounding enabled: {result.get('metadata', {}).get('grounding', False)}")
            
            if not result.get('boxes'):
                print("\n⚠️  WARNING: No boxes found!")
                print("This could mean:")
                print("1. The model didn't generate grounding tags")
                print("2. The image doesn't contain detectable text")
                print("3. The grounding parser failed")
                
                # Debug the raw output
                raw_text = result.get('raw_text', '')
                print(f"\nDebugging raw output:")
                print(f"Contains <|ref|>: {'<|ref|>' in raw_text}")
                print(f"Contains <|det|>: {'<|det|>' in raw_text}")
                print(f"Contains <|grounding|>: {'<|grounding|>' in raw_text}")
            else:
                print(f"\n✅ Found {len(result.get('boxes', []))} grounding boxes!")
                
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on localhost:9003")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_with_local_image():
    """Test with a local image file"""
    # This would require uploading a file, which is more complex
    # For now, just show how to structure the request
    print("\nTo test with a local image:")
    print("1. Upload your image to a public URL")
    print("2. Replace the URL in the test_data above")
    print("3. Run this script again")

if __name__ == "__main__":
    test_grounding_api()
    test_with_local_image()