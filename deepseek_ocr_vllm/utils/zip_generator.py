import os
import re
import io
import zipfile
import tempfile
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """Extract coordinates and label from ref text"""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception:
        return None
    return (label_type, cor_list)


def re_match(text):
    """Match ref and det patterns in text"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def crop_and_save_images(image, matches_ref, page_idx, temp_dir):
    """Crop images based on ref matches and save to temp directory"""
    image_width, image_height = image.size
    img_idx = 0
    
    for ref in matches_ref:
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                if label_type == 'image':
                    for points in points_list:
                        x1, y1, x2, y2 = points
                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)
                        
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            img_path = os.path.join(temp_dir, f"images/{page_idx}_{img_idx}.jpg")
                            cropped.save(img_path)
                            img_idx += 1
                        except Exception:
                            pass
        except:
            continue
    
    return img_idx


def generate_markdown_content(raw_result: str, images: List[Image.Image]) -> str:
    """Generate markdown content from OCR results"""
    contents = ''
    
    # Handle single image case
    if len(images) == 1:
        content = raw_result
        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')
        
        matches_ref, matches_images, matches_other = re_match(content)
        
        # Replace image references with markdown image links
        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f'![](images/0_{idx}.jpg)\n')
        
        # Clean up other matches
        for a_match_other in matches_other:
            content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
        
        contents = content
    else:
        # Handle multi-page case
        page_pattern = re.compile(r'第(\d+)页:\s*')
        page_matches = list(page_pattern.finditer(raw_result))
        
        for i, match in enumerate(page_matches):
            if i < len(images):
                start_pos = match.end()
                end_pos = page_matches[i + 1].start() if i + 1 < len(page_matches) else len(raw_result)
                content = raw_result[start_pos:end_pos].strip()
                
                if '<｜end▁of▁sentence｜>' in content:
                    content = content.replace('<｜end▁of▁sentence｜>', '')
                
                matches_ref, matches_images, matches_other = re_match(content)
                
                # Replace image references
                for idx, a_match_image in enumerate(matches_images):
                    content = content.replace(a_match_image, f'![](images/{i}_{idx}.jpg)\n')
                
                # Clean up other matches
                for a_match_other in matches_other:
                    content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
                
                page_num = f'\n<--- Page {i+1} --->'
                contents += content + f'\n{page_num}\n'
    
    return contents.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')


def create_zip_with_markdown_and_images(raw_result: str, images: List[Image.Image], filename: str) -> bytes:
    """Create a zip file containing markdown and extracted images"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create images directory
        images_dir = os.path.join(temp_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Process each image and extract sub-images
        for page_idx, image in enumerate(images):
            matches_ref, _, _ = re_match(raw_result)
            crop_and_save_images(image, matches_ref, page_idx, temp_dir)
        
        # Generate markdown content
        markdown_content = generate_markdown_content(raw_result, images)
        
        # Save markdown file
        md_filename = filename.replace('.pdf', '.md') if filename.endswith('.pdf') else f"{filename}.md"
        md_path = os.path.join(temp_dir, md_filename)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Create zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add markdown file
            zip_file.write(md_path, md_filename)
            
            # Add all images
            for root, dirs, files in os.walk(images_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zip_file.write(file_path, arcname)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()