import re
from typing import List, Dict, Any

# Match a full detection block and capture the coordinates as the entire list expression
DET_BLOCK = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*<\|det\|>\s*(?P<coords>\[.*?\])\s*<\|/det\|>",
    re.DOTALL,
)

def clean_grounding_text(text: str) -> str:
    """Remove grounding tags from text for display, keeping labels"""
    cleaned = re.sub(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\s*\[.*?\]\s*<\|/det\|>",
        r"\1",
        text,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r"<\|grounding\|>", "", cleaned)
    return cleaned.strip()

def parse_detections(text: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """Parse grounding boxes from text and scale from 0-999 normalized coords to actual image dimensions"""
    boxes: List[Dict[str, Any]] = []
    for m in DET_BLOCK.finditer(text or ""):
        label = m.group("label").strip()
        coords_str = m.group("coords").strip()

        try:
            # Clean up coords string and handle potential octal numbers
            coords_clean = coords_str.strip()
            # Replace potential octal numbers (leading zeros) with decimal
            coords_clean = re.sub(r'\b0+(\d+)', r'\1', coords_clean)
            
            import ast
            parsed = ast.literal_eval(coords_clean)

            if (
                isinstance(parsed, list)
                and len(parsed) == 4
                and all(isinstance(n, (int, float)) for n in parsed)
            ):
                box_coords = [parsed]
            elif isinstance(parsed, list):
                box_coords = parsed
            else:
                raise ValueError("Unsupported coords structure")

            for box in box_coords:
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    # Ensure coordinates are valid numbers
                    try:
                        x1 = int(float(box[0]) / 999 * image_width)
                        y1 = int(float(box[1]) / 999 * image_height)
                        x2 = int(float(box[2]) / 999 * image_width)
                        y2 = int(float(box[3]) / 999 * image_height)
                        boxes.append({"label": label, "box": [x1, y1, x2, y2]})
                    except (ValueError, TypeError):
                        continue
        except Exception:
            continue
    
    return boxes

def parse_multi_image_results(raw_result: str, images: list) -> tuple:
    """Parse results from multiple images with correct dimensions for each"""
    if len(images) <= 1:
        # Single image case
        orig_w, orig_h = images[0].size if images else (1, 1)
        boxes = parse_detections(raw_result, orig_w, orig_h) if ("<|det|>" in raw_result or "<|ref|>" in raw_result) else []
        display_text = clean_grounding_text(raw_result) if ("<|ref|>" in raw_result or "<|grounding|>" in raw_result) else raw_result
        return display_text, boxes, [{"w": orig_w, "h": orig_h}]
    
    # Multi-image case: use regex to split by page markers
    page_pattern = re.compile(r'第(\d+)页:\s*')
    page_matches = list(page_pattern.finditer(raw_result))
    
    all_boxes = []
    all_display_texts = []
    all_dims = []
    
    for i, match in enumerate(page_matches):
        if i < len(images):
            orig_w, orig_h = images[i].size
            
            # Extract content for this page
            start_pos = match.end()
            end_pos = page_matches[i + 1].start() if i + 1 < len(page_matches) else len(raw_result)
            page_content = raw_result[start_pos:end_pos].strip()
            
            page_boxes = parse_detections(page_content, orig_w, orig_h) if ("<|det|>" in page_content or "<|ref|>" in page_content) else []
            page_display_text = clean_grounding_text(page_content) if ("<|ref|>" in page_content or "<|grounding|>" in page_content) else page_content
            
            # Add page info to boxes
            for box in page_boxes:
                box["page"] = i + 1
            
            all_boxes.extend(page_boxes)
            all_display_texts.append(page_display_text)
            all_dims.append({"w": orig_w, "h": orig_h})
    
    combined_display_text = "\n\n".join([f"第{i+1}页:\n{text}" for i, text in enumerate(all_display_texts)])
    return combined_display_text, all_boxes, all_dims