import time
from vllm import LLM, AsyncLLMEngine, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from PIL import Image
from typing import List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from config import PROMPT, CROP_MODE, NUM_WORKERS

def create_sampling_params() -> SamplingParams:
    """创建采样参数"""
    logits_processors = [NoRepeatNGramLogitsProcessor(
        ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822}
    )]
    
    return SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )


def process_image_for_ocr(image: Image.Image, cropping: bool = False) -> any:
    """处理图片用于OCR"""
    # return image.convert("RGB") 
    return DeepseekOCRProcessor().tokenize_with_images(
        images=[image], bos=True, eos=True, cropping=cropping
    )

def process_single_image(image: Image.Image):

    return {
        "prompt": PROMPT,
        "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)},
    }

def process_images_batch_ocr(llm_engine: LLM, images: List[Image.Image], num_workers: int = NUM_WORKERS) -> str:
    """批量处理图片OCR"""

    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_inputs = list(tqdm(
            executor.map(process_single_image, images),
            total=len(images),
            desc="Pre-processing images"
        ))
    
    sampling_params = create_sampling_params()
    outputs_list = llm_engine.generate(batch_inputs, sampling_params=sampling_params)
    
    results = []
    for i, output in enumerate(outputs_list):
        content = output.outputs[0].text
        # 清理结束标记
        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')
        
        if len(images) > 1:
            results.append(f"第{i+1}页:\n{content}")
        else:
            results.append(content)
    
    return "\n\n".join(results)
