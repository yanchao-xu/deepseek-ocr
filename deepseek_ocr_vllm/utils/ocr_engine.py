import time
from vllm import LLM, AsyncLLMEngine, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from PIL import Image
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from config import NUM_WORKERS
import config

class OCRConfig:
    def __init__(self, crop_mode: Optional[bool] = None, base_size: Optional[int] = None, image_size: Optional[int] = None):
        self.crop_mode = crop_mode
        self.base_size = base_size
        self.image_size = image_size

def update_config_params(prompt: str = None, ocr_config: OCRConfig = None):
    """更新config参数"""
    if prompt is not None:
        config.PROMPT = prompt
    if ocr_config is not None:
        if ocr_config.crop_mode is not None:
            config.CROP_MODE = ocr_config.crop_mode
        if ocr_config.base_size is not None:
            config.BASE_SIZE = ocr_config.base_size
        if ocr_config.image_size is not None:
            config.IMAGE_SIZE = ocr_config.image_size

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


def process_single_image(image: Image.Image, prompt: str, crop_mode: bool):
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(images=[image], bos=True, eos=True, cropping=crop_mode, prompt = prompt)},
    }

def process_images_batch_ocr(llm_engine: LLM, images: List[Image.Image], prompt: str, ocr_config: OCRConfig) -> str:
    """批量处理图片OCR"""
    update_config_params(prompt, ocr_config)
    crop_mode = ocr_config.crop_mode if ocr_config.crop_mode is not None else config.CROP_MODE
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        batch_inputs = list(tqdm(
            executor.map(lambda img: process_single_image(img, prompt, crop_mode), images),
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
