import time
from vllm import LLM, AsyncLLMEngine, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from PIL import Image
from typing import List


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
    return DeepseekOCRProcessor().tokenize_with_images(
        images=[image], bos=True, eos=True, cropping=cropping
    )


def process_images_batch_ocr(llm_engine: LLM, images: List[Image.Image], prompt: str = "<image>") -> str:
    """批量处理图片OCR"""
    batch_inputs = []
    for image in images:
        cache_item = {
            "prompt": prompt,
            "multi_modal_data": {"image": process_image_for_ocr(image)},
        }
        batch_inputs.append(cache_item)
    
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


async def ocr_generate_async(engine: AsyncLLMEngine, image_features, prompt: str = "<image>") -> str:
    """异步OCR识别生成（保持向后兼容）"""
    logits_processors = [NoRepeatNGramLogitsProcessor(
        ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822}
    )]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )
    
    request_id = f"request-{int(time.time())}"
    
    if image_features and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features}
        }
    else:
        request = {"prompt": prompt}
    
    result_text = ""
    async for request_output in engine.generate(request, sampling_params, request_id):
        if request_output.outputs:
            result_text = request_output.outputs[0].text
    
    return result_text