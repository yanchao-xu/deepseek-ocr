from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import torch
from vllm import LLM
from vllm.model_executor.models.registry import ModelRegistry

from deepseek_ocr import DeepseekOCRForCausalLM
from config import MODEL_PATH, MAX_CONCURRENCY, PROMPT, BASE_SIZE, IMAGE_SIZE, CROP_MODE
from utils.file_processor import download_file, get_file_type, pdf_to_images, load_image_from_url
from utils.ocr_engine import process_images_batch_ocr, OCRConfig
from utils.callback_handler import send_callback, create_success_callback, create_error_callback
from utils.prompt import build_image_prompt, OCRMode
from utils.grounding_parser import clean_grounding_text, parse_detections, parse_multi_image_results

# 设置环境变量
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 注册模型
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

app = FastAPI(title="DeepSeek OCR API", description="图片OCR识别接口")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化OCR引擎
print("正在加载OCR模型...")
llm = LLM(
        model=MODEL_PATH,  
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},  
        block_size=256,  
        enforce_eager=False,  
        trust_remote_code=True,   
        max_model_len=8192,  # 确保这里是 8192,不是 512  
        swap_space=0,  
        max_num_seqs=MAX_CONCURRENCY,  
        tensor_parallel_size=1,  
        gpu_memory_utilization=0.3,  
        disable_mm_preprocessor_cache=True   
    )
print("OCR模型加载完成！")

class OCRBaseRequest(BaseModel):
    url: str = Field(..., description="图片或PDF文件的URL地址")
    prompt: Optional[str] = Field(None, description="自定义提示词，用于指导OCR识别")
    crop_mode: Optional[bool] = Field(CROP_MODE, description="是否启用裁剪模式，提高长文档识别精度")
    base_size: Optional[int] = Field(BASE_SIZE, description="图片预处理基础尺寸")
    image_size: Optional[int] = Field(IMAGE_SIZE, description="图片输入模型的尺寸")
    mode: Optional[OCRMode] = Field(OCRMode.plain_ocr, description="OCR识别模式")
    grounding: Optional[bool] = Field(False, description="是否启用定位功能，返回文本位置信息")
    find_term: Optional[str] = Field(None, description="查找特定词汇，配合grounding使用")
    schema: Optional[str] = Field(None, description="结构化输出模式的schema定义")
    include_caption: Optional[bool] = Field(False, description="是否包含图片描述信息")

class OCRUrlRequest(OCRBaseRequest):
    """同步OCR识别请求参数"""
    pass

class OCRAsyncRequest(OCRBaseRequest):
    callback_url: str = Field(..., description="异步处理完成后的回调URL地址")


@app.post("/ocr")
def ocr_from_url(request: OCRUrlRequest):
    """同步OCR识别接口"""
    try:
        file_bytes, filename, content_type = download_file(request.url)
        file_type = get_file_type(filename, content_type)
        
        if file_type == 'pdf':
            images = pdf_to_images(file_bytes)
        elif file_type == 'image':
            images = [load_image_from_url(request.url)]
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        if not images:
            raise HTTPException(status_code=400, detail="无法获取图片")
        
        ocr_config = OCRConfig(request.crop_mode, request.base_size, request.image_size)
        prompt = build_image_prompt(
            mode=request.mode,
            user_prompt=request.prompt or "",
            grounding=request.grounding,
            find_term=request.find_term,
            schema=request.schema,
            include_caption=request.include_caption
        )

        raw_result = process_images_batch_ocr(llm, images, prompt, ocr_config)
        
        # Parse results with correct dimensions for each image
        display_text, boxes, image_dims = parse_multi_image_results(raw_result, images)
        
        # If display text is empty after cleaning but we have boxes, show the labels
        if not display_text and boxes:
            display_text = ", ".join([b["label"] for b in boxes])
        
        # Debug information for grounding issues
        grounding_debug = {
            "prompt_used": prompt,
            "has_grounding_tag": "<|grounding|>" in prompt,
            "raw_contains_ref": "<|ref|>" in raw_result,
            "raw_contains_det": "<|det|>" in raw_result,
            "raw_contains_grounding": "<|grounding|>" in raw_result,
            "boxes_found": len(boxes)
        }
        
        return JSONResponse(content={
            "success": True,
            "text": display_text,
            "raw_text": raw_result,
            "boxes": boxes,
            "image_dims": image_dims[0] if len(image_dims) == 1 else image_dims,
            "metadata": {
                "mode": request.mode,
                "grounding": request.grounding or (request.mode in {"find_ref", "layout_map", "pii_redact"}),
                "base_size": request.base_size,
                "image_size": request.image_size,
                "crop_mode": request.crop_mode
            },
            "debug": grounding_debug
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")

def process_ocr_async(request: OCRAsyncRequest):
    """异步处理OCR任务"""
    try:
        file_bytes, filename, content_type = download_file(request.url)
        file_type = get_file_type(filename, content_type)
        
        if file_type == 'pdf':
            images = pdf_to_images(file_bytes)
        elif file_type == 'image':
            images = [load_image_from_url(request.url)]
        else:
            raise ValueError("不支持的文件格式")
        
        if not images:
            raise ValueError("无法获取图片")
        
        ocr_config = OCRConfig(request.crop_mode, request.base_size, request.image_size)
        final_prompt = build_image_prompt(
            mode=request.mode,
            user_prompt=request.prompt or "",
            grounding=request.grounding,
            find_term=request.find_term,
            schema=request.schema,
            include_caption=request.include_caption
        )
        raw_result = process_images_batch_ocr(llm, images, final_prompt, ocr_config)
        
        # Parse results with correct dimensions for each image
        display_text, boxes, image_dims = parse_multi_image_results(raw_result, images)
        
        # If display text is empty after cleaning but we have boxes, show the labels
        if not display_text and boxes:
            display_text = ", ".join([b["label"] for b in boxes])
        
        # Debug information for grounding issues
        grounding_debug = {
            "prompt_used": final_prompt,
            "has_grounding_tag": "<|grounding|>" in final_prompt,
            "raw_contains_ref": "<|ref|>" in raw_result,
            "raw_contains_det": "<|det|>" in raw_result,
            "raw_contains_grounding": "<|grounding|>" in raw_result,
            "boxes_found": len(boxes)
        }
        
        result = {
            "text": display_text,
            "raw_text": raw_result,
            "boxes": boxes,
            "image_dims": image_dims[0] if len(image_dims) == 1 else image_dims,
            "metadata": {
                "mode": request.mode,
                "grounding": request.grounding or (request.mode in {"find_ref", "layout_map", "pii_redact"}),
                "base_size": request.base_size,
                "image_size": request.image_size,
                "crop_mode": request.crop_mode
            },
            "debug": grounding_debug
        }
        callback_data = create_success_callback(request.url, result)
        
    except Exception as e:
        callback_data = create_error_callback(request.url, str(e))
    
    send_callback(request.callback_url, callback_data)

@app.post("/ocr-async")
def ocr_async(request: OCRAsyncRequest, background_tasks: BackgroundTasks):
    """异步OCR识别接口"""
    background_tasks.add_task(process_ocr_async, request)
    return JSONResponse(content={
        "success": True,
        "message": "任务已提交，结果将通过回调返回"
    })

@app.get("/")
async def root():
    """根路径"""
    return {"message": "DeepSeek OCR API 服务运行中"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)