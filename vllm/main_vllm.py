from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
from vllm import LLM
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from config import MODEL_PATH, PROMPT, MAX_CONCURRENCY
from utils.file_processor import download_file, get_file_type, pdf_to_images, load_image_from_url
from utils.ocr_engine import process_images_batch_ocr
from utils.callback_handler import send_callback, create_success_callback, create_error_callback

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

class OCRUrlRequest(BaseModel):
    url: str

class OCRAsyncRequest(BaseModel):
    url: str
    callback_url: str


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
        
        result = process_images_batch_ocr(llm, images)
        
        return JSONResponse(content={
            "success": True,
            "url": request.url,
            "result": result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")

def process_ocr_async(url: str, callback_url: str):
    """异步处理OCR任务"""
    try:
        file_bytes, filename, content_type = download_file(url)
        file_type = get_file_type(filename, content_type)
        
        if file_type == 'pdf':
            images = pdf_to_images(file_bytes)
        elif file_type == 'image':
            images = [load_image_from_url(url)]
        else:
            raise ValueError("不支持的文件格式")
        
        result = process_images_batch_ocr(llm, images, PROMPT)
        callback_data = create_success_callback(url, result)
        
    except Exception as e:
        callback_data = create_error_callback(url, str(e))
    
    send_callback(callback_url, callback_data)

@app.post("/ocr-async")
def ocr_async(request: OCRAsyncRequest, background_tasks: BackgroundTasks):
    """异步OCR识别接口"""
    background_tasks.add_task(process_ocr_async, request.url, request.callback_url)
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