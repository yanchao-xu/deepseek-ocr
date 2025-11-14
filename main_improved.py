import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
import uuid
from typing import Dict, Any
import json
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from config import MODEL_PATH
from utils.file_processor import (
    load_image_from_bytes, 
    extract_text_from_pdf, 
    extract_text_from_docx, 
    get_file_type
)
from utils.ocr_engine import ocr_generate, process_image_for_ocr

# 设置环境变量
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

app = FastAPI(title="DeepSeek OCR API", description="图片OCR识别接口")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 配置限制
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PDF_PAGES = 20
REQUEST_SEMAPHORE = asyncio.Semaphore(10)  # 最多10个并发请求
task_status: Dict[str, Dict[str, Any]] = {}

# 初始化OCR引擎
print("正在加载OCR模型...")
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    max_model_len=8192,
    enforce_eager=False,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.75,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)
print("OCR模型加载完成！")

async def process_file_async(task_id: str, file_bytes: bytes, filename: str, content_type: str):
    """异步处理文件"""
    try:
        task_status[task_id]["status"] = "processing"
        file_type = get_file_type(filename, content_type)
        
        if file_type == 'unknown':
            raise ValueError("不支持的文件格式")
        
        results = []
        
        if file_type == 'image':
            image = load_image_from_bytes(file_bytes)
            image_features = process_image_for_ocr(image)
            result = await ocr_generate(engine, image_features)
            results.append(result)
            
        elif file_type == 'pdf':
            images = extract_text_from_pdf(file_bytes)
            if len(images) > MAX_PDF_PAGES:
                raise ValueError(f"PDF页数超过限制({MAX_PDF_PAGES}页)")
            
            for i, image in enumerate(images):
                task_status[task_id]["progress"] = f"{i+1}/{len(images)}"
                image_features = process_image_for_ocr(image)
                result = await ocr_generate(engine, image_features)
                results.append(f"第{i+1}页:\n{result}")
                
        elif file_type == 'word':
            text = extract_text_from_docx(file_bytes)
            results.append(f"Word文档内容:\n{text}")
        
        final_result = "\n\n".join(results)
        
        task_status[task_id].update({
            "status": "completed",
            "result": {
                "success": True,
                "filename": filename,
                "file_type": file_type,
                "result": final_result
            }
        })
        
    except Exception as e:
        task_status[task_id].update({
            "status": "failed",
            "error": str(e)
        })

@app.post("/ocr")
async def upload_and_ocr(file: UploadFile = File(...)):
    """同步OCR接口（小文件）"""
    async with REQUEST_SEMAPHORE:
        file_bytes = await file.read()
        
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"文件过大，最大支持{MAX_FILE_SIZE//1024//1024}MB")
        
        file_type = get_file_type(file.filename, file.content_type)
        
        if file_type == 'unknown':
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        try:
            if file_type == 'image':
                image = load_image_from_bytes(file_bytes)
                image_features = process_image_for_ocr(image)
                result = await ocr_generate(engine, image_features)
                
                return JSONResponse(content={
                    "success": True,
                    "filename": file.filename,
                    "file_type": file_type,
                    "result": result
                })
            else:
                raise HTTPException(status_code=400, detail="大文件请使用异步接口 /ocr/async")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")

@app.post("/ocr/async")
async def upload_and_ocr_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """异步OCR接口（大文件）"""
    file_bytes = await file.read()
    
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"文件过大，最大支持{MAX_FILE_SIZE//1024//1024}MB")
    
    task_id = str(uuid.uuid4())
    task_status[task_id] = {
        "status": "queued",
        "progress": "0/0",
        "created_at": asyncio.get_event_loop().time()
    }
    
    background_tasks.add_task(process_file_async, task_id, file_bytes, file.filename, file.content_type)
    
    return JSONResponse(content={
        "task_id": task_id,
        "status": "queued",
        "message": "任务已提交，请使用task_id查询结果"
    })

@app.get("/ocr/status/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return JSONResponse(content=task_status[task_id])

@app.post("/ocr/stream")
async def upload_and_ocr_stream(file: UploadFile = File(...)):
    """流式OCR接口"""
    async def generate_stream():
        try:
            file_bytes = await file.read()
            file_type = get_file_type(file.filename, file.content_type)
            
            if file_type == 'pdf':
                images = extract_text_from_pdf(file_bytes)
                for i, image in enumerate(images):
                    yield f"data: {json.dumps({'type': 'progress', 'page': i+1, 'total': len(images)})}\n\n"
                    
                    image_features = process_image_for_ocr(image)
                    result = await ocr_generate(engine, image_features)
                    
                    yield f"data: {json.dumps({'type': 'result', 'page': i+1, 'content': result})}\n\n"
                    
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")

@app.get("/")
async def root():
    return {"message": "DeepSeek OCR API 服务运行中"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_tasks": len(task_status)}

# 清理过期任务
async def cleanup_tasks():
    while True:
        current_time = asyncio.get_event_loop().time()
        expired_tasks = [
            task_id for task_id, task in task_status.items()
            if current_time - task.get("created_at", 0) > 3600  # 1小时过期
        ]
        for task_id in expired_tasks:
            del task_status[task_id]
        await asyncio.sleep(300)  # 5分钟清理一次

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_tasks())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)