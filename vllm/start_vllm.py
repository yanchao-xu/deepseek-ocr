#!/usr/bin/env python3
import uvicorn
from main_vllm import app
from config import API_HOST, API_PORT

if __name__ == "__main__":
    print("启动 DeepSeek OCR API 服务...")
    print(f"API文档地址: http://localhost:{API_PORT}/docs")
    print(f"上传接口: POST http://localhost:{API_PORT}/ocr")
    
    uvicorn.run(
        app, 
        host=API_HOST, 
        port=API_PORT,
        reload=False,
        log_level="info"
    )