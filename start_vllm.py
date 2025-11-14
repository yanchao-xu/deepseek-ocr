#!/usr/bin/env python3
import uvicorn
from main_vllm import app

if __name__ == "__main__":
    print("启动 DeepSeek OCR API 服务...")
    print("API文档地址: http://localhost:8000/docs")
    print("上传接口: POST http://localhost:8000/ocr")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,
        log_level="info"
    )