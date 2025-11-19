from fastapi import FastAPI, Request, Form, File, UploadFile
import json
import zipfile
import io
import uvicorn
import os
from datetime import datetime

app = FastAPI(title="OCR Callback Test Server")

@app.post("/callback")
async def receive_callback(request: Request, callback_data: str = Form(None), zip_file: UploadFile = File(None)):
    """接收OCR异步回调结果"""
    try:
        # 解析回调数据
        if callback_data:
            data = json.loads(callback_data)
            print(f"\n收到回调: {data.get('url', 'unknown')}")
            print(f"处理结果: {'success' if data.get('success') else 'failed'}")
        
        # 处理zip文件
        if zip_file:
            zip_content = await zip_file.read()
            print(f"ZIP文件: {zip_file.filename}, 大小: {len(zip_content)} bytes")
            
            # 创建output目录
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存原始zip文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_path = os.path.join(output_dir, f"{timestamp}_{zip_file.filename}")
            with open(zip_path, 'wb') as f:
                f.write(zip_content)
            print(f"ZIP文件已保存: {zip_path}")
            
            # 解压zip文件
            with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"ZIP包含文件: {file_list}")
                
                # 创建解压目录
                extract_dir = os.path.join(output_dir, f"{timestamp}_extracted")
                os.makedirs(extract_dir, exist_ok=True)
                
                # 解压并保存每个文件
                for filename in file_list:
                    with zip_ref.open(filename) as file:
                        content = file.read()
                        
                        # 保存文件到磁盘
                        file_path = os.path.join(extract_dir, filename)
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        
                        if filename.endswith('.md'):
                            text_content = content.decode('utf-8')
                            print(f"\nMarkdown文件已保存: {file_path}")
                            print(text_content[:200] + "..." if len(text_content) > 200 else text_content)
                        
                        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            print(f"图片文件已保存: {file_path}, 大小: {len(content)} bytes")
        
        # 处理base64格式的zip文件
        elif callback_data:
            data = json.loads(callback_data)
            if 'zip_file' in data:
                import base64
                zip_data = base64.b64decode(data['zip_file']['data'])
                print(f"Base64 ZIP: {data['zip_file']['filename']}, 大小: {len(zip_data)} bytes")
                
                # 创建output目录
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                
                # 保存base64解码的zip文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_path = os.path.join(output_dir, f"{timestamp}_{data['zip_file']['filename']}")
                with open(zip_path, 'wb') as f:
                    f.write(zip_data)
                print(f"Base64 ZIP文件已保存: {zip_path}")
                
                # 解压文件
                extract_dir = os.path.join(output_dir, f"{timestamp}_base64_extracted")
                os.makedirs(extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
                    for filename in zip_ref.namelist():
                        with zip_ref.open(filename) as file:
                            content = file.read()
                            
                            file_path = os.path.join(extract_dir, filename)
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            with open(file_path, 'wb') as f:
                                f.write(content)
                            
                            if filename.endswith('.md'):
                                text_content = content.decode('utf-8')
                                print(f"\nMarkdown文件已保存: {file_path}")
                                print(f"内容预览: {text_content[:100]}...")
                            else:
                                print(f"文件已保存: {file_path}")
        
        return {"status": "success", "message": "回调处理成功，文件已保存到output目录"}
        
    except Exception as e:
        print(f"处理回调失败: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
def root():
    return {"message": "OCR Callback Test Server"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)