# DeepSeek-OCR API 启动指南

## 启动方法

### 方法 1: 使用启动脚本
```bash
python start.py
```

### 方法 2: 直接启动
```bash
python main_hf.py
```

### 方法 3: 使用 uvicorn 启动
```bash
uvicorn main_hf:app --host 0.0.0.0 --port 8000
```

## 启动前准备

1. **配置模型路径**: 编辑 `config.py` 文件，设置正确的 `MODEL_PATH`
2. **安装依赖**: 确保已安装所需的 Python 包
3. **GPU 环境**: 确保 CUDA 环境正常

## 访问服务

- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health
- OCR 接口: http://localhost:8000/api/ocr

## 配置说明

在 `config.py` 中可以修改:
- `API_HOST`: 服务监听地址 (默认: 0.0.0.0)
- `API_PORT`: 服务端口 (默认: 8000)
- `MODEL_PATH`: 模型文件路径
- `BASE_SIZE`, `IMAGE_SIZE`, `CROP_MODE`: 处理参数