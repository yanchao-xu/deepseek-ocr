# DeepSeek-OCR API 启动指南

## 启动方法

### 启动vllm

到vllm目录下

```bash
python start_vllm.py
```

### 启动hf

到hf目录下

```bash
python start_hf.py
```

## 启动前准备

1. **配置模型路径**: 编辑 `config.py` 文件，设置正确的 `MODEL_PATH`
2. **安装依赖**: 确保已安装所需的 Python 包
3. **GPU 环境**: 确保 CUDA 环境正常

## 访问服务

- API 文档: <http://localhost:9003/docs>
- 健康检查: <http://localhost:9003/health>
- OCR 接口: <http://localhost:9003/api/ocr>

## 配置说明

在 `config.py` 中可以修改:

- `API_HOST`: 服务监听地址 (默认: 0.0.0.0)
- `API_PORT`: 服务端口 (默认: 9003)
- `MODEL_PATH`: 模型文件路径
- `BASE_SIZE`, `IMAGE_SIZE`, `CROP_MODE`: 处理参数
