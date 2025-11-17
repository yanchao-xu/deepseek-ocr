# DeepSeek OCR API 使用说明

## 功能说明

基于FastAPI实现的图片OCR识别服务，支持上传图片并返回识别结果。

## 安装依赖

```bash

# uv pip install -U vllm --pre --extra-index-url https: //wheels.vllm.ai/nightly --index-strategy unsafe-best-match

uv venv --python 3.12.9
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
uv pip install ./vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl  
uv pip install flash-attn==2.7.3 --no-build-isolation --index-url https://pypi.tuna.tsinghua.edu.cn/simple
   uv pip install flash-attn==2.7.3 --no-build-isolation --verbose
```

## 启动服务

```bash
python start_vllm.py
```

服务启动后访问地址：

- API服务: <http://localhost:9003>
- API文档: <http://localhost:9003/docs>
- 健康检查: <http://localhost:9003/health>

## API接口

### 1. 图片OCR识别

- **接口**: `POST /ocr`
- **参数**: `file` (图片文件)
- **返回**: JSON格式的识别结果


#### 返回示例

```json
{
  "success": true,
  "url": "your_file_url",
  "result": "识别出的文字内容..."
}
```

## 支持的图片格式

- JPEG
- PNG  
- BMP
- TIFF
- PDF
- 其他PIL支持的格式

## 注意事项

1. 确保已正确配置 `config.py` 中的模型路径
2. 服务首次启动时会加载模型，可能需要较长时间
3. 建议使用GPU加速以获得更好的性能

### 改动

1. deepseek ocr
2. clip——sdpa
