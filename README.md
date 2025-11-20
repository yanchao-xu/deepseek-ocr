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
- 同步OCR接口: <http://localhost:9003/ocr>
- 异步OCR接口: <http://localhost:9003/ocr-async>

## 新功能：异步OCR with ZIP文件

异步OCR接口现在支持返回包含markdown和提取图片的zip文件：

```json
{
  "url": "https://example.com/document.pdf",
  "callback_url": "https://your-server.com/callback",
  "include_zip": true,
  "mode": "layout_map",
  "grounding": true
}
```

回调将包含：

- OCR识别的文本结果
- ZIP文件（base64编码），包含：
  - Markdown文件（包含OCR文本）
  - 从文档中提取的图片

## 配置说明

在 `config.py` 中可以修改:

- `API_HOST`: 服务监听地址 (默认: 0.0.0.0)
- `API_PORT`: 服务端口 (默认: 9003)
- `MODEL_PATH`: 模型文件路径
- `BASE_SIZE`, `IMAGE_SIZE`, `CROP_MODE`: 处理参数

## 推荐使用

- grounding 如果pdf中有图片，并且想要获取最终图片的位置等信息，要设置为true
- include_zip：异步接口才有用，会返回最终的md文件和图片

### 短pdf/image识别文字

- 用同步接口
- 模式选择“plain_ocr”或者不选

### 短pdf/image查找entity

eg：一张都是鸟的照片里找所有的鸟

- 同步接口
- 模式选择“find_ref”

### 描述一张图片

- 模式选择"describe"

### 长pdf文字识别

- 可以同步接口也可以异步接口
- 模式选择“markdown”

### 需要返回html格式

- 模式选择“html”

### 长pdf，图文混排

- 使用异步接口
- include_zip 为true
- 模式选择“markdown”
- grounding 为true

``` json
{
    "url": "https://example.com/document.pdf",
    "callback_url": "https://your-server.com/callback",
    "include_zip": true,
    "grounding": true,
    "mode": "markdown"
}
```
