# DeepSeek OCR 启动问题解决指南

## 问题描述
启动时出现 `RuntimeError: Engine core initialization failed` 错误，这通常是由于 vLLM 引擎初始化失败导致的。

## 解决步骤

### 1. 快速修复（推荐）
```bash
# 运行自动修复脚本
python fix_vllm_issues.py

# 运行诊断脚本
python debug_start.py

# 如果诊断通过，启动服务
python start_server.py
```

### 2. 手动排查

#### 2.1 检查环境
```bash
# 检查 CUDA 是否可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, 版本: {torch.version.cuda}')"

# 检查 GPU 内存
nvidia-smi
```

#### 2.2 安装/更新依赖
```bash
# 安装完整依赖
pip install -r requirements.txt

# 或者手动安装关键依赖
pip install vllm>=0.6.0 torch>=2.0.0 transformers>=4.40.0 einops
```

#### 2.3 环境变量设置
```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
# 不要设置 VLLM_USE_V1=0
```

### 3. 配置调整

#### 3.1 降低资源使用
在 `main.py` 中调整以下参数：
```python
llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    enforce_eager=True,          # 改为 True
    trust_remote_code=True,
    max_model_len=2048,          # 从 8192 降低到 2048
    max_num_seqs=1,              # 从 4 降低到 1
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,  # 从 0.9 降低到 0.5
    disable_mm_preprocessor_cache=True,
)
```

#### 3.2 模型路径检查
确保模型路径正确：
- 如果使用 HuggingFace 模型：`deepseek-ai/DeepSeek-OCR`
- 如果使用本地模型：确保路径存在且文件完整

### 4. 备选方案

#### 4.1 使用 CPU 模式
如果 GPU 内存不足或配置有问题：
```bash
python main_cpu.py
```

#### 4.2 使用更小的模型
临时使用更小的模型进行测试：
```bash
python test_vllm_minimal.py
```

## 常见问题

### Q1: CUDA 内存不足
**解决方案：**
- 降低 `gpu_memory_utilization` 到 0.3-0.5
- 减少 `max_model_len` 和 `max_num_seqs`
- 重启 Python 进程清理内存

### Q2: 模型下载失败
**解决方案：**
```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或者手动下载模型到本地
git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
```

### Q3: vLLM 版本兼容性
**解决方案：**
```bash
# 更新到最新版本
pip install --upgrade vllm

# 或者使用特定版本
pip install vllm==0.6.0
```

### Q4: 多进程启动失败
**解决方案：**
```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## 验证步骤

### 1. 基础功能测试
```bash
python test_vllm_minimal.py
```

### 2. 完整诊断
```bash
python debug_start.py
```

### 3. 启动服务
```bash
python start_server.py
```

### 4. 测试 API
```bash
curl -X POST "http://localhost:8000/health"
```

## 性能优化建议

1. **GPU 内存优化**
   - 根据 GPU 内存调整 `gpu_memory_utilization`
   - 8GB GPU: 0.5-0.6
   - 16GB GPU: 0.7-0.8
   - 24GB+ GPU: 0.8-0.9

2. **并发优化**
   - 单 GPU: `max_num_seqs=1-2`
   - 多 GPU: 根据 GPU 数量调整

3. **内存优化**
   - 设置合适的 `max_model_len`
   - 启用 `enforce_eager=True` 避免编译开销

## 联系支持

如果问题仍然存在，请提供以下信息：
1. 运行 `python debug_start.py` 的完整输出
2. GPU 型号和内存大小
3. CUDA 和 PyTorch 版本
4. 完整的错误日志