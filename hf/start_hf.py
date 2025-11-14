#!/usr/bin/env python3
"""
DeepSeek-OCR API 启动脚本

使用方法:
    python start.py

或者直接运行:
    python main_hf.py
"""

import subprocess
import sys
import os

def main():
    """启动 DeepSeek-OCR API 服务"""
    print("🚀 启动 DeepSeek-OCR API 服务...")
    
    # 检查是否在正确的目录
    if not os.path.exists("main_hf.py"):
        print("❌ 错误: 请在 deepseek-ocr 项目根目录下运行此脚本")
        sys.exit(1)
    
    # 检查是否存在 config.py
    if not os.path.exists("config.py"):
        print("❌ 错误: 找不到 config.py 文件")
        sys.exit(1)
    
    try:
        # 启动服务
        subprocess.run([sys.executable, "main_hf.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()