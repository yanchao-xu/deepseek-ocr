import requests
from typing import Dict, Any


def send_callback(callback_url: str, data: Dict[str, Any]) -> None:
    """发送回调请求"""
    try:
        requests.post(callback_url, json=data, timeout=10)
    except Exception:
        pass  # 静默处理回调失败


def create_success_callback(url: str, result: str) -> Dict[str, Any]:
    """创建成功回调数据"""
    return {
        "success": True,
        "url": url,
        "result": result
    }


def create_error_callback(url: str, error: str) -> Dict[str, Any]:
    """创建错误回调数据"""
    return {
        "success": False,
        "url": url,
        "error": error
    }