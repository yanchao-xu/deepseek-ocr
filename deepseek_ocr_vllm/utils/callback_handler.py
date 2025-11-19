import requests
import json
from typing import Dict, Any


def send_callback(callback_url: str, data: Dict[str, Any]) -> None:
    """发送回调请求"""
    try:
        requests.post(callback_url, json=data, timeout=10)
    except Exception:
        pass  # 静默处理回调失败


def send_callback_with_zip(callback_url: str, data: Dict[str, Any], zip_data: bytes, filename: str) -> None:
    """发送包含zip文件的回调请求"""
    try:
        # 大文件(>1MB)使用multipart，小文件可以用base64
        file_size_mb = len(zip_data) / (1024 * 1024)
        
        if file_size_mb > 1.0:
            # multipart传输 - 推荐大文件
            files = {'zip_file': (filename, zip_data, 'application/zip')}
            form_data = {'callback_data': json.dumps(data)}
            requests.post(callback_url, data=form_data, files=files, timeout=60)
        else:
            # JSON传输 - 小文件可用
            import base64
            zip_base64 = base64.b64encode(zip_data).decode('utf-8')
            data['zip_file'] = {
                'filename': filename,
                'data': zip_base64,
                'content_type': 'application/zip'
            }
            requests.post(callback_url, json=data, timeout=30)
    except Exception:
        pass  # 静默处理回调失败


def create_callback(url: str, success: bool = True, result: Any = None, error: str = None, zip_filename: str = None) -> Dict[str, Any]:
    """创建回调数据"""
    data = {
        "success": success,
        "url": url
    }
    
    if success:
        data["result"] = result
        if zip_filename:
            data["has_zip"] = True
            data["zip_filename"] = zip_filename
    else:
        data["error"] = error
    
    return data