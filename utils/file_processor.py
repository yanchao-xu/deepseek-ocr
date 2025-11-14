from PIL import Image, ImageOps
import io
import fitz
import requests
from docx import Document
import tempfile
from fastapi import HTTPException


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """从字节数据加载图片"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片加载失败: {str(e)}")


def load_image_from_url(url: str) -> Image.Image:
    """从URL加载图片"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return ImageOps.exif_transpose(image).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片下载失败: {str(e)}")


def download_file(url: str) -> tuple[bytes, str, str]:
    """下载文件并返回内容、文件名、内容类型"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        filename = url.split('/')[-1] or 'unknown'
        content_type = response.headers.get('content-type', '')
        return response.content, filename, content_type
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件下载失败: {str(e)}")


def pdf_to_images(pdf_bytes: bytes, dpi: int = 144) -> list[Image.Image]:
    """PDF转图片"""
    images = []
    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img_data = pixmap.tobytes("png")
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            images.append(image)
        
        pdf_doc.close()
        return images
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF处理失败: {str(e)}")


def extract_text_from_pdf(pdf_bytes: bytes) -> list:
    """从PDF提取图片（保持向后兼容）"""
    return pdf_to_images(pdf_bytes, dpi=144)


def extract_text_from_docx(docx_bytes: bytes) -> str:
    """从Word文档提取文本"""
    try:
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(docx_bytes)
            tmp_file.flush()
            doc = Document(tmp_file.name)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Word文档处理失败: {str(e)}")


def get_file_type(filename: str, content_type: str) -> str:
    """判断文件类型"""
    filename_lower = filename.lower() if filename else ""
    
    # 图片类型
    if (any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'jpg']) or 
        any(filename_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'])):
        return 'image'
    # PDF类型
    elif 'pdf' in content_type or filename_lower.endswith('.pdf'):
        return 'pdf'
    # Word类型
    elif (content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or 
          filename_lower.endswith(('.docx', '.doc'))):
        return 'word'
    else:
        return 'unknown'