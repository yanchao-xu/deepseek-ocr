import os
import sys
import tempfile
import shutil
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import uvicorn

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_ocr_hf import config
from deepseek_ocr_vllm.utils.prompt import build_image_prompt, OCRMode
from deepseek_ocr_vllm.utils.grounding_parser import clean_grounding_text, parse_detections
# -----------------------------
# Lifespan context for model loading
# -----------------------------
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global model, tokenizer
    
    # Environment setup
    MODEL_PATH = config.MODEL_PATH
    
    # Load model
    print(f"🚀 Loading {MODEL_PATH}...")
    torch_dtype = torch.bfloat16
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_safetensors=True,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
    ).eval().to("cuda")
    
    # Pad token setup
    try:
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    
    print("✅ Model loaded and ready!")
    
    yield
    
    # Cleanup
    print("🛑 Shutting down...")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="DeepSeek-OCR API",
    description="Blazing fast OCR with DeepSeek-OCR model 🔥",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root():
    return {"message": "DeepSeek-OCR API is running! 🚀", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/ocr")
async def ocr_inference(
    image: UploadFile = File(...),
    mode: OCRMode = Form(OCRMode.plain_ocr),
    prompt: str = Form(""),
    grounding: bool = Form(False),
    include_caption: bool = Form(False),
    find_term: Optional[str] = Form(None),
    schema: Optional[str] = Form(None),
    base_size: int = Form(config.BASE_SIZE),
    image_size: int = Form(config.IMAGE_SIZE),
    crop_mode: bool = Form(config.CROP_MODE),
    test_compress: bool = Form(False),
):
    """
    Perform OCR inference on uploaded image
    
    - **image**: Image file to process
    - **mode**: OCR mode (plain_ocr, markdown, tables_csv, etc.)
    - **prompt**: Custom prompt for freeform mode
    - **grounding**: Enable grounding boxes
    - **include_caption**: Add image description
    - **find_term**: Term to find (for find_ref mode)
    - **schema**: JSON schema (for kv_json mode)
    - **base_size**: Base processing size
    - **image_size**: Image size parameter
    - **crop_mode**: Enable crop mode
    - **test_compress**: Test compression
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Build prompt
    prompt_text = build_image_prompt(
        mode=mode,
        user_prompt=prompt,
        grounding=grounding,
        find_term=find_term,
        schema=schema,
        include_caption=include_caption,
    )
    
    tmp_img = None
    out_dir = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_img = tmp.name
        
        # Get original dimensions
        try:
            with Image.open(tmp_img) as im:
                orig_w, orig_h = im.size
        except Exception:
            orig_w = orig_h = None
        
        out_dir = tempfile.mkdtemp(prefix="dsocr_")
        
        # Run inference
        res = model.infer(
            tokenizer,
            prompt=prompt_text,
            image_file=tmp_img,
            output_path=out_dir,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=False,
            test_compress=test_compress,
            eval_mode=True,
        )
        
        # Normalize response
        if isinstance(res, str):
            text = res.strip()
        elif isinstance(res, dict) and "text" in res:
            text = str(res["text"]).strip()
        elif isinstance(res, (list, tuple)):
            text = "\n".join(map(str, res)).strip()
        else:
            text = ""
        
        # Fallback: check output file
        if not text:
            mmd = os.path.join(out_dir, "result.mmd")
            if os.path.exists(mmd):
                with open(mmd, "r", encoding="utf-8") as fh:
                    text = fh.read().strip()
        if not text:
            text = "No text returned by model."
        
        # Parse grounding boxes with proper coordinate scaling
        boxes = parse_detections(text, orig_w or 1, orig_h or 1) if ("<|det|>" in text or "<|ref|>" in text) else []
        
        # Clean grounding tags from display text, but keep the labels
        display_text = clean_grounding_text(text) if ("<|ref|>" in text or "<|grounding|>" in text) else text
        
        # If display text is empty after cleaning but we have boxes, show the labels
        if not display_text and boxes:
            display_text = ", ".join([b["label"] for b in boxes])
        
        return JSONResponse({
            "success": True,
            "text": display_text,
            "raw_text": text,  # Include raw model output for debugging
            "boxes": boxes,
            "image_dims": {"w": orig_w, "h": orig_h},
            "metadata": {
                "mode": mode,
                "grounding": grounding or (mode in {"find_ref","layout_map","pii_redact"}),
                "base_size": base_size,
                "image_size": image_size,
                "crop_mode": crop_mode
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
    
    finally:
        if tmp_img:
            try:
                os.remove(tmp_img)
            except Exception:
                pass
        if out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)

if __name__ == "__main__":
    host = config.API_HOST
    port = config.API_PORT
    uvicorn.run(app, host=host, port=port)
