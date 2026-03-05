# api/routes.py
from fastapi import APIRouter, UploadFile, File
from PIL import Image
from io import BytesIO
from typing import Any, Dict

from pipeline.analyze import Analyzer

router = APIRouter()

# You’ll initialize this in server.py and inject it
analyzer: Analyzer = None  # type: ignore

@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = await file.read()
    img = Image.open(BytesIO(content)).convert("RGB")
    result = analyzer.analyze_image(img)
    return result
