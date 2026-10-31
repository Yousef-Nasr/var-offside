"""
FastAPI application for image-based offside analysis.
"""

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from var_offside.api.service import analyze_image_bytes


app = FastAPI(
    title="VAR Offside API",
    version="1.0.0",
    description="Analyze a football freeze-frame and return structured offside results.",
)


@app.get("/health")
def health() -> dict:
    """Simple health check."""
    return {"status": "ok"}


@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    attacking_side: Optional[str] = Form(None),
    calibration_method: str = Form("auto"),
    use_automation: bool = Form(False),
) -> dict:
    """
    Analyze a single broadcast image and return the line player plus players
    ahead of that line.
    """
    if not use_automation and attacking_side not in ("left", "right"):
        raise HTTPException(
            status_code=400,
            detail="attacking_side must be 'left' or 'right' unless use_automation=true",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No image file was uploaded")

    image_bytes = await file.read()

    try:
        return analyze_image_bytes(
            image_bytes=image_bytes,
            attacking_side=attacking_side,
            calibration_method=calibration_method,
            use_automation=use_automation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
