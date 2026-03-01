import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference.video_infer import video_fake_probability
from inference.audio_infer import audio_fake_probability
from utils.metadata_infer import metadata_fake_probability
from fusion.fusion import fuse
from llm.llm_auditor import audit_decision


class LlmAuditResult(BaseModel):
    consistency: str
    confidence_level: str
    explanation: str
    warnings: Optional[list[str]] = None


class AnalysisResponse(BaseModel):
    video_prob: float
    audio_prob: Optional[float]
    meta_prob: float
    final_score: float
    label: str
    detected_type: str
    reason: str
    override_triggered: bool
    llm_audit: Optional[LlmAuditResult] = None


app = FastAPI(title="Multimodal Deepfake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_video(file: UploadFile = File(...)) -> AnalysisResponse:
    if file.content_type not in ("video/mp4", "video/x-msvideo", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an MP4 or AVI video.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        video_path = tmp.name

    video_prob = video_fake_probability(video_path)
    audio_prob = audio_fake_probability(video_path)
    meta_prob = metadata_fake_probability(video_path)

    final_score, label, dtype, reason_text = fuse(video_prob, audio_prob, meta_prob)

    override_triggered = dtype in {
        "AUDIO_DEEPFAKE",
        "VIDEO_DEEPFAKE",
        "METADATA_DEEPFAKE",
    }

    llm_result: Optional[LlmAuditResult] = None
    try:
        raw = audit_decision(
            video_prob,
            audio_prob,
            meta_prob,
            label,
            dtype,
            override_triggered,
        )
        llm_result = LlmAuditResult(**raw)
    except Exception:
        llm_result = None

    return AnalysisResponse(
        video_prob=video_prob,
        audio_prob=audio_prob,
        meta_prob=meta_prob,
        final_score=final_score,
        label=label,
        detected_type=dtype,
        reason=reason_text,
        override_triggered=override_triggered,
        llm_audit=llm_result,
    )