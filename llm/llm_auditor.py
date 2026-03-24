import requests
import json

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "meta-llama-3.1-8b-instruct"


SYSTEM_PROMPT = """
You are an AI forensic auditor for a multimodal deepfake detection system called EXPOSE AI.

You do NOT perform detection — you ONLY explain and validate the system's numeric decisions in a narrative intelligence report.

Your rules:
- Reference SPECIFIC modality values (visual tensor, spectral tensor, pipeline metadata).
- Cite threshold logic when applicable (e.g., "visual tensor 0.72 exceeded the 0.50 override threshold").
- Use precise, clinical forensic language — no casual phrasing.
- Do NOT introduce new evidence beyond what is given.
- Be authoritative. Your explanation is read by HUMAN ANALYSTS.

Respond ONLY in JSON using this exact format:
{
  "consistency": "CONSISTENT or INCONSISTENT",
  "confidence_level": "HIGH or MEDIUM or LOW",
  "explanation": "A 2-4 sentence forensic narrative summarizing which modalities drove the decision, what the tensor values indicate, and why the conclusion is reliable or uncertain.",
  "warnings": ["optional list of system/audit warning strings"]
}
"""


def audit_decision(
    video_prob,
    audio_prob,
    meta_prob,
    final_label,
    detected_type,
    override_triggered
):
    audio_str = f"{audio_prob:.2f}" if audio_prob is not None else "N/A (no audio track)"
    override_str = "YES — safeguard override triggered" if override_triggered else "NO — soft fusion applied"

    user_prompt = f"""
Forensic detection summary for analyst review:

VISUAL TENSOR:           {video_prob:.2f}  (threshold: 0.50 for hard override)
AUDIO / SPECTRAL TENSOR: {audio_str}  (threshold: 0.60 for hard override)
PIPELINE FORENSICS:      {meta_prob:.2f}  (threshold: 0.85 for isolated override)

FUSION OUTPUT:
  Final score:     {(video_prob * 0.4 + (audio_prob or 0) * 0.4 + meta_prob * 0.2):.2f}
  Label:           {final_label}
  Subtype:         {detected_type}
  Override status: {override_str}

Generate a forensic intelligence audit narrative for the above.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=10)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # Strip markdown code fences if model wraps response
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception:
        # ── Fallback: structured simulation so the UI always renders ──
        vp = round(video_prob * 100)
        ap = round((audio_prob or 0) * 100)
        mp = round(meta_prob * 100)

        if final_label == "DEEPFAKE":
            drivers = []
            if video_prob >= 0.50:
                drivers.append(f"visual tensor ({vp}%) exceeded the 0.50 hard-override threshold")
            if audio_prob is not None and audio_prob >= 0.60:
                drivers.append(f"spectral tensor ({ap}%) exceeded the 0.60 vocal-synthesis threshold")
            if meta_prob >= 0.85:
                drivers.append(f"pipeline forensics ({mp}%) flagged a synthetic encoding chain")
            driver_str = "; ".join(drivers) if drivers else "combined multimodal soft-fusion score"
            explanation = (
                f"The multimodal fusion matrix converged on a DEEPFAKE verdict. "
                f"Primary evidence: {driver_str}. "
                f"The {detected_type.replace('_', ' ')} subtype classification is consistent with the "
                f"tensor distribution — visual ({vp}%), spectral ({ap}%), pipeline ({mp}%). "
                f"Confidence remains high due to redundant cross-modal vector matching."
            )
            consistency = "CONSISTENT"
        else:
            explanation = (
                f"The multimodal fusion matrix indicates a REAL / AUTHENTIC state. "
                f"Visual tensor ({vp}%) remained below the 0.50 manipulation threshold; "
                f"spectral analysis ({ap}%) showed no vocal-synthesis fingerprints; "
                f"pipeline forensics ({mp}%) detected no synthetic encoding artifacts. "
                f"No single modality exceeded its override threshold — soft fusion applied, "
                f"yielding a {detected_type.replace('_', ' ')} classification with high confidence."
            )
            consistency = "INCONSISTENT"  # no inconsistency between modalities flagging vs result

        return {
            "consistency": consistency,
            "confidence_level": "HIGH",
            "explanation": explanation,
            "warnings": [
                "SYSTEM OFFLINE: Local LM Studio on port 1234 unreachable.",
                "FALLBACK MATRIX ENGAGED: Simulated audit generated for demonstration."
            ]
        }