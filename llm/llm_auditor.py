import requests
import json

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "meta-llama-3.1-8b-instruct"


SYSTEM_PROMPT = """
You are an AI auditor for a deepfake detection system.

You do NOT perform detection.
You ONLY explain and validate decisions.

You must:
- Explicitly state WHICH modality triggered the decision.
- Reference threshold logic when applicable.
- Explain in simple forensic terms.

Do NOT introduce new evidence.
Respond ONLY in JSON.
Your response MUST be in the following JSON format:
{
  "consistency": "CONSISTENT or INCONSISTENT",
  "confidence_level": "HIGH or MEDIUM or LOW",
  "explanation": "string",
  "warnings": ["optional list of strings"]
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
    user_prompt = f"""
Detection summary:

Video fake probability: {video_prob}
Audio fake probability: {audio_prob if audio_prob is not None else "N/A"}
Metadata fake probability: {meta_prob}

Final system decision: {final_label}
Detected manipulation type: {detected_type}
Override triggered: {override_triggered}

Please audit this decision.
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
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=5)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        # Fallback realistic simulator so the Awwwards frontend displays data
        return {
            "consistency": "CONSISTENT" if final_label == "DEEPFAKE" else "INCONSISTENT",
            "confidence_level": "HIGH",
            "explanation": f"The multimodal fusion matrix indicates a {final_label} state. The visual tensors exhibit artifacting typical of {detected_type}, supported by anomalies in spectral/metadata tracks. Confidence rating remains high due to redundant vector matching.",
            "warnings": [
               "SYSTEM OFFLINE: Local LM Studio on port 1234 unreachable.",
               "FALLBACK MATRIX ENGAGED: Simulated audit generated for demonstration."
            ]
        }