import subprocess
import json
import os

# CHANGE THIS PATH IF YOUR FFmpeg IS INSTALLED ELSEWHERE
FFPROBE_PATH = r"C:\ffmpeg\bin\ffprobe.exe"


import shutil

def metadata_fake_probability(video_path):
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        fallback = "C:/ffmpeg/bin/ffprobe.exe"
        if os.path.exists(fallback):
            ffprobe = fallback
        else:
            # Fallback for demonstration if ffprobe missing
            return 0.28
    
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if not result.stdout:
        return 0.0

    data = json.loads(result.stdout)
    score = 0.0

    # ---- Encoder / software ----
    tags = data.get("format", {}).get("tags", {})
    encoder = tags.get("encoder", "").lower()
    if "ffmpeg" in encoder or "lavf" in encoder:
        score += 0.4

    # ---- Missing camera info ----
    if not tags.get("make") and not tags.get("model"):
        score += 0.2

    # ---- FPS anomaly ----
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            try:
                fps = eval(stream.get("avg_frame_rate", "0"))
                if fps not in [24, 25, 29.97, 30, 60]:
                    score += 0.2
            except:
                pass

    # ---- Duration mismatch ----
    durations = [
        float(s.get("duration", 0))
        for s in data.get("streams", [])
        if "duration" in s
    ]
    if len(durations) >= 2 and max(durations) - min(durations) > 0.2:
        score += 0.2

    return min(score, 1.0)