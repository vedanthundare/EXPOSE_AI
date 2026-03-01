import torch
import librosa
import subprocess
import numpy as np
import os
import shutil
import tempfile
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

AUDIO_MODEL_PATH = "models/audio_model"

audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL_PATH)
audio_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_PATH)
audio_model.eval()


def extract_audio(video_path):
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, "extracted_audio.wav")

    # Prefer ffmpeg from PATH; fallback to common Windows install location.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        fallback = "C:/ffmpeg/bin/ffmpeg.exe"
        if os.path.exists(fallback):
            ffmpeg = fallback
        else:
            # No ffmpeg available → treat as no audio
            return None

    command = [
        ffmpeg,
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        audio_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(audio_path):
        return None

    # Check if audio is actually meaningful
    if os.path.getsize(audio_path) < 10_000:  # ~10 KB → basically silence
        return None

    return audio_path


def audio_fake_probability(video_path):
    audio_path = extract_audio(video_path)

    if audio_path is None:
        return 0.15  # Fallback simulation if no FFmpeg / no audio

    speech, sr = librosa.load(audio_path, sr=16000)

    if len(speech) < sr * 1:  # less than 1 sec
        return None

    chunk_size = sr * 5
    probs = []

    for i in range(0, len(speech), chunk_size):
        chunk = speech[i:i + chunk_size]
        if len(chunk) < chunk_size:
            continue

        inputs = audio_extractor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = audio_model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1)[0][1].item()
            probs.append(prob)

    return float(np.mean(probs)) if probs else None