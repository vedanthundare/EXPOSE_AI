def fuse(video_prob, audio_prob, meta_prob):
    """
    Rule-guided multimodal fusion with safeguard logic.
    """

    # ---------------- Thresholds ----------------
    AUDIO_TH = 0.60         # strong audio spoof
    VIDEO_STRONG_TH = 0.50   # strong visual manipulation
    VIDEO_SUS_TH = 0.60      # subtle visual manipulation
    META_STRONG_TH = 0.85    # strong synthetic pipeline
    META_SUS_TH = 0.50       # re-encoding / suspicious

    # ---------------- HARD OVERRIDES ----------------

    # Audio-only deepfake
    if audio_prob is not None and audio_prob >= AUDIO_TH:
        return (
            audio_prob,
            "DEEPFAKE",
            "AUDIO_DEEPFAKE",
            "Audio fake probability exceeded threshold (≥ 70%)"
        )

    # Strong video manipulation
    if video_prob >= VIDEO_STRONG_TH:
        return (
            video_prob,
            "DEEPFAKE",
            "VIDEO_DEEPFAKE",
            "Visual manipulation confidence exceeded threshold (≥ 80%)"
        )

    # Subtle video + suspicious metadata
    if video_prob >= VIDEO_SUS_TH and meta_prob >= META_SUS_TH:
        return (
            video_prob,
            "DEEPFAKE",
            "VIDEO_DEEPFAKE",
            "Moderate visual anomalies combined with suspicious metadata"
        )

    # Metadata-only (very rare, strong case)
    if meta_prob >= META_STRONG_TH:
        return (
            meta_prob,
            "DEEPFAKE",
            "METADATA_DEEPFAKE",
            "Strong metadata anomalies indicate synthetic processing"
        )

    # ---------------- SOFT FUSION ----------------

    weights = {"video": 0.4, "audio": 0.4, "meta": 0.2}

    score = video_prob * weights["video"]
    total = weights["video"]

    if audio_prob is not None:
        score += audio_prob * weights["audio"]
        total += weights["audio"]

    score += meta_prob * weights["meta"]
    total += weights["meta"]

    final_score = score / total

    if final_score > 0.5:
        return (
            final_score,
            "DEEPFAKE",
            "MULTIMODAL_DEEPFAKE",
            "Combined multimodal evidence indicates manipulation"
        )

    return (
        final_score,
        "REAL",
        "NO_MANIPULATION",
        "No modality exceeded manipulation thresholds"
    )