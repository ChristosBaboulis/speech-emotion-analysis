import os
# Import CLASS_TO_IDX from IEMOCAP to ensure consistent label mapping
# Label order: angry=0, frustrated=1, happy=2, neutral=3, sad=4
from src.data.iemocap_dataset_loader import CLASS_TO_IDX

# Map RAVDESS emotion code (EE in filename) → our 5-class schema
RAVDESS_EMOTION_MAP = {
    "01": "neutral",      # neutral
    "02": "neutral",      # calm → neutral
    "03": "happy",        # happy
    "04": "sad",          # sad
    "05": "angry",        # angry
    "06": "frustrated",   # fearful → frustrated
    "07": "frustrated",   # disgust → frustrated
    "08": "happy",        # surprised → happy
}


def load_ravdess_metadata(base_path: str):
    """
    Scan RAVDESS folder structure and return a list of samples
    compatible with IEMOCAP metadata.
    Each sample: {"path": wav_path, "emotion": mapped_name, "label": class_idx}
    """
    samples = []

    # base_path / Actor_01 ... Actor_24
    for actor in sorted(os.listdir(base_path)):
        actor_dir = os.path.join(base_path, actor)
        if not os.path.isdir(actor_dir):
            continue
        if not actor.startswith("Actor_"):
            continue

        for fname in os.listdir(actor_dir):
            if not fname.endswith(".wav"):
                continue

            parts = fname.split("-")
            if len(parts) != 7:
                # Unexpected filename format, skip
                continue

            emo_code = parts[2]  # MM-LL-EE-PP-SS-CC-OO.wav

            if emo_code not in RAVDESS_EMOTION_MAP:
                continue

            mapped = RAVDESS_EMOTION_MAP[emo_code]
            label = CLASS_TO_IDX[mapped]

            wav_path = os.path.join(actor_dir, fname)

            samples.append(
                {
                    "path": wav_path,
                    "emotion": mapped,
                    "label": label,
                }
            )

    return samples

