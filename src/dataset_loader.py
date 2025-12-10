import os

EMOTION_MAP = {
    "ang": "angry",
    "sad": "sad",
    "hap": "happy",
    "exc": "happy",         # exc + hap → happy
    "neu": "neutral",
    "fru": "frustrated",
}

CLASS_TO_IDX = {emo: i for i, emo in enumerate(sorted(set(EMOTION_MAP.values())))}


def load_iemocap_metadata(base_path: str):
    samples = []

    for sess in range(1, 6):
        session = f"Session{sess}"

        emo_dir = os.path.join(base_path, session, "dialog", "EmoEvaluation")
        wav_root = os.path.join(base_path, session, "sentences", "wav")

        if not os.path.isdir(emo_dir):
            continue

        for fname in os.listdir(emo_dir):
            # Skip Apple system files
            if not fname.endswith(".txt"):
                continue
            if fname.startswith("._"):
                continue

            # Example: Ses01F_impro01.txt
            dialog_id = os.path.splitext(fname)[0]

            emo_file = os.path.join(emo_dir, fname)
            wav_dialog_dir = os.path.join(wav_root, dialog_id)

            if not os.path.isdir(wav_dialog_dir):
                continue

            with open(emo_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line.startswith("["):
                    continue

                parts = line.replace("[", " [").split()
                
                utt_id = parts[3]
                raw_emotion = parts[4]

                if raw_emotion not in EMOTION_MAP:
                    continue

                mapped = EMOTION_MAP[raw_emotion]
                label = CLASS_TO_IDX[mapped]

                wav_path = os.path.join(wav_dialog_dir, utt_id + ".wav")

                if os.path.isfile(wav_path):
                    samples.append({
                        "path": wav_path,
                        "emotion": mapped,
                        "label": label,
                        "session": session,
                })


    return samples
