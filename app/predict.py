import os
import sys
import urllib.request

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import ASTModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

WEIGHTS_URL = (
    "https://huggingface.co/rehan-hehe/ast-model/"
    "resolve/main/ast_emotion_regression_weights.pth"
)

WEIGHTS_PATH = "ast_emotion_regression_moresamples_batch16_epochs15.pth"

SAMPLE_RATE = 44100

FIXED_LENGTH = 9830
USABLE_LENGTH = 9216

CHUNK_SIZE = 1024
NUM_CHUNKS = 9

# Exact values obtained from the training preprocessing
GLOBAL_MEAN = -33.7032
GLOBAL_STD = 37.3519

# hop_length of torchaudio's default MelSpectrogram(sample_rate=44100)
HOP_LENGTH = 200
MIN_AUDIO_SECONDS = (USABLE_LENGTH * HOP_LENGTH) / SAMPLE_RATE  # ~41.8s
CHUNK_DURATION_SECONDS = (CHUNK_SIZE * HOP_LENGTH) / SAMPLE_RATE  # ~4.64s


# ============================================================
# MODEL
# ============================================================

class ASTForEmotionRegression(nn.Module):

    def __init__(
        self,
        pretrained_model_name=MODEL_NAME
    ):
        super().__init__()

        self.ast = ASTModel.from_pretrained(
            pretrained_model_name
        )

        self.regression_head = nn.Sequential(
            nn.Linear(
                self.ast.config.hidden_size,
                128
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, input_values):

        # [B, 1, 128, 1024]
        # ->
        # [B, 128, 1024]

        input_values = input_values.squeeze(1)

        outputs = self.ast(
            input_values=input_values
        )

        hidden_states = outputs.last_hidden_state

        pooled = hidden_states.mean(dim=1)

        output = self.regression_head(
            pooled
        )

        return output


# ============================================================
# DOWNLOAD WEIGHTS
# ============================================================

def download_weights():

    if os.path.exists(WEIGHTS_PATH):

        print(
            f"Model weights already found:\n"
            f"{WEIGHTS_PATH}"
        )

        return

    print("Downloading model weights...")

    urllib.request.urlretrieve(
        WEIGHTS_URL,
        WEIGHTS_PATH
    )

    print("Download complete.")


# ============================================================
# LOAD MODEL
# ============================================================
def remap_legacy_ast_keys(state_dict):
    """
    This checkpoint was saved with an older `transformers` version, where
    ASTModel's internals were named `ast.encoder.layer.N.attention.attention.
    query...`. Newer `transformers` versions renamed these to
    `ast.layers.N.attention.q_proj...`. This remaps the old names to the new
    ones so the checkpoint loads correctly regardless of which transformers
    version is installed on the deploy target.
    """
    key_map = [
        ("attention.output.dense", "attention.o_proj"),
        ("attention.attention.query", "attention.q_proj"),
        ("attention.attention.key", "attention.k_proj"),
        ("attention.attention.value", "attention.v_proj"),
        ("intermediate.dense", "mlp.fc1"),
        ("output.dense", "mlp.fc2"),
    ]

    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("ast.encoder.layer."):
            new_key = new_key.replace("ast.encoder.layer.", "ast.layers.", 1)
            for old_sub, new_sub in key_map:
                if old_sub in new_key:
                    new_key = new_key.replace(old_sub, new_sub)
                    break
        new_state_dict[new_key] = value

    return new_state_dict
    
def load_model(device):

    download_weights()

    print("Loading AST model...")

    model = ASTForEmotionRegression()

    state_dict = torch.load(
        WEIGHTS_PATH,
        map_location=device
    )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print("Standard load failed — retrying with legacy AST key remapping...")
        model.load_state_dict(remap_legacy_ast_keys(state_dict))

    model.to(device)

    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def preprocess_audio(audio_path):

    print(f"\nLoading audio:\n{audio_path}")

    waveform, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    waveform = torch.tensor(
        waveform,
        dtype=torch.float32
    )

    print(
        f"Waveform shape: {waveform.shape}"
    )

    mel_extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE
    )

    to_db = torchaudio.transforms.AmplitudeToDB()

    # Create mel spectrogram
    spec = mel_extractor(
        waveform
    )

    spec_db = to_db(
        spec
    )

    print(
        f"Spectrogram shape: {spec_db.shape}"
    )

    # Same preprocessing used during training
    if spec_db.shape[1] < USABLE_LENGTH:

        raise ValueError(
            f"Audio is too short.\n"
            f"Need at least {USABLE_LENGTH} frames, "
            f"got {spec_db.shape[1]}"
        )

    # Training kept up to fixed length
    spec_db = spec_db[:, :FIXED_LENGTH]

    # Z-score normalization
    spec_db = (
        spec_db - GLOBAL_MEAN
    ) / GLOBAL_STD

    # Training used first 9216 frames
    spec_db = spec_db[:, :USABLE_LENGTH]

    print(
        f"Processed spectrogram shape: "
        f"{spec_db.shape}"
    )

    # Split into 9 chunks
    segments = []

    for i in range(NUM_CHUNKS):

        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE

        segment = spec_db[:, start:end]

        if segment.shape != (128, 1024):

            raise ValueError(
                f"Unexpected segment shape: "
                f"{segment.shape}"
            )

        segments.append(segment)

    # [9, 128, 1024]
    segments = torch.stack(segments)

    # [9, 1, 128, 1024]
    segments = segments.unsqueeze(1)

    print(
        f"Final model input shape: "
        f"{segments.shape}"
    )

    return segments


# ============================================================
# PREDICTION
# ============================================================

def predict_emotion(
    model,
    audio_path,
    device
):

    segments = preprocess_audio(
        audio_path
    )

    segments = segments.to(device)

    with torch.no_grad():

        predictions = model(
            segments
        )

    # predictions:
    #
    # [9, 2]
    #
    # Each row corresponds to one chunk

    predictions = predictions.cpu()

    print("\nPredictions for each chunk:")

    for i, prediction in enumerate(predictions):

        valence = prediction[0].item()
        arousal = prediction[1].item()

        print(
            f"Chunk {i + 1}: "
            f"Valence={valence:.4f}, "
            f"Arousal={arousal:.4f}"
        )

    # Average predictions across chunks

    final_prediction = predictions.mean(
        dim=0
    )

    valence = final_prediction[0].item()
    arousal = final_prediction[1].item()

    # Raw per-chunk predictions (0-1 scale), exposed so a caller (e.g. the
    # Streamlit app) can show how the prediction evolves across the clip.
    # This is additive only — it doesn't change the averaging logic above.
    chunk_predictions = predictions.numpy()

    return valence, arousal, chunk_predictions


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "\nUsage:\n"
            "python predict.py path/to/audio.mp3"
        )

        return

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):

        print(
            f"File not found:\n{audio_path}"
        )

        return

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    model = load_model(device)

    valence, arousal, _ = predict_emotion(
        model,
        audio_path,
        device
    )

    print("\n" + "=" * 50)

    print("FINAL EMOTION PREDICTION")

    print("=" * 50)

    print(
        f"Valence: {valence:.4f}"
    )

    print(
        f"Arousal: {arousal:.4f}"
    )

    # Convert back to original DEAM scale [1, 9]
    valence_original = (
        valence * 8 + 1
    )

    arousal_original = (
        arousal * 8 + 1
    )

    print("\nOriginal DEAM scale [1, 9]:")

    print(
        f"Valence: {valence_original:.2f}"
    )

    print(
        f"Arousal: {arousal_original:.2f}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()
