from __future__ import annotations

import html
import os
import random
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from huggingface_hub import InferenceClient
from pydub import AudioSegment


# ============================================================
# KONFIGURASI
# ============================================================
APP_DIR = Path(__file__).resolve().parent

SR = 16_000
N_MELS = 64
N_FFT = 1_024
HOP_LENGTH = 256
TOP_DB = 30
MAX_FRAMES = 501

INPUT_SHAPE = (N_MELS, MAX_FRAMES, 1)
DROPOUT = 0.30
MIN_BENAR = 0.50

SILENCE_RMS_DBFS = -38.0
SILENCE_PEAK = 0.015
ACTIVE_DBFS = -34.0
MIN_ACTIVE_SEC = 0.50
MIN_ACTIVE_RATIO = 0.025

QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

MODEL_FILES = [APP_DIR / "best_cnn.keras"]

METADATA_FILES = [APP_DIR / "metadata.csv",]

# UI DASAR
st.set_page_config(
    page_title="Ghunnah",
    page_icon="🎙️",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #f8fafc;
        }

        .hero {
            padding: 1.6rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #312e81, #6d28d9, #0284c7);
            color: white;
            margin-bottom: 1.25rem;
        }

        .hero h1 {
            margin-bottom: 0.3rem;
        }

        .hero p {
            margin: 0;
            opacity: 0.95;
        }


        .feedback-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: white;
            border: 1px solid #e2e8f0;
            border-left: 5px solid #7c3aed;
            line-height: 1.6;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# SECRET
def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return str(os.getenv(name, default)).strip()


HF_TOKEN = get_secret("HF_TOKEN")
HF_PROVIDER = get_secret("HF_PROVIDER", "auto") or "auto"

# MODEL CNN
def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(
        shape=INPUT_SHAPE,
        name="logmel_spectrogram",
    )

    x = inputs

    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D((2, 1))(x)

    x = tf.keras.layers.SpatialDropout2D(0.15)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT / 2)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="probability",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="GhunnahCNN",
    )


def find_model_path() -> Path | None:
    return next(
        (path for path in MODEL_FILES if path.exists()),
        None,
    )


@st.cache_resource(show_spinner="Memuat model CNN...")
def load_model(path_text: str) -> tf.keras.Model:
    path = Path(path_text)

    try:
        model = tf.keras.models.load_model(
            path,
            compile=False,
        )
    except Exception:
        model = build_model()
        model.load_weights(path)

    expected = (None, N_MELS, MAX_FRAMES, 1)

    if tuple(model.input_shape) != expected:
        raise ValueError(
            f"Input model harus {expected}, "
            f"tetapi diperoleh {model.input_shape}."
        )

    return model


# AUDIO
class SilentAudioError(Exception):
    """Dilempar ketika audio terlalu hening/tidak cukup aktif."""


def decode_audio(audio_bytes: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower() or ".audio"

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp:
        temp.write(audio_bytes)
        temp_path = Path(temp.name)

    try:
        try:
            waveform, _ = librosa.load(
                temp_path,
                sr=SR,
                mono=True,
            )
        except Exception:
            segment = (
                AudioSegment.from_file(temp_path)
                .set_channels(1)
                .set_frame_rate(SR)
            )

            samples = np.asarray(
                segment.get_array_of_samples(),
                dtype=np.float32,
            )

            scale = float(1 << (8 * segment.sample_width - 1))
            waveform = samples / max(scale, 1.0)

        return np.asarray(waveform, dtype=np.float32)

    finally:
        temp_path.unlink(missing_ok=True)


def validate_audio(waveform: np.ndarray) -> None:
    waveform = np.nan_to_num(
        np.asarray(waveform, dtype=np.float32)
    )

    if waveform.size == 0:
        raise SilentAudioError()

    peak = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(waveform**2) + 1e-12))
    rms_dbfs = 20 * np.log10(max(rms, 1e-12))

    if peak < SILENCE_PEAK or rms_dbfs < SILENCE_RMS_DBFS:
        raise SilentAudioError()

    frame_rms = librosa.feature.rms(
        y=waveform,
        frame_length=2048,
        hop_length=512,
    )[0]

    frame_dbfs = 20 * np.log10(
        np.maximum(frame_rms, 1e-12)
    )

    active = frame_dbfs >= ACTIVE_DBFS
    active_frames = int(active.sum())
    total_frames = max(len(active), 1)

    active_sec = active_frames * 512 / SR
    active_ratio = active_frames / total_frames

    if (
        active_sec < MIN_ACTIVE_SEC
        or active_ratio < MIN_ACTIVE_RATIO
    ):
        raise SilentAudioError()

    peak_to_rms = peak / max(rms, 1e-12)

    if peak_to_rms > 35 and active_sec < 1.0:
        raise SilentAudioError()


def preprocess_audio(
    audio_bytes: bytes,
    filename: str,
) -> np.ndarray:
    waveform = decode_audio(
        audio_bytes,
        filename,
    )

    validate_audio(waveform)

    waveform, _ = librosa.effects.trim(
        waveform,
        top_db=TOP_DB,
    )

    validate_audio(waveform)

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0,
    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    logmel = (
        logmel - logmel.mean()
    ) / (
        logmel.std() + 1e-8
    )

    if logmel.shape[1] > MAX_FRAMES:
        logmel = logmel[:, :MAX_FRAMES]

    elif logmel.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - logmel.shape[1]

        logmel = np.pad(
            logmel,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=float(logmel.min()),
        )

    return logmel[..., np.newaxis].astype(np.float32)

# FEEDBACK
@st.cache_data(show_spinner=False)
def load_feedback() -> dict[int, list[str]]:
    path = next(
        (p for p in METADATA_FILES if p.exists()),
        None,
    )

    if path is None:
        return {
            0: ["Ghunnah sudah terdengar jelas."],
            1: ["Ghunnah perlu diperbaiki agar sesuai kaidah tajwid."],
        }

    df = pd.read_csv(path)
    required = {"label", "error_explanation"}

    if not required.issubset(df.columns):
        raise ValueError(
            "Metadata harus memiliki kolom "
            "`label` dan `error_explanation`."
        )

    feedback: dict[int, list[str]] = {0: [], 1: []}

    for label in (0, 1):
        feedback[label] = (
            df.loc[
                df["label"].astype(int) == label,
                "error_explanation",
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

    return feedback


def one_sentence(text: str) -> str:
    text = " ".join(str(text).split()).strip()

    if not text:
        return ""

    for symbol in ".!?":
        index = text.find(symbol)

        if index >= 0:
            return text[: index + 1]

    return text.rstrip(".!?") + "."


def generate_feedback(
    status: str,
    base_feedback: str,
) -> str:
    if not HF_TOKEN:
        return base_feedback

    system_prompt = (
        "Anda adalah guru ngaji profesional. "
        "Buat tepat satu kalimat Bahasa Indonesia yang formal, natural, "
        "santun, dan memotivasi. Jangan menambah informasi baru atau "
        "mengubah inti kalimat dasar. Fokus pada bacaan ghunnah. "
        "Keluarkan hanya satu kalimat."
    )

    try:
        client = InferenceClient(
            provider=HF_PROVIDER,
            api_key=HF_TOKEN,
            timeout=60,
        )

        result = client.chat_completion(
            model=QWEN_ID,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"Status: {status}\n"
                        f"Kalimat dasar: {base_feedback}"
                    ),
                },
            ],
            max_tokens=80,
            temperature=0.4,
        )

        generated = (
            result.choices[0].message.content
            if result.choices
            else ""
        )

        return one_sentence(generated) or base_feedback

    except Exception:
        return base_feedback

# INFERENCE
def predict(
    model: tf.keras.Model,
    audio_bytes: bytes,
    filename: str,
) -> tuple[int, str]:
    x = preprocess_audio(
        audio_bytes,
        filename,
    )

    probability_wrong = float(
        model(
            np.expand_dims(x, axis=0),
            training=False,
        )
        .numpy()
        .reshape(-1)[0]
    )

    probability_wrong = float(
        np.clip(probability_wrong, 0.0, 1.0)
    )
    probability_correct = 1.0 - probability_wrong

    label = 0 if probability_correct >= MIN_BENAR else 1
    status = "BENAR" if label == 0 else "SALAH"

    # Probability tetap digunakan secara internal untuk menentukan kelas,
    # tetapi tidak ditampilkan kepada pengguna.
    return label, status


def run_analysis(
    model: tf.keras.Model,
    feedbacks: dict[int, list[str]],
    audio_bytes: bytes,
    filename: str,
) -> None:
    try:
        label, status = predict(
            model,
            audio_bytes,
            filename,
        )

        options = feedbacks.get(label) or [
            (
                "Ghunnah sudah terdengar jelas."
                if label == 0
                else "Ghunnah perlu diperbaiki agar sesuai kaidah tajwid."
            )
        ]

        base_feedback = random.choice(options)

        with st.spinner("Menyusun umpan balik..."):
            feedback = generate_feedback(
                status,
                base_feedback,
            )

        st.subheader("Umpan Balik")
        st.markdown(
            f"""
            <div class="feedback-card">
                {html.escape(feedback)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    except SilentAudioError:
        st.warning("Suara tidak terdengar atau terlalu pelan.")

    except Exception as error:
        st.error(
            f"Audio tidak dapat dianalisis. Detail: {error}"
        )


# APP
st.markdown(
    """
    <div class="hero">
        <h1>🎙️ Ghunnah</h1>
        <p>
            Analisis ketepatan bacaan ghunnah menggunakan CNN
            dengan umpan balik berbantuan Qwen2.5-1.5B.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Masukkan rekaman bacaan Surah Maryam ayat 4, 5, 7, atau 8 "
    "yang menjadi ruang lingkup sistem."
)

model_path = find_model_path()

if model_path is None:
    st.error("Model CNN belum ditemukan.")
    st.stop()

try:
    cnn_model = load_model(str(model_path))
    feedback_data = load_feedback()

except Exception as error:
    st.error(str(error))
    st.stop()

upload_tab, record_tab = st.tabs(
    ["Upload Audio", " Rekam Suara"]
)

with upload_tab:
    uploaded = st.file_uploader(
        "Pilih file audio",
        type=["wav", "mp3", "m4a", "mp4", "ogg", "flac"],
    )

    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        st.audio(audio_bytes)

        if st.button(
            "Analisis File Audio",
            type="primary",
            use_container_width=True,
            key="analyze_upload",
        ):
            run_analysis(
                cnn_model,
                feedback_data,
                audio_bytes,
                uploaded.name,
            )

with record_tab:
    recorded = st.audio_input(
        "Rekam bacaan ghunnah"
    )

    if recorded is not None:
        audio_bytes = recorded.getvalue()
        st.audio(audio_bytes)

        if st.button(
            "Analisis Rekaman",
            type="primary",
            use_container_width=True,
            key="analyze_recording",
        ):
            run_analysis(
                cnn_model,
                feedback_data,
                audio_bytes,
                getattr(
                    recorded,
                    "name",
                    "rekaman.wav",
                ),
            )

st.divider()
st.caption(
    "CNN menentukan hasil klasifikasi bacaan. "
    "Qwen hanya menyusun ulang umpan balik dan tidak mengubah keputusan CNN."
)
