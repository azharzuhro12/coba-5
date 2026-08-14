from __future__ import annotations

import html
import io
import os
import random
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import webrtcvad
from huggingface_hub import InferenceClient
from pydub import AudioSegment


# ============================================================
# KONFIGURASI SESUAI NOTEBOOK skripsi-cnn-slm-eval-fixed (13)
# ============================================================
APP_DIR = Path(__file__).resolve().parent

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
SILENCE_TOP_DB = 30

REFERENCE_DURATION = 8.0
REFERENCE_SAMPLES = int(SAMPLE_RATE * REFERENCE_DURATION)
MAX_FRAMES = 1 + REFERENCE_SAMPLES // HOP_LENGTH

CNN_INPUT_SHAPE = (N_MELS, MAX_FRAMES, 1)
MODEL_DROPOUT = 0.30

# Output sigmoid merupakan P(SALAH).
# Keputusan aplikasi memakai P(BENAR) dengan batas minimum 50%.
CLASSIFICATION_THRESHOLD = 0.50

SLM_MODEL_NAME = "Qwen2.5-1.5B"
SLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SLM_MAX_NEW_TOKENS = 80
SLM_TEMPERATURE = 0.4

# Deteksi audio hening/noise kecil sebelum CNN.
SILENCE_RMS_DBFS_THRESHOLD = -48.0
SILENCE_PEAK_THRESHOLD = 0.005

# Syarat aktivitas suara absolut.
# Bukan threshold relatif terhadap suara paling keras di rekaman.
ACTIVE_FRAME_DBFS_THRESHOLD = -44.0
MIN_ACTIVE_DURATION_SECONDS = 0.25
MIN_ACTIVE_FRAME_RATIO = 0.010

# Pengaman tambahan agar hasil yang meragukan tidak langsung disebut BENAR.
# Ini adalah rejection rule di level aplikasi, bukan threshold training CNN.
MIN_BENAR_PROBABILITY = 0.50


# ============================================================
# FILTER JENIS AUDIO / OUT-OF-DISTRIBUTION (OOD)
# ============================================================
# YAMNet hanya menjadi gerbang sebelum CNN.
# CNN skripsi, preprocessing, dan keputusan BENAR/SALAH tidak diubah.
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

# Maksimal audio yang diperiksa YAMNet agar tetap ringan.
YAMNET_CHECK_SECONDS = 10.0

# Ambang dibuat konservatif supaya tilawah/chant tidak mudah ditolak.
MIN_HUMAN_VOICE_SCORE = 0.035
ANIMAL_REJECT_SCORE = 0.28
MUSIC_REJECT_SCORE = 0.45
NONVOICE_MARGIN = 0.15


# WebRTC VAD: lapisan tambahan untuk memastikan ada pola speech/voice.
# Mode 2 = cukup agresif, tetapi masih lebih ramah terhadap bacaan
# dibanding mode 3.
WEBRTC_VAD_MODE = 2
WEBRTC_FRAME_MS = 30
MIN_VAD_SPEECH_RATIO = 0.12
MIN_VAD_SPEECH_SECONDS = 0.30

# Kategori manusia dibuat lebih presisi.
# Jangan memakai kata "vocal" karena bisa mencocokkan "animal vocalization".
SPEECH_KEYWORDS = (
    "speech",
    "speaking",
    "conversation",
    "narration",
    "whisper",
)

RECITATION_KEYWORDS = (
    "chant",
    "mantra",
    "humming",
)

SINGING_KEYWORDS = (
    "singing",
)

ANIMAL_KEYWORDS = (
    "animal",
    "dog",
    "cat",
    "bird",
    "bark",
    "meow",
    "roar",
    "growl",
    "insect",
    "cricket",
    "frog",
    "horse",
    "cow",
    "cattle",
    "pig",
    "sheep",
    "goat",
    "chicken",
    "duck",
    "goose",
)

MUSIC_KEYWORDS = (
    "music",
    "musical instrument",
    "guitar",
    "piano",
    "keyboard",
    "drum",
    "percussion",
    "violin",
    "viola",
    "cello",
    "flute",
    "saxophone",
    "trumpet",
    "orchestra",
    "rock",
    "pop music",
    "hip hop",
    "jazz",
    "classical music",
    "electronic music",
    "techno",
    "disco",
)

MODEL_CANDIDATES = [
    APP_DIR / "best_cnn.h5",
    APP_DIR / "best_cnn.keras",
    APP_DIR / "best_cnn.weights.h5",
]

METADATA_CANDIDATES = [
    APP_DIR / "metadata.csv",
    APP_DIR / "feedback_kb.csv",
]


st.set_page_config(
    page_title="Ghunnah",
    page_icon="🎙️",
    layout="centered",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(124, 58, 237, .09),
                transparent 28rem
            ),
            radial-gradient(
                circle at 92% 4%,
                rgba(14, 165, 233, .10),
                transparent 27rem
            ),
            #f8fafc;
    }

    .hero {
        padding: 1.7rem 1.8rem;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #312e81 0%,
            #6d28d9 52%,
            #0284c7 100%
        );
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 18px 48px rgba(49, 46, 129, .20);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.15rem;
        letter-spacing: -0.035em;
    }

    .hero p {
        margin: .55rem 0 0;
        opacity: .94;
        line-height: 1.55;
    }

    .feedback-card {
        padding: 1rem 1.15rem;
        border-radius: 18px;
        background: rgba(255,255,255,.92);
        border: 1px solid rgba(148,163,184,.30);
        border-left: 5px solid #7c3aed;
        line-height: 1.65;
        font-size: 1.04rem;
        margin-top: .6rem;
    }
</style>
"""

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


# ============================================================
# SECRETS
# ============================================================
def read_secret_or_env(
    name: str,
    default: str = "",
) -> str:
    try:
        return str(
            st.secrets[name]
        ).strip()
    except Exception:
        return str(
            os.getenv(
                name,
                default,
            )
        ).strip()


HF_TOKEN = read_secret_or_env(
    "HF_TOKEN"
)

HF_PROVIDER = (
    read_secret_or_env(
        "HF_PROVIDER",
        "auto",
    )
    or "auto"
)


# ============================================================
# MODEL CNN
# ============================================================
def build_ghunnah_cnn(
    input_shape=CNN_INPUT_SHAPE,
    dropout=MODEL_DROPOUT,
):
    """
    Arsitektur persis seperti notebook (13).
    Output sigmoid merupakan P(SALAH).
    """
    inputs = tf.keras.Input(
        shape=input_shape,
        name="logmel_spectrogram",
    )

    x = tf.keras.layers.Conv2D(
        16,
        kernel_size=3,
        padding="same",
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 1)
    )(x)

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 1)
    )(x)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 1)
    )(x)

    x = tf.keras.layers.SpatialDropout2D(
        0.15
    )(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(
        dropout
    )(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(
        dropout / 2
    )(x)

    probabilities = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="probability",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=probabilities,
        name="GhunnahCNN",
    )


def find_model_path() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


@st.cache_resource(
    show_spinner="Memuat model CNN..."
)
def load_best_cnn(
    model_path_text: str,
):
    model_path = Path(
        model_path_text
    )

    # Mendukung:
    # 1) full model .keras
    # 2) full model .h5
    # 3) weights-only .h5
    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )
    except Exception:
        model = build_ghunnah_cnn(
            dropout=MODEL_DROPOUT
        )
        model.load_weights(
            model_path
        )

    expected_shape = (
        None,
        N_MELS,
        MAX_FRAMES,
        1,
    )

    actual_shape = tuple(
        model.input_shape
    )

    if actual_shape != expected_shape:
        raise ValueError(
            "Input model tidak cocok dengan notebook (13). "
            f"Diharapkan {expected_shape}, diperoleh {actual_shape}."
        )

    return model


# ============================================================
# MODEL FILTER AUDIO (YAMNet)
# ============================================================
@st.cache_resource(
    show_spinner="Memuat filter jenis audio..."
)
def load_yamnet():
    """
    YAMNet mengenali kelas audio umum (speech, musik, hewan, dll).
    Dipakai hanya untuk menolak input yang jelas bukan suara bacaan.
    """
    model = hub.load(
        YAMNET_HANDLE
    )

    class_map_path = (
        model
        .class_map_path()
        .numpy()
    )

    if isinstance(
        class_map_path,
        bytes,
    ):
        class_map_path = (
            class_map_path.decode(
                "utf-8"
            )
        )

    class_names = (
        pd.read_csv(
            class_map_path
        )["display_name"]
        .astype(str)
        .tolist()
    )

    return (
        model,
        class_names,
    )


def keyword_class_score(
    class_names: list[str],
    class_scores: np.ndarray,
    keywords: tuple[str, ...],
) -> float:
    """
    Ambil skor maksimum dari kelas-kelas YAMNet yang
    nama kelasnya mengandung salah satu keyword.
    """
    indices = []

    for index, class_name in enumerate(
        class_names
    ):
        lowered = (
            class_name
            .lower()
        )

        if any(
            keyword in lowered
            for keyword in keywords
        ):
            indices.append(
                index
            )

    if not indices:
        return 0.0

    return float(
        np.max(
            class_scores[
                indices
            ]
        )
    )


def get_webrtc_vad_metrics(
    waveform: np.ndarray,
) -> dict:
    """
    Hitung proporsi frame yang dianggap speech oleh WebRTC VAD.

    Input:
    - mono
    - 16 kHz
    - float32 [-1, 1]

    WebRTC VAD membutuhkan PCM 16-bit dengan frame 10/20/30 ms.
    """
    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.size == 0:
        return {
            "vad_speech_ratio": 0.0,
            "vad_speech_seconds": 0.0,
            "vad_total_frames": 0,
            "vad_speech_frames": 0,
        }

    waveform = np.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    waveform = np.clip(
        waveform,
        -1.0,
        1.0,
    )

    pcm16 = (
        waveform
        * 32767.0
    ).astype(
        np.int16
    )

    frame_samples = int(
        SAMPLE_RATE
        * WEBRTC_FRAME_MS
        / 1000
    )

    total_frames = (
        len(pcm16)
        // frame_samples
    )

    if total_frames <= 0:
        return {
            "vad_speech_ratio": 0.0,
            "vad_speech_seconds": 0.0,
            "vad_total_frames": 0,
            "vad_speech_frames": 0,
        }

    vad = webrtcvad.Vad(
        WEBRTC_VAD_MODE
    )

    speech_frames = 0

    for frame_index in range(
        total_frames
    ):
        start = (
            frame_index
            * frame_samples
        )

        end = (
            start
            + frame_samples
        )

        frame = pcm16[
            start:end
        ].tobytes()

        try:
            is_speech = vad.is_speech(
                frame,
                SAMPLE_RATE,
            )
        except Exception:
            is_speech = False

        if is_speech:
            speech_frames += 1

    speech_ratio = (
        speech_frames
        / float(
            total_frames
        )
    )

    speech_seconds = (
        speech_frames
        * WEBRTC_FRAME_MS
        / 1000.0
    )

    return {
        "vad_speech_ratio": float(
            speech_ratio
        ),
        "vad_speech_seconds": float(
            speech_seconds
        ),
        "vad_total_frames": int(
            total_frames
        ),
        "vad_speech_frames": int(
            speech_frames
        ),
    }


def validate_ghunnah_audio_type(
    waveform: np.ndarray,
) -> dict:
    """
    Filter OOD dua lapis:
    1. WebRTC VAD -> apakah ada pola speech/voice?
    2. YAMNet -> apakah dominan musik/hewan/non-bacaan?

    CNN ghunnah hanya dijalankan jika input lolos dua pemeriksaan.
    """
    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.size == 0:
        raise SilentAudioError(
            "Suara tidak terdengar."
        )

    # --------------------------------------------------------
    # LAPIS 1: WEBRTC VAD
    # --------------------------------------------------------
    vad_metrics = (
        get_webrtc_vad_metrics(
            waveform
        )
    )

    vad_speech_ratio = float(
        vad_metrics[
            "vad_speech_ratio"
        ]
    )

    vad_speech_seconds = float(
        vad_metrics[
            "vad_speech_seconds"
        ]
    )

    # --------------------------------------------------------
    # LAPIS 2: YAMNET
    # --------------------------------------------------------
    max_samples = int(
        SAMPLE_RATE
        * YAMNET_CHECK_SECONDS
    )

    waveform_for_yamnet = waveform[
        :max_samples
    ]

    yamnet_model, class_names = (
        load_yamnet()
    )

    scores, _, _ = yamnet_model(
        tf.convert_to_tensor(
            waveform_for_yamnet,
            dtype=tf.float32,
        )
    )

    scores = np.asarray(
        scores.numpy(),
        dtype=np.float32,
    )

    if scores.size == 0:
        raise InvalidGhunnahAudioError(
            "Jenis audio tidak dapat divalidasi. Silakan rekam ulang.",
            vad_metrics,
        )

    mean_scores = np.mean(
        scores,
        axis=0,
    )

    presence_scores = np.percentile(
        scores,
        90,
        axis=0,
    )

    top_indices = np.argsort(
        mean_scores
    )[::-1][:5]

    top5 = [
        {
            "label": str(
                class_names[
                    int(index)
                ]
            ),
            "score": float(
                mean_scores[
                    int(index)
                ]
            ),
        }
        for index in top_indices
    ]

    top_label = (
        top5[0]["label"]
    )

    top_score = float(
        top5[0]["score"]
    )

    speech_score = (
        keyword_class_score(
            class_names,
            presence_scores,
            SPEECH_KEYWORDS,
        )
    )

    recitation_score = (
        keyword_class_score(
            class_names,
            presence_scores,
            RECITATION_KEYWORDS,
        )
    )

    singing_score = (
        keyword_class_score(
            class_names,
            presence_scores,
            SINGING_KEYWORDS,
        )
    )

    animal_score = (
        keyword_class_score(
            class_names,
            mean_scores,
            ANIMAL_KEYWORDS,
        )
    )

    music_score = (
        keyword_class_score(
            class_names,
            mean_scores,
            MUSIC_KEYWORDS,
        )
    )

    trusted_human_score = max(
        speech_score,
        recitation_score,
    )

    details = {
        **vad_metrics,
        "passed": True,
        "top_label": top_label,
        "top_score": top_score,
        "top5": top5,
        "speech_score": float(
            speech_score
        ),
        "recitation_score": float(
            recitation_score
        ),
        "singing_score": float(
            singing_score
        ),
        "trusted_human_score": float(
            trusted_human_score
        ),
        "animal_score": float(
            animal_score
        ),
        "music_score": float(
            music_score
        ),
    }

    lowered_top = (
        top_label
        .lower()
    )

    top_is_animal = any(
        keyword in lowered_top
        for keyword in ANIMAL_KEYWORDS
    )

    top_is_music = any(
        keyword in lowered_top
        for keyword in MUSIC_KEYWORDS
    )

    # --------------------------------------------------------
    # A. HEWAN: TOLAK TEGAS
    # --------------------------------------------------------
    # Suara hewan tidak boleh lolos hanya karena VAD salah
    # membaca beberapa frame sebagai voice.
    if (
        top_is_animal
        or animal_score >= 0.12
    ):
        # Pengecualian hanya jika indikasi bacaan manusia
        # jauh lebih kuat daripada indikasi hewan.
        if not (
            trusted_human_score >= 0.25
            and trusted_human_score
            >= animal_score * 2.0
            and vad_speech_ratio >= 0.20
        ):
            details["passed"] = False

            raise InvalidGhunnahAudioError(
                "Audio terdeteksi sebagai suara hewan atau bukan bacaan manusia. "
                "Silakan gunakan rekaman bacaan ghunnah.",
                details,
            )

    # --------------------------------------------------------
    # B. MUSIK: TOLAK TEGAS
    # --------------------------------------------------------
    # Singing tidak dijadikan bukti bacaan.
    # Lagu/vokal musik tetap harus ditolak.
    if (
        top_is_music
        or music_score >= 0.15
    ):
        # Bacaan hanya diberi pengecualian jika ada bukti kuat
        # speech/chant/mantra DAN VAD juga mendeteksi voice.
        strong_recitation = (
            trusted_human_score >= 0.22
            and vad_speech_ratio >= 0.20
            and trusted_human_score
            >= music_score * 1.25
        )

        if not strong_recitation:
            details["passed"] = False

            raise InvalidGhunnahAudioError(
                "Audio terdeteksi sebagai musik atau suara non-bacaan. "
                "Silakan gunakan rekaman bacaan ghunnah.",
                details,
            )

    # --------------------------------------------------------
    # C. HARUS ADA SPEECH/VOICE YANG CUKUP
    # --------------------------------------------------------
    # Ini membuat suara lingkungan/hewan yang lolos dari YAMNet
    # tidak langsung masuk CNN.
    if (
        vad_speech_ratio
        < MIN_VAD_SPEECH_RATIO
        or vad_speech_seconds
        < MIN_VAD_SPEECH_SECONDS
    ):
        details["passed"] = False

        raise InvalidGhunnahAudioError(
            "Tidak terdeteksi pola suara manusia yang cukup untuk "
            "dianalisis sebagai bacaan ghunnah.",
            details,
        )

    # --------------------------------------------------------
    # D. HARUS ADA INDIKASI BACAAN MANUSIA
    # --------------------------------------------------------
    # Singing saja tidak cukup agar lagu tidak lolos.
    if trusted_human_score < 0.035:
        details["passed"] = False

        raise InvalidGhunnahAudioError(
            "Audio belum dikenali sebagai suara bacaan manusia yang cukup. "
            "Silakan gunakan rekaman bacaan ghunnah yang lebih jelas.",
            details,
        )

    return details


# ============================================================
# AUDIO
# ============================================================
class SilentAudioError(Exception):
    pass


class InvalidGhunnahAudioError(Exception):
    """
    Audio terdengar, tetapi bukan input yang sesuai untuk
    klasifikasi bacaan ghunnah.
    """

    def __init__(
        self,
        message: str,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.details = details or {}


def decode_audio(
    audio_bytes: bytes,
    filename: str,
) -> np.ndarray:
    """
    Mencoba decode dengan librosa terlebih dahulu.
    Jika gagal, gunakan ffmpeg melalui pydub.

    File uploader tidak dibatasi ekstensi sehingga berbagai format
    audio dapat dicoba.
    """
    suffix = (
        Path(filename).suffix.lower()
        or ".audio"
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_file.write(
                audio_bytes
            )
            temp_path = Path(
                temp_file.name
            )

        try:
            waveform, _ = librosa.load(
                temp_path,
                sr=SAMPLE_RATE,
                mono=True,
            )

        except Exception:
            segment = AudioSegment.from_file(
                temp_path
            )

            segment = (
                segment
                .set_channels(1)
                .set_frame_rate(SAMPLE_RATE)
            )

            samples = np.array(
                segment.get_array_of_samples(),
                dtype=np.float32,
            )

            if samples.size == 0:
                return np.array(
                    [],
                    dtype=np.float32,
                )

            denominator = float(
                1 << (
                    8 * segment.sample_width - 1
                )
            )

            waveform = (
                samples
                / max(
                    denominator,
                    1.0,
                )
            )

        return np.asarray(
            waveform,
            dtype=np.float32,
        )

    finally:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            try:
                temp_path.unlink()
            except Exception:
                pass


def validate_audible_audio(
    waveform: np.ndarray,
) -> None:
    """
    Menolak audio kosong, terlalu pelan, atau hanya memiliki noise/
    bunyi singkat.

    Pemeriksaan ini menggunakan level absolut dBFS, sehingga berbeda
    dari librosa.effects.split/trim yang bersifat relatif terhadap
    bagian terkeras dari rekaman.
    """
    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.size == 0:
        raise SilentAudioError(
            "Suara tidak terdengar."
        )

    # Hilangkan NaN/Inf bila ada.
    waveform = np.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    peak = float(
        np.max(
            np.abs(
                waveform
            )
        )
    )

    if peak < SILENCE_PEAK_THRESHOLD:
        raise SilentAudioError(
            "Suara tidak terdengar."
        )

    # RMS seluruh rekaman.
    global_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    waveform
                )
            )
            + 1e-12
        )
    )

    global_rms_dbfs = float(
        20.0
        * np.log10(
            max(
                global_rms,
                1e-12,
            )
        )
    )

    if (
        global_rms_dbfs
        < SILENCE_RMS_DBFS_THRESHOLD
    ):
        raise SilentAudioError(
            "Suara tidak terdengar."
        )

    # --------------------------------------------------------
    # FRAME-LEVEL ABSOLUTE ACTIVITY CHECK
    # --------------------------------------------------------
    frame_length = 2048
    hop_length = 512

    frame_rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    frame_dbfs = (
        20.0
        * np.log10(
            np.maximum(
                frame_rms,
                1e-12,
            )
        )
    )

    active_mask = (
        frame_dbfs
        >= ACTIVE_FRAME_DBFS_THRESHOLD
    )

    active_frames = int(
        np.sum(
            active_mask
        )
    )

    total_frames = max(
        int(
            frame_dbfs.size
        ),
        1,
    )

    active_duration = (
        active_frames
        * hop_length
        / float(SAMPLE_RATE)
    )

    active_ratio = (
        active_frames
        / float(total_frames)
    )

    # Harus ada suara yang cukup lama.
    if (
        active_duration
        < MIN_ACTIVE_DURATION_SECONDS
    ):
        raise SilentAudioError(
            "Suara tidak terdengar."
        )

    # Untuk rekaman panjang, satu bunyi kecil sesaat tidak boleh
    # dianggap sebagai bacaan valid.
    if (
        active_ratio
        < MIN_ACTIVE_FRAME_RATIO
    ):
        raise SilentAudioError(
            "Suara tidak terdengar."
        )

    # --------------------------------------------------------
    # TRANSIENT / IMPULSE CHECK
    # --------------------------------------------------------
    # Bunyi klik/ketukan pendek bisa memiliki peak besar tetapi RMS
    # sangat kecil. Peak-to-RMS yang ekstrem ditolak.
    peak_to_rms = (
        peak
        / max(
            global_rms,
            1e-12,
        )
    )

    if (
        peak_to_rms > 35.0
        and active_duration < 1.0
    ):
        raise SilentAudioError(
            "Suara tidak terdengar."
        )


def load_audio_trimmed(
    waveform: np.ndarray,
) -> np.ndarray:
    """
    Sesuai notebook:
    waveform -> silence trimming top_db=30.
    """
    validate_audible_audio(
        waveform
    )

    trimmed_audio, _ = librosa.effects.trim(
        waveform,
        top_db=SILENCE_TOP_DB,
    )

    if trimmed_audio.size == 0:
        raise SilentAudioError(
            "Suara tidak terdengar."
        )

    validate_audible_audio(
        trimmed_audio
    )

    return trimmed_audio.astype(
        np.float32
    )


def pad_or_crop_spectrogram(
    logmel: np.ndarray,
    max_frames: int = MAX_FRAMES,
) -> np.ndarray:
    n_frames = logmel.shape[1]

    if n_frames > max_frames:
        logmel = logmel[
            :,
            :max_frames,
        ]

    elif n_frames < max_frames:
        right_padding = (
            max_frames
            - n_frames
        )

        pad_value = float(
            logmel.min()
        )

        logmel = np.pad(
            logmel,
            (
                (0, 0),
                (0, right_padding),
            ),
            mode="constant",
            constant_values=pad_value,
        )

    return logmel


def extract_logmel(
    waveform: np.ndarray,
) -> np.ndarray:
    """
    Persis dengan notebook:
    trimmed waveform -> full Log-Mel -> z-score -> crop/pad ke 501 frame.
    """
    mel_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0,
    )

    logmel = librosa.power_to_db(
        mel_power,
        ref=np.max,
    )

    mean = float(
        logmel.mean()
    )

    std = float(
        logmel.std()
    )

    logmel = (
        logmel - mean
    ) / (
        std + 1e-8
    )

    logmel = pad_or_crop_spectrogram(
        logmel
    )

    return logmel.astype(
        np.float32
    )


def audio_to_model_input(
    waveform: np.ndarray,
) -> np.ndarray:
    logmel = extract_logmel(
        waveform
    )

    return np.expand_dims(
        logmel,
        axis=-1,
    )


# ============================================================
# FEEDBACK
# ============================================================
@st.cache_data(
    show_spinner=False
)
def load_feedback_texts() -> dict[int, list[str]]:
    selected_path = None

    for candidate in METADATA_CANDIDATES:
        if candidate.exists():
            selected_path = candidate
            break

    if selected_path is None:
        return {
            0: [
                "Ghunnah sudah terdengar jelas."
            ],
            1: [
                "Ghunnah perlu diperbaiki agar sesuai kaidah tajwid."
            ],
        }

    df = pd.read_csv(
        selected_path
    )

    required = {
        "label",
        "error_explanation",
    }

    if not required.issubset(
        set(df.columns)
    ):
        raise ValueError(
            f"{selected_path.name} harus memiliki kolom "
            "`label` dan `error_explanation`."
        )

    result = {
        0: [],
        1: [],
    }

    for label in (
        0,
        1,
    ):
        texts = (
            df.loc[
                df["label"].astype(int) == label,
                "error_explanation",
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        result[label] = [
            text
            for text in texts.tolist()
            if text
        ]

    return result


def choose_base_feedback(
    feedback_texts: dict[int, list[str]],
    predicted_label: int,
) -> str:
    choices = feedback_texts.get(
        int(predicted_label),
        [],
    )

    if not choices:
        return (
            "Ghunnah sudah terdengar jelas."
            if int(predicted_label) == 0
            else "Ghunnah perlu diperbaiki agar sesuai kaidah tajwid."
        )

    return random.choice(
        choices
    )


def keep_one_sentence(
    text: str,
) -> str:
    normalized = " ".join(
        str(text).strip().split()
    )

    if not normalized:
        return ""

    import re

    match = re.match(
        r"^(.+?[.!?])(?:\s|$)",
        normalized,
    )

    if match:
        return match.group(1).strip()

    return (
        normalized.rstrip(
            ".!?"
        )
        + "."
    )


def build_prompt_messages(
    status: str,
    info_text: str,
):
    """
    Mengikuti aturan prompt pada notebook (13):
    satu kalimat, formal/natural, tidak menambah informasi baru.
    """
    status = str(
        status
    ).upper().strip()

    info_text = " ".join(
        str(info_text).strip().split()
    )

    if status == "SALAH":
        system_message = (
            "Anda adalah guru ngaji profesional. "
            "Buat tepat satu kalimat umpan balik dalam Bahasa Indonesia "
            "yang formal, natural, santun, dan memotivasi. "
            "Jangan menambah informasi baru, jangan mengubah inti makna "
            "kalimat dasar, dan jangan menambahkan jenis kesalahan yang "
            "tidak disebutkan. Fokus pada perbaikan bacaan ghunnah. "
            "Keluarkan hanya satu kalimat akhir tanpa judul."
        )

    else:
        system_message = (
            "Anda adalah guru ngaji profesional. "
            "Buat tepat satu kalimat apresiasi dalam Bahasa Indonesia "
            "yang formal, natural, santun, dan memotivasi. "
            "Jangan menambah informasi baru dan jangan mengubah inti "
            "makna kalimat dasar. Fokus pada apresiasi bacaan ghunnah "
            "dan hindari pujian berlebihan. "
            "Keluarkan hanya satu kalimat akhir tanpa judul."
        )

    user_message = (
        f"Kalimat dasar:\n{info_text}\n\n"
        "Susun ulang kalimat tersebut tanpa mengubah inti maknanya."
    )

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


def generate_qwen_feedback(
    status: str,
    base_feedback: str,
) -> tuple[str, str]:
    if not HF_TOKEN:
        return (
            base_feedback,
            "metadata",
        )

    try:
        client = InferenceClient(
            provider=HF_PROVIDER,
            api_key=HF_TOKEN,
            timeout=60,
        )

        result = client.chat_completion(
            model=SLM_MODEL_ID,
            messages=build_prompt_messages(
                status,
                base_feedback,
            ),
            max_tokens=SLM_MAX_NEW_TOKENS,
            temperature=SLM_TEMPERATURE,
        )

        text = ""

        if result.choices:
            text = (
                result.choices[0]
                .message
                .content
                or ""
            )

        text = keep_one_sentence(
            text
        )

        if text:
            return (
                text,
                SLM_MODEL_NAME,
            )

    except Exception:
        pass

    return (
        base_feedback,
        "metadata",
    )


# ============================================================
# INFERENCE
# ============================================================
def predict_audio(
    model,
    audio_bytes: bytes,
    filename: str,
):
    waveform = decode_audio(
        audio_bytes,
        filename,
    )

    waveform = load_audio_trimmed(
        waveform
    )

    # --------------------------------------------------------
    # FILTER JENIS AUDIO SEBELUM CNN
    # --------------------------------------------------------
    # Musik, hewan, dan input non-suara-manusia yang jelas
    # dihentikan di sini sehingga tidak dipaksa menjadi BENAR/SALAH.
    audio_type_info = (
        validate_ghunnah_audio_type(
            waveform
        )
    )

    x = audio_to_model_input(
        waveform
    )

    x_batch = np.expand_dims(
        x,
        axis=0,
    )

    probability_wrong = float(
        np.asarray(
            model(
                x_batch,
                training=False,
            )
        ).reshape(-1)[0]
    )

    probability_wrong = float(
        np.clip(
            probability_wrong,
            0.0,
            1.0,
        )
    )

    probability_correct = float(
        1.0 - probability_wrong
    )

    # ========================================================
    # ATURAN KEPUTUSAN APLIKASI
    # ========================================================
    # Gunakan SATU threshold saja, yaitu P(BENAR).
    #
    # P(BENAR) >= 0.50 -> BENAR
    # P(BENAR) <  0.50 -> SALAH
    #
    # Karena output sigmoid CNN adalah P(SALAH),
    # P(BENAR) = 1 - P(SALAH).
    predicted_label = (
        0
        if probability_correct
        >= MIN_BENAR_PROBABILITY
        else 1
    )

    status = (
        "BENAR"
        if predicted_label == 0
        else "SALAH"
    )

    return {
        "predicted_label": predicted_label,
        "status": status,
        "probability_correct": probability_correct,
        "probability_wrong": probability_wrong,
        "audio_type_info": audio_type_info,
    }


def render_result(
    prediction: dict,
    feedback_texts: dict[int, list[str]],
):
    predicted_label = int(
        prediction["predicted_label"]
    )

    status = prediction["status"]

    # Label BENAR/SALAH dan probabilitas tidak ditampilkan.
    # Hasil prediksi CNN langsung digunakan untuk memilih feedback.
    base_feedback = choose_base_feedback(
        feedback_texts,
        predicted_label,
    )

    with st.spinner(
        "Menyusun umpan balik..."
    ):
        feedback, source = generate_qwen_feedback(
            status,
            base_feedback,
        )

    st.subheader(
        "Umpan balik"
    )

    st.markdown(
        f"""
        <div class="feedback-card">
            {html.escape(feedback)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if (
        source == "metadata"
        and not HF_TOKEN
    ):
        st.caption(
            "Qwen belum aktif karena HF_TOKEN belum diisi; "
            "feedback dasar tetap ditampilkan."
        )


    # Ditampilkan untuk pengujian agar false-positive dapat dikalibrasi
    # berdasarkan skor sebenarnya, bukan menebak threshold.
    audio_type_info = prediction.get(
        "audio_type_info",
        {}
    )

    if audio_type_info:
        with st.expander(
            "Detail filter audio"
        ):
            st.json(
                audio_type_info
            )


def run_analysis(
    model,
    feedback_texts,
    audio_bytes: bytes,
    filename: str,
):
    try:
        prediction = predict_audio(
            model,
            audio_bytes,
            filename,
        )

    except SilentAudioError:
        st.warning(
            "Suara terlalu pelan atau tidak cukup jelas untuk dianalisis."
        )
        st.caption(
            "Coba dekatkan mikrofon dan baca ghunnah dengan volume normal."
        )
        return

    except InvalidGhunnahAudioError as error:
        st.warning(
            "Audio bukan input bacaan ghunnah."
        )

        st.info(
            str(error)
        )

        # Detail disembunyikan agar UI utama tetap sederhana.
        # Berguna saat pengujian/skripsi untuk melihat keputusan YAMNet.
        if error.details:
            with st.expander(
                "Detail filter audio"
            ):
                st.json(
                    error.details
                )

        return

    except Exception as error:
        st.error(
            "Audio tidak dapat dianalisis. "
            f"Detail: {error}"
        )
        return

    render_result(
        prediction,
        feedback_texts,
    )


# ============================================================
# UI
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🎙️ Ghunnah</h1>
        <p>
            Analisis bacaan ghunnah menggunakan CNN dan pemberian
            umpan balik menggunakan Qwen2.5-1.5B.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_path = find_model_path()

if model_path is None:
    st.error(
        "Best model belum ditemukan. Tambahkan salah satu file berikut "
        "ke root repository: `best_cnn.h5`, `best_cnn.keras`, "
        "atau `best_cnn.weights.h5`."
    )
    st.stop()

try:
    cnn_model = load_best_cnn(
        str(model_path)
    )
except Exception as error:
    st.error(
        "Best model gagal dimuat. Pastikan model berasal dari "
        "arsitektur notebook (13). "
        f"Detail: {error}"
    )
    st.stop()

try:
    feedback_texts = load_feedback_texts()
except Exception as error:
    st.error(
        f"File metadata feedback tidak valid: {error}"
    )
    st.stop()


upload_tab, record_tab = st.tabs(
    [
        "📁 Upload audio",
        "🎤 Rekam suara",
    ]
)


with upload_tab:
    st.subheader(
        "Upload audio"
    )

    st.caption(
        "Uploader menerima berbagai format file. "
        "File yang bukan audio akan ditolak saat proses decoding."
    )

    uploaded_audio = st.file_uploader(
        "Pilih file audio",
        type=None,
        accept_multiple_files=False,
        key="uploaded_audio",
    )

    if uploaded_audio is not None:
        audio_bytes = (
            uploaded_audio.getvalue()
        )

        st.audio(
            audio_bytes
        )

        if st.button(
            "Analisis file audio",
            type="primary",
            use_container_width=True,
            key="analyze_upload",
        ):
            run_analysis(
                cnn_model,
                feedback_texts,
                audio_bytes,
                uploaded_audio.name,
            )


with record_tab:
    st.subheader(
        "Rekam suara"
    )

    if not hasattr(
        st,
        "audio_input",
    ):
        st.error(
            "Versi Streamlit ini belum mendukung perekaman mikrofon."
        )
    else:
        recorded_audio = st.audio_input(
            "Rekam bacaan ghunnah",
            key="recorded_audio",
        )

        if recorded_audio is not None:
            recorded_bytes = (
                recorded_audio.getvalue()
            )

            st.audio(
                recorded_bytes
            )

            if st.button(
                "Analisis rekaman suara",
                type="primary",
                use_container_width=True,
                key="analyze_recording",
            ):
                filename = getattr(
                    recorded_audio,
                    "name",
                    "rekaman_ghunnah.wav",
                )

                run_analysis(
                    cnn_model,
                    feedback_texts,
                    recorded_bytes,
                    filename,
                )


st.divider()

st.caption(
    "CNN menentukan hasil bacaan. Qwen hanya menyusun ulang "
    "kalimat feedback dan tidak mengubah keputusan CNN."
)
