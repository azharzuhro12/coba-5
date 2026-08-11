# GhunnahSense — Streamlit dari notebook (13)

Paket ini dibuat mengikuti parameter inference pada
`skripsi-cnn-slm-eval-fixed (13).ipynb`.

## Konfigurasi CNN yang dipakai

- Sample rate: 16 kHz
- Silence trimming: `top_db=30`
- Log-Mel: 64 Mel bins
- FFT: 1024
- Hop length: 256
- Durasi referensi: 8 detik
- Input CNN: `(64, 501, 1)`
- Dropout utama: 0.30
- Output: sigmoid
- Nilai sigmoid dibaca sebagai `P(SALAH)`
- Threshold klasifikasi tetap: 0.50

Preprocessing mengikuti notebook:
audio -> trim silence -> Log-Mel penuh -> z-score penuh ->
ambil 501 frame pertama / right-padding.

## Best model

Notebook (13) menyimpan best model ke:

```text
/kaggle/working/best_cnn.keras
```

Download file tersebut dari Kaggle lalu taruh di root repository GitHub.

Aplikasi juga mendukung nama:

```text
best_cnn.h5
best_cnn.keras
best_cnn.weights.h5
```

Untuk `.h5`, aplikasi mencoba memuat sebagai full Keras model terlebih
dahulu. Jika itu weights-only, aplikasi membangun arsitektur CNN notebook
terlebih dahulu lalu menjalankan `load_weights()`.

## Feedback

Jika `metadata.csv` tersedia di root repository dan memiliki kolom:

```text
label,error_explanation
```

aplikasi akan menggunakannya.

Jika tidak ada, aplikasi menggunakan `feedback_kb.csv` bawaan sebagai
fallback sederhana.

## Streamlit Community Cloud

Upload isi folder ini ke repository GitHub.

Main file:

```text
app.py
```

Gunakan Python 3.11.

Advanced settings / Secrets:

```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
HF_PROVIDER = "auto"
```

`HF_TOKEN` diperlukan hanya untuk feedback Qwen melalui Hugging Face.
Jika token belum diisi, CNN tetap dapat dipakai dan aplikasi menampilkan
feedback dasar.

## Tampilan aplikasi

Sesuai permintaan:

- label hasil BENAR tidak ditampilkan;
- probabilitas tidak ditampilkan;
- hasil SALAH tetap ditampilkan;
- jika audio hening, tampil `Suara tidak terdengar.`;
- audio hening tidak diteruskan ke CNN/Qwen;
- upload menerima berbagai format dan `ffmpeg` disertakan melalui
  `packages.txt`.


## Model sudah disertakan

Paket ini sudah berisi:

```text
best_cnn.keras
```

Model tervalidasi sebagai full Keras archive.

- Keras version: 3.13.2
- Input: `(64, 501, 1)`
- Conv2D filters: `16 -> 32 -> 64`
- Dropout utama: `0.30`
- Output: `Dense(1, activation="sigmoid")`
- Sigmoid digunakan sebagai `P(SALAH)` sesuai notebook (13)
- Threshold klasifikasi aplikasi: `0.50`
