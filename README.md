# Belajar Python — Mini GPT

Proyek belajar membuat LLM kecil (Transformer gaya GPT) dari nol dengan PyTorch.
Modelnya kecil dan tokenizer-nya tingkat karakter (1 huruf = 1 token) supaya mudah dipahami,
tapi arsitekturnya pada dasarnya sama dengan LLM sungguhan.

## Isi

| File | Keterangan |
| --- | --- |
| [mini_gpt.py](mini_gpt.py) | Definisi model, tokenizer, training, dan contoh generate teks |
| [pakai_mini_gpt.py](pakai_mini_gpt.py) | Memakai model yang sudah dilatih secara interaktif (tanpa melatih ulang) |
| `mini_gpt.pt` | Bobot model hasil training (dibuat oleh `mini_gpt.py`) |

## Persiapan

```bash
pip install torch
```

Jika ada GPU dengan CUDA, model otomatis memakainya; jika tidak, berjalan di CPU.

## Cara Pakai

### 1. Melatih model

```bash
python mini_gpt.py              # latih dengan teks contoh bawaan (cerita tentang Raka)
python mini_gpt.py data.txt     # latih dengan teks milikmu sendiri
```

Selama training, loss train dan validasi dicetak setiap 500 langkah. Setelah selesai,
bobot model disimpan ke `mini_gpt.pt` dan contoh teks hasil generate ditampilkan.

### 2. Memakai model

```bash
python pakai_mini_gpt.py            # jika model dilatih dengan teks bawaan
python pakai_mini_gpt.py data.txt   # jika model dilatih dengan data.txt
```

Ketik awalan kalimat, model akan melanjutkannya. Ketik `keluar` untuk berhenti.

> **Penting:** gunakan file teks yang sama dengan saat training. Tokenizer dibangun ulang
> dari teks tersebut, jadi teks yang berbeda akan menghasilkan kosakata yang tidak cocok
> dengan bobot model. Karakter yang tidak ada di data training juga tidak bisa dipakai di prompt.

## Arsitektur

1. **Data & tokenizer** — teks dipecah per karakter, 90% untuk training dan 10% untuk validasi.
2. **Embedding** — token dan posisinya diubah menjadi vektor.
3. **Blok Transformer** (×4) — masing-masing berisi:
   - *Causal multi-head self-attention*: tiap token hanya bisa "melihat" token sebelumnya.
   - *Feed-forward* dengan aktivasi GELU.
   - *Residual connection* dan *LayerNorm*.
4. **Output head** — vektor diubah menjadi skor untuk tiap token di kosakata.
5. **Generate** — model menebak token berikutnya satu per satu, dengan `temperature` dan `top_k`
   untuk mengatur variasi hasil.

## Hyperparameter

Bisa diubah di bagian atas [mini_gpt.py](mini_gpt.py):

| Nama | Nilai | Arti |
| --- | --- | --- |
| `BATCH` | 32 | Jumlah potongan teks yang dilatih sekaligus |
| `CONTEXT` | 64 | Panjang konteks (token yang "diingat" model) |
| `D_MODEL` | 128 | Ukuran vektor embedding |
| `N_HEADS` | 4 | Jumlah attention head |
| `N_LAYERS` | 4 | Jumlah blok Transformer |
| `DROPOUT` | 0.1 | Peluang dropout |
| `STEPS` | 3000 | Jumlah langkah training |
| `LR` | 3e-4 | Learning rate (optimizer AdamW) |

Jika hyperparameter model diubah, latih ulang dulu sebelum menjalankan `pakai_mini_gpt.py`.
