"""
pakai_mini_gpt.py — memakai model hasil training (mini_gpt.pt) tanpa melatih ulang.

Cara pakai:
    python pakai_mini_gpt.py              # model dilatih dengan teks contoh bawaan
    python pakai_mini_gpt.py data.txt     # model dilatih dengan data.txt (harus file yang sama!)

Ketik awalan kalimat, model akan melanjutkannya. Ketik 'keluar' untuk berhenti.
"""
import torch

# import dari mini_gpt.py: kode training tidak ikut jalan karena ada di bawah
# `if __name__ == "__main__"`, tapi tokenizer dibangun ulang dari teks yang sama
from mini_gpt import MiniGPT, encode, decode, stoi, DEVICE

model = MiniGPT().to(DEVICE)
model.load_state_dict(torch.load("mini_gpt.pt", map_location=DEVICE))
model.eval()  # matikan dropout
print("Model dimuat. Ketik awalan kalimat ('keluar' untuk berhenti).\n")

while True:
    prompt = input("prompt> ").lstrip("﻿")  # buang BOM dari input PowerShell
    if prompt.strip().lower() == "keluar":
        break
    unknown = sorted(set(c for c in prompt if c not in stoi))
    if unknown:
        print(f"Karakter tidak dikenal model: {unknown}\n")
        continue
    if not prompt:
        prompt = "\n"
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, 200, temperature=0.8, top_k=10)
    print(decode(out[0].tolist()), "\n")
