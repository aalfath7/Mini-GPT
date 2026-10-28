"""
mini_gpt.py — LLM kecil (Transformer gaya GPT) untuk belajar.

Cara pakai:
    pip install torch
    python mini_gpt.py                 # latih dengan teks contoh bawaan
    python mini_gpt.py data.txt        # latih dengan teks milikmu sendiri

Tokenizer-nya tingkat karakter (1 huruf = 1 token) supaya mudah dipahami.
LLM sungguhan memakai tokenizer subword (BPE), model jauh lebih besar,
dan data miliaran token — tapi arsitekturnya pada dasarnya sama.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- hyperparameter
BATCH = 32          # berapa potongan teks dilatih sekaligus
CONTEXT = 64        # panjang konteks (berapa token yang "diingat" model)
D_MODEL = 128       # ukuran vektor embedding
N_HEADS = 4         # jumlah attention head
N_LAYERS = 4        # jumlah blok Transformer
DROPOUT = 0.1
STEPS = 3000
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ---------------------------------------------------------------- 1. data
SAMPLE_TEXT = """
Di sebuah desa kecil di kaki gunung, hiduplah seorang anak bernama Raka.
Setiap pagi Raka pergi ke sawah bersama ayahnya. Mereka menanam padi,
menyiram sayur, dan memberi makan bebek. Raka suka bertanya tentang banyak hal.
Mengapa langit berwarna biru? Mengapa hujan turun dari awan? Ayahnya selalu
menjawab dengan sabar. Suatu hari Raka ingin membuat mesin yang bisa menulis
cerita sendiri. Ia belajar membaca, belajar menghitung, dan belajar menulis.
Ia tahu bahwa mesin itu harus membaca banyak sekali cerita terlebih dahulu,
lalu menebak huruf berikutnya, satu demi satu, sampai tebakannya menjadi pintar.
""" * 30  # diulang supaya ada cukup data untuk contoh

text = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else SAMPLE_TEXT

# ---------------------------------------------------------------- 2. tokenizer
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


def get_batch(source):
    # ambil potongan acak; target = input digeser 1 token (tebak token berikutnya)
    ix = torch.randint(len(source) - CONTEXT - 1, (BATCH,))
    x = torch.stack([source[i : i + CONTEXT] for i in ix])
    y = torch.stack([source[i + 1 : i + CONTEXT + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ---------------------------------------------------------------- 3. model
class SelfAttention(nn.Module):
    """Multi-head causal self-attention: tiap token 'melihat' token sebelumnya."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        # mask segitiga: token ke-t tidak boleh mengintip token masa depan
        self.register_buffer("mask", torch.tril(torch.ones(CONTEXT, CONTEXT)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(D_MODEL, dim=2)
        # pecah jadi beberapa head: (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)   # skor kemiripan
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))                  # jadi bobot 0..1
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(out))


class Block(nn.Module):
    """Satu blok Transformer: attention + feed-forward, dengan residual & LayerNorm."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = SelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL),
            nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)    # token -> vektor
        self.pos_emb = nn.Embedding(CONTEXT, D_MODEL)  # posisi -> vektor
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)          # vektor -> skor tiap token

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -CONTEXT:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)   # ambil sampel token berikutnya
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ---------------------------------------------------------------- 4. training
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for name, src in (("train", train_data), ("val", val_data)):
        out[name] = sum(model(*get_batch(src))[1].item() for _ in range(20)) / 20
    model.train()
    return out


if __name__ == "__main__":
    model = MiniGPT().to(DEVICE)
    print(f"device={DEVICE}  vocab={VOCAB}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for step in range(STEPS + 1):
        if step % 500 == 0:
            l = estimate_loss(model)
            print(f"step {step:5d} | train loss {l['train']:.3f} | val loss {l['val']:.3f}")
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save(model.state_dict(), "mini_gpt.pt")

    # ---------------------------------------------------------------- 5. generate
    prompt = "Raka "
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    print("\n--- hasil generate ---")
    print(decode(model.generate(idx, 300, temperature=0.8, top_k=10)[0].tolist()))
