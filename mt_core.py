import os, json, time, math, random, gc, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import sacrebleu

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

VOCAB_SIZE = 2000
MAXLEN     = 14
BATCH      = 64
LR         = 1e-3
EPOCHS     = 5
TRAIN_N    = 8000
MAX_WORDS  = 10
TEST_N     = 100
D_MODEL    = 64
N_HEAD     = 2
N_LAYER    = 1
D_FF       = 128
PAD, UNK, BOS, EOS = 0, 1, 2, 3

BASE_URL = 'https://huggingface.co/datasets/Helsinki-NLP/opus-100/resolve/main/en-eo'

def download():
    os.makedirs('data', exist_ok=True)
    for sp in ['train', 'validation', 'test']:
        p = f'data/en-eo_{sp}.parquet'
        if not os.path.exists(p):
            urllib.request.urlretrieve(f'{BASE_URL}/{sp}-00000-of-00001.parquet', p)
    return True

def load_pairs():
    import pyarrow as pa, pyarrow.parquet as pq
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    def subset(split, n, shuffle=True):
        t = pq.read_table(f'data/en-eo_{split}.parquet').flatten()
        idx = sorted(random.Random(SEED).sample(range(t.num_rows), min(n, t.num_rows))) if shuffle \
              else list(range(min(n, t.num_rows)))
        rows = []
        for r in t.take(pa.array(idx)).to_pylist():
            en = (r['translation.en'] or '').strip(); eo = (r['translation.eo'] or '').strip()
            if en and eo:
                rows.append({'en': en, 'eo': eo})
        return rows
    return subset('train', TRAIN_N), subset('validation', 500, False), subset('test', TEST_N, False)

def keep(r, max_words=MAX_WORDS, max_ratio=2.0):
    a, b = len(r['en'].split()), len(r['eo'].split())
    if a == 0 or b == 0 or a > max_words or b > max_words:
        return False
    return max(a, b) / min(a, b) <= max_ratio

def build_tokenizer(train):
    import sentencepiece as spm
    with open('data/spm_corpus.txt', 'w', encoding='utf-8') as f:
        for r in train:
            f.write(r['en'] + '\n'); f.write(r['eo'] + '\n')
    spm.SentencePieceTrainer.train(
        input='data/spm_corpus.txt', model_prefix='data/spm', vocab_size=VOCAB_SIZE,
        model_type='bpe', character_coverage=1.0, pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece='<pad>', unk_piece='<unk>', bos_piece='<s>', eos_piece='</s>', num_threads=1)
    return spm.SentencePieceProcessor(model_file='data/spm.model')

def make_encdec(sp, pairs):
    S = torch.zeros(len(pairs), MAXLEN, dtype=torch.long)
    T = torch.zeros(len(pairs), MAXLEN, dtype=torch.long)
    for i, r in enumerate(pairs):
        a = sp.encode(r['en'], out_type=int)[:MAXLEN-1]
        b = sp.encode(r['eo'], out_type=int)[:MAXLEN-2]
        S[i, 0] = BOS; S[i, 1:1+len(a)] = torch.tensor(a)
        T[i, 0] = BOS; T[i, 1:1+len(b)] = torch.tensor(b); T[i, 1+len(b)] = EOS
    return S, T

PROMPT = 'translate English to Esperanto: '
SEP = ' => '

def make_causal(sp, pairs):
    L2 = 2 * MAXLEN
    X = torch.zeros(len(pairs), L2, dtype=torch.long)
    Y = torch.full((len(pairs), L2), -100, dtype=torch.long)
    for i, r in enumerate(pairs):
        p = sp.encode(PROMPT + r['en'] + SEP, out_type=int)[:MAXLEN]
        t = sp.encode(r['eo'], out_type=int)[:MAXLEN-1]
        seq = (p + t + [EOS])[:L2]
        X[i, :len(seq)] = torch.tensor(seq)
        Y[i, len(p):len(seq)] = torch.tensor(seq[len(p):])
    return X, Y

from transformers import (T5Config, T5ForConditionalGeneration,
                          LlamaConfig, LlamaForCausalLM,
                          GemmaConfig, GemmaForCausalLM)

class BasicTransformer(nn.Module):
    def __init__(self, vocab, d=D_MODEL, heads=N_HEAD, ff=D_FF, maxlen=MAXLEN, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(maxlen, d)
        self.tr = nn.Transformer(d_model=d, nhead=heads, num_encoder_layers=N_LAYER,
                                 num_decoder_layers=N_LAYER, dim_feedforward=ff,
                                 dropout=drop, batch_first=True)
        self.out = nn.Linear(d, vocab)
    def forward(self, src, tgt):
        s = self.emb(src) + self.pos(torch.arange(src.size(1)).unsqueeze(0))
        t = self.emb(tgt) + self.pos(torch.arange(tgt.size(1)).unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
        h = self.tr(s, t, tgt_mask=mask, src_key_padding_mask=(src == PAD),
                    tgt_key_padding_mask=(tgt == PAD))
        return self.out(h)

def build_model(name, V):
    if name == 'Basic Transformer':
        return BasicTransformer(V), 'encdec'
    if name == 'mT5-style':
        cfg = T5Config(vocab_size=V, d_model=D_MODEL, d_ff=D_FF, num_layers=N_LAYER,
                       num_decoder_layers=N_LAYER, num_heads=N_HEAD,
                       decoder_start_token_id=PAD, pad_token_id=PAD, eos_token_id=EOS)
        return T5ForConditionalGeneration(cfg), 'hf_encdec'
    if name == 'LLaMA-style':
        cfg = LlamaConfig(vocab_size=V, hidden_size=D_MODEL, intermediate_size=D_FF,
                          num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
                          num_key_value_heads=N_HEAD, max_position_embeddings=4*MAXLEN,
                          pad_token_id=PAD)
        return LlamaForCausalLM(cfg), 'causal'
    if name == 'Gemma-style':
        cfg = GemmaConfig(vocab_size=V, hidden_size=D_MODEL, intermediate_size=D_FF,
                          num_hidden_layers=N_LAYER, num_attention_heads=N_HEAD,
                          num_key_value_heads=N_HEAD, head_dim=D_MODEL // N_HEAD,
                          max_position_embeddings=4*MAXLEN, pad_token_id=PAD)
        return GemmaForCausalLM(cfg), 'causal'
    raise ValueError(name)

def batch_loss(model, kind, src, tgt, V):
    if kind == 'encdec':
        out = model(src, tgt[:, :-1])
        return F.cross_entropy(out.reshape(-1, V), tgt[:, 1:].reshape(-1), ignore_index=PAD)
    if kind == 'hf_encdec':
        labels = tgt.clone(); labels[labels == PAD] = -100
        return model(input_ids=src, labels=labels).loss
    return model(input_ids=src, labels=tgt).loss

def prepare(sp, pairs, kind):
    return make_encdec(sp, pairs) if kind in ('encdec', 'hf_encdec') else make_causal(sp, pairs)

def train_model(name, sp, V, train, epochs=EPOCHS, bs=BATCH, lr=LR, verbose=True):
    torch.manual_seed(SEED)
    model, kind = build_model(name, V)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    t0 = time.time(); curve = []
    for ep in range(1, epochs + 1):
        data = train[:]
        random.Random(SEED + ep).shuffle(data)
        tot, nb = 0.0, 0
        for i in range(0, len(data), bs):
            src, tgt = prepare(sp, data[i:i+bs], kind)
            opt.zero_grad()
            loss = batch_loss(model, kind, src, tgt, V)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        curve.append(tot / nb)
        if verbose:
            print(f'  epoch {ep}/{epochs}  train loss {curve[-1]:.4f}')
    return model, kind, time.time() - t0, curve

def _src_tensor(sp, srcs):
    S = torch.zeros(len(srcs), MAXLEN, dtype=torch.long)
    for i, s in enumerate(srcs):
        a = sp.encode(s, out_type=int)[:MAXLEN-1]
        S[i, 0] = BOS; S[i, 1:1+len(a)] = torch.tensor(a)
    return S

def translate_greedy(model, kind, sp, srcs, V):
    model.eval()
    if kind == 'encdec':
        S = _src_tensor(sp, srcs)
        ys = torch.full((len(srcs), 1), BOS, dtype=torch.long)
        done = torch.zeros(len(srcs), dtype=torch.bool)
        with torch.no_grad():
            for _ in range(MAXLEN):
                nxt = model(S, ys)[:, -1].argmax(-1)
                ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
                done |= (nxt == EOS)
                if bool(done.all()):
                    break
        out = []
        for row in ys[:, 1:]:
            toks = []
            for t in row.tolist():
                if t == EOS: break
                toks.append(t)
            out.append(sp.decode(toks))
        return out
    if kind == 'hf_encdec':
        S = _src_tensor(sp, srcs)
        with torch.no_grad():
            gen = model.generate(input_ids=S, attention_mask=(S != PAD), max_new_tokens=MAXLEN,
                                 num_beams=1, do_sample=False, pad_token_id=PAD, eos_token_id=EOS)
        return [sp.decode([t for t in row.tolist() if t not in (PAD, EOS)]) for row in gen]
    out = []
    for s in srcs:
        p = sp.encode(PROMPT + s + SEP, out_type=int)[:MAXLEN]
        with torch.no_grad():
            gen = model.generate(input_ids=torch.tensor([p]), max_new_tokens=MAXLEN,
                                 do_sample=False, pad_token_id=PAD, eos_token_id=EOS)
        toks = []
        for t in gen[0, len(p):].tolist():
            if t == EOS: break
            toks.append(t)
        out.append(sp.decode(toks))
    return out

def safe(name):
    return name.lower().replace(' ', '_').replace('-', '_')