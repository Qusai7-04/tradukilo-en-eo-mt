"""Train one machine-translation model and save its checkpoint.

Usage:  python train_all.py "Basic Transformer"
        python train_all.py "mT5-style"
        python train_all.py "LLaMA-style"
        python train_all.py "Gemma-style"
"""
import sys, os, json
import torch
from mt_core import (download, load_pairs, keep, build_tokenizer, train_model, safe, EPOCHS)

name = sys.argv[1]
os.makedirs('models', exist_ok=True)

download()
train_raw, val_raw, test_raw = load_pairs()
train = [r for r in train_raw if keep(r)]
print('training pairs:', len(train))

import sentencepiece as spm
if os.path.exists('data/spm.model'):
    sp = spm.SentencePieceProcessor(model_file='data/spm.model')
else:
    sp = build_tokenizer(train)
V = sp.get_piece_size()
print('vocab:', V)

torch.manual_seed(42)
model, kind, t, curve = train_model(name, sp, V, train, epochs=EPOCHS)
torch.save(model.state_dict(), f'models/{safe(name)}.pt')

LOG = 'models/train_log.json'
log = json.load(open(LOG)) if os.path.exists(LOG) else {}
log[name] = {'train_s': t, 'loss_curve': curve, 'params': sum(p.numel() for p in model.parameters())}
json.dump(log, open(LOG, 'w'), indent=2)
print(f'DONE {name}: {t:.0f}s, final loss {curve[-1]:.4f}')