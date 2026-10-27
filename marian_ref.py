"""Evaluate the pretrained MarianMT English-Esperanto reference model.

Runs in its own process so that its memory footprint does not collide with the
four small models held by the notebook. Reads data/test_srcs.json and writes
data/marian.json.
"""
import json, time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

d = json.load(open('data/test_srcs.json', encoding='utf-8'))
srcs, refs = d['srcs'], d['refs']

tok = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-eo')
model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-eo')
model.eval()

t0 = time.time()
hyps = []
for i in range(0, len(srcs), 16):
    batch = tok(srcs[i:i+16], return_tensors='pt', padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=28)
    hyps += [tok.decode(x, skip_special_tokens=True) for x in out]

res = {'hyps': hyps,
       'bleu': sacrebleu.corpus_bleu(hyps, [refs]).score,
       'chrf': sacrebleu.corpus_chrf(hyps, [refs]).score,
       'infer_s': time.time() - t0,
       'params': sum(p.numel() for p in model.parameters())}
json.dump(res, open('data/marian.json', 'w'), ensure_ascii=False)
print('MarianMT BLEU %.2f  chrF %.2f  params %.1fM  infer %.1fs'
      % (res['bleu'], res['chrf'], res['params']/1e6, res['infer_s']))