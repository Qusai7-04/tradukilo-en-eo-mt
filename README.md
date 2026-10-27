# Tradukilo: English-to-Esperanto Machine Translation

**Foundations of Generative AI (ICT4403), Assignment 3**  
Dataset: [OPUS-100, en-eo (Helsinki-NLP)](https://huggingface.co/datasets/Helsinki-NLP/opus-100)

Tradukilo is a resource-constrained comparison of encoder-decoder and decoder-only machine-translation architectures. It includes a basic Transformer, compact T5/LLaMA/Gemma-style architectures trained from scratch, and a pretrained MarianMT reference model. The notebook records BLEU, chrF, parameter count, training time, inference time, example translations, decoding-temperature analysis, and responsible-AI observations.

> **Important model distinction:** the four headline experimental models in MTSystems_en_eo.ipynb are compact architecture reproductions trained from scratch. They are not pretrained mT5, LLaMA, or Gemma checkpoints, and the reported table must not be read as a benchmark of those pretrained models. Because the original runtime had a very small memory budget and this packaging environment cannot download packages or checkpoints, the current saved scores are for the architecture-only experiment. An additional pretrained_adaptation.py path is included to run genuine pretrained mT5 fine-tuning and prompt-only TinyLlama/Gemma evaluation in a network-enabled environment with adequate memory. Those additional model runs have not been executed here, so no results are claimed for them.

## Dataset and preprocessing

The public OPUS-100 en-eo configuration contains English–Esperanto parallel text. The experiment downloads the train, validation, and test parquet files directly from the dataset repository and filters out empty pairs, sentences longer than 10 whitespace-delimited words on either side, and pairs with a source/target length ratio above 2.0.

The fixed subset settings are 8,000 raw training rows, 500 validation rows, and 100 test rows before filtering. With seed 42, the saved run contains:

| Split | Pairs after filtering |
|---|---:|
| Training | 6,496 |
| Validation | 403 |
| Test | 82 |

The validation and test subsets are taken from the beginning of their respective source split files, not randomly sampled across the full split. This makes the current result reproducible but may introduce sample-selection bias; the 82-sentence test set is also small. Results should therefore be interpreted as a small experiment, not a robust leaderboard estimate.

A shared SentencePiece BPE vocabulary of 2,000 pieces is trained on the filtered training pairs. Each compact architecture uses a maximum sequence length of 14 tokens, batch size 64, Adam with learning rate 0.001, five epochs, and a fixed seed of 42.

## Reported baseline results

These are the metrics saved in results.json and reproduced in the notebook. BLEU and chrF are corpus-level SacreBLEU scores on the same 82 filtered examples. The MarianMT model is a pretrained reference, not one of the four trained-from-scratch variants.

| System | BLEU | chrF | Parameters | Training time | Inference time for 82 examples |
|---|---:|---:|---:|---:|---:|
| Basic Transformer | 0.71 | 10.93 | 0.343M | 118 s | 1.8 s |
| T5 / mT5-style (scratch) | 0.60 | 4.12 | 0.260M | 144 s | 0.3 s |
| LLaMA-style (scratch) | 1.22 | 11.37 | 0.297M | 141 s | 3.5 s |
| Gemma-style (scratch) | 0.56 | 7.61 | 0.169M | 157 s | 3.8 s |
| MarianMT (pretrained reference) | 52.79 | 71.58 | 74.972M | n/a | 34.6 s |

The four compact scratch models do not produce reliable translations: they tend to emit fragments, repetitive strings, or text that is semantically unrelated to the English source. LLaMA-style has the highest BLEU and chrF of the four scratch models, but 1.22 BLEU is still extremely poor translation quality. The pretrained MarianMT reference performs much better on this subset. Since model size, pretraining, tokenization, training data and adaptation differ, these values do not isolate architecture as a causal factor.

### Decoding-temperature study

For the Basic Transformer, the saved run compares greedy decoding, temperature sampling at 0.7, 1.0 and 1.3, and top-k sampling at k=20:

| Decoding | BLEU | chrF | distinct-2 |
|---|---:|---:|---:|
| Greedy | 0.71 | 10.93 | 0.973 |
| T = 0.7 | 1.10 | 12.24 | 0.998 |
| T = 1.0 | 0.56 | 11.48 | 1.000 |
| T = 1.3 | 0.23 | 11.73 | 1.000 |
| T = 1.0, top-k = 20 | 0.94 | 11.12 | 1.000 |

These sampling values are from a single run. Higher distinct-2 shows more varied bigrams, not necessarily more accurate translations; here quality remains poor at every setting.

## Run the baseline experiment

Use Python 3.10–3.12 if possible. A CPU is sufficient for the compact scratch models; internet access is required to download OPUS-100 and the MarianMT reference checkpoint.

1. Create and activate a virtual environment.
2. Install dependencies with python -m pip install -r requirements.txt.
3. Open MTSystems_en_eo.ipynb in Jupyter and run the cells from top to bottom.

To train checkpoints individually from the command line, run each of these commands:

- python train_all.py "Basic Transformer"
- python train_all.py "mT5-style"
- python train_all.py "LLaMA-style"
- python train_all.py "Gemma-style"

The scripts create data/, models/, and figures/ locally. Dataset files, tokenizer cache, and model checkpoints are intentionally not committed; the notebook or trainer downloads/builds them. Running each training command from scratch takes a few minutes per model on a modest CPU. The notebook will reuse locally available checkpoints.

## Run actual pretrained-family adaptation

pretrained_adaptation.py provides a separate path for the specific pretrained-model requirement:

- **mT5:** fine-tunes google/mt5-small on a bounded subset of the training split and evaluates it on the same filtered test subset.
- **LLaMA family:** prompts the pretrained TinyLlama/TinyLlama-1.1B-Chat-v1.0 checkpoint to translate; no weight fine-tuning is performed for this prompt-based run.
- **Gemma family:** prompts google/gemma-3-270m-it; no weight fine-tuning is performed for this prompt-based run.

Example commands:

- python pretrained_adaptation.py --mode mt5 --train-samples 512 --epochs 1
- python pretrained_adaptation.py --mode llama
- python pretrained_adaptation.py --mode gemma

Use a network-enabled machine with adequate RAM (a GPU runtime is recommended for mT5 fine-tuning). Google Gemma model access may require accepting its model terms on Hugging Face. These optional runs download real pretrained weights and are separate from the saved scratch-model scores above. The script writes predictions and metrics to pretrained_adaptation_results.json; those results must be generated and reviewed before claiming the pretrained-model requirement has been experimentally completed.

## Responsible and ethical AI

The qualitative outputs illustrate hallucination, omission, repetition collapse, and incorrect handling of names. Surface fluency does not guarantee semantic fidelity. A real translation service should preserve names where appropriate, display source and target text side by side, flag uncertain or malformed outputs, test culturally specific and sensitive examples, audit the dataset for harmful or private content, and require qualified human review for legal, medical, safety-critical, or other consequential translations.

## Repository structure

- MTSystems_en_eo.ipynb: executable baseline notebook.
- mt_core.py: dataset, tokenizer, architecture, training, and decoding utilities.
- train_all.py: standalone trainer for one compact architecture.
- marian_ref.py: evaluates the pretrained MarianMT reference.
- pretrained_adaptation.py: optional genuine pretrained-family fine-tuning/prompting runner.
- requirements.txt: Python dependencies.
- results.json: saved baseline metrics, decoding scores, and qualitative examples.
- models/train_log.json: training loss/time log from the supplied baseline run.
- figures/: generated charts are embedded in notebook outputs; re-run the notebook to export PNGs.

## Acknowledgements and references

- OPUS-100: [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
- Pretrained translation reference: [Helsinki-NLP/opus-mt-en-eo](https://huggingface.co/Helsinki-NLP/opus-mt-en-eo)
- mT5: [google/mt5-small](https://huggingface.co/google/mt5-small)
- TinyLlama: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Gemma: [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)

Student: Qusai  
Registration number: 230953480  
Class: FGAI-F
