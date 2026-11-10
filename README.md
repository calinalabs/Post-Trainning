# Calina Labs: post-training an open embedding model

Fine-tunes a small open-weight embedding model on synthetic questions, on a laptop, and measures whether it finds the right documentation passages more often. Published by [Calina Labs](https://calinalabs.com), an independent UK consultancy that builds agentic processes, private knowledge bases and process automation.

- Write-up: [Fine-tuning a small open embedding model on a laptop](https://calinalabs.com/blog/fine-tune-embedding-model/)
- Related experiment (same data, Gemini Embedding 2, re-ranking, BM25): [calinalabs/rag](https://github.com/calinalabs/rag) · [write-up](https://calinalabs.com/blog/gemini-embedding-rag/)
- Website: [calinalabs.com](https://calinalabs.com) · [Contact](https://calinalabs.com/contact/)

## What it does

1. `gen_questions.py` asks Claude Haiku 5.5 for 3 questions per passage: 693 training questions for 232 passages. The 100 test questions are never read.
2. `finetune.py` fine-tunes [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) (33M parameters) with sentence-transformers, using a contrastive loss (cached multiple-negatives ranking, batch 32, 3 epochs). Then it scores the base model and the fine-tuned model on the same 100 human-written test questions.
3. Checks that the gain is real:
   - **Leakage:** how close any training question is to any test question. There are no exact matches, and 15 test questions have a training question with similarity above 0.9. The result holds when those 15 are dropped.
   - **Held-out passages:** a second model is trained on questions for only half of the passages, then scored separately on test questions about seen and unseen passages.

## Results

Top 3 passages per question, 100 test questions.

| Model | Recall@3 | MRR |
|---|---|---|
| BM25 keyword search | 0.60 | 0.75 |
| bge-small, untuned | 0.62 | 0.73 |
| **bge-small, fine-tuned** | **0.71** | **0.87** |
| Gemini Embedding 2 (for reference) | 0.74 | 0.90 |

| Check | Untuned → fine-tuned (recall / MRR) |
|---|---|
| Without the 15 near-duplicate test questions (n=85) | 0.61 → 0.70 / 0.74 → 0.87 |
| Half-trained model, passages it saw (n=30) | 0.62 → 0.75 / 0.69 → 0.79 |
| Half-trained model, passages it never saw (n=29) | 0.64 → 0.79 / 0.72 → 0.93 |

Most of the gain carries over to passages the model never trained on. It learned how users phrase questions about this kind of documentation, not just answers to specific questions. The held-out groups are small (about 30 questions each), so treat those numbers as indicative. Every number is in `results/finetune_results.json`.

## Run it

```bash
git clone https://github.com/calinalabs/Post-Trainning && cd Post-Trainning
./get_data.sh                                   # public evaluation set (MIT licence) into ./data
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...                    # only needed to regenerate the questions
python3 gen_questions.py --data data --out data/train_questions.json   # optional: the questions we used are committed
.venv/bin/python finetune.py --data data --train data/train_questions.json --out results
```

`gen_questions.py` uses only the Python standard library. Training takes about 15 minutes for both runs on an Apple M3 with 16 GB. It uses the Mac GPU (MPS) when available and falls back to the CPU otherwise. The fine-tuned model is saved to `models/bge-small-finetuned`.

Notes for Apple silicon:
- Attention is set to `eager`, because the fused attention kernel on MPS has no dropout support in training mode.
- The cached loss computes each batch of 32 in chunks of 8, so it fits in 16 GB with the same result as a plain batch of 32.

---

Built by [Calina Labs](https://calinalabs.com). Want a retrieval model tuned to your own documents? [Get in touch](https://calinalabs.com/contact/).
