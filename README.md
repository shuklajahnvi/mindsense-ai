# MindSense AI

**An Explainable AI and Generative AI Framework for Mental Well-being Analysis**

MindSense AI is a *non-clinical* journaling and self-reflection platform. It analyses the language of journal
entries for emotion-related and stress-related patterns, explains individual predictions with LIME, tracks
patterns over time, and offers reflective guidance grounded in a curated knowledge base via Retrieval-Augmented
Generation (RAG).

> **Not a medical device.** MindSense AI does not diagnose depression, anxiety, PTSD or any other condition,
> does not recommend medication, and cannot respond to emergencies. Model outputs describe patterns in
> language, not a person's mental health.

---

## Architecture

```
frontend/  React + Vite + TypeScript + Tailwind + Recharts + Lucide
   └─ /api (Vite proxy) ─────────────────────────────┐
backend/                                             ▼
  app/            FastAPI + Pydantic + SQLAlchemy (SQLite)
    api/          entries · insights · chat (RAG) · system/models · demo
    services/
      classifiers.py   loads trained models (transformer → baseline fallback)
      explain.py       LIME explanations (emotion + stress)
      safety.py        risk-phrase screen, crisis resources, LLM output guardrail
      trends.py        daily / weekly / monthly longitudinal analysis
      topics.py        recurring themes: sentence embeddings + K-Means + c-TF-IDF
      embeddings.py    Sentence-Transformers (all-MiniLM-L6-v2)
      rag.py           chunk → embed → ChromaDB → retrieve
      llm.py           provider-independent LLM interface (OpenAI / Gemini / none)
      assistant.py     grounded chat + personalised entry reflections
  ml/             research pipeline (data, EDA, baselines, transformer fine-tuning, metrics)
  knowledge_base/ curated wellbeing documents with source URLs (NHS, WHO, NIMH, CDC, APA, Mind, crisis lines)
  artifacts/      models/ (trained weights + meta.json)  reports/ (metrics JSON, EDA figures)  logs/
  tests/          unit + API tests
```

## Quick start (Windows, PowerShell)

Prerequisites: Python 3.10+ (3.12 used), Node 18+.

```powershell
# 1. Backend environment
cd backend
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # or CUDA build
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env        # optional: add OPENAI_API_KEY or GEMINI_API_KEY

# 2. Data + models (see "Research pipeline" below for options)
.venv\Scripts\python -m ml.eda
.venv\Scripts\python -m ml.train_baselines
.venv\Scripts\python -m ml.train_transformer --task stress
.venv\Scripts\python -m ml.train_transformer --task emotion --epochs 2 --evals-per-epoch 2

# 3. Run the API
.venv\Scripts\python -m uvicorn app.main:app --port 8000

# 4. Run the frontend (new terminal)
cd frontend
npm install
npm run dev          # http://localhost:3004
```

Or start both with `.\run.ps1` from the repository root. API docs are at http://localhost:8000/docs.

On first start the backend downloads the sentence-embedding model and builds the ChromaDB index (a few seconds).
Use **History → Load demo entries** to populate 29 synthetic entries (hand-written texts of a fictional student
over six weeks) so the Insights page has data. Their *predictions are real model outputs*.

## Research pipeline (`backend/ml`)

| Step | Command | Output |
|---|---|---|
| Download + EDA | `python -m ml.eda` | `artifacts/reports/{emotion,stress}/eda.json`, PNG figures |
| Baselines | `python -m ml.train_baselines [--task emotion\|stress] [--train-limit N]` | `artifacts/models/{emotion_baseline,stress_baseline_lr,stress_baseline_svm}` |
| Transformer fine-tuning | `python -m ml.train_transformer --task {emotion,stress} [--epochs] [--batch-size] [--grad-accum] [--max-len] [--lr] [--train-limit N] [--eval-limit N] [--patience] [--evals-per-epoch] [--model CHECKPOINT]` | `artifacts/models/{task}_transformer` |

Every trained model writes `meta.json` (algorithm, hyper-parameters, data sizes, whether it was fine-tuned,
whether a dev subset was used, validation + test metrics, training history). The **Models & evaluation**
page renders these files directly; no metric in the UI is hard-coded.

### Emotion — GoEmotions (Demszky et al., ACL 2020)

* `google-research-datasets/go_emotions`, *simplified* config, official split (43,410 / 5,426 / 5,427).
  27 emotions + neutral, **multi-label** (mean 1.18 labels per comment).
* Baseline: TF-IDF (1–2-grams) + One-vs-Rest Logistic Regression (`class_weight=balanced`), C chosen on validation.
* Transformer: `distilroberta-base` fine-tuned with **28 sigmoid outputs and `BCEWithLogitsLoss`**
  (not softmax). Length-bucketed dynamic padding, linear warm-up/decay, gradient clipping, early stopping on
  validation macro-F1.
* **Per-label decision thresholds** are tuned on validation, then frozen for the single test evaluation.
* Metrics: micro/macro/weighted/sample F1, micro/macro precision & recall, subset accuracy, Hamming loss,
  macro ROC-AUC, per-class P/R/F1/support, per-class TP/FP/FN/TN, and the most frequent gold→wrong-label
  confusions.

### Stress — Dreaddit (Turcan & McKeown, 2019)

* `andreagasparini/dreaddit` mirror with the authors' original train/test split (labelled subset, ~3.5k segments).
* **Leakage control:** posts present in both train and test and exact duplicate texts are removed from train;
  validation is carved from train with `StratifiedGroupKFold` grouped by `post_id` (376 training posts have
  multiple segments, so a naive split would leak). The test set is left untouched for comparability.
* Baselines: TF-IDF + Logistic Regression; TF-IDF + linear SVM (sigmoid-calibrated for probabilities).
* Transformer: `distilroberta-base`, 2-way head with cross-entropy, early stopping on validation F1.
* Metrics: accuracy, precision, recall, F1, macro-F1, ROC-AUC, confusion matrix.

### Hardware note

The development machine has no CUDA GPU, so all training ran on CPU (Intel i5-1340P, ~7 s per optimiser step
for distilroberta at batch 16 × 128 tokens). The scripts use CUDA automatically if available. CPU budget options:

* `--train-limit N` / `--eval-limit N` — development subsets (the test set is never subsampled). A model trained on
  a subset is flagged `is_subset: true` and labelled "dev subset" in the UI.
* `--freeze-embeddings --freeze-layers K` — partial fine-tuning (top 6−K layers + head). Recorded in
  `meta.json → fine_tuning_scope`.
* `--threads`, `--batch-size`, `--grad-accum`, `--max-len`, `--patience`, `--evals-per-epoch`.

Commands used for the shipped models (CPU):

```powershell
python -m ml.train_transformer --task stress  --epochs 3 --patience 2 --threads 6 --freeze-embeddings --freeze-layers 2
python -m ml.train_transformer --task emotion --epochs 2 --evals-per-epoch 2 --patience 2 --train-limit 20000 --threads 6 --freeze-embeddings --freeze-layers 2 --head-lr 5e-4
```

For the multi-label emotion head, the output bias is initialised to each label's prior log-odds and the
(randomly initialised) head uses a higher learning rate (`--head-lr`). Without this, a short CPU run stays in
the "predict every label negative" regime for a long time (validation macro-F1 was 0.10 after 250 steps).

Full research run for GoEmotions (all 43,410 training examples, full fine-tuning — recommended on a GPU):

```powershell
python -m ml.train_transformer --task emotion --epochs 3 --evals-per-epoch 2
```

**Long entries at inference.** Journal entries are often longer than the training texts (GoEmotions comments are
truncated to 64 tokens). Rather than truncating, the transformer service packs whole sentences into windows that
fit `max_len` and aggregates per label: *max* across windows for emotions (an emotion expressed anywhere counts),
*mean* for stress.

### Results (test sets, measured on this machine — see `artifacts/reports/*/*_metrics.json`)

Thresholds were tuned on validation only; each test split was evaluated once.

**GoEmotions (5,427 test comments, 28 labels)**

| Model | Train data | Micro-F1 | Macro-F1 | Micro-P | Micro-R | Macro ROC-AUC |
|---|---|---|---|---|---|---|
| TF-IDF + One-vs-Rest LR | full 43,410 | 0.533 | 0.458 | 0.458 | 0.638 | 0.895 |
| distilroberta-base, partial fine-tune (top 4/6 layers), 2 epochs | **dev subset 20,000** | **0.590** | **0.498** | 0.521 | 0.679 | 0.933 |

The transformer beats the baseline on 25 of 28 labels. Best labels: gratitude (F1 0.92), amusement (0.83),
love (0.79); weakest: grief (0.14, 6 test examples), realization (0.18), relief (0.22), nervousness (0.29).
For reference, the GoEmotions paper reports macro-F1 0.46 for full BERT-base fine-tuning.

**Dreaddit (715 test segments)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.722 | 0.690 | 0.837 | 0.756 | 0.823 |
| TF-IDF + Linear SVM (calibrated) | 0.736 | 0.713 | 0.816 | 0.761 | 0.821 |
| distilroberta-base, partial fine-tune (top 4/6 layers), best of 3 epochs | **0.807** | **0.797** | **0.840** | **0.818** | **0.886** |

These scores measure agreement with crowd annotations of Reddit text, not the ability to detect anyone's emotional
or mental state.

> Windows tip: if training slows down sharply in a hidden/background window, Windows 11 may have put the process in
> Efficiency mode. Raising its priority (`(Get-Process -Id <pid>).PriorityClass = 'AboveNormal'`) or running it in a
> foreground terminal restores full speed.

Trained weights are git-ignored (`artifacts/models/*/model.safetensors`, `*.joblib`); `meta.json` and reports are kept.

## Explainable AI

LIME (`LimeTextExplainer`) explains the top predicted emotion labels and the stress prediction for each entry,
for whichever model variant is selected (transformer or baseline — LIME only needs `predict_proba`, so it works
with both). For multi-label emotion, each label's independent sigmoid score is explained. The UI highlights words
that pushed the score up (orange) or down (blue), shows weights, the local surrogate's R², and which
model/label the explanation belongs to. Wording: *"These words contributed to the model's prediction."*
Explanations never claim to establish a psychological state or causality.

## Longitudinal analysis

Each entry stores text, timestamp, optional mood / energy / sleep check-ins, all 28 emotion scores, the stress
prediction, model provenance, LIME output, the safety screen result and its sentence embedding.
`/api/insights` provides daily series, weekly and monthly summaries with descriptive comparisons
("Stress-related language has increased across your recent entries."), emotion distribution, GoEmotions sentiment
groups, and a descriptive sleep-vs-stress-language correlation when enough data exists. Score changes are never
framed as clinical improvement or deterioration.

Recurring themes (`/api/insights/topics`) use Sentence-Transformer embeddings, K-Means with *k* chosen by cosine
silhouette, and BERTopic-style class-based TF-IDF keywords. Full BERTopic (UMAP + HDBSCAN) was not used because it
is unstable on the tens-of-documents corpora a single journaling user produces.

## RAG and generative AI

1. `knowledge_base/*.md` — 7 curated documents (stress management, sleep hygiene, emotional wellbeing,
   self-reflection/journaling, healthy routines, academic stress, professional support & crisis lines), each
   paraphrased from and citing reputable sources (NHS, WHO, NIMH, CDC, APA, Mind, 988 Lifeline, Samaritans,
   Tele-MANAS, Find A Helpline).
2. Heading-aware chunking (~170 words, 30-word overlap) → `all-MiniLM-L6-v2` embeddings → persistent **ChromaDB**
   collection (cosine). The index is rebuilt automatically when the documents change.
3. Top-k retrieval with a similarity floor → numbered context → LLM with a non-clinical system prompt requiring
   inline citations → output guardrail that strips diagnostic or medication statements.
4. Optional personalisation: a short, non-clinical summary of the user's recent language patterns.

**Providers:** set `OPENAI_API_KEY` (any OpenAI-compatible endpoint via `OPENAI_BASE_URL`) or `GEMINI_API_KEY`
in `backend/.env`; `LLM_PROVIDER=auto|openai|gemini|none`. Without a key the assistant runs in an explicitly
labelled **retrieval-only mode** (retrieved passages and fixed reflection prompts). Nothing is presented as
generated text unless an LLM produced it.

## Safety

* Rule-based risk-phrase screen on every entry and chat message (self-harm/suicide, harm to others,
  abuse/danger, hopelessness), with negation handling. High-risk messages bypass generation and return a fixed
  supportive message plus crisis resources (US/Canada 988, Samaritans 116 123, Tele-MANAS 14416, Lifeline
  13 11 14, Find A Helpline, emergency numbers). This is a conservative screen, **not** a clinical risk assessment.
* LLM output guardrail removes diagnostic claims and medication advice.
* Persistent non-clinical notice and a "Get support" drawer on every page.

## Tests

```powershell
cd backend
.venv\Scripts\python -m pytest tests -q
cd ..\frontend
npm run build
```

## Limitations

* Both datasets are English Reddit text; journaling differs in register, topic and demographics, so domain
  shift is expected. Annotator agreement in GoEmotions is moderate, which caps achievable scores.
* LIME explanations are local approximations and can vary with the random perturbation sample.
* The safety screen is keyword-based and will miss some risk language and flag some benign text.
* Single local profile; no authentication — intended for local, personal or academic use.

## References

* Demszky, D. et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* ACL. https://aclanthology.org/2020.acl-main.372/
* Turcan, E. & McKeown, K. (2019). *Dreaddit: A Reddit Dataset for Stress Analysis in Social Media.* LOUHI. https://aclanthology.org/D19-6213/
* Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD.
* Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794.
* Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
