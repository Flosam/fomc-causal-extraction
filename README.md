# FOMC Causal Extraction

A research pipeline for evaluating how reliably LLMs extract causal knowledge from Federal Reserve meeting minutes, and whether failure rates vary systematically across linguistic complexity and economic regime.

**CPSC 532 Final Project**

---

## Research Question

How faithfully can an LLM extract causal triples from expert institutional text (FOMC minutes), and do failure rates and failure modes differ across:
- **Linguistic complexity** (simple explicit causality → dense hedged implicit causality)
- **Economic regime** (Great Moderation → Post-Crisis ZLB → Post-COVID Surge)

---

## Pipeline Overview

```
FOMC corpus (Kaggle)
       ↓
data_pipeline.py      — filter to 30 meetings across 3 economic periods
       ↓
preprocessor.py       — extract target sections, segment into passages (NLTK)
       ↓
extractor.py          — LLM extracts causal triples {cause, connector, effect, hedge, direction}
       ↓
judge.py              — LLM-as-judge scores each triple for complexity (1–5) + faithfulness (0/1)
       ↓
build_annotation_csv.py — merges everything into annotated.csv for human review
       ↓
03_analysis_and_viz.ipynb — charts + metrics
```

---

## Economic Periods

| Period | Years | Description |
|---|---|---|
| Great Moderation | 1994–2007 | Conventional monetary policy language |
| Post-Crisis ZLB | 2008–2019 | QE, forward guidance, new vocabulary |
| Post-COVID Surge | 2020–2023 | Supply chain, pandemic demand language |

10 meetings sampled per period (30 total).

---

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Flosam/fomc-causal-extraction.git
cd fomc-causal-extraction

# Install dependencies and create virtual environment
uv sync

# Copy and fill in API keys
cp .env.example .env
```

Edit `.env`:
```
GEMINI_API_KEY=...
OPENAI_API_KEY=...   # optional fallback
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

---

## Running the Pipeline

```bash
# 1. Download and filter the corpus
uv run python -m src.data_pipeline

# 2. Extract sections and segment into passages
uv run python -m src.preprocessor

# 3. Verify the corpus contains canonical causal relationships
uv run python -m src.sanity_check

# 4. Run extraction (held-out set first for prompt refinement)
uv run python -m src.extractor --held-out-only
uv run python -m src.extractor --resume          # full run

# 5. Run LLM-as-judge
uv run python -m src.judge --resume

# 6. Build annotation spreadsheet
uv run python -m src.build_annotation_csv
```

Then open `outputs/annotated.csv`, fill in `human_complexity_score` and `human_faithful` where you want to override the LLM judge, and run `notebooks/03_analysis_and_viz.ipynb`.

---

## Switching LLM Provider

Edit `config.yaml`:
```yaml
model_provider: openai   # or: gemini
```

No code changes needed.

---

## Outputs

| File | Description |
|---|---|
| `outputs/passages.csv` | All segmented passages |
| `outputs/extractions.csv` | Raw LLM extraction results |
| `outputs/judge_scores.csv` | LLM-as-judge complexity + faithfulness scores |
| `outputs/annotated.csv` | Merged sheet for human review |
| `outputs/held_out_passages.csv` | 20-passage prompt refinement set |
| `outputs/fig_*.png` | Generated charts |

---

## Project Structure

```
src/
  config.py                 # loads config.yaml + .env
  data_pipeline.py          # corpus download + period filtering
  preprocessor.py           # section extraction + NLTK segmentation
  sanity_check.py           # verify canonical causal pairs
  prompts.py                # all LLM prompt templates
  extractor.py              # extraction orchestration loop
  judge.py                  # LLM-as-judge orchestration loop
  build_annotation_csv.py   # merge pipeline outputs
  models/
    base.py                 # abstract LLMModel + data classes
    gemini.py               # Gemini 1.5 Pro adapter
    openai_model.py         # OpenAI adapter
notebooks/
  01_data_exploration.ipynb
  02_extraction_run.ipynb
  03_analysis_and_viz.ipynb
```
