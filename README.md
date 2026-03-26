# FOMC Causal Extraction

A research pipeline for evaluating how reliably LLMs extract causal knowledge from Federal Reserve meeting minutes, and whether failure rates vary systematically across linguistic complexity and economic regime.


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

### 1. Download and Prepare Data

```bash
# 1. Download and filter the corpus
uv run python -m src.data_pipeline

# 2. Extract sections and segment into passages
uv run python -m src.preprocessor

# 3. Verify the corpus contains canonical causal relationships
uv run python -m src.sanity_check
```

### 2. Run Extraction (Batch API)

The extraction process uses **Gemini's Batch API** for 50% cost savings. Processing happens asynchronously and may take several hours.

#### First Time: Test on Held-Out Set

```bash
# Test extraction on 20-passage held-out set (recommended first run)
uv run python -m src.extractor --held-out-only
```

This submits a small batch job (~10 passages) to test your setup. The script will:
1. Build and submit batch requests to Gemini
2. Begin polling for completion (displays progress every 1-10 minutes)
3. Save results to `outputs/extractions.csv` when complete

**You can safely interrupt polling** (Ctrl+C) - the batch job continues on Google's servers.

#### Resume Polling an In-Progress Batch

If you interrupted polling or want to check on your batch:

```bash
# Resume polling existing batch jobs
uv run python -m src.extractor --poll
```

This will:
- Detect any pending/running batch jobs
- Show their status and progress
- Retrieve results if complete

#### Run Full Extraction

Once the held-out test succeeds:

```bash
# Full extraction on all passages
uv run python -m src.extractor
```

If you need to extract additional passages later (skipping already-extracted ones):

```bash
# Skip passages already in extractions.csv
uv run python -m src.extractor --skip-extracted
```

### 3. Run Judgment (Batch API)

After extraction completes, run LLM-as-judge to score complexity and faithfulness:

```bash
# Submit judgment batch job
uv run python -m src.judge

# If interrupted, resume polling with:
uv run python -m src.judge --poll
```

### 4. Build Annotation Spreadsheet

```bash
# Merge extraction and judgment results
uv run python -m src.build_annotation_csv
```

Then open `outputs/annotated.csv`, fill in `human_complexity_score` and `human_faithful` columns for manual review, and run `notebooks/03_analysis_and_viz.ipynb` for analysis.

---

## Batch API Workflow

### Command Reference

| Command | Purpose |
|---------|---------|
| `uv run python -m src.extractor` | Submit new extraction batch job |
| `uv run python -m src.extractor --held-out-only` | Extract held-out set only (for testing) |
| `uv run python -m src.extractor --poll` | Resume polling existing batch jobs |
| `uv run python -m src.extractor --skip-extracted` | Skip passages already in extractions.csv |
| `uv run python -m src.judge` | Submit new judgment batch job |
| `uv run python -m src.judge --poll` | Resume polling judgment batch jobs |

### Typical Workflow

```bash
# Day 1: Submit batch job
uv run python -m src.extractor --held-out-only
# Output: "Batch job created: batches/abc123..."
# Wait time shown, polling begins automatically

# Press Ctrl+C to safely interrupt (optional)
# Batch continues running on Google's servers

# Day 2: Check status and retrieve results
uv run python -m src.extractor --poll
# If complete: Results saved to outputs/extractions.csv
# If still running: Shows progress (e.g., "523/1000 requests complete")

# Once extraction completes, run judgment
uv run python -m src.judge
# Press Ctrl+C if needed, resume later with --poll

# After judgment completes
uv run python -m src.build_annotation_csv
```

### Monitoring Progress

**During polling**, you'll see:
```
⏳ Polling for results (this may take hours)...
   [Poll #1] State: PENDING | 0/1234 requests | Elapsed: 0h 1m
   [Poll #2] State: RUNNING | 150/1234 requests | Elapsed: 0h 5m
   [Poll #15] State: SUCCEEDED | 1234/1234 requests | Elapsed: 3h 24m
✅ Batch job completed successfully!
```

**Check batch status** anytime:
```bash
# View batch_jobs.json to see all active/pending jobs
cat outputs/batch_jobs.json
```

---

## Batch API Processing

The pipeline uses **Gemini's Batch API** for extraction and judgment:

### Workflow

1. **Submit**: Build batch requests and submit to Gemini
2. **Poll**: Script polls job status with exponential backoff (1min → 10min intervals)
3. **Retrieve**: Download and parse results when complete

### Interrupting and Resuming

- Press `Ctrl+C` to safely interrupt polling
- Batch job continues running on Google's servers
- Resume later with `--resume` flag to continue polling

### Monitoring

Progress updates show:
- Elapsed time
- Job state (PENDING → RUNNING → SUCCEEDED)
- Completed/failed request counts

Batch state is saved to `outputs/batch_jobs.json` for tracking.

---

## Switching LLM Provider

Edit `config.yaml`:
```yaml
model_provider: openai   # or: gemini, github
```

**Note**: Batch API is only available for Gemini. Other providers will use individual API calls.

---

## Configuration

Key settings in `config.yaml`:

```yaml
# Batch API polling configuration
batch_api:
  poll_initial_delay: 60      # First poll wait (seconds)
  poll_max_interval: 600      # Max wait between polls (seconds)
  poll_backoff_factor: 1.5    # Exponential backoff multiplier
```

---

## Outputs

| File | Description |
|---|---|
| `outputs/passages.csv` | All segmented passages |
| `outputs/extractions.csv` | Raw LLM extraction results |
| `outputs/judge_scores.csv` | LLM-as-judge complexity + faithfulness scores |
| `outputs/annotated.csv` | Merged sheet for human review |
| `outputs/held_out_passages.csv` | 20-passage prompt refinement set |
| `outputs/batch_jobs.json` | Batch API job state tracking |
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
    gemini.py               # Gemini 3 flash adapter
    openai_model.py         # OpenAI adapter
notebooks/
  01_data_exploration.ipynb
  02_extraction_run.ipynb
  03_analysis_and_viz.ipynb
```
