# Batch API Workflow Quick Reference

**Last Updated**: March 26, 2026

---

## Quick Start

### First Run (Recommended)

```bash
# 1. Test on small held-out set
uv run python -m src.extractor --held-out-only

# 2. Check status (can be run anytime)
uv run python -m src.extractor --poll

# 3. Once complete, run full extraction
uv run python -m src.extractor

# 4. (Optional) Run judgment for LLM-as-a-judge scoring
uv run python -m src.judge

# 5. Build annotation CSV (works with or without judge scores)
uv run python -m src.build_annotation_csv
```

---

## Command Reference

### Extraction Commands

| Command | What It Does |
|---------|--------------|
| `uv run python -m src.extractor` | Submit new extraction batch job for all passages |
| `uv run python -m src.extractor --held-out-only` | Extract only 20-passage held-out set (for testing) |
| `uv run python -m src.extractor --poll` | **Resume polling** existing batch jobs |
| `uv run python -m src.extractor --skip-extracted` | Extract only new passages (skips ones in extractions.csv) |

### Judgment Commands (Optional)

| Command | What It Does |
|---------|--------------|
| `uv run python -m src.judge` | Submit new judgment batch job (optional step) |
| `uv run python -m src.judge --poll` | **Resume polling** existing judgment batch jobs |

**Note**: Judgment is optional. You can build the annotation CSV and manually annotate extracted triples without running LLM-as-a-judge.

### Monitoring Commands

| Command | What It Does |
|---------|--------------|
| `cat outputs/batch_jobs.json` | View all active batch jobs and their status |
| `cat outputs/extractions.csv` | View extraction results |
| `cat outputs/judge_scores.csv` | View judgment results (if judge was run) |

---

## Common Workflows

### Scenario 1: First Time Setup

```bash
# Step 1: Test with small batch
uv run python -m src.extractor --held-out-only

# Output shows:
# ✅ Batch job created: extraction_10_passages_20260326_123456
#    Job name: batches/abc123xyz789
#    Total requests: 10
# ⏳ Polling for results (this may take hours)...
#    [Poll #1] State: PENDING | 0/10 requests | Elapsed: 0h 1m

# Step 2: Press Ctrl+C to interrupt (optional)
# Batch continues running on Google's servers

# Step 3: Later, check if complete
uv run python -m src.extractor --poll

# If complete, you'll see:
# ✅ Batch job completed successfully!
# Results saved to outputs/extractions.csv
```

### Scenario 2: Interrupted Polling

```bash
# You ran this earlier:
uv run python -m src.extractor

# Polling started, then you pressed Ctrl+C
# Now you want to check if it's done:

uv run python -m src.extractor --poll

# Script detects existing batch and continues polling
```

### Scenario 3: Adding More Passages

```bash
# You already extracted 500 passages (in extractions.csv)
# Now you want to extract 500 more without re-processing the first 500:

uv run python -m src.extractor --skip-extracted

# Script will:
# 1. Load extractions.csv
# 2. Skip those 500 passage IDs
# 3. Create batch job for remaining passages only
```

### Scenario 4: Full Pipeline (Multi-Day)

```bash
# Day 1 - Morning
uv run python -m src.data_pipeline
uv run python -m src.preprocessor
uv run python -m src.sanity_check
uv run python -m src.extractor --held-out-only
# (Wait or interrupt - batch runs on Google's servers)

# Day 1 - Evening
uv run python -m src.extractor --poll
# Still running: 3/10 complete

# Day 2 - Morning  
uv run python -m src.extractor --poll
# ✅ Complete! Results in extractions.csv

# Run full extraction
uv run python -m src.extractor
# (Wait or interrupt)

# Day 3
uv run python -m src.extractor --poll
# ✅ Complete! 1234 triples extracted

# (Optional) Run judgment
uv run python -m src.judge
# (Wait or interrupt)

# Day 4
uv run python -m src.judge --poll
# ✅ Complete! All triples judged

# Build final dataset (works with or without judge scores)
uv run python -m src.build_annotation_csv
```

---

## Understanding Batch Job States

| State | Meaning | What to Do |
|-------|---------|------------|
| `PENDING` | Job is in Google's queue | Wait, or interrupt and check later |
| `RUNNING` | Job is actively processing | Wait, or interrupt and check later |
| `SUCCEEDED` | Job completed successfully | Results will be retrieved automatically |
| `FAILED` | Job failed with errors | Check error messages, debug requests |
| `CANCELLED` | Job was manually cancelled | N/A (not implemented yet) |

---

## Troubleshooting

### "Found existing batch job(s)" prompt

When you run a command, if there are pending/running jobs, you'll be asked:

```
⚠️  Found 1 in-progress extraction batch job(s):
  - extraction_1234_passages_20260325_202728 (batches/abc123...)
    State: RUNNING, Created: 2026-03-25T20:27:29

Continue polling existing batch? [Y/n]:
```

**Answer Y** to resume polling that job  
**Answer n** to skip it and start a new batch job (not recommended - causes duplicates)

### Batch stuck in PENDING for hours

This is **normal**. Google processes batch jobs during off-peak times. Wait time can be:
- Small jobs (10-50 passages): 2-8 hours
- Medium jobs (100-500 passages): 8-16 hours  
- Large jobs (1000+ passages): 24+ hours

### How to check status without running script

```bash
# View batch_jobs.json
cat outputs/batch_jobs.json

# Look for "state" field:
# - "JOB_STATE_PENDING" = waiting
# - "JOB_STATE_RUNNING" = processing
# - "JOB_STATE_SUCCEEDED" = done (run --poll to retrieve)
```

### Accidentally created duplicate batch jobs

If you see multiple jobs for the same passages in `batch_jobs.json`:

```bash
# Cancel the unwanted job (replace with actual job name)
uv run python -c "
from src.models import get_model
model = get_model('gemini')
model.cancel_batch('batches/UNWANTED_JOB_NAME')
"

# Remove from tracking file
uv run python -c "
from src.batch_state import BatchStateManager
state_manager = BatchStateManager()
state_manager.delete('batches/UNWANTED_JOB_NAME')
"
```

---

## Key Differences: --poll vs --skip-extracted

### `--poll` (Resume Polling)

**Use when:** You have an in-progress batch job and want to check its status

```bash
uv run python -m src.extractor --poll
```

**What it does:**
1. Checks `batch_jobs.json` for pending/running jobs
2. If found, resumes polling that job
3. If complete, retrieves and saves results
4. Does NOT create a new batch job

### `--skip-extracted` (Skip Completed Work)

**Use when:** You want to extract NEW passages without re-processing old ones

```bash
uv run python -m src.extractor --skip-extracted
```

**What it does:**
1. Loads `extractions.csv` to see what's already extracted
2. Filters those passage IDs from the input
3. Creates a NEW batch job for remaining passages only
4. Does NOT check for existing in-progress jobs

### `--resume` (Deprecated - Does Both)

**Old flag:** Still works for backward compatibility

```bash
uv run python -m src.extractor --resume
```

**What it does:** Both `--poll` AND `--skip-extracted` together

**Recommendation:** Use the specific flag you need (`--poll` or `--skip-extracted`) for clarity

---

## Expected Processing Times

### Extraction

| Passages | Requests | Estimated Time | Cost (Batch API) |
|----------|----------|----------------|------------------|
| 10 (held-out) | 10 | 2-8 hours | ~$0.05 |
| 100 | 100 | 8-12 hours | ~$0.50 |
| 500 | 500 | 12-18 hours | ~$2.50 |
| 1000 | 1000 | 18-24 hours | ~$5.00 |

### Judgment

Each triple requires 2 requests (complexity + faithfulness)

| Triples | Requests | Estimated Time | Cost (Batch API) |
|---------|----------|----------------|------------------|
| 50 | 100 | 8-12 hours | ~$0.30 |
| 200 | 400 | 12-18 hours | ~$1.20 |
| 1000 | 2000 | 18-24 hours | ~$6.00 |

*Times are estimates - Google's queue varies by demand*

---

## Tips

✅ **Always test with `--held-out-only` first** before running full extraction  
✅ **Press Ctrl+C freely** - batch jobs continue on Google's servers  
✅ **Check `batch_jobs.json`** to see all active jobs  
✅ **Use `--poll` daily** to check if jobs completed  
✅ **Don't create duplicate jobs** - always answer 'Y' when prompted about existing batches  

❌ **Don't use `--skip-extracted` when polling** - use `--poll` instead  
❌ **Don't cancel existing jobs** unless they're truly duplicates  
❌ **Don't expect immediate results** - batch API trades time for 50% cost savings

---

## Files Generated

| File | Created By | Purpose |
|------|------------|---------|
| `outputs/batch_jobs.json` | Batch submission | Tracks all batch job states |
| `outputs/passages.csv` | Preprocessor | Input passages for extraction |
| `outputs/held_out_passages.csv` | First `--held-out-only` run | 20-passage test set |
| `outputs/extractions.csv` | Extraction completion | Raw extraction results |
| `outputs/judge_scores.csv` | Judgment completion (optional) | Complexity + faithfulness scores |
| `outputs/annotated.csv` | build_annotation_csv | Final merged dataset for analysis |

---

## See Also

- **README.md** - Full pipeline overview
- **BATCH_API_GUIDE.md** - Detailed technical documentation
- **config.yaml** - Configuration settings
- **src/extractor.py** - Extraction implementation
- **src/judge.py** - Judgment implementation
