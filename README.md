# TimeRouter

State-of-the-art GIFT-EVAL submission. TimeRouter does end-to-end inference with
**4 frozen time-series foundation models (TSFMs)** — Chronos-2, FlowState,
PatchTST-FM, Sundial — combined by an **XGBoost router** with a
margin/diversity gate and a CV-inverse-weighted fallback.

**Result: LB MASE = 0.6746** on the full 97-config GIFT-EVAL test suite.

---

**Requirements:** an NVIDIA GPU.
All model weights download from HuggingFace on first run, the 4 TSFMs and the router
checkpoints ([`nkh/timerouter-v1`](https://huggingface.co/nkh/timerouter-v1)).

---

## 1. Build the conda environments

### 1a. Main env (`timerouter`)

```bash
conda create -n timerouter python=3.11 -y && conda activate timerouter
pip install --no-cache-dir \
  torch==2.9.1 \
  transformers==4.57.6 tokenizers==0.22.2 safetensors==0.5.3 accelerate==1.12.0 \
  numpy==1.26.4 pandas==2.3.3 scipy==1.11.4 einops==0.7.0 \
  xgboost==3.1.3 gluonts==0.15.1 chronos-forecasting==2.2.2 \
  datasets==2.17.1 utilsforecast==0.2.14 python-dotenv==1.0.0 toolz==0.12.1
```

> The GIFT-EVAL data loader is vendored (`gift_eval/_data.py`), so `timecopilot` is
> **not** a dependency — it pins an incompatible `transformers==4.40.1` and
> eager-imports a heavy agent stack. `python-dotenv`/`toolz` are what that vendored
> loader needs; `utilsforecast` is used by the Chronos backend.

### 1b. Sundial env (`sundial`)

```bash
conda create -n sundial python=3.10 -y && conda activate sundial
pip install --no-cache-dir \
  torch==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir \
  transformers==4.40.1 tokenizers==0.19.1 safetensors==0.5.3 accelerate==1.13.0 \
  numpy==1.26.4 pandas==2.1.4 scipy==1.15.3 einops==0.7.0 \
  xgboost==3.1.3 gluonts==0.16.2 datasets==4.8.5
```

---

## 2. Clone the two granite-tsfm branches

FlowState and PatchTST-FM are on separate unmerged branches of IBM's `granite-tsfm`:

```bash
git clone --branch patchtst-fm    https://github.com/ibm-granite/granite-tsfm.git /tmp/granite_patchtst
git clone --branch gift-flowstate https://github.com/ibm-granite/granite-tsfm.git /tmp/granite_tsfm_clone
```

These are added to `sys.path` at runtime (no `pip install` needed). The PatchTST-FM
path is overridable via `--granite-patchtst`; the FlowState path is hard-coded to
`/tmp/granite_tsfm_clone`.

---

## 3. GIFT-EVAL data

Download the GIFT-EVAL test corpus per the official instructions:
<https://huggingface.co/spaces/Salesforce/GIFT-Eval>. Then point the loader at it,
either via the `--gift-eval-storage` flag or the `$GIFT_EVAL` env var (one is required):

```bash
export GIFT_EVAL=/path/to/GIFT-EVAL    # or pass --gift-eval-storage /path/to/GIFT-EVAL
```

---

## 4. Run

Always run from the **`timerouter`** env; it spawns the Sundial subprocess into
`sundial` automatically. Point `--sundial-python` at your `sundial` env's python.

```bash
conda activate timerouter
SUNDIAL_PY=$(conda run -n sundial which python)   # path to the sundial env python
```

### 4a. Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python gift_eval/run_eval.py \
  --datasets m4_weekly --terms short \
  --sundial-python "$SUNDIAL_PY" \
  --gift-eval-storage /path/to/GIFT-EVAL \
  --out-csv /tmp/smoke.csv
```

Expect `m4_weekly/W/short` MASE ≈ **1.99**.

### 4b. Full 97-config run

**Single GPU** (simplest, slow):

```bash
CUDA_VISIBLE_DEVICES=0 python gift_eval/run_eval.py \
  --sundial-python "$SUNDIAL_PY" \
  --gift-eval-storage /path/to/GIFT-EVAL \
  --out-csv gift_eval/all_results.csv
```

**4 GPUs** (shard + merge; ~3.5 h wall-clock on 4× A6000):

```bash
for SH in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$((SH-1)) python gift_eval/run_eval.py \
    --num-shards 4 --shard $SH \
    --sundial-python "$SUNDIAL_PY" \
    --gift-eval-storage /path/to/GIFT-EVAL \
    --out-csv gift_eval/all_results.csv \
    > /tmp/shard${SH}.log 2>&1 &
done
wait
python gift_eval/merge_shards.py 'gift_eval/all_results_shard*of4.csv' gift_eval/all_results.csv
```

Each shard runs its own Sundial subprocess on its assigned GPU. Per-shard LB MASE is
meaningless (different denominators) — only the merged result is valid.

### Outputs

| File | Contents |
|---|---|
| `gift_eval/all_results.csv` | Leaderboard CSV — 97 rows × 15 cols, standard 11-metric list. |
| `gift_eval/lb_mase.txt` | Achieved LB MASE (geometric mean of per-config MASE / seasonal-naive MASE). **Should read `0.6746`.** |
