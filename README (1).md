# Octavia-training

Current **Octavia / o44** training scripts — a self-improving *World Model* harness.
An LLM agent (`o44`) doesn't just answer questions: it **writes, trains and registers
reusable Python World Models (WMs)** to a local registry, then at inference time plans
with hypotheses that call those WMs, and a decoder model turns the trace into a final
answer. A benchmark runner scores the whole pipeline against a plain baseline LLM with
a judge model.

You need **your own [OpenRouter](https://openrouter.ai) API key** — none is shipped.

License: Apache-2.0

---

## What's in the repo

| File / dir | What it is |
|---|---|
| `Training.py` | **Desktop GUI** (tkinter) for building the WM registry. Two modes: *questions* (build WMs needed for one question) and *TGT* (triggered-geometry-training: extract a batch of WMs from a concept cloud). |
| `Inference.py` | **Local web UI** (Flask, `http://127.0.0.1:5000`). Cognition → hypotheses → executable probes → decoder answer. |
| `BenchmarkRun.py` | **Local web UI** (Flask, `http://127.0.0.1:5001`). Runs benchmark suites, LLM-judge grading, o44 vs baseline comparison. |
| `WorldModels.py` | The harness library: registry API (`save_wm_metadata`, `ping_*`, `call_program`, `call_rl`), folder layout, the full WM-creation guide (`@harness`). |
| `LLM_Rules.py` | System prompts (training / TGT), the ping protocol, and the streaming OpenRouter client shared by everything. |
| `benchmarks/` | Benchmark suites as JSON (math, logic, chess, blackjack, finance, science, refusal, …). |
| `benchmarks.txt` | Latest recorded results (Octavia-V3 run) — see below. |
| `datasets/` | Datasets / self-test / eval artifacts used by WMs and benchmarks. |
| `scripts/`, `jsons/` | WM training scripts and WM metadata registry (`jsons/<name>.json`). Created automatically; empty until you train WMs. |
| `probes/` | Inference-time probe scripts written by the cognition loop (auto-created). |

### The core idea: World Models

Every WM is a small, reusable artifact built by the agent in one of five families:

1. **Symbolic AI** — rules, logic, constraint solvers (z3, pysat, networkx, …)
2. **Classical ML** — scikit-learn / xgboost / lightgbm
3. **Deep learning** — PyTorch (MLP, CNN, RNN, transformer, VAE, JEPA)
4. **Programs** — a plain callable Python module (chess engine, Sudoku solver, sympy evaluator, simulator, …)
5. **RL programs** — Stable-Baselines3 + Gymnasium policies (Blackjack, CartPole, …)

Each WM produces four artifacts:

```
jsons/<wm_name>.json           metadata: family, description, I/O schema, metrics, deps
scripts/<wm_name>_train.py     the script that builds it
models/<wm_name>.<ext>         joblib / .pt / .zip  (null for pure symbolic/program)
datasets/<wm_name>.<ext>       csv / npz / json / pt (null if none)
```

The agent talks to the harness through a **ping protocol** while it works:

- `@harness` → the full WM-creation guide (families, folder layout, standards)
- `@jsons` → registry summary of already-built WMs
- `@<name>` → full metadata for one WM (includes an `_api` block with real signatures)
- `@src:<name>` / `@api:<name>` / `@family:<family>` → source / API / family listing

Training scripts are run as subprocesses with the repo root as cwd, stdout/stderr are
streamed live, failures (traceback, timeout, non-zero exit) are fed back to the model
for self-correction. A WM only counts as done when its JSON actually appears in
`jsons/` — `=== DONE ===` without it is rejected.

---

## Setup

**Requirements:** Python 3.10+, and a desktop display if you want `Training.py` (tkinter).

```bash
git clone https://github.com/MarioKartPlayer12/Octavia-training.git
cd Octavia-training
pip install flask requests
```

**1. Set your OpenRouter API key.** Open `LLM_Rules.py` and fill in line 11 (it is
published blank, which is also a syntax error — nothing runs until you set it):

```python
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

**2. (Optional) install WM-family dependencies** you plan to use:

```bash
pip install numpy scipy scikit-learn xgboost lightgbm sympy networkx z3-solver joblib
pip install torch                      # deep learning family
pip install gymnasium stable-baselines3 sb3-contrib   # RL family
```

**3. Run one of the three entry points.**

```bash
python Training.py       # desktop GUI  — build/extend the WM registry
python Inference.py      # web UI       — http://127.0.0.1:5000
python BenchmarkRun.py   # web UI       — http://127.0.0.1:5001
```

Both web UIs auto-open your browser ~1 s after startup.

---

## Configuration (environment variables)

All optional — defaults shown.

| Variable | Default | Used by |
|---|---|---|
| `O44_MODEL` | `z-ai/glm-5.3-flash` | training default model |
| `O44_COGNITION_MODEL` | `nvidia/nemotron-3.5-lightning` | cognition / planner |
| `O44_DECODER_MODEL` | `nvidia/nemotron-3.5-lightning` | final-answer decoder |
| `O44_JUDGE_MODEL` | `nvidia/nemotron-3.5-lightning` | benchmark judge |
| `O44_BASELINE_MODEL` | `O44_MODEL` value | benchmark baseline |
| `O44_MAX_WMS` | `3` | max WMs built per question (training) |
| `O44_MAX_WMS_PER_TRIGGER` | `10` | max WMs per trigger (TGT mode) |
| `O44_MAX_HYP` | `4` | max hypotheses per question |
| `O44_MAX_ONLINE` | `1` (Inference) / `2` (BenchmarkRun) | online WM create/update quota per question; `0` disables |
| `O44_MAX_TURNS_PER_HYP` | `36` | turn cap per hypothesis |
| `O44_SCRIPT_TIMEOUT` | `300` | subprocess timeout, seconds |
| `O44_HOST` / `O44_PORT` | `127.0.0.1` / `5000` | Inference UI |
| `O44_BENCH_HOST` / `O44_BENCH_PORT` | `127.0.0.1` / `5001` | Benchmark UI |

Example:

```bash
O44_COGNITION_MODEL=anthropic/claude-sonnet-4.5 O44_JUDGE_MODEL=openai/gpt-4o python BenchmarkRun.py
```

Note: running the web UIs on a non-localhost host exposes your API-key-backed
harness to the network — keep `127.0.0.1` unless you know why you're changing it.

---

## Adding a benchmark

Drop a JSON file into `benchmarks/`; it appears in the Benchmark UI automatically
(if the folder is empty, a starter suite is generated on first run).

```json
{
  "name": "My Suite",
  "description": "What this suite measures.",
  "version": "1",
  "created_at": "2026-10-31T12:00:00Z",
  "questions": [
    {
      "id": "q1",
      "question": "What is the sum of the integers from 1 to 100 inclusive?",
      "answer": "5050",
      "type": "numeric",
      "tolerance": 1e-6,
      "tags": ["math", "arithmetic"],
      "notes": "Gauss trick: n(n+1)/2",
      "rubric": "Accept 5050 or '5050'. Reject off-by-one answers."
    }
  ]
}
```

- `type` is one of: `exact`, `contains`, `numeric` (uses `tolerance`), `fuzzy`, `list`
  (order-insensitive). All questions need `id`, `question`, `answer`, `type`, `tags`.
- `rubric` (optional) is what the judge grades against; if absent, `notes` is used.

---

## Latest results (Octavia-V3)

From `benchmarks.txt` — judge-graded, o44 pipeline:

| Benchmark | Domain | Accuracy |
|---|---|---|
| BusinessBench | practical business + business math | 100% |
| LogiCheck | complex propositional logic | 100% |
| HilbertMana | advanced math / proof-level | 100% |
| BusinessFinance_ProofBench | advanced financial math | 100% |
| arc_challenge_mini | scientific reasoning | 94.1% |
| EzMath | moderate math | 93.3% |
| U-Math Mini | university math problems | 92.1% (mean 0.921) |
| JustLogic | deep logic questions | 87.0% (mean 0.874) |
| MonteBench | long blackjack strategy / math | 73.3% |
| RefusalBench | refusal of unsafe prompts — **0% answered, 18/18 refused (lower = better)** | — |

---

## Security / cost notes

- Your API key lives in `LLM_Rules.py` — don't commit it back. The harness executes
  LLM-generated Python locally (`scripts/`, `probes/`), so treat the registry output
  like any untrusted code before running it.
- Training and benchmarks stream many completions per question (hypotheses × turns);
  check OpenRouter usage before long benchmark sessions.