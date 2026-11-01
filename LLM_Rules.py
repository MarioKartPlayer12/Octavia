 

import os
import re
import json
import requests

import WorldModels as wm

 
OPENROUTER_API_KEY = 
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODEL = os.environ.get("O44_MODEL", "z-ai/glm-5.3-flash")
MAX_WMS_PER_QUESTION = int(os.environ.get("O44_MAX_WMS", "3"))
MAX_WMS_PER_TRIGGER  = int(os.environ.get("O44_MAX_WMS_PER_TRIGGER", "10"))

 
FAMILIES_HUMAN = """
WM FAMILIES  (pick one per WM)
  1. Symbolic AI       logic / rules / constraint solvers
  2. Classical ML      scikit-learn, xgboost, lightgbm
  3. Deep Learning     PyTorch: MLPs, CNNs, RNNs, Transformers, VAEs, JEPA
  4. Programs          self-contained Python callable module
                        (chess engines, sudoku solvers, physics sims,
                         sympy evaluators, file utilities, ...)
  5. RL Programs       Stable-Baselines3 + Gymnasium agents
                        (Blackjack, CartPole, custom GridWorlds, ...)
Ping @harness for the detailed breakdown.
"""


PATH_RULES = """
PATH RULES  (harness runs your script with the WM root as cwd)
  - Use RELATIVE paths only.
      os.makedirs("models", exist_ok=True)            ✓
      open("datasets/foo.csv", "w")                   ✓
      joblib.dump(model, "models/foo.joblib")         ✓
  - DO NOT use /models, /datasets, /jsons, /scripts as absolute
    roots — those point at the filesystem root and need sudo.
  - The harness auto-rewrites leading-slash folder paths to relative
    ones as a safety net, but always write RELATIVE in the first place.
"""


CODE_BLOCK_FORMAT = """
CODE BLOCK FORMAT
  Every training-script code block must begin with a header comment
  naming the WM:

      # @wm: atlantic_sst_forecaster
      import ...
      ...

  The harness uses this name to save the script to
  /scripts/<wm_name>_train.py and to register the resulting WM.

  For Programs (family 4): the SAME block can additionally write
      /scripts/<wm_name>.py
  as the callable module (with `def run(*args, **kwargs):` etc.)
  if you want to keep the training script and the executable
  module separate.

  For RL Programs (family 5): save the model with
      model.save("models/<wm_name>.zip")
  and the JSON with `wm.save_wm_metadata({...})` at the end.
"""


SELF_CORRECTION = """
SELF-CORRECTION RULES
  - If your script returns non-zero / raises / times out, the
    harness gives you back: returncode, elapsed seconds, full
    stdout, full stderr (incl. traceback), and a heuristic hint.
    READ the feedback, FIX the code, emit a corrected ```python```
    block in your next reply.
  - The harness streams stdout/stderr live to the operator's UI.
  - DO NOT emit `=== DONE ===` unless the WM's JSON actually
    appeared in /jsons/. If you emit DONE without a JSON, the
    harness rejects it and forces self-correction.
  - Common pitfalls (the harness hints at them too):
      * numpy scalar (e.g. numpy.float64) cannot multiply a list.
        Use `np.array(seq)` before multiplying, or `float(scalar)`.
      * numpy aggregators on empty arrays.
      * shape mismatches between train/test splits.
      * undefined names, typos, wrong scope.
      * division by zero / NaN.
      * file paths: ALWAYS relative. NEVER `/models/...`.
      * blocking I/O: input(), urllib without timeout, network waits.
      * forgetting to call `wm.save_wm_metadata({...})` — without
        it no JSON appears and DONE will be rejected.
"""

 
def build_training_system_prompt(max_wms: int = MAX_WMS_PER_QUESTION) -> str:
    return f"""You are o44 in QUESTIONS-TRAINING mode.

Your job is NOT to answer the user's question. Your job is to BUILD
the World Models (WMs) that would be needed to answer it well. These
WMs get saved to disk and reused by future o44 instances (and by
you, in later inference mode).

You build WMs by writing Python code (sklearn / PyTorch / SB3 +
Gymnasium / plain Python / etc.) and saving the artifacts to the
canonical folders.

------------------------------------------------------------
PING PROTOCOL
------------------------------------------------------------
Emit pings in your output. The harness resolves them and feeds the
content back as the next turn.

  @harness   →  full WM-creation guide (5 families, folder layout,
                synthetic-data standards).
  @jsons     →  the registered WMs already in the registry.
  @<name>    →  full JSON metadata for one specific WM.

Typical flow:
  1. (Optional) @harness to refresh on creation standards.
  2. @jsons to see what WMs already exist for the domain.
  3. Identify gaps.
  4. For each gap, write a ```python``` block tagged with the WM
     name. The harness saves it to /scripts/ and executes it.
  5. When the WM's JSON sits in /jsons/, emit:

         === DONE ===

------------------------------------------------------------
LIMITS
------------------------------------------------------------
You may build AT MOST {max_wms} WMs per question. If you would
need more, prioritise the most general / reusable WMs.

Each training script runs as a subprocess with a default timeout
of 300 s (5 min). The harness kills anything that exceeds it and
feeds the failure back. Plan accordingly.

{FAMILIES_HUMAN}
{PATH_RULES}
{CODE_BLOCK_FORMAT}
{SELF_CORRECTION}

NOTE on iteration: each conversation with you builds exactly ONE WM.
When you emit === DONE ===, the conversation ends and a fresh one
starts for the next WM. Plan accordingly.
"""

 
def build_tgt_training_system_prompt(
        max_wms_per_trigger: int = MAX_WMS_PER_TRIGGER) -> str:
    return f"""You are o44 in TRIGGERED-GEOMETRY-TRAINING (TGT) mode.

You do NOT receive a question. You receive a TRIGGER — a concept
cloud that lights up specific regions of your internal geometry.
Your job is to extract AS MANY distinct World Models as you
genuinely have access to for that trigger, building them all into
the registry. You're a one-shot extractor. Be dense.

A trigger is a comma-space-separated bag of related terms, e.g.

    "first-order logic, constraint propagation, SAT solvers,
     propositional variables, DPLL, CNF, graph colouring, zebra
     puzzles, Einstein's riddle, latin squares, Knapsack,
     python-constraint, z3, ortools.SatCp"

Concept triggers activate broader representational regions than
specific questions. EXPECT to pull out many WMs per trigger.

------------------------------------------------------------
PING PROTOCOL
------------------------------------------------------------
  @harness   →  full WM creation guide (5 families).
  @jsons     →  registered WMs in the current registry.
  @<name>    →  full JSON metadata for one WM.

------------------------------------------------------------
WORKFLOW
------------------------------------------------------------
  1. @harness once if you need a refresher on families / format.
  2. @jsons to see what's already registered — DO NOT duplicate.
  3. Plan the BATCH. Aim for width: distinct sub-skills the
     trigger implies. Examples for the trigger above:
        - cnfsat_solver            family=classical (uses pysat)
        - zebra_puzzle_solver       family=program
        - constraint_propagation    family=symbolic
        - ksatisfiability_eval      family=classical (eval harness)
  4. Emit ONE ```python``` block per WM, each with `# @wm: <name>`.
  5. The harness saves & executes each in turn. Failures get fed
     back to you as next-turn user messages — FIX and re-emit.
  6. Continue until you've produced all WM JSONs or you've reached
     the per-trigger cap ({max_wms_per_trigger}).
  7. Emit exactly:

         === DONE ===

ONLY when at least one new JSON is in /jsons/.

{FAMILIES_HUMAN}
{PATH_RULES}
{CODE_BLOCK_FORMAT}
{SELF_CORRECTION}

------------------------------------------------------------
TGT-SPECIFIC GUIDANCE
------------------------------------------------------------
  - Plan wide, not deep. ONE great program per sub-skill beats one
    gigantic super-WM.
  - For Programs (family 4): the module under /scripts/<name>.py is
    the artifact people will CALL at inference. Make it ergonomic.
  - For RL Programs (family 5): keep sample efficiency in mind.
    If the env is slow or sample-inefficient, pair the RL agent
    with a known-good Program WM under family 4 (cf. Blackjack RL +
    basic-strategy lookup) and tag the JSON accordingly.
  - Synthetic data generators go in /datasets/ and are referenced
    from the JSON.
  - One training script may register MANY WMs (use multiple
    `# @wm:` blocks, or one script that calls
    `wm.save_wm_metadata({...})` for several names).
"""
 

PING_PATTERN    = re.compile(r"@([a-zA-Z0-9_\-\.]+)")
CODE_PATTERN    = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
WM_NAME_PATTERN = re.compile(r"^\s*#\s*@wm[:\s]+([a-zA-Z0-9_\-]+)",
                             re.MULTILINE)
DONE_TOKEN      = "=== DONE ==="


def extract_pings(text: str) -> list:
    return PING_PATTERN.findall(text)


def extract_code_blocks(text: str) -> list:
    return CODE_PATTERN.findall(text)


def extract_wm_name(code: str):
    m = WM_NAME_PATTERN.search(code)
    return m.group(1) if m else None


def is_done(text: str) -> bool:
    return DONE_TOKEN in text

def resolve_ping(name: str) -> str:
    name = name.strip()

    if name == "harness":
        return wm.ping_harness()

    if name.startswith("family:"):
        fam = name.split(":", 1)[1]
        return json.dumps(wm.ping_family(fam), indent=2)
 
    if name.startswith(("src:", "code:")):
        target = name.split(":", 1)[1]
        return wm.ping_source(target)

 
    if name.startswith("api:"):
        target = name.split(":", 1)[1]
        meta = wm.ping_wm(target)
        return json.dumps(meta.get("_api") or
                          {"error": meta.get("error", "no _api")},
                          indent=2)

    if name.lower() in ("jsons", "registry"):
        return json.dumps(wm.ping_registry_summary(), indent=2)
    if name.lower() in ("registry_full", "jsons_full"):
        return json.dumps(wm.ping_registry(), indent=2)

    return json.dumps(wm.ping_wm(name), indent=2)

 

def call_llm_streaming(
    history: list,
    on_chunk,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 32000,
    reasoning_effort: str = "none",
    system_prompt: str = None,
) -> str:
    """Stream an OpenRouter chat-completion.

    history:  list of {role, content} dicts (no system).
    on_chunk: callable(delta: str) called for each content delta.
    Returns the full assistant message.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set.")

    sys_content = (system_prompt
                   if system_prompt is not None
                   else build_training_system_prompt())
    messages = [{"role": "system", "content": sys_content}]
    messages.extend(history)

    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      True,
    }

    if reasoning_effort == "low":
        payload["reasoning"]  = {"effort": "low"}
        payload["max_tokens"] = max_tokens + 4000
    elif reasoning_effort == "medium":
        payload["reasoning"]  = {"effort": "medium"}
        payload["max_tokens"] = max_tokens + 8000
    elif reasoning_effort == "high":
        payload["reasoning"]  = {"effort": "high"}
        payload["max_tokens"] = max_tokens + 16000
    elif reasoning_effort == "max":
        payload["reasoning"]  = {"effort": "high", "max_tokens": 32000}
        payload["max_tokens"] = max_tokens + 32000

    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type":  "application/json",
        },
        json=payload,
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    resp.encoding = "utf-8"

    accumulated = ""
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        body = raw_line[len("data: "):]
        if body.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(body)
            delta = (chunk.get("choices", [{}])[0]
                          .get("delta", {})
                          .get("content"))
            if delta:
                accumulated += delta
                on_chunk(delta)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue

    return accumulated