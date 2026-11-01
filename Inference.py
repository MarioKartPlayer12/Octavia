 

import os
import sys
import json
import queue
import threading
import subprocess
import re
import webbrowser
from datetime import datetime

import requests
from flask import Flask, Response, request, jsonify

import WorldModels as wm
import LLM_Rules as lr
 
OPENROUTER_API_KEY  = lr.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = lr.OPENROUTER_BASE_URL

COGNITION_MODEL = os.environ.get(
    "O44_COGNITION_MODEL", "nvidia/nemotron-3.5-lightning")
DECODER_MODEL = os.environ.get(
    "O44_DECODER_MODEL", "nvidia/nemotron-3.5-lightning")

DEFAULT_MAX_HYP    = int(os.environ.get("O44_MAX_HYP", "4"))
DEFAULT_MAX_ONLINE = int(os.environ.get("O44_MAX_ONLINE", "1"))
MAX_TURNS_PER_HYP  = int(os.environ.get("O44_MAX_TURNS_PER_HYP", "36"))

PROBES_DIR = os.path.join(wm.WM_ROOT, "probes")
os.makedirs(PROBES_DIR, exist_ok=True)

 
DEMO_EXAMPLES = [
    ("Einstein's riddle",
     "Solve the classic Einstein / Zebra puzzle: there are five houses in "
     "a row, each painted a different colour, with a person of a different "
     "nationality, a different drink, a different cigarette brand, and a "
     "different pet. Use the 15 standard clues. Who owns the fish?"),
    ("Best chess move",
     "Starting from the standard chess starting position, what is the "
     "best move for White? Use your chess engine to evaluate and explain."),
    ("Play one hand of Blackjack",
     "I have a 16 vs the dealer's 10. Should I hit, stand, double, or "
     "split? Use your blackjack policy (RL) AND the basic-strategy lookup "
     "if it exists. Justify with both."),
    ("Sudoku (easy)",
     "Solve this Sudoku puzzle using your constraint solver. "
     "Puzzle: ..5..4.6. | .3..7...1 | 8.....2. | ...9..8.. | "
     ".6...4... | ..5..2... | .2.....4 | 1...3..7. | .8.1..5.."),
    ("Projectile w/ drag",
     "A 0.145-kg baseball is hit at 45 m/s at a 30° launch angle. Air "
     "drag coefficient C·ρ·A/(2m) ≈ 0.012 /m. Compute the range, "
     "apex, and time-of-flight numerically. Show work."),
    ("Linear algebra",
     "Compute the eigenvalues and eigenvectors of the matrix "
     "[[3,1,0],[1,2,1],[0,1,3]]. Express them symbolically."),
    ("Logic puzzle",
     "Three friends (Alice, Bob, Carol) each have a different favourite "
     "colour (red, blue, green) and a different pet (cat, dog, fish). "
     "Alice does not have the cat. The person who has the dog loves blue. "
     "Carol does not have green. Bob has green. Who has the fish?"),
    ("Constrain math",
     "Is there an integer solution to x^3 + y^3 + z^3 = 42 with "
     "-100 ≤ x,y,z ≤ 100? Prove it or find one."),
]
 

def call_llm_streaming(messages, model, temperature=0.2,
                       max_tokens=32000, reasoning_effort="none",
                       on_chunk=None) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens,
               "stream": True}

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
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                 "Content-Type":  "application/json"},
        json=payload, stream=True, timeout=600,
    )
    resp.raise_for_status()
    resp.encoding = "utf-8"

    accumulated = ""
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        body = raw_line[len("data: "):]
        if body.strip() == "[DONE]": break
        try:
            chunk = json.loads(body)
            delta = (chunk.get("choices", [{}])[0]
                          .get("delta", {}).get("content"))
            if delta:
                accumulated += delta
                if on_chunk: on_chunk(delta)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
    return accumulated


 
class EventBus:
    def __init__(self):
        self.listeners = []
        self.lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue(maxsize=2048)
        with self.lock: self.listeners.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.listeners: self.listeners.remove(q)

    def emit(self, event_type, data=None):
        payload = {"type": event_type, "data": data,
                   "ts": datetime.now().isoformat()}
        with self.lock: listeners = list(self.listeners)
        for q in listeners:
            try: q.put_nowait(payload)
            except queue.Full: pass

 

PATH_REWRITE_RE = re.compile(
    r'''(["\'])/(models|datasets|jsons|scripts|probes)(["'/])''')


def _rewrite_absolute_paths(code: str) -> str:
    return PATH_REWRITE_RE.sub(r"\1\2\3", code)
 
PROBE_HELPERS = """
# Probe helpers — auto-injected ABOVE every probe block.  o44 may use them.
import wm_helpers as _h
#  _h.call_program(name, *args, **kwargs)   → Program WMs
#  _h.call_rl(name, obs, deterministic=True) → RL WMs
#  _h.call_sklearn(name, *args, **kwargs)   → Classical WMs (joblib)
#  _h.call_torch(name, *args, **kwargs)     → Deep WMs (.pt)
#  _h.call_symbolic(name, *args, **kwargs)  → import + call
"""


def _inject_helpers(code: str) -> str:
    """Prepend a probe with the WM-helper shim if it doesn't import wm."""
    if "WM_HELPERS_OK" in code:
        return code
    return PROBE_HELPERS + "\nWM_HELPERS_OK = True\n\n" + code


def execute_code(code: str, wm_name: str = None) -> dict:
    code = _rewrite_absolute_paths(code)
    code = _inject_helpers(code)

    helper_module = os.path.join(PROBES_DIR, "_wm_helpers.py")
    with open(helper_module, "w", encoding="utf-8") as f:
        f.write(_make_helper_module_text())

   
    frontend = os.path.join(wm.WM_ROOT, "wm_helpers.py")
    with open(frontend, "w", encoding="utf-8") as f:
        f.write(_make_helper_module_text())

    if wm_name:
        script_path = os.path.join(PROBES_DIR, f"{wm_name}_probe.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        cmd = [sys.executable, script_path]
    else:
        script_path = None
        cmd = [sys.executable, "-c", code]

    try:
        proc = subprocess.run(cmd, cwd=wm.WM_ROOT,
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=900)
        return {"script_path": script_path,
                "stdout": proc.stdout, "stderr": proc.stderr,
                "returncode": proc.returncode}
    except subprocess.TimeoutExpired:
        return {"script_path": script_path,
                "stdout": "", "stderr": "TIMEOUT after 900s",
                "returncode": -1}


def _make_helper_module_text() -> str:
    return '''"""Auto-generated WM call helpers used by inference probes."""
import WorldModels as _wm

def call_program(name, *args, **kwargs):
    return _wm.call_program(name, *args, **kwargs)

def call_rl(name, obs, deterministic=True):
    return _wm.call_rl(name, obs, deterministic=deterministic)

def call_sklearn(name, *args, **kwargs):
    import joblib
    meta = _wm.ping_wm(name)
    p = _wm.WM_ROOT + "/" + (meta["model_path"] or f"models/{name}.joblib")
    return joblib.load(p)(*args, **kwargs)

def call_torch(name, *args, **kwargs):
    import torch
    meta = _wm.ping_wm(name)
    p = _wm.WM_ROOT + "/" + (meta["model_path"] or f"models/{name}.pt")
    obj = torch.load(p, map_location="cpu")
    return obj(*args, **kwargs) if callable(obj) else obj

def call_symbolic(name, *args, **kwargs):
    meta = _wm.ping_wm(name)
    p = meta.get("script_path", f"scripts/{name}_train.py")
    return _wm.call_program(name, *args, **kwargs)

def call_any(name, *args, **kwargs):
    meta = _wm.ping_wm(name)
    fam = (meta.get("family") or "").lower()
    if   fam == "program":  return call_program(name, *args, **kwargs)
    elif fam == "rl":       return call_rl(name, *args, **kwargs)
    elif fam == "classical":return call_sklearn(name, *args, **kwargs)
    elif fam == "deep":     return call_torch(name, *args, **kwargs)
    elif fam == "symbolic": return call_symbolic(name, *args, **kwargs)
    raise RuntimeError(f"Unknown family for WM {name!r}: {fam!r}")
'''
 

COGNITION_SYSTEM_PROMPT = """You are o44 in COGNITION MODE (inference).

You do NOT answer the user's question directly. You PLAN how to answer
it by inspecting available World Models (WMs), forming HYPOTHESES, and
running METHODS that call those WMs. The harness executes the
```python``` blocks you emit and feeds results back on your next turn.

WMs are listed via pings. There are FIVE families:
  1. Symbolic AI        (logic / rules)
  2. Classical ML       (sklearn / xgboost)
  3. Deep Learning      (PyTorch)
  4. Programs           (self-contained callable Python module)
  5. RL Programs        (Stable-Baselines3 + Gymnasium)

------------------------------------------------------------
PING PROTOCOL
------------------------------------------------------------
  @harness    →  full WM-creation guide
  @jsons      →  summary of all registered WMs
  @<name>     →  full JSON metadata for one WM
  @family:rl  →  names of all RL-family WMs
  @family:program   /  @family:classical  /  @family:deep  /  @family:symbolic

------------------------------------------------------------
CALLING A WM IN A PROBE
------------------------------------------------------------
Probes are auto-injected with `import wm_helpers as _h` and the helpers
`call_program`, `call_rl`, `call_sklearn`, `call_torch`,
`call_symbolic`, `call_any`.

Examples:
  result = _h.call_program("sudoku_solver", puzzle_grid)
  action, _ = _h.call_rl("blackjack_ppo", obs, deterministic=True)
  pred    = _h.call_any("surrogate_x", X)

------------------------------------------------------------
HYPOTHESES
------------------------------------------------------------
A HYPOTHESIS is a candidate answer strategy bundling one or more METHODS
(calls to specific WMs or fresh probe code). Up to {max_hyp} per query.

For each hypothesis:
  - State which WMs you'll use & why.
  - Emit any pings you need.
  - Emit one or more ```python``` blocks to run those WMs.
  - Read harness stdout/stderr feedback on your next turn.
  - When you have the evidence you need, emit:

      === HYPOTHESIS COMPLETE ===

------------------------------------------------------------
NEW ONLINE WORLD MODELS
------------------------------------------------------------
You MAY create a new WM online if a hypothesis genuinely needs a bridge
the registry doesn't cover. State it explicitly:

    NEW ONLINE WORLD MODEL: <name>

Such WMs MUST be:
  - small (<1M params equivalent)
  - trainable on simple synthetic data
  - quick to fit (<30s)

Limit: at most {max_online} per query. Strongly prefer existing WMs.

------------------------------------------------------------
PLAN FORMAT (Turn 1)
------------------------------------------------------------
After setup pings, emit:

    PLAN
    Number of hypotheses: <N>
    Hypothesis 1: <one-sentence description>
      Method 1: <wm_name_or_probe_name> — <what it does>
      Method 2: ...
    Hypothesis 2: ...

------------------------------------------------------------
SUMMARY FORMAT (final cognition turn)
------------------------------------------------------------
    SUMMARY
    Hypothesis 1: <description>
      Methods used: <wm_names>
      Key results: <bullet list of evidence>
      Confidence: high/medium/low — <one-sentence justification>
    ...
    Overall Best Answer: <one-paragraph synthesis>

    === COGNITION DONE ===
"""


DECODER_SYSTEM_PROMPT = """You are the o44 DECODER.

You receive the full cognition trace (plan + hypothesis executions +
summary) and convert it into a polished final answer for the user.

Rules:
  - Markdown freely: headings, bullets, tables, code, math.
  - Be concise but complete.
  - Cite confidence levels from the cognition trace.
  - If hypotheses disagreed, say which is more credible and why.
  - STRIP harness internals: raw @ping tokens, `# @wm:` headers,
    === HYPOTHESIS COMPLETE ===, `## PING RESULT` headers, etc.
  - Translate technical findings into natural language the user
    actually wants to read.

Output the final user-facing answer only.
"""


def _system_prompt(max_hyp, max_online):
    return COGNITION_SYSTEM_PROMPT.format(
        max_hyp=max_hyp, max_online=max_online,
    )
 
def phase_plan(question, max_hyp, max_online,
               model, reasoning, bus):
    bus.emit("phase", {"name": "PLAN"})
    bus.emit("status", f"Planning with {model}…")
    messages = [
        {"role": "system",
         "content": _system_prompt(max_hyp, max_online)},
        {"role": "user",
         "content": f"Question: {question}\n\nBegin Cognition Mode."},
    ]
    bus.emit("llm_start", {"phase": "plan"})

    def on_chunk(d): bus.emit("llm_delta", {"phase": "plan", "delta": d})

    text = call_llm_streaming(messages, model=model,
                              reasoning_effort=reasoning, on_chunk=on_chunk)
    bus.emit("llm_end", {"phase": "plan"})
    return text, messages


def phase_hypotheses(plan_text, messages, max_hyp,
                     model, reasoning, bus):
    bus.emit("phase", {"name": "HYPOTHESES"})
    m = re.search(r"Number of hypotheses:\s*(\d+)", plan_text)
    n_hyp = min(int(m.group(1)) if m else 1, max_hyp)
    bus.emit("status", f"Executing {n_hyp} hypothesis(es).")
    messages.append({"role": "assistant", "content": plan_text})

    for h_idx in range(n_hyp):
        bus.emit("hypothesis_start", {"idx": h_idx + 1, "total": n_hyp})
        messages.append({
            "role": "user",
            "content": (f"Execute hypothesis {h_idx + 1} now. "
                       f"Ping any WMs you need, write python code to "
                       f"call them, and report results. When done, "
                       f"emit === HYPOTHESIS COMPLETE ===.")
        })

        done = False
        for turn in range(MAX_TURNS_PER_HYP):
            bus.emit("llm_start", {"phase": "hyp", "hyp_idx": h_idx + 1})

            def on_chunk(d, _h=h_idx + 1):
                bus.emit("llm_delta",
                         {"phase": "hyp", "hyp_idx": _h, "delta": d})

            text = call_llm_streaming(messages, model=model,
                                      reasoning_effort=reasoning,
                                      on_chunk=on_chunk)
            bus.emit("llm_end", {"phase": "hyp", "hyp_idx": h_idx + 1})
            messages.append({"role": "assistant", "content": text})

            for name in lr.extract_pings(text):
                try:
                    content = lr.resolve_ping(name)
                    bus.emit("ping", {"name": name})
                    messages.append({
                        "role": "user",
                        "content": (f"## PING RESULT @{name}\n"
                                   f"```\n{content}\n```")})
                except Exception as e:
                    bus.emit("error", f"Ping @{name} failed: {e}")

            for code in lr.extract_code_blocks(text):
                wm_name = lr.extract_wm_name(code)
                bus.emit("status", f"Executing {wm_name or 'inline'}…")
                result = execute_code(code, wm_name)
                bus.emit("code_result", result)
                messages.append({
                    "role": "user",
                    "content": (f"## CODE RESULT (rc={result['returncode']})\n"
                               f"stdout:\n```\n{result['stdout']}\n```\n"
                               f"stderr:\n```\n{result['stderr']}\n```")})

            if "=== HYPOTHESIS COMPLETE ===" in text:
                bus.emit("hypothesis_done", {"idx": h_idx + 1})
                done = True; break

        if not done:
            bus.emit("error",
                     f"Max turns reached for hypothesis {h_idx + 1}")
    return messages


def phase_summary(messages, model, reasoning, bus):
    bus.emit("phase", {"name": "SUMMARY"})
    bus.emit("status", "Summarizing hypotheses…")
    messages.append({
        "role": "user",
        "content": ("Now emit the SUMMARY block: each hypothesis with "
                   "its methods, key results, confidence, + Overall "
                   "Best Answer paragraph. Then === COGNITION DONE ===.")})
    bus.emit("llm_start", {"phase": "summary"})

    def on_chunk(d):
        bus.emit("llm_delta", {"phase": "summary", "delta": d})

    text = call_llm_streaming(messages, model=model,
                              reasoning_effort=reasoning, on_chunk=on_chunk)
    bus.emit("llm_end", {"phase": "summary"})
    messages.append({"role": "assistant", "content": text})
    return text


def phase_decoder(question, cognition_trace, model, reasoning, bus):
    bus.emit("phase", {"name": "DECODER"})
    bus.emit("status", f"Decoding with {model}…")
    messages = [
        {"role": "system", "content": DECODER_SYSTEM_PROMPT},
        {"role": "user",
         "content": (f"User's question:\n{question}\n\n"
                    f"Cognition trace:\n{cognition_trace}\n\n"
                    f"Produce the final user-facing answer.")}]
    bus.emit("llm_start", {"phase": "decoder"})

    def on_chunk(d):
        bus.emit("llm_delta", {"phase": "decoder", "delta": d})

    text = call_llm_streaming(messages, model=model,
                              reasoning_effort=reasoning, on_chunk=on_chunk)
    bus.emit("llm_end", {"phase": "decoder"})
    return text
 
def run_inference(question, max_hyp, max_online,
                  cog_model, dec_model, reasoning, bus):
    try:
        bus.emit("status", f"Question: {question}")
        bus.emit("status",
                 f"o44  cognition={cog_model}  decoder={dec_model}  "
                 f"max_hyp={max_hyp}  max_online={max_online}  "
                 f"reasoning={reasoning}")

        plan_text, messages = phase_plan(
            question, max_hyp, max_online, cog_model, reasoning, bus)

        messages = phase_hypotheses(
            plan_text, messages, max_hyp, cog_model, reasoning, bus)

        summary_text = phase_summary(messages, cog_model, reasoning, bus)

        trace = [f"## PLAN\n{plan_text}"]
        for m in messages:
            if m["role"] == "assistant":
                c = m["content"]
                if c != plan_text and c != summary_text:
                    trace.append(c)
        trace.append(f"## SUMMARY\n{summary_text}")
        full_trace = "\n\n---\n\n".join(trace)

        final = phase_decoder(question, full_trace, dec_model,
                              reasoning, bus)

        bus.emit("done", {"answer": final})
    except Exception as e:
        bus.emit("error", f"Inference crashed: {e}")
    finally:
        bus.emit("status", "(ready for next question)")

 

app = Flask(__name__)
bus = EventBus()
worker_thread = None
worker_lock   = threading.Lock()


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>o44 Inference</title>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
<style>
  :root {
    --bg: #0e1116;  --panel: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #7d8590;
    --accent: #58a6ff; --green: #3fb950; --red: #f85149;
    --purple: #d2a8ff; --orange: #ffa657;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Helvetica, Arial, sans-serif; line-height: 1.55;
  }
  .app { max-width: 1200px; margin: 0 auto; padding: 20px; }
  h1 {
    margin: 0 0 4px 0; font-size: 22px;
    display: flex; align-items: center; gap: 10px;
  }
  h1::before { content: "◆"; color: var(--accent); }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 18px; }

  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px; margin-bottom: 14px;
  }
  .settings {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 10px;
  }
  .setting label {
    display: block; font-size: 11px; color: var(--muted);
    margin-bottom: 3px; text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .setting input, .setting select {
    width: 100%; padding: 6px 8px;
    background: var(--bg); border: 1px solid var(--border);
    color: var(--text); border-radius: 4px;
    font-family: inherit; font-size: 13px;
  }
  .setting input:focus, .setting select:focus {
    outline: none; border-color: var(--accent);
  }
  textarea {
    width: 100%; min-height: 70px; padding: 10px;
    background: var(--bg); border: 1px solid var(--border);
    color: var(--text); border-radius: 4px;
    font-family: inherit; font-size: 14px; resize: vertical;
  }
  textarea:focus { outline: none; border-color: var(--accent); }

  button {
    padding: 9px 18px;
    background: var(--accent); color: #0e1116;
    border: none; border-radius: 4px;
    font-weight: 600; cursor: pointer; font-size: 13px;
  }
  button:hover:not(:disabled) { opacity: 0.85; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  button.secondary { background: var(--border); color: var(--text); }
  button.chip {
    background: var(--panel); color: var(--text);
    border: 1px solid var(--border); font-weight: 400;
    font-size: 12px; padding: 4px 10px;
  }
  button.chip:hover:not(:disabled) { border-color: var(--accent); }

  #trace {
    min-height: 300px; max-height: 65vh;
    overflow-y: auto; padding: 10px;
    background: var(--bg); border-radius: 4px;
    border: 1px solid var(--border); font-size: 13.5px;
  }
  .phase-header {
    font-weight: 600; font-size: 12px;
    margin: 14px 0 6px 0; padding: 6px 10px;
    background: rgba(88, 166, 255, 0.1);
    border-left: 3px solid var(--accent);
    border-radius: 0 4px 4px 0;
    text-transform: uppercase; letter-spacing: 1px;
  }
  .phase-header.hyp { border-left-color: var(--purple);
                      background: rgba(210, 168, 255, 0.1); }
  .phase-header.sum { border-left-color: var(--orange);
                      background: rgba(255, 166, 87, 0.1); }
  .phase-header.dec { border-left-color: var(--green);
                      background: rgba(63, 185, 80, 0.1); }
  .hyp-header {
    color: var(--purple); font-size: 12px;
    margin: 8px 0 4px 14px; padding: 3px 8px;
    background: rgba(210, 168, 255, 0.05);
    border-left: 2px solid var(--purple);
    font-family: monospace;
  }
  .msg { margin: 6px 0; padding: 8px 12px; border-radius: 4px; }
  .msg.llm {
    background: rgba(63, 185, 80, 0.05);
    border-left: 2px solid var(--green);
  }
  .msg.llm.is-decoder {
    background: rgba(63, 185, 80, 0.12);
    border: 1px solid var(--green);
    border-left-width: 3px;
  }
  .msg.status { color: var(--muted); font-size: 12.5px; padding: 3px 8px; }
  .msg.ping { color: var(--purple); font-size: 12.5px;
              font-family: monospace; padding: 3px 8px; }
  .msg.code {
    background: rgba(255, 166, 87, 0.05);
    border-left: 2px solid var(--orange);
    padding: 6px 10px; font-size: 12.5px;
  }
  .msg.code_err {
    background: rgba(248, 81, 73, 0.05);
    border-left: 2px solid var(--red);
    padding: 6px 10px; font-size: 12.5px;
  }
  .msg.error {
    background: rgba(248, 81, 73, 0.1);
    border-left: 2px solid var(--red); color: var(--red);
  }
  .msg.hyp-done {
    color: var(--green); font-size: 12.5px;
    padding: 4px 10px; margin-left: 14px; font-family: monospace;
  }
  pre {
    background: #0d1117; border: 1px solid var(--border);
    border-radius: 4px; padding: 8px;
    overflow-x: auto; font-size: 12px; line-height: 1.4;
  }
  code { font-family: "SF Mono", Menlo, Consolas, monospace; }
  pre code { background: transparent; padding: 0; }
  p code, li code {
    background: rgba(110, 118, 129, 0.4);
    padding: 2px 4px; border-radius: 3px; font-size: 90%;
  }
  table { border-collapse: collapse; margin: 8px 0; }
  th, td {
    border: 1px solid var(--border);
    padding: 5px 10px; text-align: left;
  }
  th { background: var(--panel); }
  blockquote {
    border-left: 3px solid var(--border);
    padding-left: 10px; color: var(--muted); margin: 6px 0;
  }
  .running::after {
    content: "▌"; color: var(--accent);
    animation: blink 1s infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }
  .status-bar {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 8px;
  }
  .status-bar strong { font-size: 13px; }
  .status-bar #status {
    color: var(--muted); font-size: 12.5px;
    font-family: monospace;
  }
  .status-bar #status.running  { color: var(--accent); }
  .status-bar #status.complete { color: var(--green); }
  .status-bar #status.error    { color: var(--red); }
  .chip-row {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin-top: 8px;
  }
</style>
</head>
<body>
<div class="app">
  <h1>o44 Inference</h1>
  <div class="subtitle">
    Explicit Intelligence Architecture · EIA + TGT · 5 WM families
    (Symbolic · Classical · Deep · Programs · RL Programs)
  </div>

  <div class="panel">
    <div class="settings">
      <div class="setting">
        <label>Max Hypotheses</label>
        <input type="number" id="max_hyp" value="4" min="1" max="20">
      </div>
      <div class="setting">
        <label>Max Online WMs</label>
        <input type="number" id="max_online" value="1" min="0" max="5">
      </div>
      <div class="setting">
        <label>Cognition Model</label>
        <input type="text" id="cog_model" value="__COG_MODEL__">
      </div>
      <div class="setting">
        <label>Decoder Model</label>
        <input type="text" id="dec_model" value="__DEC_MODEL__">
      </div>
      <div class="setting">
        <label>Reasoning</label>
        <select id="reasoning">
          <option value="none"   selected>none</option>
          <option value="low">low</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
          <option value="max">max</option>
        </select>
      </div>
    </div>
  </div>

  <div class="panel">
    <div class="setting">
      <label>Question / Prompt</label>
      <textarea id="question"
                placeholder="Ask anything. Or pick a demo below."></textarea>
    </div>
    <div style="margin-top: 10px; display: flex; gap: 8px;">
      <button id="submit">Run Inference (Ctrl+Enter)</button>
      <button id="clear" class="secondary">Clear Trace</button>
    </div>
    <div class="chip-row" id="demo_chips"></div>
  </div>

  <div class="panel">
    <div class="status-bar">
      <strong>Trace</strong>
      <span id="status">idle</span>
    </div>
    <div id="trace"></div>
  </div>
</div>

<script>
  const trace = document.getElementById("trace");
  const statusEl  = document.getElementById("status");
  const submitBtn = document.getElementById("submit");
  const questionBox = document.getElementById("question");
  const chipRow = document.getElementById("demo_chips");

  marked.setOptions({ breaks: true, gfm: true,
                      headerIds: false, mangle: false });

  const DEMOS = __DEMOS__;

  DEMOS.forEach(([title, q]) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = title;
    b.onclick = () => { questionBox.value = q; questionBox.focus(); };
    chipRow.appendChild(b);
  });

  let llmBuffer = "", llmDiv = null, renderTimer = null;

  function esc(s) { return String(s).replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
  }[c]));}

  function appendRaw(html, cls) {
    const div = document.createElement("div");
    div.className = "msg " + (cls || "");
    div.innerHTML = html;
    trace.appendChild(div);
    trace.scrollTop = trace.scrollHeight;
    return div;
  }
  function appendPhase(name) {
    const cls = name === "HYPOTHESES" ? "hyp" :
                name === "SUMMARY"    ? "sum" :
                name === "DECODER"    ? "dec" : "";
    const div = document.createElement("div");
    div.className = "phase-header " + cls;
    div.textContent = name;
    trace.appendChild(div);
    trace.scrollTop = trace.scrollHeight;
  }
  function startLLM(phase) {
    flushRender(); llmBuffer = "";
    const isDec = (phase === "decoder");
    llmDiv = document.createElement("div");
    llmDiv.className = "msg llm" + (isDec ? " is-decoder" : "");
    trace.appendChild(llmDiv);
    const cur = document.createElement("span");
    cur.className = "running"; llmDiv.appendChild(cur);
    trace.scrollTop = trace.scrollHeight;
  }
  function scheduleRender() {
    if (renderTimer) clearTimeout(renderTimer);
    renderTimer = setTimeout(doRender, 60);
  }
  function flushRender() {
    if (renderTimer) { clearTimeout(renderTimer); renderTimer = null; }
    doRender();
  }
  function doRender() {
    if (!llmDiv) return;
    const cur = llmDiv.querySelector(".running");
    if (cur) cur.remove();
    llmDiv.innerHTML = marked.parse(llmBuffer);
    const newCur = document.createElement("span");
    newCur.className = "running"; llmDiv.appendChild(newCur);
    trace.scrollTop = trace.scrollHeight;
  }
  function finishLLM() {
    flushRender();
    if (llmDiv) {
      const cur = llmDiv.querySelector(".running");
      if (cur) cur.remove();
    }
    llmDiv = null;
  }
  function setStatus(text, kind) {
    statusEl.textContent = text;
    statusEl.className = kind || "";
  }

  let es = null;
  function connectSSE() {
    if (es) es.close();
    es = new EventSource("/stream");
    es.onmessage = (e) => {
      try { handleEvent(JSON.parse(e.data)); }
      catch (err) { console.error(err); }
    };
    es.onerror = () => setStatus("disconnected", "error");
    es.onopen  = () => setStatus("connected", "");
  }

  function handleEvent(evt) {
    const { type, data } = evt;
    switch (type) {
      case "phase":
        appendPhase(data.name);
        setStatus(data.name.toLowerCase(),
                  data.name === "DECODER" ? "complete" : "running");
        break;
      case "status":
        appendRaw(esc(data), "status"); break;
      case "llm_start":
        startLLM(data.phase); break;
      case "llm_delta":
        llmBuffer += data.delta; scheduleRender(); break;
      case "llm_end":
        finishLLM(); break;
      case "ping":
        appendRaw("🔔 PING → @" + esc(data.name), "ping"); break;
      case "code_result": {
        const ok = data.returncode === 0;
        const head = (ok ? "✓ CODE OK" : "✗ CODE FAILED rc=" + data.returncode) +
                     "  (" + esc(data.script_path || "inline") + ")";
        const body = "stdout:\n" + (data.stdout || "(empty)") +
                     "\nstderr:\n" + (data.stderr || "(empty)");
        appendRaw("<strong>" + head + "</strong><br>" +
                  "<pre><code>" + esc(body) + "</code></pre>",
                  ok ? "code" : "code_err");
        break;
      }
      case "hypothesis_start":
        appendRaw("--- Hypothesis " + data.idx + " / " + data.total +
                  " ---", "hyp-header"); break;
      case "hypothesis_done":
        appendRaw("✓ Hypothesis " + data.idx + " complete", "hyp-done");
        break;
      case "error":
        appendRaw("❌ " + esc(data), "error");
        setStatus("error", "error"); break;
      case "done":
        setStatus("complete", "complete");
        submitBtn.disabled = false; break;
    }
    trace.scrollTop = trace.scrollHeight;
  }

  function submit() {
    const question = questionBox.value.trim();
    if (!question) return;
    submitBtn.disabled = true;
    trace.innerHTML = "";
    llmBuffer = ""; llmDiv = null;
    fetch("/ask", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        question:  question,
        max_hyp:   parseInt(document.getElementById("max_hyp").value),
        max_online:parseInt(document.getElementById("max_online").value),
        cog_model: document.getElementById("cog_model").value,
        dec_model: document.getElementById("dec_model").value,
        reasoning: document.getElementById("reasoning").value,
      })
    })
    .then(r => r.json())
    .then(r => {
      if (r.error) {
        appendRaw("❌ " + esc(r.error), "error");
        submitBtn.disabled = false;
        setStatus("error", "error");
      } else setStatus("starting…", "running");
    })
    .catch(err => {
      appendRaw("❌ " + esc(err.toString()), "error");
      submitBtn.disabled = false;
    });
  }

  document.getElementById("submit").onclick = submit;
  document.getElementById("clear").onclick = () => {
    trace.innerHTML = ""; llmBuffer = ""; llmDiv = null;
    setStatus("idle", "");
  };
  questionBox.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault(); submit();
    }
  });

  connectSSE();
</script>
</body>
</html>
"""

HTML = HTML.replace("__COG_MODEL__", COGNITION_MODEL)
HTML = HTML.replace("__DEC_MODEL__", DECODER_MODEL)
HTML = HTML.replace("__DEMOS__", json.dumps(DEMO_EXAMPLES, ensure_ascii=False))
 

@app.route("/")
def index():
    return Response(HTML.encode("utf-8"),
        headers={"Content-Type": "text/html; charset=utf-8"})

@app.route("/ask", methods=["POST"])
def ask():
    global worker_thread
    body = request.get_json(force=True, silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    with worker_lock:
        if worker_thread and worker_thread.is_alive():
            return jsonify({"error":
                "Inference already running."}), 409

        max_hyp    = int(body.get("max_hyp",    DEFAULT_MAX_HYP))
        max_online = int(body.get("max_online", DEFAULT_MAX_ONLINE))
        cog_model  = body.get("cog_model") or COGNITION_MODEL
        dec_model  = body.get("dec_model") or DECODER_MODEL
        reasoning  = body.get("reasoning") or "none"

        worker_thread = threading.Thread(
            target=run_inference,
            args=(question, max_hyp, max_online,
                  cog_model, dec_model, reasoning, bus),
            daemon=True)
        worker_thread.start()
        return jsonify({"ok": True})


@app.route("/stream")
def stream():
    def gen():
        q = bus.subscribe()
        try:
            yield ("data: " + json.dumps(
                {"type": "status", "data": "connected"},
                ensure_ascii=False) + "\n\n").encode("utf-8")
            while True:
                try:
                    evt = q.get(timeout=30)
                    yield ("data: " +
                           json.dumps(evt, ensure_ascii=False) +
                           "\n\n").encode("utf-8")
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            bus.unsubscribe(q)

    return Response(gen(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Content-Type":      "text/event-stream; charset=utf-8",
        })
 

def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    port = int(os.environ.get("O44_PORT", "5000"))
    host = os.environ.get("O44_HOST", "127.0.0.1")

    print(f"\n  o44 Inference UI:  http://{host}:{port}\n")

    threading.Timer(1.2,
        lambda: webbrowser.open(f"http://{host}:{port}")
    ).start()

    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()