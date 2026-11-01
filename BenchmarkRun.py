 
import os
import sys
import json
import queue
import time
import re
import threading
import subprocess
import webbrowser
import requests

from datetime import datetime, timezone

from flask import Flask, Response, request, jsonify

import WorldModels as wm
import LLM_Rules as lr
 
from Inference import (
    call_llm_streaming,
    EventBus,
    COGNITION_SYSTEM_PROMPT,
    DECODER_SYSTEM_PROMPT,
    MAX_TURNS_PER_HYP,
    _make_helper_module_text,
    _inject_helpers,
    _rewrite_absolute_paths,
)


class QuestionAborted(Exception):
    """Raised inside run_o44_question when the user clicks Skip."""

_PING_RE = re.compile(
    r"@([a-zA-Z0-9_\-\.]+(?::[a-zA-Z0-9_\-\.]+)?)")


def _extract_pings(text):
    out, seen = [], set()
    for m in _PING_RE.finditer(text or ""):
        name = m.group(1).rstrip(".,;:!?")
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out
 
DEFAULT_RETRY_MSG = (
    "You are stuck retrying the same failing approach. STOP. "
    "Try a FUNDAMENTALLY DIFFERENT approach this turn. Do NOT call "
    "any WM whose call already failed. Options: (a) use a different "
    "registered WM you have NOT tried, (b) create a fresh online WM "
    "under a NEW name — do not reuse names that already failed, "
    "(c) solve directly in probe code with no WM at all. "
    "Emit exactly one probe now."
)


def _check_q_abort(cfg):
    ev = cfg.get("q_abort")
    if ev is not None and ev.is_set():
        raise QuestionAborted()


def _drain_injections(cfg, messages):
    q = cfg.get("q_inject")
    if q is None:
        return 0
    n = 0
    while True:
        try:
            msg = q.get_nowait()
            messages.append({"role": "user", "content": msg})
            n += 1
        except queue.Empty:
            break
    return n


OPENROUTER_API_KEY  = lr.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = lr.OPENROUTER_BASE_URL

COGNITION_MODEL = os.environ.get("O44_COGNITION_MODEL",
                                 "nvidia/nemotron-3.5-lightning")
DECODER_MODEL   = os.environ.get("O44_DECODER_MODEL",
                                 "nvidia/nemotron-3.5-lightning")
JUDGE_MODEL     = os.environ.get("O44_JUDGE_MODEL",
                                 "nvidia/nemotron-3.5-lightning")
BASELINE_MODEL  = os.environ.get("O44_BASELINE_MODEL", lr.DEFAULT_MODEL)

DEFAULT_MAX_HYP    = int(os.environ.get("O44_MAX_HYP", "4"))
DEFAULT_MAX_ONLINE = int(os.environ.get("O44_MAX_ONLINE", "2"))
DEFAULT_SCRIPT_TIMEOUT = int(os.environ.get("O44_SCRIPT_TIMEOUT", "300"))

BENCHMARKS_DIR = os.path.join(wm.WM_ROOT, "benchmarks")
os.makedirs(BENCHMARKS_DIR, exist_ok=True)
os.makedirs(wm.PROBES_DIR, exist_ok=True)

 
ONLINE_WM_PROTOCOL = """

------------------------------------------------------------
ONLINE WORLD MODELS  (create or update, mid-run)
------------------------------------------------------------
If a hypothesis genuinely needs a World Model the registry does
not cover, you may CREATE one online. If an existing WM is wrong
or missing capability, you may UPDATE it.

  To create:
      NEW ONLINE WORLD MODEL: <name>
      ```python
      # @wm: <name>
      ...
      ```

  To update:
      UPDATE WORLD MODEL: <name>
      ```python
      # @wm: <name>
      ...
      ```

  The harness will:
    1. save the block to scripts/<name>_train.py
    2. execute it with the WM root as cwd
    3. stream stdout/stderr back to you
    4. register /jsons/<name>.json if the script calls
       wm.save_wm_metadata({...})

  Hard rules:
    - <name> must be a valid Python identifier, snake_case.
    - The script MUST call wm.save_wm_metadata({...}) at the end.
    - Runtime < 30 s, parameters < 1 M.
    - Use RELATIVE paths ("models/x.joblib", never "/models/x").
    - If it fails, READ the traceback and emit a corrected block.
    - Quota: at most {max_online} create/update operations per
      question. Set max_online=0 in the UI to disable entirely.

      
"""
WmApiDiscipline = """

------------------------------------------------------------
WM API DISCIPLINE  (READ THIS BEFORE WRITING ANY PROBE)
------------------------------------------------------------
You are FORBIDDEN from guessing a WM's parameter names.

Rule 1: Before you call any WM, ping `@<name>`. The response now
        includes an `_api` block with the REAL function signatures
        and the REAL `op=` values for any dispatch table. Copy the
        signature verbatim.

Rule 2: If `_api.functions` lists `sum_series(expr: str, **opts)`,
        then you call `_h.call_program("<wm>", op="sum_series",
        expr="<real expression string>")`. Nothing else. No
        `start=`, no `end=`, no `n=`, unless the signature says so.

Rule 3: If `_api.dispatch_tables` shows `_OPS: [...]`, the ONLY
        legal `op=` values are the strings inside that list.

Rule 4: If `@<name>` is still unclear, ping `@src:<name>` for the
        FULL source. Read the function that handles your `op=` and
        copy its keyword names exactly.

Rule 5: If you get a `TypeError`, the error message now contains
        the correct signature. ONE retry using that exact signature.
        Do NOT guess a second variant.

Rule 6: For your NEW online WMs — write the entry function with
        EXPLICIT, well-named kwargs (e.g. `sum_series(expr)`), and
        add each op name to a module-level `_OPS = {...}` dict so
        the auto-introspector can expose them.
"""

def _fill(template: str, **kwargs) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _bench_cognition_prompt(max_hyp, max_online):
    base = _fill(COGNITION_SYSTEM_PROMPT,
                 max_hyp=max_hyp, max_online=max_online)
    return (base
            + _fill(ONLINE_WM_PROTOCOL, max_online=max_online)
            + WmApiDiscipline)
 

JUDGE_SYSTEM_PROMPT = """You are an impartial benchmark judge.

You will receive:
  • QUESTION          — what was asked
  • REFERENCE         — the expected answer
  • RUBRIC            — optional grading guidance (may be empty)
  • CANDIDATE         — the system-under-test's final answer
  • CANDIDATE TRACE   — the system's reasoning / tool-use trace
                        (may be truncated; use it for context only)

Your job: decide whether CANDIDATE is CORRECT, PARTIAL, INCORRECT,
or REFUSED, and give a score in [0, 1].

GRADING RULES
  1. Judge MEANING, not surface form. "The German" == "the German
     national" == "German (4th house)". "2.5" == "5/2". Ignore
     formatting, LaTeX, markdown, and whitespace differences.
  2. Numbers: allow small rounding / tolerance unless the rubric
     explicitly forbids it. 1e-2 relative tolerance is a sane
     default.
  3. Lists / sets: order does not matter unless the rubric says so.
     Extra correct items should not fail a "name all of" question
     unless the rubric says "exactly".
  4. If CANDIDATE is empty, silent, or explicitly says it cannot
     answer → verdict REFUSED, score 0.
  5. If CANDIDATE gets the method right but the final value wrong →
     PARTIAL with a score reflecting how close.
  6. If CANDIDATE is essentially correct but worded differently →
     CORRECT. Do NOT be pedantic.
  7. Do NOT reward verbosity. Do NOT penalise brevity.
  8. If the reference itself is ambiguous, say so in `reason` and
     grade leniently.

Return STRICT JSON ONLY — no prose, no markdown fences:

{
  "verdict": "correct" | "partial" | "incorrect" | "refused",
  "score": <float 0..1>,
  "reason": "<one or two sentences>",
  "matched": ["<claim that matched>", ...],
  "missed":  ["<claim that was missing or wrong>", ...]
}
"""



_HELPERS_WRITTEN = False


def _ensure_helpers():
    global _HELPERS_WRITTEN
    if _HELPERS_WRITTEN:
        return
    text = _make_helper_module_text()
 
    for p in (os.path.join(wm.PROBES_DIR, "wm_helpers.py"),
              os.path.join(wm.WM_ROOT, "wm_helpers.py")):
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass
    _HELPERS_WRITTEN = True
 
def _script_prelude():
    return (
        "# --- harness prelude (auto-injected) ---\n"
        "import os, sys\n"
        f"_HARNESS_ROOT = {wm.WM_ROOT!r}\n"
        "if _HARNESS_ROOT not in sys.path:\n"
        "    sys.path.insert(0, _HARNESS_ROOT)\n"
        "import WorldModels as wm\n"
        "import wm_helpers as _h\n"
        "# --- end prelude ---\n\n"
    )


def _run_script_streaming(code, wm_name, bus, timeout,
                          is_training, abort_event):
    """Execute a probe or a training script with live stdout/stderr."""
    code = _rewrite_absolute_paths(code)
    _ensure_helpers()

    prelude = _script_prelude()

    if is_training and wm_name:
        path = os.path.join(wm.SCRIPTS_DIR, f"{wm_name}_train.py")
        payload = prelude + code
    else:
        path = os.path.join(wm.PROBES_DIR,
                            f"probe_{int(time.time() * 1000)}.py")
        payload = prelude + code

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)
    except Exception as e:
        return {"script_path": path, "stdout": "",
                "stderr": f"FAILED TO WRITE SCRIPT: {e}",
                "returncode": -1, "elapsed": 0.0,
                "timed_out": False, "aborted": False}

    cmd = [sys.executable, "-u", path]
    popen_kwargs = dict(cwd=wm.WM_ROOT,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, bufsize=1,
                        encoding="utf-8", errors="replace")
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NO_WINDOW", 0)

    start = time.time()
    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        return {"script_path": path, "stdout": "",
                "stderr": f"FAILED TO START SUBPROCESS: {e}",
                "returncode": -1, "elapsed": 0.0,
                "timed_out": False, "aborted": False}

    out_buf, err_buf = [], []

    def reader(stream, sink, evt_name):
        try:
            for line in iter(stream.readline, ""):
                sink.append(line)
                bus.emit(evt_name, {"line": line})
                if abort_event and abort_event.is_set():
                    break
        except Exception:
            pass
        finally:
            try: stream.close()
            except Exception: pass

    t_out = threading.Thread(target=reader,
        args=(proc.stdout, out_buf, "subprocess_stdout"), daemon=True)
    t_err = threading.Thread(target=reader,
        args=(proc.stderr, err_buf, "subprocess_stderr"), daemon=True)
    t_out.start(); t_err.start()

    timed_out = aborted = False
    deadline = time.time() + timeout
    while proc.poll() is None:
        if abort_event and abort_event.is_set():
            aborted = True
            try: proc.kill()
            except Exception: pass
            break
        if time.time() > deadline:
            timed_out = True
            try: proc.kill()
            except Exception: pass
            break
        time.sleep(0.1)

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try: proc.kill(); proc.wait()
        except Exception: pass

    t_out.join(timeout=2); t_err.join(timeout=2)
    elapsed = time.time() - start
    rc = proc.returncode if proc.returncode is not None else -1
    stdout = "".join(out_buf)
    stderr = "".join(err_buf)

    if timed_out:
        stderr += f"\n\n[TIMEOUT after {timeout:.0f}s — process killed]"
    if aborted:
        stderr += "\n\n[ABORTED BY USER — process killed]"

    return {"script_path": path, "stdout": stdout, "stderr": stderr,
            "returncode": rc, "elapsed": elapsed,
            "timed_out": timed_out, "aborted": aborted}
 
_CREATE_RE = re.compile(r"NEW\s+ONLINE\s+WORLD\s+MODEL:\s*([A-Za-z_]\w*)",
                        re.I)
_UPDATE_RE = re.compile(r"UPDATE\s+WORLD\s+MODEL:\s*([A-Za-z_]\w*)", re.I)


def _online_op_kind(text, wm_name):
    """Return 'create' | 'update' | None for a given WM name."""
    if not wm_name:
        return None
    for m in _CREATE_RE.finditer(text):
        if m.group(1) == wm_name:
            return "create"
    for m in _UPDATE_RE.finditer(text):
        if m.group(1) == wm_name:
            return "update"
    return None


def _code_feedback(res, is_training, wm_name):
    kind = "TRAINING SCRIPT" if is_training else "PROBE"
    head = f"## {kind} RESULT"
    if wm_name:
        head += f" (WM: `{wm_name}`)"
    ok = res["returncode"] == 0
    parts = [
        head,
        f"- returncode: `{res['returncode']}`",
        f"- elapsed: `{res['elapsed']:.1f}s`",
        f"- timed_out: `{res['timed_out']}`",
        f"- aborted: `{res['aborted']}`",
        f"- script: `{res['script_path']}`",
    ]
    if res["stdout"]:
        parts.append(f"**stdout:**\n```\n{res['stdout'][-4000:]}\n```")
    if res["stderr"]:
        parts.append(f"**stderr:**\n```\n{res['stderr'][-4000:]}\n```")
    if not ok:
        parts.append("⚠️ READ the traceback above, FIX the code, and emit "
                     "a corrected ```python``` block in your next reply.")
    return "\n".join(parts)

 
def list_benchmarks():
    out = []
    for fname in sorted(os.listdir(BENCHMARKS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(BENCHMARKS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            out.append({
                "file": fname,
                "name": obj.get("name", fname[:-5]),
                "description": obj.get("description", ""),
                "n_questions": len(obj.get("questions", [])),
            })
        except Exception as e:
            out.append({"file": fname, "name": fname[:-5],
                        "description": f"<load error: {e}>",
                        "n_questions": 0})
    return out


def load_benchmark(fname):
    path = os.path.join(BENCHMARKS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No benchmark file: {fname}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_sample_benchmark():
    """Drop a small sample benchmark if the folder is empty."""
    if any(f.endswith(".json") for f in os.listdir(BENCHMARKS_DIR)):
        return
    sample = {
        "name": "o44 Starter Suite",
        "description": ("A tiny mixed-domain suite to exercise the "
                        "judge + online-WM pipeline."),
        "questions": [
            {
                "id": "zebra_1",
                "question": ("Solve the classic Einstein / Zebra puzzle: "
                             "five houses in a row, each a different "
                             "colour, with a person of a different "
                             "nationality, a different drink, a different "
                             "cigarette brand, and a different pet. Use "
                             "the 15 standard clues. Who owns the fish?"),
                "answer": "The German owns the fish.",
                "rubric": ("Full credit for 'German' / 'the German'. "
                           "Partial credit for a correct full table with "
                           "the wrong final attribution."),
            },
            {
                "id": "linalg_1",
                "question": ("Compute the eigenvalues of the matrix "
                             "[[3,1,0],[1,2,1],[0,1,3]]. Express them "
                             "symbolically and numerically."),
                "answer": ("Eigenvalues: 2, 2+sqrt(2)≈3.4142, "
                           "2−sqrt(2)≈0.5858."),
                "rubric": ("Accept 2, 2±√2. Numeric forms within 1e-3. "
                           "Symbolic exact form preferred."),
            },
            {
                "id": "blackjack_1",
                "question": ("I have a 16 vs the dealer's 10 in "
                             "blackjack. Should I hit, stand, double, or "
                             "split? Justify with basic strategy."),
                "answer": ("Hit. Basic strategy for a hard 16 vs a "
                           "dealer 10 is to hit."),
                "rubric": ("'Hit' is correct. 'Stand' is a common "
                           "mistake — mark incorrect. Any of "
                           "hit/stand/double/split is a valid answer "
                           "token, but only 'hit' is right here."),
            },
        ],
    }
    with open(os.path.join(BENCHMARKS_DIR, "starter.json"),
              "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

 

def _parse_judge_json(text):
    if not text:
        return {"verdict": "incorrect", "score": 0.0,
                "reason": "empty judge output",
                "matched": [], "missed": []}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"verdict": "incorrect", "score": 0.0,
                "reason": f"judge produced no JSON: {text[:200]}",
                "matched": [], "missed": []}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return {"verdict": "incorrect", "score": 0.0,
                "reason": f"judge JSON parse error: {e}",
                "matched": [], "missed": []}
    obj.setdefault("verdict", "incorrect")
    obj.setdefault("score", 0.0)
    obj.setdefault("reason", "")
    obj.setdefault("matched", [])
    obj.setdefault("missed", [])
    if obj["verdict"] not in ("correct", "partial",
                              "incorrect", "refused"):
        obj["verdict"] = "incorrect"
    try:
        obj["score"] = max(0.0, min(1.0, float(obj["score"])))
    except (TypeError, ValueError):
        obj["score"] = 0.0
    return obj


def run_judge(question, expected, rubric, candidate, trace,
              judge_model, reasoning, bus):
    bus.emit("phase", {"name": "JUDGE"})
    bus.emit("status",
             {"text": f"Judging with {judge_model}…"})

    trace_snip = (trace or "")[-8000:]
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE:\n{expected}\n\n"
        f"RUBRIC:\n{rubric or '(none)'}\n\n"
        f"CANDIDATE:\n{candidate if candidate else '(empty)'}\n\n"
        f"CANDIDATE TRACE (truncated):\n{trace_snip or '(none)'}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    sink = []
    bus.emit("llm_start", {"phase": "judge"})

    def on_chunk(d):
        sink.append(d)
        bus.emit("llm_delta", {"phase": "judge", "delta": d})

    try:
        text = call_llm_streaming(messages, model=judge_model,
                                  reasoning_effort=reasoning,
                                  on_chunk=on_chunk)
    except Exception as e:
        bus.emit("llm_end", {"phase": "judge"})
        bus.emit("error", {"text": f"Judge call failed: {e}"})
        return {"verdict": "incorrect", "score": 0.0,
                "reason": f"judge error: {e}",
                "matched": [], "missed": []}

    bus.emit("llm_end", {"phase": "judge"})
    verdict = _parse_judge_json(text)
    bus.emit("judge_result", verdict)
    return verdict


 
def _call_llm_streaming_stoppable(messages, model,
                                  temperature=0.2,
                                  max_tokens=32000,
                                  reasoning_effort="none",
                                  on_chunk=None,
                                  stop_markers=None):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens,
               "stream": True}
    if reasoning_effort == "low":
        payload["reasoning"] = {"effort": "low"}
        payload["max_tokens"] = max_tokens + 4000
    elif reasoning_effort == "medium":
        payload["reasoning"] = {"effort": "medium"}
        payload["max_tokens"] = max_tokens + 8000
    elif reasoning_effort == "high":
        payload["reasoning"] = {"effort": "high"}
        payload["max_tokens"] = max_tokens + 16000
    elif reasoning_effort == "max":
        payload["reasoning"] = {"effort": "high", "max_tokens": 32000}
        payload["max_tokens"] = max_tokens + 32000

    if stop_markers:
        if isinstance(stop_markers, str):
            stop_markers = [stop_markers]
        stop_markers = [m for m in stop_markers if m]
    else:
        stop_markers = []

    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                 "Content-Type": "application/json"},
        json=payload, stream=True, timeout=600,
    )
    resp.raise_for_status()
    resp.encoding = "utf-8"

    accumulated = ""
    try:
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            body = raw_line[len("data: "):]
            if body.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(body)
                delta = (chunk.get("choices", [{}])[0]
                              .get("delta", {}).get("content"))
            except (json.JSONDecodeError, KeyError,
                    IndexError, TypeError):
                continue
            if not delta:
                continue
            accumulated += delta
            if on_chunk:
                on_chunk(delta)
            if stop_markers and any(m in accumulated
                                    for m in stop_markers):
                break
    finally:
        try: resp.close()
        except Exception: pass

    return accumulated


def _stream_llm(messages, model, reasoning, bus, phase,
                hyp_idx=None, stop_markers=None):
    bus.emit("llm_start", {"phase": phase, "hyp_idx": hyp_idx})

    def on_chunk(d):
        bus.emit("llm_delta",
                 {"phase": phase, "hyp_idx": hyp_idx, "delta": d})

    text = _call_llm_streaming_stoppable(
        messages, model=model, reasoning_effort=reasoning,
        on_chunk=on_chunk, stop_markers=stop_markers)
    bus.emit("llm_end", {"phase": phase, "hyp_idx": hyp_idx})
    return text


def _parse_n_hyp(plan_text, max_hyp):
    m = re.search(r"Number of hypotheses:\s*(\d+)", plan_text or "", re.I)
    n = int(m.group(1)) if m else 1
    return max(1, min(n, max_hyp))
_HYP_TERMINATORS = ("=== HYPOTHESIS COMPLETE ===", "=== COGNITION DONE ===")


def run_o44_question(question, cfg, bus):
    """Full cognition → decoder pipeline with online WM support."""
    t0 = time.time()
    cog_model   = cfg["cog_model"]
    dec_model   = cfg["dec_model"]
    reasoning   = cfg["reasoning"]
    max_hyp     = cfg["max_hyp"]
    max_online  = cfg["max_online"]
    online_on   = cfg["allow_online"] and max_online > 0
    script_to   = cfg["script_timeout"]
    abort_ev    = cfg["abort"]

    created_wms, updated_wms = [], []
    online_ops = 0
    assistant_msgs = []
    cog_trace = ""
    answer = None
    error = None

    sys_prompt = _bench_cognition_prompt(max_hyp, max_online)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",
         "content": f"Question: {question}\n\nBegin Cognition Mode."},
    ]

    try:
       
        bus.emit("phase", {"name": "PLAN"})
        
        plan = _stream_llm(messages, cog_model, reasoning, bus, "plan")
        messages.append({"role": "assistant", "content": plan})
        assistant_msgs.append(plan)

        n_hyp = _parse_n_hyp(plan, max_hyp)
        bus.emit("status",
                 {"text": f"Plan parsed: {n_hyp} hypothesis(es)."})

 
        for h in range(n_hyp):
            _check_q_abort(cfg)
            if abort_ev.is_set():
                raise RuntimeError("aborted")

            bus.emit("hypothesis_start", {"idx": h + 1, "total": n_hyp})



            online_hint = (
                f"Online WM creation is ENABLED "
                f"({max_online - online_ops} slot(s) left)."
                if online_on else
                "Online WM creation is DISABLED for this run."
            )
            messages.append({
                "role": "user",
                "content": (
                    f"Execute hypothesis {h + 1} now. Ping any WMs you "
                    f"need, write python code to call them, and report "
                    f"results. {online_hint} When done, emit "
                    f"=== HYPOTHESIS COMPLETE ==="
                ),
            })

            done = False
            for turn in range(MAX_TURNS_PER_HYP):
                _check_q_abort(cfg)
                if abort_ev.is_set():
                    raise RuntimeError("aborted")

                text = _stream_llm(
                    messages, cog_model, reasoning, bus, "hyp",
                    hyp_idx=h + 1,
                    stop_markers=_HYP_TERMINATORS)
                messages.append({"role": "assistant", "content": text})
                assistant_msgs.append(text)

                
                _check_q_abort(cfg)
                if _drain_injections(cfg, messages) > 0:
                    bus.emit("injection_delivered", {"idx": h + 1})

                 
                for name in _extract_pings(text):
                    try:
                        content = lr.resolve_ping(name)
                        bus.emit("ping", {"name": name})
                        messages.append({
                            "role": "user",
                            "content": (f"## PING RESULT @{name}\n"
                                        f"```\n{content}\n```")})
                    except Exception as e:
                        bus.emit("error",
                                 {"text": f"Ping @{name} failed: {e}"})

               
                for code in lr.extract_code_blocks(text):
                    if abort_ev.is_set():
                        raise RuntimeError("aborted")

                    wm_name = lr.extract_wm_name(code)
                    kind = _online_op_kind(text, wm_name)
                    is_training = kind is not None

                    if is_training and not online_on:
                        bus.emit("error", {"text":
                            f"Online WM '{wm_name}' blocked "
                            f"(disabled or quota reached)."})
                        messages.append({
                            "role": "user",
                            "content": (f"## ONLINE WM BLOCKED\n"
                                        f"Online WM creation is disabled "
                                        f"or the quota ({max_online}) is "
                                        f"exhausted. Solve without it.")})
                        continue

                    if is_training and online_ops >= max_online:
                        bus.emit("error", {"text":
                            f"Online WM quota reached ({max_online})."})
                        messages.append({
                            "role": "user",
                            "content": (f"## ONLINE WM QUOTA REACHED\n"
                                        f"You may not create/update more "
                                        f"WMs this question.")})
                        continue

                    label = (f"training script ({kind})"
                             if is_training else "probe")
                    bus.emit("status", {"text":
                        f"Running {label}: {wm_name or 'inline'}…"})

                    res = _run_script_streaming(
                        code,
                        wm_name if is_training else None,
                        bus, timeout=script_to,
                        is_training=is_training,
                        abort_event=abort_ev)

                    if is_training:
                        online_ops += 1
                        json_path = os.path.join(
                            wm.JSONS_DIR, f"{wm_name}.json")
                        if os.path.exists(json_path):
                            if kind == "create":
                                created_wms.append(wm_name)
                                bus.emit("wm_created",
                                         {"name": wm_name})
                            else:
                                updated_wms.append(wm_name)
                                bus.emit("wm_updated",
                                         {"name": wm_name})
                        else:
                            bus.emit("error", {"text":
                                f"Training script for '{wm_name}' "
                                f"did not register a JSON."})

                    bus.emit("code_result", {
                        "script_path": res["script_path"],
                        "stdout":      res["stdout"],
                        "stderr":      res["stderr"],
                        "returncode":  res["returncode"],
                        "elapsed":     res["elapsed"],
                        "timed_out":   res["timed_out"],
                        "aborted":     res["aborted"],
                        "kind":        "training" if is_training else "probe",
                        "wm_name":     wm_name,
                    })

                    messages.append({
                        "role": "user",
                        "content": _code_feedback(res, is_training,
                                                  wm_name)})

                if any(tok in text for tok in _HYP_TERMINATORS):
                    bus.emit("hypothesis_done", {"idx": h + 1})
                    done = True
                    break

            if not done:
                bus.emit("error", {"text":
                    f"Max turns reached for hypothesis {h + 1}"})

       
        bus.emit("phase", {"name": "SUMMARY"})
        messages.append({
            "role": "user",
            "content": ("Now emit the SUMMARY block: each hypothesis "
                        "with its methods, key results, confidence, "
                        "+ Overall Best Answer paragraph. Then "
                        "=== COGNITION DONE ===")})
        
        _check_q_abort(cfg)
        summary = _stream_llm(
            messages, cog_model, reasoning, bus, "summary",
            stop_markers="=== COGNITION DONE ===")
        if "=== COGNITION DONE ===" in summary:
            head, _, _tail = summary.partition("=== COGNITION DONE ===")
            summary = head.rstrip() + "\n\n=== COGNITION DONE ==="
        messages.append({"role": "assistant", "content": summary})
        assistant_msgs.append(summary)
        _check_q_abort(cfg)

        
        cog_trace = "\n\n---\n\n".join(assistant_msgs)
        bus.emit("phase", {"name": "DECODER"})
        dec_messages = [
            {"role": "system", "content": DECODER_SYSTEM_PROMPT},
            {"role": "user",
             "content": (f"User's question:\n{question}\n\n"
                         f"Cognition trace:\n{cog_trace}\n\n"
                         f"Produce the final user-facing answer.")},
        ]
        answer = _stream_llm(dec_messages, dec_model, reasoning,
                             bus, "decoder")
    except QuestionAborted:
        error = "aborted by user (skip)"
        cog_trace = "\n\n---\n\n".join(assistant_msgs)
    except Exception as e:
        error = str(e)
        cog_trace = "\n\n---\n\n".join(assistant_msgs)

    elapsed = time.time() - t0
    return {
        "answer":       answer,
        "trace":        cog_trace,
        "elapsed":      elapsed,
        "error":        error,
        "created_wms":  created_wms,
        "updated_wms":  updated_wms,
    }


 
def run_baseline_question(question, cfg, bus):
    t0 = time.time()
    bus.emit("phase", {"name": "BASELINE"})
    messages = [{"role": "user", "content": question}]
    sink = []

    def on_chunk(d):
        sink.append(d)
        bus.emit("llm_delta", {"phase": "baseline", "delta": d})

    bus.emit("llm_start", {"phase": "baseline"})
    error = None
    answer = None
    try:
        answer = call_llm_streaming(messages, model=cfg["baseline_model"],
                                    reasoning_effort=cfg["reasoning"],
                                    on_chunk=on_chunk)
    except Exception as e:
        error = str(e)
    bus.emit("llm_end", {"phase": "baseline"})

    return {
        "answer":       answer,
        "trace":        "".join(sink),
        "elapsed":      time.time() - t0,
        "error":        error,
        "created_wms":  [],
        "updated_wms":  [],
    }

 
app = Flask(__name__)
bus = EventBus()

RUN_LOCK = threading.Lock()
STATE = {
    "running": False,
    "abort":   threading.Event(),
    "question_abort":  None,
    "question_inject": None,
    "results": [],
    "worker":  None,
    "cfg":     {},
    "benchmark": None,
}


def _run_loop(benchmark_obj, questions, cfg):
    results = STATE["results"]
    abort_ev = cfg["abort"]

    bus.emit("run_start", {
        "mode":       cfg["mode"],
        "benchmark":  benchmark_obj.get("name", "?"),
        "n_questions": len(questions),
        "cfg":        {k: v for k, v in cfg.items() if k != "abort"},
    })

    try:
        for i, q in enumerate(questions):
            if abort_ev.is_set():
                bus.emit("status", {"text": "⛔ Aborted."})
                break

            qid = str(q.get("id", i))

            # Fresh per-question control state
            cfg["q_abort"]  = threading.Event()
            cfg["q_inject"] = queue.Queue()
            STATE["question_abort"]  = cfg["q_abort"]
            STATE["question_inject"] = cfg["q_inject"]

            bus.emit("question_start",
                     {"idx": i + 1, "total": len(questions), "qid": qid})
            bus.emit("status", {"text":
                f"[{i+1}/{len(questions)}] {qid} — running…"})

            try:
                if cfg["mode"] == "o44":
                    run_res = run_o44_question(q.get("question", ""),
                                               cfg, bus)
                else:
                    run_res = run_baseline_question(q.get("question", ""),
                                                    cfg, bus)
            except Exception as e:
                run_res = {"answer": None, "trace": "",
                           "elapsed": 0.0, "error": str(e),
                           "created_wms": [], "updated_wms": []}

            skipped = cfg["q_abort"].is_set()

            if abort_ev.is_set():
                break

            # ---- Judge ----
            candidate = run_res["answer"]
            if skipped or not candidate or not str(candidate).strip():
                verdict = {
                    "verdict": "skipped" if skipped else "refused",
                    "score": 0.0,
                    "reason": ("skipped by user"
                               if skipped else
                               "no candidate answer produced"),
                    "matched": [], "missed": []}
                bus.emit("judge_result", verdict)
            else:
                verdict = run_judge(
                    question=q.get("question", ""),
                    expected=str(q.get("answer", "")),
                    rubric=q.get("rubric", "") or q.get("notes", ""),
                    candidate=str(candidate),
                    trace=run_res["trace"],
                    judge_model=cfg["judge_model"],
                    reasoning=cfg["reasoning"],
                    bus=bus)

            row = {
                "qid":          qid,
                "question":     q.get("question", ""),
                "expected":     str(q.get("answer", "")),
                "rubric":       q.get("rubric", ""),
                "notes":        q.get("notes", ""),
                "answer":       candidate,
                "verdict":      verdict["verdict"],
                "score":        verdict["score"],
                "reason":       verdict["reason"],
                "matched":      verdict["matched"],
                "missed":       verdict["missed"],
                "elapsed":      run_res["elapsed"],
                "error":        run_res["error"],
                "trace":        run_res["trace"],
                "created_wms":  run_res["created_wms"],
                "updated_wms":  run_res["updated_wms"],
            }
            results.append(row)
            bus.emit("question_done", row)
            STATE["question_abort"]  = None
            STATE["question_inject"] = None

    except Exception as e:
        bus.emit("error", {"text": f"Run crashed: {e}"})
    finally:
        n_ok      = sum(1 for r in results if r["verdict"] == "correct")
        n_partial = sum(1 for r in results if r["verdict"] == "partial")
        n_bad     = sum(1 for r in results if r["verdict"] == "incorrect")
        n_ref     = sum(1 for r in results if r["verdict"] == "refused")
        bus.emit("run_end", {
            "correct":   n_ok,
            "partial":   n_partial,
            "incorrect": n_bad,
            "refused":   n_ref,
            "total":     len(results),
        })
        with RUN_LOCK:
            STATE["running"] = False
 
def build_report(results, cfg, benchmark_name):
    lines = []
    lines.append("=" * 72)
    lines.append(f"o44 BENCHMARK REPORT — {benchmark_name}")
    lines.append("=" * 72)
    lines.append(f"Generated:       {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Mode:            {cfg.get('mode','?').upper()}")
    lines.append(f"Judge model:     {cfg.get('judge_model','?')}")
    if cfg.get("mode") == "o44":
        lines.append(f"Cognition model: {cfg.get('cog_model','?')}")
        lines.append(f"Decoder model:   {cfg.get('dec_model','?')}")
        lines.append(f"Online WMs:      "
                     f"{'ENABLED' if cfg.get('allow_online') else 'disabled'}"
                     f" (max {cfg.get('max_online', 0)})")
    else:
        lines.append(f"Baseline model:  {cfg.get('baseline_model','?')}")
    lines.append(f"Reasoning:       {cfg.get('reasoning','?')}")

    total   = len(results)
    n_ok    = sum(1 for r in results if r["verdict"] == "correct")
    n_part  = sum(1 for r in results if r["verdict"] == "partial")
    n_bad   = sum(1 for r in results if r["verdict"] == "incorrect")
    n_ref   = sum(1 for r in results if r["verdict"] == "refused")
    score   = (sum(r["score"] for r in results) / total) if total else 0.0

    lines.append(f"Total questions: {total}")
    lines.append(f"Correct:         {n_ok}")
    lines.append(f"Partial:         {n_part}")
    lines.append(f"Incorrect:       {n_bad}")
    lines.append(f"Refused:         {n_ref}")
    lines.append(f"Accuracy:        "
                 f"{(n_ok / total if total else 0.0):.1%}")
    lines.append(f"Mean score:      {score:.3f}")
    lines.append(f"Total elapsed:   "
                 f"{sum(r['elapsed'] for r in results):.1f}s")
    lines.append("")

    lines.append("-" * 72)
    lines.append("PER-QUESTION RESULTS")
    lines.append("-" * 72)
    for r in results:
        badge = {"correct": "✓", "partial": "◐",
                 "incorrect": "✗", "refused": "⊘"}[r["verdict"]]
        lines.append(f"\n[{r['qid']}] {badge} {r['verdict'].upper()}  "
                     f"(score={r['score']:.2f}, {r['elapsed']:.1f}s)")
        lines.append(f"  Q:        {r['question']}")
        lines.append(f"  Expected: {r['expected']}")
        if r["answer"]:
            ans = r["answer"]
            if len(ans) > 600:
                ans = ans[:600] + "…"
            lines.append(f"  Answer:   {ans}")
        else:
            lines.append(f"  Answer:   <NONE>"
                         + (f" ({r['error']})" if r['error'] else ""))
        lines.append(f"  Judge:    {r['reason']}")
        if r["matched"]:
            lines.append(f"  Matched:  {r['matched']}")
        if r["missed"]:
            lines.append(f"  Missed:   {r['missed']}")
        if r["created_wms"]:
            lines.append(f"  + WMs:    {r['created_wms']}")
        if r["updated_wms"]:
            lines.append(f"  ↻ WMs:    {r['updated_wms']}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF REPORT")
    lines.append("=" * 72)
    return "\n".join(lines)

 
@app.route("/")
def index():
    return Response(HTML.encode("utf-8"),
        headers={"Content-Type": "text/html; charset=utf-8"})


@app.route("/benchmarks")
def route_benchmarks():
    return jsonify(list_benchmarks())


@app.route("/benchmark")
def route_benchmark():
    fname = request.args.get("file", "")
    try:
        return jsonify(load_benchmark(fname))
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/run", methods=["POST"])
def route_run():
    body = request.get_json(force=True, silent=True) or {}
    fname = body.get("benchmark")
    qids  = body.get("qids") or []

    try:
        obj = load_benchmark(fname)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    all_q = obj.get("questions", [])
    if qids:
        wanted = set(str(x) for x in qids)
        questions = [q for q in all_q
                     if str(q.get("id", "")) in wanted]
    else:
        questions = all_q

    if not questions:
        return jsonify({"error": "No questions selected."}), 400

    with RUN_LOCK:
        if STATE["running"]:
            return jsonify({"error":
                "A run is already in progress."}), 409

        cfg = {
            "mode":           body.get("mode", "o44"),
            "cog_model":      body.get("cog_model") or COGNITION_MODEL,
            "dec_model":      body.get("dec_model") or DECODER_MODEL,
            "judge_model":    body.get("judge_model") or JUDGE_MODEL,
            "baseline_model": body.get("baseline_model") or BASELINE_MODEL,
            "reasoning":      body.get("reasoning", "none"),
            "max_hyp":        int(body.get("max_hyp", DEFAULT_MAX_HYP)),
            "max_online":     int(body.get("max_online", DEFAULT_MAX_ONLINE)),
            "allow_online":   bool(body.get("allow_online", True)),
            "script_timeout": int(body.get("script_timeout",
                                           DEFAULT_SCRIPT_TIMEOUT)),
            "abort":          threading.Event(),
        }

        STATE["running"]   = True
        STATE["abort"]     = cfg["abort"]
        STATE["cfg"]       = cfg
        STATE["benchmark"] = obj.get("name", fname)

        STATE["worker"] = threading.Thread(
            target=_run_loop, args=(obj, questions, cfg), daemon=True)
        STATE["worker"].start()

    return jsonify({"ok": True, "n_questions": len(questions)})


@app.route("/abort", methods=["POST"])
def route_abort():
    with RUN_LOCK:
        if STATE["running"]:
            STATE["abort"].set()
            return jsonify({"ok": True})
    return jsonify({"ok": False, "reason": "not running"})

@app.route("/skip_question", methods=["POST"])
def route_skip_question():
    with RUN_LOCK:
        ev = STATE.get("question_abort")
        if STATE["running"] and ev is not None:
            ev.set()
            return jsonify({"ok": True})
    return jsonify({"ok": False, "reason": "not running"})


@app.route("/try_something_else", methods=["POST"])
def route_try_something_else():
    body = request.get_json(force=True, silent=True) or {}
    msg = (body.get("message") or "").strip() or DEFAULT_RETRY_MSG
    with RUN_LOCK:
        q = STATE.get("question_inject")
        if STATE["running"] and q is not None:
            q.put(msg)
            return jsonify({"ok": True, "message": msg})
    return jsonify({"ok": False, "reason": "not running"})

@app.route("/results")
def route_results():
    return jsonify({
        "results": STATE["results"],
        "running": STATE["running"],
    })


@app.route("/clear", methods=["POST"])
def route_clear():
    with RUN_LOCK:
        if STATE["running"]:
            return jsonify({"error": "cannot clear while running"}), 409
        STATE["results"] = []
    return jsonify({"ok": True})


@app.route("/report")
def route_report():
    results = STATE["results"]
    if not results:
        return jsonify({"error": "no results"}), 400
    report = build_report(results, STATE["cfg"],
                          STATE["benchmark"] or "(unknown)")
    return Response(report, mimetype="text/plain; charset=utf-8")


@app.route("/stream")
def route_stream():
    def gen():
        q = bus.subscribe()
        try:
            yield ("data: " + json.dumps(
                {"type": "status", "data": "connected"},
                ensure_ascii=False) + "\n\n").encode("utf-8")
            while True:
                try:
                    evt = q.get(timeout=30)
                    yield ("data: " + json.dumps(evt,
                                                 ensure_ascii=False)
                           + "\n\n").encode("utf-8")
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

 
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>o44 Benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
<style>
  :root {
    --bg:#0e1116; --panel:#161b22; --border:#30363d;
    --text:#e6edf3; --muted:#7d8590;
    --accent:#58a6ff; --green:#3fb950; --red:#f85149;
    --purple:#d2a8ff; --orange:#ffa657; --yellow:#e3b341;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",
                     Helvetica,Arial,sans-serif; line-height:1.55; }
  .app { max-width:1700px; margin:0 auto; padding:16px; }
  h1 { margin:0 0 4px 0; font-size:22px;
       display:flex; align-items:center; gap:10px; }
  h1::before { content:"◆"; color:var(--accent); }
  .subtitle { color:var(--muted); font-size:13px; margin-bottom:14px; }

  .panel { background:var(--panel); border:1px solid var(--border);
           border-radius:8px; padding:12px; margin-bottom:12px; }

  .settings { display:grid;
              grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
              gap:10px; }
  .setting label { display:block; font-size:11px; color:var(--muted);
                   margin-bottom:3px; text-transform:uppercase;
                   letter-spacing:0.5px; }
  .setting input, .setting select {
    width:100%; padding:6px 8px; background:var(--bg);
    border:1px solid var(--border); color:var(--text);
    border-radius:4px; font-family:inherit; font-size:13px; }
  .setting input:focus, .setting select:focus {
    outline:none; border-color:var(--accent); }
  .setting.checkbox { display:flex; align-items:center; gap:6px;
                      padding-top:16px; }
  .setting.checkbox input { width:auto; }
  .setting.checkbox label { text-transform:none; letter-spacing:0;
                            margin:0; font-size:13px; color:var(--text); }

  .shell { display:grid; grid-template-columns:330px 1fr; gap:12px; }

  button {
    padding:8px 14px; background:var(--accent); color:#0e1116;
    border:none; border-radius:4px; font-weight:600;
    cursor:pointer; font-size:13px; }
  button:hover:not(:disabled) { opacity:0.85; }
  button:disabled { opacity:0.4; cursor:not-allowed; }
  button.secondary { background:var(--border); color:var(--text); }
  button.danger { background:var(--red); color:#fff; }

  .bench-list { display:flex; flex-direction:column; gap:4px;
                max-height:150px; overflow-y:auto; }
  .bench-item { padding:6px 9px; border-radius:4px; cursor:pointer;
                font-size:12.5px; border:1px solid transparent; }
  .bench-item:hover { background:rgba(88,166,255,0.08); }
  .bench-item.active { background:rgba(88,166,255,0.15);
                       border-color:var(--accent); }
  .bench-item .n { color:var(--muted); font-size:11px; }

  .q-list { display:flex; flex-direction:column; gap:3px;
            max-height:340px; overflow-y:auto; margin-top:6px; }
  .q-item { display:flex; gap:7px; padding:5px 7px; border-radius:4px;
            font-size:12.5px; cursor:pointer; align-items:flex-start; }
  .q-item:hover { background:rgba(88,166,255,0.06); }
  .q-item input { margin-top:3px; }
  .q-item .qtext { flex:1; }
  .q-item .qid { color:var(--purple); font-family:monospace;
                 font-size:11px; }
  .q-item .qtype { color:var(--muted); font-size:11px; }

  .trace { min-height:280px; max-height:52vh; overflow-y:auto;
           padding:10px; background:var(--bg); border-radius:4px;
           border:1px solid var(--border); font-size:13.5px; }

  .phase-header { font-weight:600; font-size:12px; margin:14px 0 6px 0;
                  padding:6px 10px; background:rgba(88,166,255,0.1);
                  border-left:3px solid var(--accent);
                  border-radius:0 4px 4px 0; text-transform:uppercase;
                  letter-spacing:1px; }
  .phase-header.hyp { border-left-color:var(--purple);
                      background:rgba(210,168,255,0.1); }
  .phase-header.sum { border-left-color:var(--orange);
                      background:rgba(255,166,87,0.1); }
  .phase-header.dec { border-left-color:var(--green);
                      background:rgba(63,185,80,0.1); }
  .phase-header.jdg { border-left-color:var(--yellow);
                      background:rgba(227,179,65,0.1); }
  .phase-header.q   { border-left-color:var(--accent);
                      background:rgba(88,166,255,0.15); font-size:13px; }

  .msg { margin:6px 0; padding:8px 12px; border-radius:4px; }
  .msg.llm { background:rgba(63,185,80,0.05);
             border-left:2px solid var(--green); }
  .msg.llm.is-decoder { background:rgba(63,185,80,0.12);
                        border:1px solid var(--green);
                        border-left-width:3px; }
  .msg.llm.is-judge { background:rgba(227,179,65,0.06);
                      border-left:2px solid var(--yellow); }
  .msg.status { color:var(--muted); font-size:12.5px; padding:3px 8px; }
  .msg.ping { color:var(--purple); font-size:12.5px;
              font-family:monospace; padding:3px 8px; }
  .msg.code { background:rgba(255,166,87,0.05);
              border-left:2px solid var(--orange);
              padding:6px 10px; font-size:12.5px; }
  .msg.code_err { background:rgba(248,81,73,0.05);
                  border-left:2px solid var(--red);
                  padding:6px 10px; font-size:12.5px; }
  .msg.error { background:rgba(248,81,73,0.1);
               border-left:2px solid var(--red); color:var(--red); }
  .msg.hyp-done { color:var(--green); font-size:12.5px;
                  padding:4px 10px; margin-left:14px;
                  font-family:monospace; }
  .msg.wm { background:rgba(210,168,255,0.12);
            border-left:2px solid var(--purple); padding:6px 10px;
            font-size:12.5px; font-family:monospace; }
  .msg.judge { background:rgba(227,179,65,0.08);
               border-left:2px solid var(--yellow); padding:8px 12px; }

  pre { background:#0d1117; border:1px solid var(--border);
        border-radius:4px; padding:8px; overflow-x:auto;
        font-size:12px; line-height:1.4; }
  code { font-family:"SF Mono",Menlo,Consolas,monospace; }
  pre code { background:transparent; padding:0; }
  p code, li code { background:rgba(110,118,129,0.4);
                    padding:2px 4px; border-radius:3px; font-size:90%; }
  table { border-collapse:collapse; margin:8px 0; }
  th, td { border:1px solid var(--border); padding:5px 10px;
           text-align:left; }
  th { background:var(--panel); }
  blockquote { border-left:3px solid var(--border);
               padding-left:10px; color:var(--muted); margin:6px 0; }

  .running::after { content:"▌"; color:var(--accent);
                    animation:blink 1s infinite; }
  @keyframes blink { 50% { opacity:0; } }

  .status-bar { display:flex; justify-content:space-between;
                align-items:center; margin-bottom:8px; }
  .status-bar strong { font-size:13px; }
  .status-bar #status { color:var(--muted); font-size:12.5px;
                        font-family:monospace; }
  .status-bar #status.running  { color:var(--accent); }
  .status-bar #status.complete { color:var(--green); }
  .status-bar #status.error    { color:var(--red); }

  .results-table { width:100%; border-collapse:collapse;
                   font-size:12.5px; }
  .results-table th { background:var(--panel); position:sticky; top:0;
                      cursor:pointer; user-select:none; }
  .results-table td { padding:5px 8px; border-bottom:1px solid var(--border);
                      vertical-align:top; }
  .results-table tr { cursor:pointer; }
  .results-table tr:hover { background:rgba(88,166,255,0.06); }
  .results-table tr.sel { background:rgba(88,166,255,0.12); }
  .badge { display:inline-block; padding:1px 7px; border-radius:10px;
           font-size:11px; font-weight:600; }
  .badge.correct   { background:rgba(63,185,80,0.2); color:var(--green); }
  .badge.partial   { background:rgba(227,179,65,0.2); color:var(--yellow); }
  .badge.incorrect { background:rgba(248,81,73,0.2); color:var(--red); }
  .badge.refused   { background:rgba(125,133,144,0.25); color:var(--muted); }

  .tab-row { display:flex; gap:4px; margin-bottom:8px; }
  .tab { padding:6px 14px; border-radius:4px 4px 0 0; cursor:pointer;
         font-size:13px; background:transparent; color:var(--muted);
         border:1px solid transparent; border-bottom:none; }
  .tab.active { background:var(--panel); color:var(--text);
                border-color:var(--border); }
  .tab-body { display:none; }
  .tab-body.active { display:block; }

  .report-box { width:100%; min-height:320px; max-height:52vh;
                background:var(--bg); color:var(--text);
                border:1px solid var(--border); border-radius:4px;
                font-family:"SF Mono",Menlo,Consolas,monospace;
                font-size:12px; padding:10px; white-space:pre-wrap;
                overflow-y:auto; }

  .summary-chips { display:flex; gap:8px; flex-wrap:wrap;
                   margin-bottom:8px; }
  .chip { padding:4px 12px; border-radius:14px; font-size:12px;
          font-weight:600; }
  .chip.correct   { background:rgba(63,185,80,0.2); color:var(--green); }
  .chip.partial   { background:rgba(227,179,65,0.2); color:var(--yellow); }
  .chip.incorrect { background:rgba(248,81,73,0.2); color:var(--red); }
  .chip.refused   { background:rgba(125,133,144,0.25); color:var(--muted); }
  .chip.score     { background:rgba(88,166,255,0.2); color:var(--accent); }

  .detail { padding:10px; background:var(--bg); border-radius:4px;
            border:1px solid var(--border); font-size:13px;
            max-height:52vh; overflow-y:auto; margin-top:8px; }
  .detail h3 { margin:8px 0 4px 0; font-size:13px;
               color:var(--accent); }
  .detail .lbl { color:var(--muted); font-size:11px;
                 text-transform:uppercase; letter-spacing:0.5px; }
</style>
</head>
<body>
<div class="app">
  <h1>o44 Benchmark</h1>
  <div class="subtitle">
    LLM-judged · full cognition trace · online World Models
  </div>

  <!-- ============ SETTINGS ============ -->
  <div class="panel">
    <div class="settings">
      <div class="setting">
        <label>Mode</label>
        <select id="mode">
          <option value="o44" selected>o44 (full pipeline)</option>
          <option value="baseline">baseline LLM</option>
        </select>
      </div>
      <div class="setting">
        <label>Cognition model</label>
        <input id="cog_model" value="__COG_MODEL__">
      </div>
      <div class="setting">
        <label>Decoder model</label>
        <input id="dec_model" value="__DEC_MODEL__">
      </div>
      <div class="setting">
        <label>Judge model</label>
        <input id="judge_model" value="__JUDGE_MODEL__">
      </div>
      <div class="setting">
        <label>Baseline model</label>
        <input id="baseline_model" value="__BASELINE_MODEL__">
      </div>
      <div class="setting">
        <label>Reasoning</label>
        <select id="reasoning">
          <option value="none" selected>none</option>
          <option value="low">low</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
          <option value="max">max</option>
        </select>
      </div>
      <div class="setting">
        <label>Max hypotheses</label>
        <input type="number" id="max_hyp" value="__MAX_HYP__" min="1" max="20">
      </div>
      <div class="setting">
        <label>Max online WMs</label>
        <input type="number" id="max_online" value="__MAX_ONLINE__" min="0" max="10">
      </div>
      <div class="setting">
        <label>Script timeout (s)</label>
        <input type="number" id="script_timeout" value="__SCRIPT_TIMEOUT__"
               min="10" max="3600">
      </div>
      <div class="setting checkbox">
        <input type="checkbox" id="allow_online" checked>
        <label for="allow_online">Allow online WM creation</label>
      </div>
    </div>
  </div>

  <!-- ============ SHELL ============ -->
  <div class="shell">

    <!-- LEFT: benchmarks + questions -->
    <div>
      <div class="panel">
        <div style="display:flex; justify-content:space-between;
                    align-items:center; margin-bottom:6px;">
          <strong style="font-size:13px;">Benchmarks</strong>
          <button class="secondary" id="refresh_bench"
                  style="padding:3px 8px; font-size:11px;">↻</button>
        </div>
        <div class="bench-list" id="bench_list"></div>
      </div>

      <div class="panel">
        <div style="display:flex; justify-content:space-between;
                    align-items:center; margin-bottom:4px;">
          <strong style="font-size:13px;">Questions</strong>
          <div style="display:flex; gap:4px;">
            <button class="secondary" id="sel_all"
                    style="padding:2px 7px; font-size:11px;">all</button>
            <button class="secondary" id="sel_none"
                    style="padding:2px 7px; font-size:11px;">none</button>
          </div>
        </div>
        <div class="q-list" id="q_list"></div>
      </div>

      <div class="panel">
        <div style="display:flex; flex-direction:column; gap:6px;">
          <button id="run_selected">▶ Run Selected</button>
          <button id="run_all">▶ Run All</button>
          <button class="danger" id="abort_btn">⛔ Abort</button>
          <button class="secondary" id="skip_q_btn">⏭ Skip question</button>
          <button class="secondary" id="retry_btn">🔄 Try something else</button>
          <button class="secondary" id="clear_btn">🗑 Clear Results</button>
        </div>
      </div>
    </div>

    <!-- RIGHT: trace + results + report -->
    <div>
      <div class="panel">
        <div class="status-bar">
          <strong>Live trace</strong>
          <span id="status">idle</span>
        </div>
        <div class="trace" id="trace"></div>
      </div>

      <div class="panel">
        <div class="tab-row">
          <div class="tab active" data-tab="results">Results</div>
          <div class="tab" data-tab="detail">Detail</div>
          <div class="tab" data-tab="report">Report</div>
        </div>

        <div class="tab-body active" id="tab-results">
          <div class="summary-chips" id="summary_chips"></div>
          <div style="max-height:44vh; overflow-y:auto;">
            <table class="results-table" id="results_table">
              <thead>
                <tr>
                  <th data-sort="idx">#</th>
                  <th data-sort="qid">ID</th>
                  <th data-sort="verdict">Verdict</th>
                  <th data-sort="score">Score</th>
                  <th data-sort="elapsed">Time</th>
                  <th>Answer</th>
                  <th>Judge reason</th>
                </tr>
              </thead>
              <tbody id="results_body"></tbody>
            </table>
          </div>
        </div>

        <div class="tab-body" id="tab-detail">
          <div class="detail" id="detail_box">
            <em style="color:var(--muted);">
              Select a row in the Results tab to see the full trace.
            </em>
          </div>
        </div>

        <div class="tab-body" id="tab-report">
          <div style="display:flex; gap:6px; margin-bottom:8px;">
            <button class="secondary" id="gen_report">📊 Generate</button>
            <button class="secondary" id="save_report">💾 Save .txt</button>
          </div>
          <div class="report-box" id="report_box"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  marked.setOptions({ breaks:true, gfm:true, headerIds:false,
                      mangle:false });

  // ---------- state ----------
  let BENCHMARKS = [];
  let CURRENT    = null;      // {file, name, questions}
  let SELECTED   = new Set(); // question ids
  let RESULTS    = [];
  let SORT_KEY   = "idx", SORT_ASC = true;

  // ---------- dom ----------
  const $ = id => document.getElementById(id);
  const traceEl   = $("trace");
  const statusEl  = $("status");
  const benchList = $("bench_list");
  const qList     = $("q_list");
  const resultsBody = $("results_body");
  const detailBox = $("detail_box");
  const reportBox = $("report_box");
  const summaryChips = $("summary_chips");

  // ---------- helpers ----------
  function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
    }[c]));
  }
  function setStatus(text, kind) {
    statusEl.textContent = text;
    statusEl.className = kind || "";
  }

  // ---------- tabs ----------
  document.querySelectorAll(".tab").forEach(t => {
    t.onclick = () => {
      document.querySelectorAll(".tab").forEach(x =>
        x.classList.remove("active"));
      document.querySelectorAll(".tab-body").forEach(x =>
        x.classList.remove("active"));
      t.classList.add("active");
      $("tab-" + t.dataset.tab).classList.add("active");
    };
  });

  // ---------- benchmark list ----------
  async function loadBenchmarks() {
    const r = await fetch("/benchmarks");
    BENCHMARKS = await r.json();
    benchList.innerHTML = "";
    if (!BENCHMARKS.length) {
      benchList.innerHTML =
        "<em style='color:var(--muted); font-size:12px;'>" +
        "No benchmarks. Drop JSON files into ./benchmarks/</em>";
      return;
    }
    BENCHMARKS.forEach(b => {
      const div = document.createElement("div");
      div.className = "bench-item";
      div.innerHTML =
        `<div>${esc(b.name)}</div>` +
        `<div class="n">${b.n_questions} questions · ${esc(b.file)}</div>`;
      div.onclick = () => selectBenchmark(b.file, div);
      benchList.appendChild(div);
    });
  }

  async function selectBenchmark(file, el) {
    document.querySelectorAll(".bench-item").forEach(x =>
      x.classList.remove("active"));
    if (el) el.classList.add("active");
    const r = await fetch("/benchmark?file=" + encodeURIComponent(file));
    const obj = await r.json();
    if (obj.error) { alert(obj.error); return; }
    CURRENT = { file, name: obj.name, questions: obj.questions || [] };
    SELECTED = new Set(CURRENT.questions.map(q => String(q.id)));
    renderQuestions();
  }

  function renderQuestions() {
    qList.innerHTML = "";
    if (!CURRENT) {
      qList.innerHTML =
        "<em style='color:var(--muted); font-size:12px;'>" +
        "(select a benchmark)</em>";
      return;
    }
    CURRENT.questions.forEach(q => {
      const qid = String(q.id);
      const row = document.createElement("label");
      row.className = "q-item";
      row.innerHTML =
        `<input type="checkbox" ${SELECTED.has(qid) ? "checked" : ""}>` +
        `<span class="qtext"><span class="qid">[${esc(qid)}]</span> ` +
        `${esc((q.question || "").slice(0, 110))}` +
        `${q.rubric ? '<div class="qtype">rubric ✓</div>' : ''}</span>`;
      row.querySelector("input").onchange = e => {
        if (e.target.checked) SELECTED.add(qid);
        else SELECTED.delete(qid);
      };
      qList.appendChild(row);
    });
  }

  $("sel_all").onclick  = () => { if (CURRENT) { SELECTED =
      new Set(CURRENT.questions.map(q => String(q.id))); renderQuestions(); } };
  $("sel_none").onclick = () => { SELECTED = new Set(); renderQuestions(); };
  $("refresh_bench").onclick = loadBenchmarks;

  // ---------- run ----------
  async function startRun(all) {
    if (!CURRENT) { alert("Select a benchmark first."); return; }
    const qids = all
      ? CURRENT.questions.map(q => String(q.id))
      : Array.from(SELECTED);
    if (!qids.length) { alert("No questions selected."); return; }

    const body = {
      benchmark:      CURRENT.file,
      qids:           qids,
      mode:           $("mode").value,
      cog_model:      $("cog_model").value,
      dec_model:      $("dec_model").value,
      judge_model:    $("judge_model").value,
      baseline_model: $("baseline_model").value,
      reasoning:      $("reasoning").value,
      max_hyp:        parseInt($("max_hyp").value),
      max_online:     parseInt($("max_online").value),
      allow_online:   $("allow_online").checked,
      script_timeout: parseInt($("script_timeout").value),
    };

    const r = await fetch("/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (j.error) { alert(j.error); return; }
    setStatus("running…", "running");
  }

  $("run_selected").onclick = () => startRun(false);
  $("run_all").onclick      = () => startRun(true);
  $("abort_btn").onclick    = async () => {
    await fetch("/abort", {method:"POST"});
    setStatus("aborting…", "error");
  };
  $("clear_btn").onclick    = async () => {
    const r = await fetch("/clear", {method:"POST"});
    const j = await r.json();
    if (j.error) { alert(j.error); return; }
    RESULTS = [];
    renderResults();
    traceEl.innerHTML = "";
    reportBox.textContent = "";
    detailBox.innerHTML =
      "<em style='color:var(--muted);'>Select a row…</em>";
  };
  $("skip_q_btn").onclick = async () => {
    const r = await fetch("/skip_question", {method:"POST"});
    const j = await r.json();
    if (!j.ok) alert("Not running.");
    else setStatus("skipping current question…", "error");
  };

  $("retry_btn").onclick = async () => {
    const msg = prompt(
      "Message to inject into the current question's loop " +
      "(leave blank for the default 'try a different approach'):",
      "");
    if (msg === null) return;   // user cancelled
    const r = await fetch("/try_something_else", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: msg}),
    });
    const j = await r.json();
    if (!j.ok) alert("Not running.");
    else setStatus("injected — model will see it on next turn", "running");
  };
  
  // ---------- report ----------
  $("gen_report").onclick = async () => {
    const r = await fetch("/report");
    if (!r.ok) { alert("No results yet."); return; }
    reportBox.textContent = await r.text();
  };
  $("save_report").onclick = async () => {
    const r = await fetch("/report");
    if (!r.ok) { alert("No results yet."); return; }
    const text = await r.text();
    const blob = new Blob([text], {type:"text/plain;charset=utf-8"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `o44_benchmark_${Date.now()}.txt`;
    a.click();
  };

  // ---------- results table ----------
  function verdictBadge(v) {
    return `<span class="badge ${v}">${v}</span>`;
  }

  function renderResults() {
    // summary chips
    const n = RESULTS.length;
    const c = {correct:0, partial:0, incorrect:0, refused:0};
    RESULTS.forEach(r => c[r.verdict] = (c[r.verdict] || 0) + 1);
    const mean = n ? (RESULTS.reduce((a,r)=>a+r.score,0)/n) : 0;
    summaryChips.innerHTML =
      `<span class="chip correct">✓ ${c.correct}</span>` +
      `<span class="chip partial">◐ ${c.partial}</span>` +
      `<span class="chip incorrect">✗ ${c.incorrect}</span>` +
      `<span class="chip refused">⊘ ${c.refused}</span>` +
      `<span class="chip score">mean ${mean.toFixed(2)}</span>`;

    // sorted rows
    const rows = RESULTS.map((r, i) => ({...r, idx: i+1}));
    rows.sort((a,b) => {
      const av = a[SORT_KEY], bv = b[SORT_KEY];
      if (av < bv) return SORT_ASC ? -1 : 1;
      if (av > bv) return SORT_ASC ?  1 : -1;
      return 0;
    });

    resultsBody.innerHTML = "";
    rows.forEach(r => {
      const tr = document.createElement("tr");
      const ans = r.answer ? r.answer.slice(0, 140) : "<none>";
      tr.innerHTML =
        `<td>${r.idx}</td>` +
        `<td>${esc(r.qid)}</td>` +
        `<td>${verdictBadge(r.verdict)}</td>` +
        `<td>${r.score.toFixed(2)}</td>` +
        `<td>${r.elapsed.toFixed(1)}s</td>` +
        `<td>${esc(ans)}</td>` +
        `<td>${esc(r.reason || "")}</td>`;
      tr.onclick = () => {
        document.querySelectorAll("#results_body tr").forEach(x =>
          x.classList.remove("sel"));
        tr.classList.add("sel");
        showDetail(r);
      };
      resultsBody.appendChild(tr);
    });
  }

  function showDetail(r) {
    let html = "";
    html += `<h3>Question [${esc(r.qid)}]</h3><div>${esc(r.question)}</div>`;
    html += `<h3>Reference answer</h3><div>${esc(r.expected)}</div>`;
    if (r.rubric) html += `<h3>Rubric</h3><div>${esc(r.rubric)}</div>`;
    html += `<h3>Model answer</h3><div>${esc(r.answer || "<none>")}</div>`;
    html += `<h3>Judge</h3><div>` +
            `${verdictBadge(r.verdict)} score=${r.score.toFixed(2)}` +
            `<div style="margin-top:4px;">${esc(r.reason)}</div>`;
    if (r.matched && r.matched.length)
      html += `<div class="lbl" style="margin-top:6px;">matched</div>` +
              `<ul>${r.matched.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>`;
    if (r.missed && r.missed.length)
      html += `<div class="lbl" style="margin-top:6px;">missed</div>` +
              `<ul>${r.missed.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>`;
    html += `</div>`;
    if (r.created_wms && r.created_wms.length)
      html += `<h3>WMs created</h3><div>${r.created_wms.join(", ")}</div>`;
    if (r.updated_wms && r.updated_wms.length)
      html += `<h3>WMs updated</h3><div>${r.updated_wms.join(", ")}</div>`;
    if (r.error)
      html += `<h3>Error</h3><div style="color:var(--red);">${esc(r.error)}</div>`;
    if (r.trace)
      html += `<h3>Full cognition trace</h3>` +
              `<pre><code>${esc(r.trace)}</code></pre>`;
    detailBox.innerHTML = html;

    // switch to detail tab
    document.querySelectorAll(".tab").forEach(x =>
      x.classList.remove("active"));
    document.querySelectorAll(".tab-body").forEach(x =>
      x.classList.remove("active"));
    document.querySelector('[data-tab="detail"]').classList.add("active");
    $("tab-detail").classList.add("active");
  }

  document.querySelectorAll("th[data-sort]").forEach(th => {
    th.onclick = () => {
      const k = th.dataset.sort;
      if (SORT_KEY === k) SORT_ASC = !SORT_ASC;
      else { SORT_KEY = k; SORT_ASC = true; }
      renderResults();
    };
  });

  // ---------- SSE trace ----------
  let llmBuffer = "", llmDiv = null, renderTimer = null;

  function appendRaw(html, cls) {
    const div = document.createElement("div");
    div.className = "msg " + (cls || "");
    div.innerHTML = html;
    traceEl.appendChild(div);
    traceEl.scrollTop = traceEl.scrollHeight;
    return div;
  }
  function appendPhase(name, cls) {
    const div = document.createElement("div");
    div.className = "phase-header " + (cls || "");
    div.textContent = name;
    traceEl.appendChild(div);
    traceEl.scrollTop = traceEl.scrollHeight;
  }
  function startLLM(phase) {
    flushRender(); llmBuffer = "";
    const cls = phase === "decoder" ? " is-decoder"
              : phase === "judge"   ? " is-judge" : "";
    llmDiv = document.createElement("div");
    llmDiv.className = "msg llm" + cls;
    traceEl.appendChild(llmDiv);
    const cur = document.createElement("span");
    cur.className = "running"; llmDiv.appendChild(cur);
    traceEl.scrollTop = traceEl.scrollHeight;
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
    const nc = document.createElement("span");
    nc.className = "running"; llmDiv.appendChild(nc);
    traceEl.scrollTop = traceEl.scrollHeight;
  }
  function finishLLM() {
    flushRender();
    if (llmDiv) {
      const cur = llmDiv.querySelector(".running");
      if (cur) cur.remove();
    }
    llmDiv = null;
  }

  let es = null;
  function connectSSE() {
    if (es) es.close();
    es = new EventSource("/stream");
    es.onmessage = e => {
      try { handleEvent(JSON.parse(e.data)); }
      catch (err) { console.error(err); }
    };
    es.onerror = () => setStatus("disconnected", "error");
    es.onopen  = () => setStatus("connected", "");
  }

  function handleEvent(evt) {
    const { type, data } = evt;
    switch (type) {
      case "run_start":
        traceEl.innerHTML = "";
        setStatus("running", "running");
        appendPhase(`RUN — ${data.benchmark} (${data.n_questions} Q, ` +
                    `mode=${data.mode})`, "q");
        break;
      case "question_start":
        appendPhase(`QUESTION ${data.idx}/${data.total} — ${data.qid}`, "q");
        break;
      case "question_done":
        RESULTS.push(data);
        renderResults();
        break;
      case "run_end":
        setStatus(`done  (✓${data.correct} ◐${data.partial} ` +
                  `✗${data.incorrect} ⊘${data.refused})`, "complete");
        break;
      case "phase": {
        const cls = data.name === "HYPOTHESES" ? "hyp"
                  : data.name === "SUMMARY"    ? "sum"
                  : data.name === "DECODER"    ? "dec"
                  : data.name === "JUDGE"      ? "jdg" : "";
        appendPhase(data.name, cls);
        break;
      }
      case "status":
        appendRaw(esc(typeof data === "string" ? data : data.text),
                  "status");
        break;
      case "llm_start":
        startLLM(data.phase); break;
      case "llm_delta":
        llmBuffer += data.delta; scheduleRender(); break;
      case "llm_end":
        finishLLM(); break;
      case "ping":
        appendRaw("🔔 PING → @" + esc(data.name), "ping"); break;
      case "hypothesis_start":
        appendRaw(`--- Hypothesis ${data.idx} / ${data.total} ---`,
                  "hyp-header"); break;
      case "hypothesis_done":
        appendRaw(`✓ Hypothesis ${data.idx} complete`, "hyp-done"); break;
      case "subprocess_stdout":
        appendRaw(`<pre><code>${esc(data.line)}</code></pre>`, "code");
        break;
      case "subprocess_stderr":
        appendRaw(`<pre><code>${esc(data.line)}</code></pre>`, "code_err");
        break;
      case "code_result": {
        const ok = data.returncode === 0;
        const head = (ok ? "✓" : "✗") +
          ` ${data.kind.toUpperCase()}` +
          (data.wm_name ? ` [${esc(data.wm_name)}]` : "") +
          ` rc=${data.returncode} (${data.elapsed.toFixed(1)}s)`;
        let body = "";
        if (data.stdout) body += "stdout:\n" + data.stdout + "\n";
        if (data.stderr) body += "stderr:\n" + data.stderr;
        if (!body) body = "(no output)";
        appendRaw(`<strong>${head}</strong><br>` +
                  `<pre><code>${esc(body)}</code></pre>`,
                  ok ? "code" : "code_err");
        break;
      }
      case "wm_created":
        appendRaw(`★ WM CREATED: ${esc(data.name)}`, "wm"); break;
      case "wm_updated":
        appendRaw(`↻ WM UPDATED: ${esc(data.name)}`, "wm"); break;
      case "judge_result":
        appendRaw(
          `<strong>JUDGE → ${verdictBadge(data.verdict)} ` +
          `score=${(data.score || 0).toFixed(2)}</strong><br>` +
          `${esc(data.reason || "")}` +
          (data.matched && data.matched.length
            ? `<div style="margin-top:4px;"><span class="lbl">matched</span> ` +
              data.matched.map(esc).join(", ") + `</div>` : "") +
          (data.missed && data.missed.length
            ? `<div style="margin-top:4px;"><span class="lbl">missed</span> ` +
              data.missed.map(esc).join(", ") + `</div>` : ""),
          "judge");
        break;
      case "error":
        appendRaw("❌ " + esc(typeof data === "string" ? data : data.text),
                  "error");
        break;
    }
  }

  // ---------- boot ----------
  loadBenchmarks();
  connectSSE();
</script>
</body>
</html>
"""

HTML = HTML.replace("__COG_MODEL__",       COGNITION_MODEL)
HTML = HTML.replace("__DEC_MODEL__",       DECODER_MODEL)
HTML = HTML.replace("__JUDGE_MODEL__",     JUDGE_MODEL)
HTML = HTML.replace("__BASELINE_MODEL__",  BASELINE_MODEL)
HTML = HTML.replace("__MAX_HYP__",         str(DEFAULT_MAX_HYP))
HTML = HTML.replace("__MAX_ONLINE__",      str(DEFAULT_MAX_ONLINE))
HTML = HTML.replace("__SCRIPT_TIMEOUT__",  str(DEFAULT_SCRIPT_TIMEOUT))


 
def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    ensure_sample_benchmark()

    port = int(os.environ.get("O44_BENCH_PORT", "5001"))
    host = os.environ.get("O44_BENCH_HOST", "127.0.0.1")

    print(f"\n  o44 Benchmark UI:  http://{host}:{port}\n"
          f"  Benchmarks dir:    {BENCHMARKS_DIR}\n"
          f"  Judge model:       {JUDGE_MODEL}\n")

    threading.Timer(1.2,
        lambda: webbrowser.open(f"http://{host}:{port}")
    ).start()

    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()