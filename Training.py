 
import os
import sys
import json
import queue
import threading
import subprocess
import time
import re
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox

import WorldModels as wm
import LLM_Rules as lr
 
PATH_REWRITE_RE = re.compile(
    r'''(["\'])/(models|datasets|jsons|scripts)(["'/])''')
DEFAULT_SCRIPT_TIMEOUT = int(os.environ.get("O44_SCRIPT_TIMEOUT", "300"))

TRAINING_MODES = ("questions", "tgt")

 

def _rewrite_absolute_paths(code: str) -> str:
    return PATH_REWRITE_RE.sub(r"\1\2\3", code)


def stream_training_script(
    code: str,
    wm_name: str = None,
    on_event=None,
    abort_event: threading.Event = None,
    timeout: float = DEFAULT_SCRIPT_TIMEOUT,
) -> dict:
    code = _rewrite_absolute_paths(code)
    abort_event = abort_event or threading.Event()

    if wm_name:
        script_path = os.path.join(wm.SCRIPTS_DIR, f"{wm_name}_train.py")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            return {"script_path": script_path,
                    "stdout": "",
                    "stderr": f"FAILED TO WRITE SCRIPT FILE: {e}",
                    "returncode": -1, "elapsed": 0.0,
                    "timed_out": False, "aborted": False,
                    "timeout": timeout}
        cmd = [sys.executable, "-u", script_path]
    else:
        script_path = None
        cmd = [sys.executable, "-u", "-c", code]

    cwd = wm.WM_ROOT
    start = time.time()

    popen_kwargs = dict(cwd=cwd, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, bufsize=1)
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NO_WINDOW", 0)

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        return {"script_path": script_path, "stdout": "",
                "stderr": f"FAILED TO START SUBPROCESS: {e}",
                "returncode": -1, "elapsed": 0.0,
                "timed_out": False, "aborted": False, "timeout": timeout}

    full_stdout, full_stderr = [], []

    def reader(stream, sink, evt_name):
        try:
            for line in iter(stream.readline, ""):
                sink.append(line)
                if on_event:
                    try: on_event(evt_name, line)
                    except Exception: pass
                if abort_event.is_set():
                    break
        except Exception as e:
            sink.append(f"[reader error: {e}]\n")
        finally:
            try: stream.close()
            except Exception: pass

    t_out = threading.Thread(target=reader,
        args=(proc.stdout, full_stdout, "subprocess_stdout"), daemon=True)
    t_err = threading.Thread(target=reader,
        args=(proc.stderr, full_stderr, "subprocess_stderr"), daemon=True)
    t_out.start(); t_err.start()

    timed_out = aborted = False
    deadline = time.time() + timeout
    while proc.poll() is None:
        if abort_event.is_set():
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

    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try: proc.kill(); proc.wait()
        except Exception: pass

    t_out.join(timeout=2); t_err.join(timeout=2)
    elapsed = time.time() - start
    rc = proc.returncode if proc.returncode is not None else -1
    stdout = "".join(full_stdout); stderr = "".join(full_stderr)

    if timed_out:
        stderr += (f"\n\n[TIMEOUT after {timeout:.0f}s — process killed]"
                   f"\n[Common: infinite loop, slow import (torch), "
                   f"blocking I/O.]")
    if aborted:
        stderr += "\n\n[ABORTED BY USER — process killed]"

    return {"script_path": script_path, "stdout": stdout,
            "stderr": stderr, "returncode": rc, "elapsed": elapsed,
            "timed_out": timed_out, "aborted": aborted,
            "timeout": timeout}


def open_folder(path):
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')


 
def _summarize_traceback(stderr: str) -> str:
    if not stderr.strip():
        return "(no stderr)"
    idx = stderr.rfind("Traceback (most recent call last):")
    return stderr[idx:].strip() if idx != -1 else stderr.strip()


def _diagnose_error(stderr: str) -> str:
    s = stderr.lower()
    hints = []
    if "can't multiply sequence by non-int of type 'numpy.float" in s:
        hints.append("• numpy scalar can't multiply a list/sequence.\n"
                     "  FIX: `np.array(seq)` first OR `float(scalar)`.")
    if "can't multiply sequence by non-int of type 'float'" in s:
        hints.append("• A plain Python float can't multiply a list. "
                     "Cast both sides to np.array(...) or build floats.")
    if "operands could not be broadcast together" in s:
        hints.append("• numpy broadcast failed — shapes mismatch. "
                     "Use np.broadcast_arrays or np.ndindex loop.")
    if "name '" in s and "is not defined" in s:
        hints.append("• Undefined name. Make sure everything is imported.")
    if "modulenotfounderror" in s or "no module named" in s:
        hints.append("• Missing module. Add to `dependencies` JSON OR "
                     "use only stdlib + sklearn + torch + gymnasium + "
                     "stable_baselines3.")
    if "filenotfounderror" in s or "no such file or directory" in s:
        hints.append("• Wrong path. Use RELATIVE ('models/x.joblib'), "
                     "NEVER absolute ('/models/x.joblib').")
    if "syntaxerror" in s:
        hints.append("• Syntax error. Check indentation, brackets, f-strings.")
    if "importerror" in s:
        hints.append("• Import error — probably circular or wrong path.")
    if "valueerror" in s and "shape" in s:
        hints.append("• Shape mismatch. Check tensor shapes / train_test_split.")
    if "permissionerror" in s:
        hints.append("• Permission denied — path likely uses filesystem "
                     "root. Switch to RELATIVE paths.")
    if "typeerror: unsupported operand type" in s:
        hints.append("• Type mismatch in arithmetic. Cast both sides "
                     "(`float()`, `np.array()`).")
    if "attributeerror" in s and "predict" in s:
        hints.append("• SB3 model attribute error — make sure you "
                     "instantiate `model = PPO(...)` and call "
                     "`model.learn(...)` to train before `.predict()`.")
    if "gymnasium" in s and "not found" in s:
        hints.append("• Gymnasium env not found. Check spelling, "
                     "and make sure gym is installed, not just gym.")
    return "\n".join(hints) if hints else \
        "(no specific hint available — read the traceback)"


def build_feedback(ping_results, code_results, new_wms,
                    turn, max_turns):
    parts = []

    if ping_results:
        parts.append("## PING RESULTS")
        for name, content in ping_results.items():
            trimmed = content if len(content) <= 4000 \
                else content[:4000] + "\n... (truncated)"
            parts.append(f"### @{name}\n```\n{trimmed}\n```")

    if code_results:
        parts.append("## SCRIPT EXECUTION RESULTS")
        for i, r in enumerate(code_results):
            wm_name = r.get("wm_name") or "(inline)"
            ok = r["returncode"] == 0
            status = "✓ SUCCESS" if ok else "✗ FAILED"
            parts.append(
                f"### Script {i+1}: `{wm_name}` — {status}\n"
                f"- returncode: `{r['returncode']}`\n"
                f"- elapsed: {r.get('elapsed', 0):.1f}s\n"
                f"- timed_out: `{r.get('timed_out', False)}`\n"
                f"- aborted:  `{r.get('aborted', False)}`\n"
                f"- script_path: `{r.get('script_path')}`\n")
            if r.get("timed_out"):
                parts.append("⚠️ **YOUR SCRIPT TIMED OUT — process killed.**")
            if r.get("aborted"):
                parts.append("\n⛔ **USER ABORTED.**\n")
            if not ok:
                tb = _summarize_traceback(r["stderr"])
                parts.append(f"**Traceback:**\n```\n{tb}\n```\n"
                             f"**Hints:**\n{_diagnose_error(r['stderr'])}\n")
            elif r["stdout"].strip():
                parts.append(f"\n**stdout (last 2k chars):**\n```\n"
                             f"{r['stdout'][-2000:]}\n```\n")

    if new_wms:
        parts.append("## NEW WMS DETECTED IN /jsons")
        parts.extend(f"- {n}" for n in new_wms)

    if not parts:
        parts.append("(No action detected. If finished, emit === DONE ===. "
                     "Otherwise, ping @harness or write code.)")

    if code_results and any(r["returncode"] != 0 for r in code_results):
        remaining = max_turns - turn - 1
        parts.append(f"\n## ⚠️ REQUIRED ACTION\n"
                     f"Your script(s) failed above. READ the traceback + "
                     f"hints. Emit CORRECTED ```python``` block(s) (with "
                     f"`# @wm: <name>` headers) in your next reply.\n"
                     f"DO NOT emit `=== DONE ===` until at least one new "
                     f"JSON lands in /jsons/ for this conversation.\n"
                     f"(Turn {turn + 1} / {max_turns}, "
                     f"{remaining} attempt(s) remaining.)")

    return "\n\n".join(parts)

 
MAX_TURNS_PER_WM = 8


def build_wm_prompt(question, wm_idx, max_wms, existing_wms):
    existing_list = "\n".join(f"  - {n}" for n in existing_wms) \
                    or "  (none yet)"
    return f"""Question: {question}

You are building world model #{wm_idx + 1} of {max_wms} for this question.

Currently registered in /jsons (do NOT duplicate any of these):
{existing_list}

Workflow (one WM per conversation):
  1. (Optional) @harness to refresh on standards.
  2. (Optional) @<name> to inspect an existing WM.
  3. Plan your new WM.
  4. Write ONE training script as a ```python``` block tagged with
     `# @wm: <name>`.
  5. The harness saves and executes it.
  6. When the WM JSON appears in /jsons/, emit `=== DONE ===`.

Build WM #{wm_idx + 1} now."""


def run_one_wm(question, wm_idx, max_wms, model, reasoning,
                on_event, abort_event, script_timeout):
    existing_wms = list(wm.list_wms())
    prompt = build_wm_prompt(question, wm_idx, max_wms, existing_wms)
    history = [{"role": "user", "content": prompt}]

    for turn in range(MAX_TURNS_PER_WM):
        if abort_event.is_set():
            on_event("status", "Aborted by user before turn.")
            return False

        on_event("status", f"Turn {turn + 1}/{MAX_TURNS_PER_WM}: calling LLM...")
        on_event("llm_start")

        def emit_chunk(d): on_event("llm_delta", d)

        try:
            assistant_text = lr.call_llm_streaming(
                history, on_chunk=emit_chunk, model=model,
                reasoning_effort=reasoning,
            )
        except Exception as e:
            on_event("llm_end", "")
            on_event("error", f"LLM call failed: {e}")
            return False

        on_event("llm_end", assistant_text)
        if not assistant_text:
            on_event("error",
                     "Empty response from LLM. Bump max_tokens or simplify.")
            return False
        history.append({"role": "assistant", "content": assistant_text})

        ping_results = {}
        for name in lr.extract_pings(assistant_text):
            if abort_event.is_set(): break
            try:
                ping_results[name] = lr.resolve_ping(name)
                on_event("ping", name)
            except Exception as e:
                ping_results[name] = f"ERROR: {e}"
                on_event("error", ping_results[name])

        code_results = []
        for code in lr.extract_code_blocks(assistant_text):
            if abort_event.is_set(): break
            wm_name = lr.extract_wm_name(code)
            on_event("status", f"Executing {wm_name or 'inline script'} "
                               f"(timeout {script_timeout}s)...")
            result = stream_training_script(code, wm_name, on_event,
                                            abort_event, timeout=script_timeout)
            code_results.append({"wm_name": wm_name, **result})
            on_event("code_result", result)

        current = set(wm.list_wms())
        new = sorted(current - set(existing_wms))
        for n in new: on_event("wm_created", n)
        if new: return True

        if lr.is_done(assistant_text):
            on_event("status", "=== DONE === rejected — no new WM in "
                               "/jsons/. Code must have failed.")
            history.append({"role": "user", "content": (
                "You emitted === DONE === but no new JSON appeared in "
                "/jsons/. That means your training script failed.\n\n"
                + build_feedback(ping_results, code_results, new,
                                  turn, MAX_TURNS_PER_WM))})
            continue

        feedback = build_feedback(ping_results, code_results, new,
                                   turn, MAX_TURNS_PER_WM)
        history.append({"role": "user", "content": feedback})
        on_event("feedback", feedback[:2000] + ("..." if len(feedback) > 2000 else ""))

    on_event("status", f"Max turns ({MAX_TURNS_PER_WM}) reached for "
                       f"WM #{wm_idx + 1}.")
    return False
 

MAX_TURNS_PER_TRIGGER = 24  


def build_trigger_prompt(trigger, max_wms, existing_wms):
    existing_list = "\n".join(f"  - {n}" for n in existing_wms) \
                    or "  (none yet)"
    return f"""TGT TRIGGER:
{trigger}

You are in TRIGGERED GEOMETRY TRAINING mode. Extract up to {max_wms}
distinct World Models relevant to this trigger.

Currently registered in /jsons (do NOT duplicate):
{existing_list}

Workflow:
  1. (Optional) @harness to refresh on standards.
  2. @jsons to inspect what's already there.
  3. Plan a BATCH of distinct WMs (different families / sub-skills).
  4. Emit ONE ```python``` block per WM, each with `# @wm: <name>`.
  5. The harness saves & executes each in turn.
  6. Read stdout/stderr / failure feedback on next turn; FIX and re-emit.
  7. When you've produced all the WM JSONs you can, emit `=== DONE ===`.
     (DONE is rejected if NO new JSONs landed — you MUST register at
     least one WM in this conversation.)

Begin extraction now."""


def run_tgt_trigger(trigger, model, reasoning,
                     on_event, abort_event, script_timeout,
                     max_wms_per_trigger):
    existing_wms = list(wm.list_wms())
    prompt = build_trigger_prompt(trigger, max_wms_per_trigger,
                                   existing_wms)
    history = [{"role": "user", "content": prompt}]

    initial_count = len(existing_wms)

    for turn in range(MAX_TURNS_PER_TRIGGER):
        if abort_event.is_set():
            on_event("status", "Aborted by user before turn.")
            return []

        on_event("status",
                 f"[TGT] Turn {turn + 1}/{MAX_TURNS_PER_TRIGGER}: calling LLM…")
        on_event("llm_start")

        def emit_chunk(d): on_event("llm_delta", d)

        try:
            assistant_text = lr.call_llm_streaming(
                history, on_chunk=emit_chunk, model=model,
                reasoning_effort=reasoning,
                system_prompt=lr.build_tgt_training_system_prompt(
                    max_wms_per_trigger),
            )
        except Exception as e:
            on_event("llm_end", "")
            on_event("error", f"LLM call failed: {e}")
            return []

        on_event("llm_end", assistant_text)
        if not assistant_text:
            return []
        history.append({"role": "assistant", "content": assistant_text})

        # Pings
        ping_results = {}
        for name in lr.extract_pings(assistant_text):
            if abort_event.is_set(): break
            try:
                ping_results[name] = lr.resolve_ping(name)
                on_event("ping", name)
            except Exception as e:
                ping_results[name] = f"ERROR: {e}"
                on_event("error", ping_results[name])

        # Code blocks
        code_results = []
        for code in lr.extract_code_blocks(assistant_text):
            if abort_event.is_set(): break
            wm_name = lr.extract_wm_name(code)
            on_event("status", f"Executing {wm_name or 'inline'} "
                               f"(timeout {script_timeout}s)…")
            result = stream_training_script(code, wm_name, on_event,
                                            abort_event, timeout=script_timeout)
            code_results.append({"wm_name": wm_name, **result})
            on_event("code_result", result)

        # New WMs?
        current = set(wm.list_wms())
        new = sorted(current - set(existing_wms))
        for n in new: on_event("wm_created", n)

        # DONE?
        if lr.is_done(assistant_text):
            if not new:
                on_event("status",
                         "=== DONE === rejected — no new WM in /jsons/.")
                history.append({"role": "user", "content": (
                    "You emitted === DONE === but no new JSON appeared in "
                    "/jsons/. Code must have failed. Fix and re-emit.\n\n"
                    + build_feedback(ping_results, code_results, new,
                                      turn, MAX_TURNS_PER_TRIGGER))})
                continue
            on_event("status",
                     f"[TGT] Done. WMs produced: {new}")
            return new

        # If hit the per-trigger quota, force DONE
        if len(new) >= max_wms_per_trigger:
            on_event("status",
                     f"[TGT] Per-trigger cap "
                     f"({max_wms_per_trigger}) reached.")
            return new

        feedback = build_feedback(ping_results, code_results, new,
                                   turn, MAX_TURNS_PER_TRIGGER)
        history.append({"role": "user", "content": feedback})
        on_event("feedback",
                 feedback[:2000] + ("..." if len(feedback) > 2000 else ""))

    on_event("status",
             f"[TGT] Max turns ({MAX_TURNS_PER_TRIGGER}) reached.")
    return list(set(wm.list_wms()) - set(existing_wms))


 
def run_training(prompt_body, mode, max_wms, model, reasoning,
                  on_event, abort_event, script_timeout):
    on_event("status",
             f"o44 TRAINER — mode={mode.upper()}   "
             f"prompt={prompt_body[:80]}{'...' if len(prompt_body)>80 else ''}")
    on_event("status",
             f"max_wms={max_wms}   model={model}   "
             f"reasoning={reasoning}   script_timeout={script_timeout}s")

    if mode == "tgt":
        return _run_tgt(prompt_body, max_wms, model, reasoning,
                         on_event, abort_event, script_timeout)
    return _run_questions(prompt_body, max_wms, model, reasoning,
                           on_event, abort_event, script_timeout)


def _run_questions(question, max_wms, model, reasoning,
                    on_event, abort_event, script_timeout):
    wms_created = []
    for wm_idx in range(max_wms):
        if abort_event.is_set():
            on_event("status", "Aborted by user."); break
        on_event("status", f"=== Building WM {wm_idx + 1}/{max_wms} ===")
        before = set(wm.list_wms())
        success = run_one_wm(question, wm_idx, max_wms, model, reasoning,
                              on_event, abort_event, script_timeout)
        if not success:
            on_event("status", "Stopping early.")
            break
        wms_created.extend(sorted(set(wm.list_wms()) - before))
    on_event("status", f"Done. WMs created this run: {wms_created}")
    return {"wms_created": wms_created}


def _run_tgt(trigger, max_wms, model, reasoning,
              on_event, abort_event, script_timeout):
    wms = run_tgt_trigger(trigger, model, reasoning,
                           on_event, abort_event, script_timeout,
                           max_wms)
    on_event("status", f"Done. WMs created this run: {wms}")
    return {"wms_created": wms}


 
class CantoTrainerGUI:
    POLL_MS = 80

    def __init__(self, root):
        self.root = root
        self.root.title("o44 Trainer")
        self.root.geometry("1350x900")

        self._live_active = False
        self.abort_event = threading.Event()
        self.worker_thread = None

        self._build_params(root)
        self._build_chat(root)
        self._build_input(root)
        self._build_abort(root)

        self.queue = queue.Queue()
        self.root.after(self.POLL_MS, self._drain_queue)

    def _build_params(self, parent):
        bar = ttk.Frame(parent, padding=5)
        bar.pack(fill=tk.X)

        ttk.Label(bar, text="Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="tgt")
        ttk.Combobox(bar, textvariable=self.mode_var, width=10,
                     values=TRAINING_MODES, state="readonly"
                     ).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(bar, text="Max WMs:").pack(side=tk.LEFT)
        max_for_mode = 10 if True else 3   # both default
        self.max_wms_var = tk.IntVar(value=max_for_mode)
        ttk.Spinbox(bar, from_=1, to=20, width=5,
                    textvariable=self.max_wms_var
                    ).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(bar, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=lr.DEFAULT_MODEL)
        ttk.Entry(bar, textvariable=self.model_var, width=32
                  ).pack(side=tk.LEFT, padx=5)

        ttk.Label(bar, text="Reasoning:").pack(side=tk.LEFT, padx=(15, 0))
        self.reasoning_var = tk.StringVar(value="none")
        ttk.Combobox(bar, textvariable=self.reasoning_var, width=7,
                     values=["none", "low", "medium", "high", "max"],
                     state="readonly").pack(side=tk.LEFT, padx=5)

        ttk.Label(bar, text="Script timeout (s):").pack(
            side=tk.LEFT, padx=(15, 0))
        self.timeout_var = tk.IntVar(value=DEFAULT_SCRIPT_TIMEOUT)
        ttk.Spinbox(bar, from_=30, to=3600, width=6,
                    textvariable=self.timeout_var).pack(side=tk.LEFT, padx=5)

        ttk.Button(bar, text="Open /jsons",
                   command=lambda: open_folder(wm.JSONS_DIR)
                   ).pack(side=tk.RIGHT, padx=3)
        ttk.Button(bar, text="Open /scripts",
                   command=lambda: open_folder(wm.SCRIPTS_DIR)
                   ).pack(side=tk.RIGHT, padx=3)
        ttk.Button(bar, text="Open /models",
                   command=lambda: open_folder(wm.MODELS_DIR)
                   ).pack(side=tk.RIGHT, padx=3)

    def _build_chat(self, parent):
        self.chat = scrolledtext.ScrolledText(parent, wrap=tk.WORD,
                                              state=tk.DISABLED,
                                              font=("Consolas", 11))
        self.chat.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.chat.tag_config("user",        foreground="#0050a0")
        self.chat.tag_config("llm",         foreground="#1f6f3f")
        self.chat.tag_config("ping",        foreground="#7a3ea0")
        self.chat.tag_config("code",        foreground="#a06020")
        self.chat.tag_config("code_err",    foreground="#aa0000",
                             font=("Consolas", 11, "bold"))
        self.chat.tag_config("code_stderr", foreground="#aa0000",
                             font=("Consolas", 10))
        self.chat.tag_config("code_stdout", foreground="#208020",
                             font=("Consolas", 10))
        self.chat.tag_config("wm",          foreground="#0060a0",
                             font=("Consolas", 11, "bold"))
        self.chat.tag_config("status",      foreground="#666666",
                             font=("Consolas", 10, "italic"))
        self.chat.tag_config("error",       foreground="#aa0000",
                             font=("Consolas", 11, "bold"))
        self.chat.tag_config("feedback",    foreground="#444444",
                             font=("Consolas", 10))

    def _build_input(self, parent):
        frame = ttk.Frame(parent, padding=5)
        frame.pack(fill=tk.X)
        self.input_box = tk.Text(frame, height=4, wrap=tk.WORD,
                                 font=("Consolas", 11))
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_box.bind("<Control-Return>", lambda e: self.submit())
        self.submit_btn = ttk.Button(frame, text="Submit (Ctrl+Enter)",
                                      command=self.submit)
        self.submit_btn.pack(side=tk.LEFT, padx=5)

    def _build_abort(self, parent):
        self.abort_btn = ttk.Button(
            parent,
            text="⛔  ABORT  (kill running script + LLM call)",
            command=self.abort)
        self.abort_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=3)

    def abort(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.abort_event.set()
            self.log("⛔ ABORT requested — will kill subprocess.", "error")
        else:
            self.log("(nothing to abort.)", "status")

    def _enter_live(self):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, "◀ LLM\n", "llm")
        self.chat.config(state=tk.DISABLED)

    def _append_live(self, text):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text, "llm")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def _finalize_live(self):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, "\n\n")
        self.chat.config(state=tk.DISABLED)

    def submit(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Busy",
                "A training run is in progress. ABORT or wait.")
            return
        body = self.input_box.get("1.0", tk.END).strip()
        if not body: return
        self.input_box.delete("1.0", tk.END)

        mode      = self.mode_var.get()
        max_wms   = self.max_wms_var.get()
        model     = self.model_var.get()
        reasoning = self.reasoning_var.get()
        timeout   = self.timeout_var.get()

        self.log(f"▶ {mode.upper()} PROMPT\n{body}\n"
                 f"(reasoning={reasoning}, timeout={timeout}s)",
                 "user")

        self.abort_event.clear()
        self.submit_btn.config(state=tk.DISABLED)
        self.worker_thread = threading.Thread(
            target=self._run,
            args=(body, mode, max_wms, model, reasoning, timeout),
            daemon=True)
        self.worker_thread.start()

    def _run(self, body, mode, max_wms, model, reasoning, timeout):
        try:
            run_training(body, mode, max_wms, model, reasoning,
                         on_event=self._enqueue,
                         abort_event=self.abort_event,
                         script_timeout=timeout)
        except Exception as e:
            self._enqueue("error", f"Run crashed: {e}")
        finally:
            self._enqueue("status", "(ready for next)")
            self.root.after(0,
                lambda: self.submit_btn.config(state=tk.NORMAL))

    def _enqueue(self, event, payload=None):
        self.queue.put((event, payload))

    def _drain_queue(self):
        events = []
        try:
            while True: events.append(self.queue.get_nowait())
        except queue.Empty: pass

        text_buffer = []
        for event, payload in events:
            if event == "llm_delta":
                text_buffer.append(payload)
            else:
                if text_buffer:
                    if self._live_active:
                        self._append_live("".join(text_buffer))
                    text_buffer = []
                self._render(event, payload)
        if text_buffer and self._live_active:
            self._append_live("".join(text_buffer))

        self.root.after(self.POLL_MS, self._drain_queue)

    def _render(self, event, payload):
        if event == "llm_start":
            if self._live_active: self._finalize_live()
            self._live_active = True; self._enter_live()
        elif event == "llm_end":
            if self._live_active:
                self._finalize_live(); self._live_active = False
        elif event == "ping":
            self.log(f"  PING  →  @{payload}", "ping")
        elif event == "subprocess_stdout":
            self.chat.config(state=tk.NORMAL)
            self.chat.insert(tk.END, payload, "code_stdout")
            self.chat.see(tk.END); self.chat.config(state=tk.DISABLED)
        elif event == "subprocess_stderr":
            self.chat.config(state=tk.NORMAL)
            self.chat.insert(tk.END, payload, "code_stderr")
            self.chat.see(tk.END); self.chat.config(state=tk.DISABLED)
        elif event == "code_result":
            if payload["returncode"] == 0:
                self.log(f"  ✓ CODE OK  ({payload['script_path'] or 'inline'})"
                         f"  [{payload.get('elapsed', 0):.1f}s]\n"
                         f"stdout: {payload['stdout'].strip() or '(empty)'}",
                         "code")
            else:
                msg = (f"  ✗ CODE FAILED rc={payload['returncode']}  "
                       f"({payload['script_path'] or 'inline'})  "
                       f"[{payload.get('elapsed', 0):.1f}s]\n")
                if payload.get("timed_out"): msg += "  ⚠️ TIMED OUT.\n"
                if payload.get("aborted"):   msg += "  ⛔ ABORTED.\n"
                msg += f"stderr:\n{payload['stderr'].strip()}"
                self.log(msg, "code_err")
                self.log("  ↳ Error fed back to LLM for self-correction.",
                         "feedback")
        elif event == "wm_created":
            self.log(f"  ★ WM CREATED: {payload}", "wm")
        elif event == "status":
            self.log(f"  · {payload}", "status")
        elif event == "error":
            self.log(f"  ✗ ERROR: {payload}", "error")
        elif event == "feedback":
            self.log(f"  ↳ FEEDBACK SENT TO LLM (first 2k):\n{payload}",
                     "feedback")

    def log(self, text, tag=None):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text + "\n\n", tag)
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

 
def main():
    if not lr.OPENROUTER_API_KEY:
        root = tk.Tk(); root.withdraw()
        messagebox.showerror(
            "OpenRouter API key missing",
            "Set OPENROUTER_API_KEY before running Training.py.",
        )
        return

    root = tk.Tk()
    CantoTrainerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()