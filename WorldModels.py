 
import os
import json
import importlib.util
import sys
from datetime import datetime, timezone

 

WM_ROOT      = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR  = os.path.join(WM_ROOT, "scripts")
MODELS_DIR   = os.path.join(WM_ROOT, "models")
DATASETS_DIR = os.path.join(WM_ROOT, "datasets")
JSONS_DIR    = os.path.join(WM_ROOT, "jsons")
PROBES_DIR   = os.path.join(WM_ROOT, "probes")   
for _d in (SCRIPTS_DIR, MODELS_DIR, DATASETS_DIR, JSONS_DIR, PROBES_DIR):
    os.makedirs(_d, exist_ok=True)


 
HARNESS_TEXT = """
# o44 WORLD MODEL HARNESS
You build World Models (WMs) by writing Python code that constructs,
trains, or implements a system, then saving the artifacts to disk.
A WM takes structured inputs and produces structured outputs.

There are FIVE families. Pick the one that fits the problem.

------------------------------------------------------------
1. SYMBOLIC AI  (regular Python — no ML libraries required)
------------------------------------------------------------
1.1  Expert Systems            Knowledge Base + Inference Engine + Working Memory
1.2  Propositional Logic       boolean vars, AND/OR/NOT/IMPLIES, SAT (pysat, z3)
1.3  First-Order Logic         predicates, quantifiers (forall, exists)
1.4  Higher-Order Logic        quantification over predicates/functions
1.5  Modal Logic               necessity (□) and possibility (◇) operators
1.6  Description Logics        concepts / roles / individuals (owlready2)
1.7  Semantic Networks         nodes + edges (networkx)
1.8  Frame-Based Systems       slots, inheritance hierarchies
1.9  Recognise-Act Rules       IF condition THEN action, match-act cycle
1.10 Inductive Logic Prog.     induce rules from examples (ILP)
1.11 Case-Based Reasoning      retrieve / reuse / revise / retain
1.12 Constraint Solvers        python-constraint, z3, ortools
1.13 Theorem Provers           z3 / sympy.prover
1.14 Synthetic data (Symbolic) numpy-generated test cases; structured
       (dicts / tuples) matching the system's input schema; saved as
       .npz or .json in /datasets/.

------------------------------------------------------------
2. CLASSICAL MACHINE LEARNING  (scikit-learn / xgboost / lightgbm)
------------------------------------------------------------
2.1  Linear / Logistic / Ridge / Lasso Regression
2.2  SVM (SVC, SVR)
2.3  Decision Trees / Random Forest / Extra Trees
2.4  Gradient Boosting (XGBoost, LightGBM, CatBoost)
2.5  k-NN
2.6  Naive Bayes (Gaussian / Multinomial / Bernoulli)
2.7  Clustering (KMeans, Agglomerative, DBSCAN)
2.8  Dim Reduction (PCA, t-SNE, UMAP)
2.9  Density Estimation (GMM, KDE)
2.10 Anomaly Detection (Isolation Forest, One-Class SVM)
2.11 Synthetic data (Classical) numpy; match real-data distribution;
       train/test split via sklearn.model_selection.train_test_split;
       saved as .csv or .npz.

------------------------------------------------------------
3. DEEP LEARNING  (PyTorch)
------------------------------------------------------------
3.1  Multilayer Perceptrons      most applicable for tabular / vector input
3.2  RNN / LSTM / GRU            sequential data
3.3  Transformers                language, long sequences, attention
3.4  CNNs / ResNets              image / spatial data
3.5  VAEs / Autoencoders         generation, reconstruction, anomaly
3.6  Graph Neural Networks       torch_geometric, graph-structured data
3.7  JEPA Networks               joint embedding predictive architecture
3.8  Synthetic data (DL)         numpy; match expected shape; for images
       use numpy; save as .pt or .npz.

------------------------------------------------------------
4. PROGRAMS  (general Python programs / tools)
------------------------------------------------------------
A Program WM is a self-contained callable Python module — NOT an ML
model, NOT a symbolic rule base. The harness treats it as a WM because
it saves its metadata JSON and the inference harness dispatches to it
via a named entry function.

Interface contract — pick ONE and document it in the JSON:
  (a) `<wm_name>.run(*args, **kwargs) -> result`
  (b) `<wm_name>.solve(problem: dict) -> solution: dict`
  (c) `<wm_name>.main(argv) -> int`  (CLI-style; not generally used at inference)

Examples:
  4.1  Chess engine            python-chess + (optional) Stockfish
  4.2  Sudoku / KenKen solver  backtracking + constraint propagation
  4.3  Logic puzzle solver     zebra puzzles, Einstein's riddle, Knapsack
  4.4  Symbolic math           sympy / z3 evaluators, equation solvers
  4.5  Numerical simulator     scipy.integrate.solve_ivp (projectiles, etc.)
  4.6  Numerical optimiser     scipy.optimize.brute / minimize / linprog
  4.7  Differential eq solver  numeric or sympy
  4.8  Computer algebra        sympy simplification, expansion, factoring
  4.9  Graph algorithms        networkx: shortest path, MST, max flow
  4.10 String / sequence utils regex, edit distance, hashing, parsing

DISPATCH: at inference the harness loads
  /scripts/<wm_name>_train.py
(or /scripts/<wm_name>/__init__.py) and calls `mod.run(...)` /
`mod.solve(...)`. Declare which in the JSON metadata.

------------------------------------------------------------
5. RL PROGRAMS  (Stable-Baselines3 + Gymnasium)
------------------------------------------------------------
An RL WM trains a policy on a Gymnasium environment using SB3, saves
the policy to disk, and proxies inference through the policy's
predict(obs) method while still allowing access to the underlying
env for resets / step-by-step simulation.

Stack:
  gymnasium          env interface + many built-in envs
  stable-baselines3  PPO / DQN / A2C / SAC / TD3 / DDPG
  sb3-contrib        QR-DQN, RecurrentPPO, TQC, etc. (optional)

Examples:
  5.1  Blackjack         gymnasium.envs.toy_text.Blackjack-v1 (built-in)
  5.2  CartPole          gymnasium.envs.classic_control.CartPole-v1
  5.3  FrozenLake        gymnasium.envs.toy_text.FrozenLake-v1
  5.4  CliffWalking      gymnasium.envs.toy_text.CliffWalking-v1
  5.5  Taxi              gymnasium.envs.toy_text.Taxi-v3
  5.6  Custom GridWorld  hand-written env for specific RL problem
  5.7  Atari             only when budget permits
  5.8  Hybrid RL+Program RL agent that ALSO queries a known-optimal
        rule for some states (e.g. Blackjack RL that consults a
        basic-strategy lookup). Cite both "rl" and "program" families
        in metadata if you do this.

Cheap-training constraints:
  - env step ≤ 5 ms  (use DummyVecEnv / SubprocVecEnv)
  - total training steps ≤ 1e6 per WM
  - CPU training is fine for tabular / toy envs
  - GPU only when image inputs are involved

DISPATCH: at inference the harness loads the .zip via
  `algo.load("models/<wm_name>.zip")` and calls
  `model.predict(obs, deterministic=True)`.

------------------------------------------------------------
6. CREATION STANDARDS  (apply to EVERY WM you build)
------------------------------------------------------------
Every WM produces FOUR artifacts:

6.1  /jsons/<wm_name>.json
     Metadata describing the WM. Required fields:
       name          (string, unique)
       family        "symbolic" | "classical" | "deep" | "program" | "rl"
       description   (one-paragraph plain English)
       script_path   "scripts/<wm_name>_train.py"
       model_path    "models/<wm_name>.<ext>"   or null if symbolic
       dataset_path  "datasets/<wm_name>.<ext>" or null if none
       inputs        {"schema": {...}, "examples": [...]}
       outputs       {"schema": {...}, "examples": [...]}
       tags          ["domain", "type", ...]
       dependencies  ["scikit-learn==1.4", "torch>=2.0", ...]
       version       "1"
       created_at    ISO-8601 timestamp
       metrics       {"rmse": 0.42, "r2": 0.91, "mean_reward": -0.04} or {}

     RL-only optional fields:
       env           "Blackjack-v1"
       algorithm     "PPO"
       policy        "MlpPolicy"
       training_steps        (int)
       total_episodes        (int, optional)

     Program-only optional fields:
       entry         "run" | "solve" | "main"
       module_path   "scripts/<wm_name>.py"  (if separate from script_path)

6.2  /scripts/<wm_name>_train.py
     Training script (or the script that builds the system).
     For programs: it should ALSO build /scripts/<wm_name>.py (the
     callable module) if you want entry-point separation.
     Must save the model to /models/, write the JSON via
     `wm.save_wm_metadata({...})`, and print final metrics to stdout.

6.3  /models/<wm_name>.<ext>
     joblib / pickle for sklearn;
     .pt / .pth for torch;
     .zip for SB3;
     null for pure symbolic / pure program (programs may have no model).

6.4  /datasets/<wm_name>.<ext>
     csv / npz / json / pt as appropriate;
     null if the WM is pure rules / pure program.

------------------------------------------------------------
7. WORKFLOW
------------------------------------------------------------
1. Decide which family fits.
2. Write /scripts/<wm_name>_train.py that builds the WM.
3. Run it. It saves the model, emits the JSON, prints metrics.
4. Confirm the JSON is in /jsons/ and the artifacts are in the right folders.
5. The WM is now pingable.
"""

 
 
def ping_harness() -> str:
    return HARNESS_TEXT


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

import ast as _ast
import inspect as _inspect


def _doc_short(doc, n=240):
    if not doc:
        return ""
    first = doc.strip().split("\n\n")[0]
    return first[:n] + ("…" if len(first) > n else "")


def _sig_from_ast(node):
    """Render an AST FunctionDef as a callable-looking signature string."""
    a = node.args
    parts = []
    pos_args = (a.posonlyargs or []) + (a.args or [])
    defaults = [None] * (len(pos_args) - len(a.defaults)) + list(a.defaults)
    for i, arg in enumerate(pos_args):
        name = arg.arg
        d = defaults[i]
        if d is not None:
            try:
                dv = _ast.unparse(d)
            except Exception:
                dv = "..."
            parts.append(f"{name}={dv}")
        else:
            parts.append(name)
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, d in zip(a.kwonlyargs, a.kw_defaults or []):
        if d is not None:
            try: dv = _ast.unparse(d)
            except Exception: dv = "..."
            parts.append(f"{arg.arg}={dv}")
        else:
            parts.append(arg.arg)
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return f"{node.name}({', '.join(parts)})"


def _introspect_module(path):
    """Parse a Python file and return a machine-readable API summary."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        return {"error": f"could not read {path}: {e}"}

    try:
        tree = _ast.parse(src)
    except SyntaxError as e:
        return {"error": f"syntax error in {path}: {e}"}

    api = {"functions": [], "classes": [], "imports": [],
           "dispatch_tables": {}, "doc": ""}
    try:
        api["doc"] = _ast.get_docstring(tree) or ""
        if api["doc"]:
            api["doc"] = _doc_short(api["doc"], 400)
    except Exception:
        pass

    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            prefix = "async " if isinstance(node, _ast.AsyncFunctionDef) else ""
            api["functions"].append({
                "name": node.name,
                "signature": prefix + _sig_from_ast(node),
                "doc": _doc_short(_ast.get_docstring(node)),
            })
        elif isinstance(node, _ast.ClassDef):
            api["classes"].append({
                "name": node.name,
                "doc": _doc_short(_ast.get_docstring(node)),
            })
        elif isinstance(node, _ast.Import):
            for n in node.names:
                api["imports"].append(n.name.split(".")[0])
        elif isinstance(node, _ast.ImportFrom):
            if node.module:
                api["imports"].append(node.module.split(".")[0])
        elif isinstance(node, _ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, _ast.Name)
                        and isinstance(node.value, _ast.Dict)):
                    keys = []
                    for k in node.value.keys:
                        if isinstance(k, _ast.Constant) and isinstance(k.value, str):
                            keys.append(k.value)
                    if keys:
                        api["dispatch_tables"][tgt.id] = keys

    api["imports"] = sorted(set(api["imports"]))
    return api

def ping_registry() -> dict:
    """Full /jsons registry as {name: metadata}."""
    registry = {}
    for fname in os.listdir(JSONS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            entry = _load_json(os.path.join(JSONS_DIR, fname))
            registry[entry.get("name", fname[:-5])] = entry
        except (json.JSONDecodeError, OSError):
            registry[fname[:-5]] = {"error": "invalid / unreadable JSON"}
    return registry


def ping_registry_summary() -> dict:
    out = {}
    for name, entry in ping_registry().items():
        if isinstance(entry, dict) and "error" not in entry:
            out[name] = {
                "family":      _norm_family(entry.get("family"))
                               or "unspecified",
                "description": entry.get("description", ""),
                "tags":        entry.get("tags", []),
            }
    return out


def ping_wm(name: str) -> dict:
    path = os.path.join(JSONS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return {"error": f"No world model named '{name}' in /jsons."}

    try:
        meta = _load_json(path)
    except Exception as e:
        return {"error": f"could not load {name}.json: {e}"}

    if not isinstance(meta, dict):
        return {"error": f"{name}.json is not a JSON object."}

    family = _norm_family(meta.get("family"))

    if family in ("program", "symbolic"):
        module_rel = (meta.get("module_path")
                      or meta.get("script_path")
                      or f"scripts/{name}.py")
        module_abs = module_rel if os.path.isabs(module_rel) \
                     else os.path.join(WM_ROOT, module_rel)
        if not os.path.exists(module_abs):
            alt = os.path.join(SCRIPTS_DIR, f"{name}.py")
            if os.path.exists(alt):
                module_abs = alt
                module_rel = os.path.relpath(alt, WM_ROOT)
        if os.path.exists(module_abs):
            try:
                api = _introspect_module(module_abs)
                api["source_path"] = module_rel
                meta["_api"] = api
            except Exception as e:
                meta["_api"] = {"error": f"introspection failed: {e}"}

    return meta


def ping_source(name: str) -> str:
    """Return the FULL source text of a WM's module."""
    meta = _load_json(os.path.join(JSONS_DIR, f"{name}.json")) \
           if os.path.exists(os.path.join(JSONS_DIR, f"{name}.json")) \
           else {"error": f"No world model named '{name}' in /jsons."}
    if isinstance(meta, dict) and "error" in meta:
        return meta["error"]

    module_rel = (meta.get("module_path")
                  or meta.get("script_path")
                  or f"scripts/{name}.py")
    module_abs = module_rel if os.path.isabs(module_rel) \
                 else os.path.join(WM_ROOT, module_rel)
    if not os.path.exists(module_abs):
        alt = os.path.join(SCRIPTS_DIR, f"{name}.py")
        if os.path.exists(alt):
            module_abs = alt
    if not os.path.exists(module_abs):
        return f"No source file found for '{name}'."
    with open(module_abs, "r", encoding="utf-8") as f:
        return f"# SOURCE OF {name}  ({os.path.relpath(module_abs, WM_ROOT)})\n" \
               + f.read()



def ping_family(family: str) -> list:
    fam = _norm_family(family)
    return sorted(
        name for name, entry in ping_registry().items()
        if isinstance(entry, dict)
        and _norm_family(entry.get("family")) == fam
    )
 

def save_wm_metadata(metadata: dict) -> str:
    """Write a WM's metadata JSON to /jsons. Returns the path written."""
    if "name" not in metadata:
        raise ValueError("metadata must include 'name'")
    metadata.setdefault("family",
                        _infer_family(metadata))
    metadata.setdefault("created_at",
                        datetime.now(timezone.utc).isoformat())
    metadata.setdefault("version", "1")
    path = os.path.join(JSONS_DIR, f"{metadata['name']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return path


def _infer_family(metadata: dict) -> str:
    sp = (metadata.get("script_path") or "").lower()
    mp = (metadata.get("model_path") or "").lower()
    if mp.endswith(".zip"):
        return "rl"
    if mp.endswith((".pt", ".pth")):
        return "deep"
    if mp.endswith((".joblib", ".pkl", ".pickle")):
        return "classical"
    if not mp:
        if any(k in metadata for k in ("rule_base", "rules", "facts")):
            return "symbolic"
        return "program"
    return "unspecified"
 
_FAMILY_NUM = {1: "symbolic", 2: "classical", 3: "deep",
               4: "program",  5: "rl"}

_FAMILY_ALIAS = {
    "programs": "program",
    "program":  "program",
    "symbolics": "symbolic",
    "symbolic ai": "symbolic",
    "classical ml": "classical",
    "classical machine learning": "classical",
    "deep learning": "deep",
    "rl programs": "rl",
    "rl program": "rl",
    "reinforcement learning": "rl",
}

def _norm_family(val) -> str:
    """Coerce any family value to a canonical lowercase string.

    Accepts strings ('program', 'programs', 'Programs'), ints
    (1..5, from the numbered HARNESS_TEXT list), and anything else
    by falling back to str(val).lower(). Never raises.
    """
    if val is None:
        return ""
    if isinstance(val, bool):            
        return str(val).lower()
    if isinstance(val, (int, float)):
        try:
            iv = int(val)
            return _FAMILY_NUM.get(iv, str(val)).lower()
        except Exception:
            return str(val).lower()
    s = str(val).strip().lower()
    return _FAMILY_ALIAS.get(s, s)

def list_wms() -> list:
    return sorted(
        f[:-5] for f in os.listdir(JSONS_DIR) if f.endswith(".json")
    ) 
def call_program(wm_name: str, *args, **kwargs):
    meta = ping_wm(wm_name)
    if isinstance(meta, dict) and meta.get("error"):
        raise RuntimeError(meta["error"])

    module_path = (meta.get("module_path")
                   or meta.get("script_path")
                   or f"scripts/{wm_name}.py")
    module_path = module_path.replace("\\", "/")
    if not os.path.isabs(module_path):
        module_path = os.path.join(WM_ROOT, module_path)

    if not os.path.exists(module_path):
        alt = os.path.join(SCRIPTS_DIR, f"{wm_name}.py")
        if os.path.exists(alt):
            module_path = alt
        else:
            raise FileNotFoundError(f"Program module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"o44_wm_{wm_name}", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    entry = meta.get("entry")
    guesses = [entry] if entry else []
    guesses += ["run", "solve", "main"]
    fn = next((getattr(mod, g) for g in guesses if hasattr(mod, g)), None)
    if fn is None:
        raise RuntimeError(
            f"Program WM '{wm_name}' has no callable entry "
            f"(tried: {guesses}).")

    try:
        return fn(*args, **kwargs)
    except TypeError as e:
       
        api = meta.get("_api") or {}
        hints = [f"→ {wm_name}.{fn.__name__}{_safe_sig(fn)}"]
        for tbl_name, keys in (api.get("dispatch_tables") or {}).items():
            hints.append(f"   dispatch table {tbl_name}: ops = {keys}")
            for k in keys:
                for fninfo in api.get("functions", []):
                    if fninfo["name"] == k:
                        hints.append(f"     {fninfo['signature']}")
                        if fninfo.get("doc"):
                            hints.append(f"       {fninfo['doc']}")
        raise TypeError(
            str(e) + "\n"
            + "\n".join(hints) + "\n"
            + f"Tip: ping @{wm_name} for the full _api summary, "
              f"or @src:{wm_name} for the source."
        ) from None


def _safe_sig(fn):
    try:
        return str(_inspect.signature(fn))
    except Exception:
        return "(...) "


def call_rl(wm_name: str, obs, deterministic: bool = True):
    """Load a Stable-Baselines3 policy and return (action, state)."""
    try:
        from stable_baselines3 import PPO, DQN, A2C, SAC, TD3, DDPG
    except ImportError as e:
        raise RuntimeError(
            "stable-baselines3 not installed. "
            "`pip install stable-baselines3 gymnasium`."
        ) from e

    ALGOS = {"PPO": PPO, "DQN": DQN, "A2C": A2C,
             "SAC": SAC, "TD3": TD3, "DDPG": DDPG}

    meta = ping_wm(wm_name)
    algo_name = (meta.get("algorithm") or "PPO").upper()
    AlgoCls = ALGOS.get(algo_name)
    if AlgoCls is None:
        raise RuntimeError(f"Unknown SB3 algorithm: {algo_name}")

    model_rel = meta.get("model_path") or f"models/{wm_name}.zip"
    if not os.path.isabs(model_rel):
        model_rel = os.path.join(WM_ROOT, model_rel)
    if not os.path.exists(model_rel):
     
        if os.path.exists(model_rel + ".zip"):
            model_rel = model_rel + ".zip"
        else:
            raise FileNotFoundError(f"RL model not found: {model_rel}")

    model = AlgoCls.load(model_rel)
    action, state = model.predict(obs, deterministic=deterministic)
    return action, state