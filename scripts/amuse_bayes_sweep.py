#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PARAMS = (
    ("lr_scale", "log", 0.25, 2.0),
    ("adam_lr_scale", "log", 0.25, 2.0),
    ("warmup_steps", "int", 5, 250),
    ("start_ratio", "linear", 0.0, 0.50),
    ("amuse_beta1", "linear", 0.10, 0.80),
    ("amuse_rho", "linear", 0.20, 1.00),
)

EVAL_RE = re.compile(r"(?:eval step=\d+|heldout_eval split=val) .*?val_loss=([0-9.eE+-]+)")
ELAPSED_RE = re.compile(r"elapsed_s=([0-9.eE+-]+)")
LOSS_RE = re.compile(r"\bloss=([0-9.eE+-]+)")
STEP_RE = re.compile(r"step=(\d+)")
RUN_DIR_RE = re.compile(r"run_dir=(.*)")


@dataclass
class Trial:
    index: int
    params: dict
    val_loss: float
    train_loss: float | None
    elapsed_s: float | None
    completed_steps: int | None
    returncode: int
    log_path: str
    run_dir: str | None


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    binary = root / "target" / "release" / "rust-kernels"
    sweep_dir = args.out_dir or root / "target" / "sweeps" / f"amuse_bayes_{utc_stamp()}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    if args.build:
        run_checked(["cargo", "build", "--release"], root)
    if not binary.exists():
        print(f"missing binary: {binary}", file=sys.stderr)
        print("run with --build or build target/release/rust-kernels first", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    save_config(sweep_dir, args)
    trials = load_trials(sweep_dir / "trials.jsonl")
    print(f"sweep_dir={sweep_dir}")
    print(f"loaded_trials={len(trials)}")

    for index in range(len(trials), args.trials):
        params = suggest(trials, rng, args.init_random, args.candidates)
        trial = run_trial(index, params, args, root, binary, sweep_dir)
        trials.append(trial)
        append_jsonl(sweep_dir / "trials.jsonl", trial.__dict__)
        if successful(trials):
            write_best(sweep_dir, trials)
        print_trial(trial, maybe_best_trial(trials))
        sys.stdout.flush()

    best = maybe_best_trial(trials)
    if best is None:
        print("best=null")
        return 1
    print("best=" + json.dumps(best.__dict__, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian AMUSE hyperparameter sweep")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--init-random", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--dataset", default="shakespeare")
    parser.add_argument("--device", default=os.environ.get("CUDA_DEVICE_INDEX", "0"))
    parser.add_argument("--seed", type=int, default=0xA11CE)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--log-interval", type=int, default=0)
    return parser.parse_args()


def run_trial(
    index: int,
    params: dict,
    args: argparse.Namespace,
    root: Path,
    binary: Path,
    sweep_dir: Path,
) -> Trial:
    log_path = sweep_dir / f"trial_{index:03d}.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_INDEX": args.device,
            "TRAIN_DATASET": args.dataset,
            "TRAIN_STEPS": str(args.steps),
            "TRAIN_LOG_INTERVAL": str(args.log_interval or max(args.steps - 1, 1)),
            "TRAIN_EVAL_INTERVAL": str(max(args.steps + 1, 1)),
            "TRAIN_LR_SCALE": f"{params['lr_scale']:.8g}",
            "TRAIN_ADAM_LR_SCALE": f"{params['adam_lr_scale']:.8g}",
            "TRAIN_LR_WARMUP_STEPS": str(params["warmup_steps"]),
            "TRAIN_LR_START_RATIO": f"{params['start_ratio']:.8g}",
            "TRAIN_AMUSE_BETA1": f"{params['amuse_beta1']:.8g}",
            "TRAIN_AMUSE_RHO": f"{params['amuse_rho']:.8g}",
        }
    )
    if args.max_seconds > 0.0:
        env["TRAIN_MAX_SECONDS"] = f"{args.max_seconds:.8g}"

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(binary)],
            cwd=root,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - start
    parsed = parse_log(log_path)
    val_loss = parsed["val_loss"]
    if not math.isfinite(val_loss):
        val_loss = 1.0e9
    return Trial(
        index=index,
        params=params,
        val_loss=val_loss,
        train_loss=parsed["train_loss"],
        elapsed_s=parsed["elapsed_s"] or elapsed,
        completed_steps=parsed["completed_steps"],
        returncode=proc.returncode,
        log_path=str(log_path),
        run_dir=parsed["run_dir"],
    )


def parse_log(log_path: Path) -> dict:
    val_loss = math.inf
    train_loss = None
    elapsed_s = None
    completed_steps = None
    run_dir = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := RUN_DIR_RE.search(line):
            run_dir = match.group(1).strip()
        if match := EVAL_RE.search(line):
            val_loss = float(match.group(1))
        if line.startswith("step="):
            if match := STEP_RE.search(line):
                completed_steps = int(match.group(1)) + 1
            if match := LOSS_RE.search(line):
                train_loss = float(match.group(1))
            if match := ELAPSED_RE.search(line):
                elapsed_s = float(match.group(1))
    return {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "elapsed_s": elapsed_s,
        "completed_steps": completed_steps,
        "run_dir": run_dir,
    }


def suggest(trials: list[Trial], rng: random.Random, init_random: int, candidates: int) -> dict:
    if len(successful(trials)) < init_random:
        return sample_params(rng)

    observed = successful(trials)
    x = np.array([encode(t.params) for t in observed], dtype=np.float64)
    y = np.array([t.val_loss for t in observed], dtype=np.float64)
    y_mean = float(y.mean())
    y_std = float(y.std() or 1.0)
    y_norm = (y - y_mean) / y_std

    cand_params = [sample_params(rng) for _ in range(candidates)]
    cand_x = np.array([encode(params) for params in cand_params], dtype=np.float64)
    mean, std = gp_predict(x, y_norm, cand_x)
    best = float(y_norm.min())
    ei = expected_improvement(mean, std, best)
    return cand_params[int(np.argmax(ei))]


def successful(trials: list[Trial]) -> list[Trial]:
    return [trial for trial in trials if trial.returncode == 0 and math.isfinite(trial.val_loss)]


def sample_params(rng: random.Random) -> dict:
    return {name: decode_one(rng.random(), name, kind, lo, hi) for name, kind, lo, hi in PARAMS}


def encode(params: dict) -> list[float]:
    values = []
    for name, kind, lo, hi in PARAMS:
        value = float(params[name])
        if kind == "log":
            values.append((math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo)))
        else:
            values.append((value - lo) / (hi - lo))
    return values


def decode_one(unit: float, name: str, kind: str, lo: float, hi: float):
    unit = min(max(unit, 0.0), 1.0)
    if kind == "log":
        return math.exp(math.log(lo) + unit * (math.log(hi) - math.log(lo)))
    value = lo + unit * (hi - lo)
    if kind == "int":
        return int(round(value))
    return value


def gp_predict(x: np.ndarray, y: np.ndarray, cand_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengthscale = 0.28
    noise = 1.0e-5
    k_xx = rbf(x, x, lengthscale) + noise * np.eye(x.shape[0])
    k_cx = rbf(cand_x, x, lengthscale)
    try:
        chol = np.linalg.cholesky(k_xx)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        mean = k_cx @ alpha
        v = np.linalg.solve(chol, k_cx.T)
        var = np.maximum(1.0 - np.sum(v * v, axis=0), 1.0e-9)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(k_xx)
        mean = k_cx @ inv @ y
        var = np.maximum(1.0 - np.sum((k_cx @ inv) * k_cx, axis=1), 1.0e-9)
    return mean, np.sqrt(var)


def rbf(a: np.ndarray, b: np.ndarray, lengthscale: float) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    dist2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return np.exp(-0.5 * dist2 / (lengthscale * lengthscale))


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float) -> np.ndarray:
    improvement = best - mean
    z = improvement / np.maximum(std, 1.0e-9)
    return improvement * normal_cdf(z) + std * normal_pdf(z)


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def load_trials(path: Path) -> list[Trial]:
    if not path.exists():
        return []
    trials = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        trials.append(Trial(**data))
    return trials


def append_jsonl(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, sort_keys=True) + "\n")


def save_config(sweep_dir: Path, args: argparse.Namespace) -> None:
    config = vars(args).copy()
    config["out_dir"] = str(config["out_dir"]) if config["out_dir"] else None
    config["params"] = PARAMS
    (sweep_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def write_best(sweep_dir: Path, trials: list[Trial]) -> None:
    best = best_trial(trials)
    (sweep_dir / "best.json").write_text(json.dumps(best.__dict__, indent=2, sort_keys=True) + "\n")


def best_trial(trials: list[Trial]) -> Trial:
    return min(successful(trials), key=lambda trial: trial.val_loss)


def maybe_best_trial(trials: list[Trial]) -> Trial | None:
    ok = successful(trials)
    if not ok:
        return None
    return min(ok, key=lambda trial: trial.val_loss)


def print_trial(trial: Trial, best: Trial | None) -> None:
    best_index = "n/a" if best is None else best.index
    best_val = "n/a" if best is None else f"{best.val_loss:.6f}"
    print(
        "trial={index} val_loss={val:.6f} train_loss={train} steps={steps} "
        "elapsed_s={elapsed:.3f} best_trial={best_index} best_val={best_val} params={params}".format(
            index=trial.index,
            val=trial.val_loss,
            train="n/a" if trial.train_loss is None else f"{trial.train_loss:.6f}",
            steps="n/a" if trial.completed_steps is None else trial.completed_steps,
            elapsed=0.0 if trial.elapsed_s is None else trial.elapsed_s,
            best_index=best_index,
            best_val=best_val,
            params=json.dumps(trial.params, sort_keys=True),
        )
    )


def run_checked(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
