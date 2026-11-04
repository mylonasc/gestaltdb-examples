"""Persistent Optuna workflow for comparable YAGO KGE experiments.

Each trial runs training in a fresh process so TensorFlow releases GPU memory.
Trial artifacts and logs are retained under the study directory; large embedding
exports and checkpoints are disabled during search.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import optuna


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", default="yago45-kge")
    ap.add_argument("--study-dir", default="optuna_runs/yago45-kge")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5_000)
    ap.add_argument("--eval-every", type=int, default=1_000)
    ap.add_argument("--timeout", type=int, default=None, help="Whole-study timeout in seconds")
    ap.add_argument("--baseline-negative-samplers", action="store_true",
                    help="Queue controlled engine-vs-hybrid trials before TPE trials")
    ap.add_argument("--model-variant", choices=["gnn", "shallow", "tune"], default="gnn")
    ap.add_argument("--sampler-search", choices=["engine", "hybrid", "both"], default="both")
    ap.add_argument("--semantic-search", action="store_true",
                    help="Tune cosine semantic loss weight and temperature")
    ap.add_argument("--objective", choices=["mrr", "semantic", "combined"], default="mrr")
    ap.add_argument("--fixed-kge-recipe", action="store_true",
                    help="Keep the published masked KGE hyperparameters fixed")
    ap.add_argument("--init-embeddings", default=None,
                    help="Warm-start every trial from exported base tables")
    ap.add_argument("--finetune-recipe", action="store_true",
                    help="Tune a conservative LR while fixing other KGE settings")
    args = ap.parse_args()
    if args.objective != "mrr" and not args.semantic_search:
        ap.error("--objective semantic/combined requires --semantic-search")
    if args.fixed_kge_recipe and args.finetune_recipe:
        ap.error("--fixed-kge-recipe and --finetune-recipe are mutually exclusive")

    study_dir = Path(args.study_dir).resolve()
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_dir / 'study.sqlite3'}"
    sampler = optuna.samplers.TPESampler(seed=13, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    if args.baseline_negative_samplers and not study.trials:
        baseline = {
            "learning_rate": 3e-4,
            "batch_size": 128,
            "negatives": 16,
            "adv_temp": 1.0,
            "aux_weight": 1.0,
        }
        study.enqueue_trial({**baseline, "negative_sampler": "engine"})
        study.enqueue_trial({**baseline, "negative_sampler": "hybrid"})
    train_py = Path(__file__).with_name("train.py")

    def objective(trial: optuna.Trial) -> float:
        if args.finetune_recipe:
            params = {"learning_rate": trial.suggest_float("learning_rate", 1e-5, 2e-4, log=True),
                      "batch_size": 128, "negatives": 16,
                      "adv_temp": 1.413563694141917, "aux_weight": 1.0}
        elif args.fixed_kge_recipe:
            params = {"learning_rate": 0.0005038894433618876, "batch_size": 128,
                      "negatives": 16, "adv_temp": 1.413563694141917,
                      "aux_weight": 1.0032679431822509}
        else:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 8e-4, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
                "negatives": trial.suggest_categorical("negatives", [8, 16, 32]),
                "adv_temp": trial.suggest_float("adv_temp", 0.5, 2.0),
                "aux_weight": trial.suggest_float("aux_weight", 0.25, 1.5, log=True),
            }
        params["negative_sampler"] = (
            trial.suggest_categorical("negative_sampler", ["engine", "hybrid"])
            if args.sampler_search == "both" else args.sampler_search
        )
        if args.semantic_search:
            params["semantic_weight"] = trial.suggest_float("semantic_weight", 0.05, 2.0, log=True)
            params["semantic_temperature"] = trial.suggest_float(
                "semantic_temperature", 0.04, 0.2, log=True)
        model_variant = (trial.suggest_categorical("model_variant", ["gnn", "shallow"])
                         if args.model_variant == "tune" else args.model_variant)
        params["model_variant"] = model_variant
        run_dir = study_dir / f"trial_{trial.number:04d}"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "params.json").write_text(json.dumps(params, indent=2))
        cmd = [
            sys.executable, str(train_py),
            "--data-root", args.data_root,
            "--run-dir", str(run_dir),
            "--steps", str(args.steps),
            "--eval-every", str(args.eval_every),
            "--score", "complex",
            "--learning-rate", str(params["learning_rate"]),
            "--batch-size", str(params["batch_size"]),
            "--negatives", str(params["negatives"]),
            "--adv-temp", str(params["adv_temp"]),
            "--aux-weight", str(params["aux_weight"]),
            "--negative-sampler", params["negative_sampler"],
            "--no-export", "--no-checkpoints",
        ]
        if model_variant == "shallow":
            cmd.append("--no-gnn")
        if args.init_embeddings:
            cmd.extend(["--init-embeddings", args.init_embeddings])
        if args.semantic_search:
            cmd.extend([
                "--semantic-weight", str(params["semantic_weight"]),
                "--semantic-temperature", str(params["semantic_temperature"]),
            ])
        with (run_dir / "train.log").open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        if proc.returncode:
            tail = (run_dir / "train.log").read_text(errors="replace").splitlines()[-30:]
            raise RuntimeError("training failed:\n" + "\n".join(tail))
        summary = json.loads((run_dir / "summary.json").read_text())
        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("model_variant", model_variant)
        trial.set_user_attr("negative_sampler", params["negative_sampler"])
        trial.set_user_attr("elapsed_seconds", summary["elapsed_seconds"])
        trial.set_user_attr("best_step", summary["best_step"])
        trial.set_user_attr("final_valid_mrr", summary["final_valid_mrr"])
        trial.set_user_attr("final_semantic_top1", summary.get("final_semantic_top1"))
        trial.set_user_attr("final_semantic_margin", summary.get("final_semantic_margin"))
        if args.objective == "semantic":
            return float(summary["final_semantic_top1"])
        if args.objective == "combined":
            return float(summary["best_valid_mrr"] + 0.2 * summary["final_semantic_top1"])
        return float(summary["best_valid_mrr"])

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, catch=(RuntimeError,))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"all trials failed; inspect {study_dir}/trial_*/train.log")
    rows = []
    for trial in study.trials:
        row = {"number": trial.number, "state": trial.state.name, "value": trial.value}
        row.update({f"param_{k}": v for k, v in trial.params.items()})
        row.update({f"attr_{k}": v for k, v in trial.user_attrs.items()})
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (study_dir / "trials.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    best = {"number": study.best_trial.number, "value": study.best_value,
            "params": study.best_trial.params, "user_attrs": study.best_trial.user_attrs}
    (study_dir / "best_trial.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
