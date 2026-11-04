"""Persistent, validation-only Optuna search for the inductive GNN recipe."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import optuna


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--eval-triples", type=int, default=300)
    args = parser.parse_args()
    args.study_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="inductive-gnn-validation", direction="maximize",
        storage=f"sqlite:///{(args.study_dir / 'study.sqlite').resolve()}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=13))
    if not study.trials:
        for policy, rate, temperature in [("uniform", .001, .1), ("mixed", .001, .1),
                                          ("mixed", .003, .05), ("relation-range", .001, .1)]:
            study.enqueue_trial(dict(negative_policy=policy, learning_rate=rate, temperature=temperature))

    def objective(trial):
        rate = trial.suggest_float("learning_rate", .0001, .003, log=True)
        temperature = trial.suggest_float("temperature", .03, .2, log=True)
        policy = trial.suggest_categorical("negative_policy", ["uniform", "mixed", "relation-range"])
        run = args.study_dir / f"trial_{trial.number}"
        command = [sys.executable, "-m", "yago.inductive_train", "--dataset", str(args.dataset.resolve()),
                   "--run-dir", str(run.resolve()), "--steps", str(args.steps),
                   "--batch-size", "64", "--negatives", "8", "--fanouts", "8", "4",
                   "--hidden-dim", "64", "--learning-rate", str(rate), "--temperature", str(temperature),
                   "--negative-policy", policy, "--validation-policy", "relation-range",
                   "--feature-mode", "full", "--eval-triples", str(args.eval_triples),
                   "--checkpoint-every", "500", "--eval-every", "500", "--require-gpu", "--no-export"]
        trial.set_user_attr("command", command)
        log_path = args.study_dir / f"trial_{trial.number}.log"
        with log_path.open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        best = json.loads((run / "best.json").read_text())
        trial.set_user_attr("best_step", best["step"])
        trial.set_user_attr("run_dir", str(run.resolve()))
        return best["mrr"]

    study.optimize(objective, n_trials=args.trials)
    with (args.study_dir / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial", "state", "value", "learning_rate", "temperature", "negative_policy", "best_step"])
        writer.writeheader()
        for trial in study.trials:
            writer.writerow(dict(trial=trial.number, state=trial.state.name, value=trial.value,
                                 **trial.params, best_step=trial.user_attrs.get("best_step")))
    result = dict(best_trial=study.best_trial.number, best_valid_mrr=study.best_value,
                  params=study.best_params, run_dir=study.best_trial.user_attrs["run_dir"],
                  selection_protocol="validation only, relation-range sampled-filtered tail MRR")
    (args.study_dir / "best_recipe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
