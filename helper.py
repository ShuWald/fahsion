import time
from pathlib import Path

from modelevaluator import Evaluator


def timefunc(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return end - start, result


def log_model_run(model_name, train_time, eval_time, evals_result):
    er = evals_result
    log_dir = Path("models/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_name.lower().replace(' ', '_')}.txt"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Model: {model_name}\n")
        handle.write(f"Training Time: {train_time:.6f}\n")
        handle.write(f"Evaluation Time: {eval_time:.6f}\n")
        handle.write(f"Accuracy: {er.accuracy}\n")
        handle.write(f"Precision: {er.precision}\n")
        handle.write(f"Recall: {er.recall}\n")
        handle.write(f"F1: {er.f1}\n")
        handle.write(f"Confusion Matrix: {er.cm}\n")
        handle.write(f"Class Probabilities: {er.class_probabilities}\n")
        handle.write(f"OVR: {er.ovr}\n")
        handle.write(f"OVO Best: {er.ovo_best}\n")
        handle.write(f"OVO Worst: {er.ovo_worst}\n")
        handle.write(f"Evals: {er!r}\n")


def run_model_evaluation(model, model_name, X_train, y_train, X_test, y_test, color="viridis"):
    evaluator = Evaluator(model, model_name)
    train_time, _ = timefunc(evaluator.classifier.fit, X_train, y_train)
    eval_time, evals_result = timefunc(evaluator.evals, X_test, y_test, color=color)
    log_model_run(model_name, train_time, eval_time, evals_result)
    return evaluator, evals_result, train_time, eval_time