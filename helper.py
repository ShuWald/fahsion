import time
from pathlib import Path
from modelevaluator import Evaluator
from modelevaluator import Evals


def timefunc(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return end - start, result


def log_model_run(model_name, stage, **payload):
    log_dir = Path("models/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_name.lower().replace(' ', '_')}.txt"
    mode = "w" if stage == "train" else "a"
    with log_path.open(mode, encoding="utf-8") as handle:
        handle.write(f"Model: {model_name}\n")
        handle.write(f"Stage: {stage}\n")
        for key, value in payload.items():
            if isinstance(value, Evals):
                handle.write(f"{key}.accuracy: {value.accuracy}\n")
                handle.write(f"{key}.precision: {value.precision}\n")
                handle.write(f"{key}.recall: {value.recall}\n")
                handle.write(f"{key}.f1: {value.f1}\n")
                handle.write(f"{key}.cm: {value.cm}\n")
                handle.write(f"{key}.class_probabilities: {value.class_probabilities}\n")
                handle.write(f"{key}.ovr: {value.ovr}\n")
                handle.write(f"{key}.ovo_best: {value.ovo_best}\n")
                handle.write(f"{key}.ovo_worst: {value.ovo_worst}\n")
            else:
                handle.write(f"{key}: {value}\n")
        handle.write("\n")


def train_model(model, model_name, X_train, y_train):
    train_time, history = timefunc(model.fit, X_train, y_train)
    log_model_run(model_name, "train", train_time=train_time)
    return train_time, history


def evaluate_model(model, model_name, X, y, stage, color="viridis", nbest=3, nworst=5):
    evaluator = Evaluator(model, model_name)
    eval_time, evals_result = timefunc(evaluator.evals, X, y, color=color, nbest=nbest, nworst=nworst)
    log_model_run(model_name, stage, eval_time=eval_time, evals=evals_result)
    return evaluator, evals_result, eval_time