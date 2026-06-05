import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataevaluator import DataEvaluator
from modelevaluator import Evaluator
from models.knn import KNN
from models.mlp import MLP
from models.logistic import LogReg

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

data = DataEvaluator('dataset')
X_train, y_train, X_test, y_test = data.get_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
data.plot_clothing_distribution()

# Initialize models

l = LogReg()
k = KNN()
f = MLP()


# Logistic Regression

le = Evaluator(l, "Logistic Regression")
ltime, _ = timefunc(le.classifier.fit, X_train, y_train)
print(f"Training time for Logistic Regression: {ltime:.2f} seconds")
letime, levals = timefunc(le.evals, X_test, y_test, color="YlGnBu")
log_model_run("Logistic Regression", ltime, letime, levals)



# k-Nearest Neighbors

ke = Evaluator(k, "kNN")
kfit_time, _ = timefunc(ke.classifier.fit, X_train, y_train)
k_eval_time, kevals = timefunc(ke.evals, X_test, y_test, color="magma")
print(f"Evaluation time for kNN: {k_eval_time:.2f} seconds")
log_model_run("kNN", kfit_time, k_eval_time, kevals)


# FeedForward Neural Network

fe = Evaluator(f, "MLP")
ftime, _ = timefunc(fe.classifier.fit, X_train, y_train)
print(f"Training time for MLP: {ftime:.2f} seconds")
fetime, fevals = timefunc(fe.evals, X_test, y_test, color="viridis")
log_model_run("MLP", ftime, fetime, fevals)

plt.plot(f.classifier.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('MLP Loss Curve')
plt.savefig('figures/mlp_loss_curve.png')
plt.show()

models = ["Logistic Regression", "kNN", "MLP"]
catcounts = {"Logistic Regression": np.diag(levals.cm), "kNN": np.diag(kevals.cm), "MLP": np.diag(fevals.cm)}
cataccs = {model: counts / len(y_test) for model, counts in catcounts.items()}

print("\n\n")
print("Logistic Regression Evals:")
print(levals)
print("kNN Confusion Evals:")
print(kevals)
print("MLP Confusion Evals:")
print(fevals)
print("\n\n")

x = np.arange(len(data.clothing_types))
width = 0.275
[plt.bar(x - width + i * width, cataccs[model], width, label=model) for i, model in enumerate(models)]
plt.xlabel('Clothing Category')
plt.ylabel('Number of Correct Predictions')
plt.title('Accuracy of Models by Clothing Category')
plt.xticks(x, [data.clothing_types[i] for i in data.clothing_types.keys()], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('figures/model_category_accuracies.png')
plt.show()

[print(f"{model} Category Accuracy: {cataccs[model]}") for model in models]
totaccs = {model: np.round(np.sum(cataccs[model]) / 10, 4) for model in models}
print("Sorted by Overall Accuracy:", sorted([{"Logistic Regression": totaccs["Logistic Regression"]}, {"kNN": totaccs["kNN"]}, {"MLP": totaccs["MLP"]}], key=lambda x: list(x.values())[0], reverse=True))

performance_by_category = pd.DataFrame({model: cataccs[model] for model in models}, index=[data.clothing_types[i] for i in data.clothing_types.keys()])
print("Performance by Category:", performance_by_category, sep="\n")
print("\nSorted by Category Accuracy:")
for category in performance_by_category.index:
    sorted_models = performance_by_category.loc[category].sort_values(ascending=False)
    print(f"Category {category}: {sorted_models.to_dict()}")

best_model_by_category = performance_by_category.idxmax(axis=1)
print(f"Best model by category:", best_model_by_category, sep="\n")
print("\nImportant Note: Sneakers category was actually a tie, but marked as a kNN win.")
print("Regardless, kNN still wins 5 of 9 categories.\n")
print(f"Model best performace counts by category: \n{best_model_by_category.value_counts()}")
print(f"Model with most category wins: {best_model_by_category.value_counts().idxmax()}")
