import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataevaluator import DataEvaluator
from helper import run_model_evaluation
from models.knn import KNN
from models.mlp import MLP
from models.logistic import LogReg

data = DataEvaluator('dataset')
X_train, y_train, X_test, y_test = data.get_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
data.plot_clothing_distribution()

# Initialize models

l = LogReg()
k = KNN()
f = MLP()


# Logistic Regression

le, levals, ltime, letime = run_model_evaluation(l, "Logistic Regression", X_train, y_train, X_test, y_test, color="YlGnBu")
print(f"Training time for Logistic Regression: {ltime:.2f} seconds")



# k-Nearest Neighbors

ke, kevals, kfit_time, k_eval_time = run_model_evaluation(k, "kNN", X_train, y_train, X_test, y_test, color="magma")
print(f"Training time for kNN: {kfit_time:.2f} seconds")
print(f"Evaluation time for kNN: {k_eval_time:.2f} seconds")


# FeedForward Neural Network

fe, fevals, ftime, fetime = run_model_evaluation(f, "MLP", X_train, y_train, X_test, y_test, color="viridis")
print(f"Training time for MLP: {ftime:.2f} seconds")
print(f"Evaluation time for MLP: {fetime:.2f} seconds")

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
