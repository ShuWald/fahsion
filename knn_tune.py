import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataevaluator import DataEvaluator
from helper import evaluate_model, train_model
from models.knn import KNN
from models.mlp import MLP
from models.logistic import LogReg
from models.random_forest import RandomForest


data = DataEvaluator('dataset')
X_train, y_train, X_val, y_val, X_test, y_test = data.get_data()
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
clothing_types = data.clothing_types

# Initialize models
ks = [1, 3, 5, 7, 10, 15, 20, 25]
knns = {k:KNN(k=k) for k in ks}

# Train models
kfits = {}
for k, knn in knns.items():
    kfits[k] = train_model(knn, f'kNN (k={k})', X_train, y_train)
    print(f'Training time for kNN (k={k}): {kfits[k]} seconds')


# Validate models
ke = {}
kevals = {}
k_eval_times = {}
for k, knn in knns.items():
    ke[k], kevals[k], k_eval_times[k] = evaluate_model(knn, f'kNN (k={k})', X_val, y_val, stage='validation', color='magma', nbest=0, nworst=0)
    ke[k].name = f'kNN (k={k})'
    print(f'Validation time for kNN (k={k}): {k_eval_times[k]} seconds')

print('kNN Confusion Evals:')
print(kevals)

# Plot k vs accuracy
accuracies = {k: kevals[k].accuracy for k in ks}
plt.plot(ks, list(accuracies.values()), marker='o')
plt.xlabel('k in kNN')
plt.ylabel('Validation Accuracy')
plt.title('kNN Validation Accuracy vs k')
plt.xticks(ks)
plt.grid()
plt.savefig('figures/knn_k_vs_accuracy.png')
plt.show()

# Compare all models across categories

models = ks
catcounts = {kevals[k].name: np.diag(kevals[k].cm) for k in ks}
cataccs = {model: counts/(len(y_test)/len(clothing_types)) for model, counts in catcounts.items()}

x = np.arange(len(clothing_types))
width = 0.2
[plt.bar(x - width + i*width, catcounts[model], width, label=model) for i, model in enumerate(models)]
plt.xlabel('Clothing Category')
plt.ylabel('Number of Correct Predictions')
plt.title('Accuracy of Models by Clothing Category')
plt.xticks(x, [clothing_types[i] for i in clothing_types.keys()], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('figures/model_category_accuracies.png')
plt.show()
[print(f'{model} Category Accuracy: {cataccs[model]}') for model in models]
totaccs = {model: np.round(np.mean(cataccs[model]), 4) for model in models}
print('Sorted by Overall Accuracy:', sorted([{'Logistic Regression': totaccs['Logistic Regression']}, {'kNN': totaccs['kNN']}, {'MLP': totaccs['MLP']}, {'Random Forest': totaccs['Random Forest']}], key=lambda x: list(x.values())[0], reverse=True))

performance_by_category = pd.DataFrame({model: cataccs[model] for model in models}, index=[data.clothing_types[i] for i in data.clothing_types.keys()])
print('Performance by Category:', performance_by_category, sep='\n')
print('\nSorted by Category Accuracy:')
for category in performance_by_category.index:
    sorted_models = performance_by_category.loc[category].sort_values(ascending=False)
    print(f'Category {category}: {sorted_models.to_dict()}')

best_model_by_category = performance_by_category.idxmax(axis=1)
print(f'Best model by category:', best_model_by_category, sep='\n')
print(f'Model best performace counts by category: \n{best_model_by_category.value_counts()}')
print(f'Model with most category wins: {best_model_by_category.value_counts().idxmax()}')