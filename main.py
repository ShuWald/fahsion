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
l = LogReg()
k = KNN()
f = MLP()
r = RandomForest()

# Train models
ltime = train_model(l, 'Logistic Regression', X_train, y_train)
print(f'Training time for Logistic Regression: {ltime:.2f} seconds')
kfit_time = train_model(k, 'kNN', X_train, y_train)
print(f'Training time for kNN: {kfit_time:.2f} seconds')
ftime = train_model(f, 'MLP', X_train, y_train)
print(f'Training time for MLP: {ftime:.2f} seconds')
rtime = train_model(r, 'Random Forest', X_train, y_train)
print(f'Training time for Random Forest: {rtime:.2f} seconds')

# Validate models
le, levals, letime = evaluate_model(l, 'Logistic Regression', X_val, y_val, stage='validation', color='YlGnBu')
print(f'Validation time for Logistic Regression: {letime:.2f} seconds')
ke, kevals, k_eval_time = evaluate_model(k, 'kNN', X_val, y_val, stage='validation', color='magma')
print(f'Validation time for kNN: {k_eval_time:.2f} seconds')
fe, fevals, fetime = evaluate_model(f, 'MLP', X_val, y_val, stage='validation', color='viridis')
print(f'Validation time for MLP: {fetime:.2f} seconds')
re, revals, retime = evaluate_model(r, 'Random Forest', X_val, y_val, stage='validation', color='plasma')
print(f'Validation time for Random Forest: {retime:.2f} seconds')


plt.plot(f.classifier.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('MLP Loss Curve')
plt.savefig('figures/mlp_loss_curve.png')
plt.show()

print('\n\n')
print('Logistic Regression Evals:')
print(levals)
print('kNN Confusion Evals:')
print(kevals)
print('MLP Confusion Evals:')
print(fevals)
print('Random Forest Confusion Evals:')
print(revals)
print('\n\n')


# Test models
le_test, le_test_vals, le_test_time = evaluate_model(l, 'Logistic Regression', X_test, y_test, stage='test', color='YlGnBu')
print(f'Test time for Logistic Regression: {le_test_time:.2f} seconds')
ke_test, ke_test_vals, ke_test_time = evaluate_model(k, 'kNN', X_test, y_test, stage='test', color='magma')
print(f'Test time for kNN: {ke_test_time:.2f} seconds')
fe_test, fe_test_vals, fe_test_time = evaluate_model(f, 'MLP', X_test, y_test, stage='test', color='viridis')
print(f'Test time for MLP: {fe_test_time:.2f} seconds')
re_test, re_test_vals, re_test_time = evaluate_model(r, 'Random Forest', X_test, y_test, stage='test', color='plasma')
print(f'Test time for Random Forest: {re_test_time:.2f} seconds')

# Compare all models across categories
models = ["Logistic Regression", "kNN", "MLP", "Random Forest"]
catcounts = {"Logistic Regression": np.diag(levals.cm) , "kNN": np.diag(kevals.cm) , "MLP": np.diag(fevals.cm), "Random Forest": np.diag(revals.cm)}
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