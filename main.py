# Python file version of main.ipynb
# ideally we will be using the notebook since its easier to run and edit without rerunning all code

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay

import mnist_reader
from modelevaluator import Evaluator
from models.knn import KNN
from models.mlp import MLP
from models.logistic import LogReg
from seed import seed

def timefit(model, X_train, y_train):
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    return end - start

X_train, y_train = mnist_reader.load_mnist('dataset', kind='train')
X_test, y_test = mnist_reader.load_mnist('dataset', kind='t10k')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

clothing_types = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

l = LogReg()
k = KNN()
f = MLP()


# Logistic Regression

ltime = timefit(l, X_train, y_train)
le = Evaluator(l, "Logistic Regression")
print(f"Training time for Logistic Regression: {ltime:.2f} seconds")
print("Logistic Regression Accuracy:", le.score(X_test, y_test))
le.cm(X_test, y_test, "YlGnBu")



# k-Nearest Neighbors

k.fit(X_train, y_train)
ke = Evaluator(k, "kNN")
start = time.time()
print("kNN Accuracy:", ke.score(X_test, y_test))
end = time.time()
ktime = end - start
print(f"Evaluation time for kNN: {ktime:.2f} seconds")
ke.cm(X_test, y_test, "magma")


# FeedForward Neural Network

ftime = timefit(f, X_train, y_train)
fe = Evaluator(f, "MLP")
print(f"Training time for MLP: {ftime:.2f} seconds")
print("MLP Accuracy:", fe.score(X_test, y_test))
fe.cm(X_test, y_test, "viridis")

plt.plot(f.classifier.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('MLP Loss Curve')
plt.savefig('figures/mlp_loss_curve.png')

print("\n\n")
print("Logistic Regression Confusion Matrix:")
print(le.cm)
print("kNN Confusion Matrix:")
print(ke.cm)
print("MLP Confusion Matrix:")
print(fe.cm)