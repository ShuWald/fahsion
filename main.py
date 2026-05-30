# Python file version of main.ipynb
# ideally we will be using the notebook since its easier to run and edit without rerunning all code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import mnist_reader
from models.knn import KNN
from models.mlp import MLP
from models.logistic import LogReg
from seed import seed

X_train, y_train = mnist_reader.load_mnist('dataset', kind='train')
X_test, y_test = mnist_reader.load_mnist('dataset', kind='t10k')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
