# Basic Logistic Regression
from seed import seed
from sklearn.linear_model import LogisticRegression
import numpy as np

class LogReg:
    def __init__(self, C=1.0, solver='lbfgs', max_iter=1000, tol=1e-4):
        self.classifier = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            random_state=seed
        )

    def fit(self, X, y):
        self.X_train, self.y_train = X, y
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.classifier.predict(X)