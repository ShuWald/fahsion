# Basic Logistic Regression
from seed import seed
from sklearn.linear_model import LogisticRegression

class LogReg:
    def __init__(self, max_iter=1000):
        self.classifier = LogisticRegression(max_iter=max_iter, random_state=seed)

    def fit(self, X, y):
        self.X_train, self.y_train = X, y
        self.classifier.fit(self.X_train, self.y_train)
    
    def predict(self, X):
        return self.classifier.predict(X)