# Basic kNN
from seed import seed
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.classifier = KNeighborsClassifier(n_neighbors=self.k)

    def fit(self, X, y):
        self.X_train, self.y_train = X, y
        self.classifier.fit(self.X_train, self.y_train)
    
    def predict(self, X):
        return self.classifier.predict(X)

