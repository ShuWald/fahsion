# Basic Random Forest
from seed import seed
from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', n_jobs=-1):
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=seed
        )

    def fit(self, X, y):
        self.X_train, self.y_train = X, y
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.classifier.predict(X)
