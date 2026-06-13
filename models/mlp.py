# Basic Multi-layer Perceptron
from seed import seed
from sklearn.neural_network import MLPClassifier

class MLP:
    def __init__(self, hidden=(100, 100), lr=1e-3, max_iter=200, batch_size=128, alpha=1e-4, activation='relu'):
        self.classifier = MLPClassifier(
            hidden_layer_sizes=hidden, 
            learning_rate_init=lr,
            max_iter=max_iter, 
            batch_size=batch_size, 
            alpha=alpha, 
            activation=activation, 
            random_state=seed
            )

    def fit(self, X, y):
        self.X_train, self.y_train = X, y
        self.classifier.fit(self.X_train, self.y_train)
    
    def predict(self, X):
        return self.classifier.predict(X)