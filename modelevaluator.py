import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

class Evaluator:
    def __init__(self, model, name):
        self.model, self.modelname = model, name
        self.classifier = model.classifier
        self.cache = {}
    
    def cached_pred(self, X):
        x_id = id(X)
        if x_id in self.cache:
            return self.cache[x_id]

        y_pred = self.classifier.predict(X)
        self.cache[x_id] = y_pred
        if len(self.cache) > 50:
            self.cache.pop(next(iter(self.cache)))
        return y_pred

    def cm(self, X, y_true, color="viridis"):
        if self.modelname == "kNN":
            y_pred = self.cached_pred(X)
        else:
            y_pred = self.classifier.predict(X)
        self.cm = confusion_matrix(y_true, y_pred)
        self.cmdisp = ConfusionMatrixDisplay(confusion_matrix=self.cm)
        self.cmdisp.plot(cmap=color)
        plt.title(f'Confusion Matrix for {self.modelname}')
        plt.savefig(f'figures/{self.modelname}_confusion_matrix.png')
        plt.show()

    def score(self, X, y_true):
        if self.modelname == "kNN":
            y_pred = self.cached_pred(X)
        else:
            y_pred = self.classifier.predict(X)
        return accuracy_score(y_true, y_pred)
    
    def evals(self, X, y_true, color="viridis"):
        score = self.score(X, y_true)
        print(f"{self.modelname} Accuracy:", score)
        self.cm(X, y_true, color=color)
        return score, self.cm

