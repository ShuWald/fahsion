import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize


@dataclass
class Evals:
    cm: np.ndarray
    class_probabilities: np.ndarray
    y_pred: np.ndarray
    accuracy: float
    precision: float
    recall: float
    f1: float
    ovr: dict
    ovo_best: list
    ovo_worst: list

class Evaluator:
    def __init__(self, model, name):
        self.model, self.modelname = model, name
        self.classifier = model.classifier

    def cm(self, y_pred, y_true, color="viridis"):
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        self.cmdisp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix)
        self.cmdisp.plot(cmap=color)
        plt.title(f'Confusion Matrix for {self.modelname}')
        plt.savefig(f'figures/{self.modelname}_confusion_matrix.png')
        plt.show()
        self.confusion_histogram()
        return self.confusion_matrix

    def confusion_histogram(self):
        classes = self.classifier.classes_
        pairs = list(combinations(range(len(classes)), 2))
        if not pairs:
            return None

        values = [(self.confusion_matrix[i, j], self.confusion_matrix[j, i]) for i, j in pairs]
        labels = [f"{classes[i]} vs {classes[j]}" for i, j in pairs]
        x = np.arange(len(pairs))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(14, len(pairs) * 0.55), 6))
        ax.bar(x - width / 2, [pair[0] for pair in values], width, label='a as b')
        ax.bar(x + width / 2, [pair[1] for pair in values], width, label='b as a')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_xlabel('Class Pair')
        ax.set_ylabel('Count')
        ax.set_title(f'Pairwise Confusions for {self.modelname}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'figures/{self.modelname}_confusion_histogram.png')
        plt.show()
        return values

# Accuracy, Precision, Recall, and F1 Score
    def score(self, y_pred, y_true):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"{self.modelname} Accuracy:", acc, " Precision:", prec, " Recall:", rec, " F1 Score:", f1)
        return acc, prec, rec, f1
    
# Ideally the optimal function to evaluate a model without recomputing predictions or other metrics
    def evals(self, X, y_true, color="viridis", nbest=3, nworst=5):
        y_pred = self.classifier.predict(X)
        if hasattr(self.classifier, "predict_proba"):
            y_score = self.classifier.predict_proba(X)
        else:
            y_score = self.classifier.predict(X)
        acc, prec, rec, f1 = self.score(y_pred, y_true)
        cm = self.cm(y_pred, y_true, color=color)
        ovr = self.rocauc_allclass(y_score, y_true)
        ovo_best = self.rocauc_worstn(y_score, y_true, y_pred, n=nbest, worst=False)
        ovo_worst = self.rocauc_worstn(y_score, y_true, y_pred, n=nworst, worst=True)
        return Evals(cm, y_score, y_pred, acc, prec, rec, f1, ovr, ovo_best, ovo_worst)

# for each class, run roc/auc against all other classes
    def rocauc_allclass(self, y_score, y_true):
        classes = self.classifier.classes_
        y_true_bin = label_binarize(y_true, classes=classes)
        values = {}
        fig, ax = plt.subplots()

        for i, cls in enumerate(classes):
            display = RocCurveDisplay.from_predictions(
                y_true_bin[:, i],
                y_score[:, i],
                name=str(cls),
                ax=ax,
            )
            values[cls] = display.roc_auc
            print(f"{self.modelname} class {cls} AUC:", values[cls])
        ax.set_title(f"ROC Curves for {self.modelname} (OVR)")
        fig.savefig(f'figures/{self.modelname}_roc_auc_ovr.png')
        plt.show()
        return values

# runs roc/auc on pairs of classes, outputting roc/auc of the n best or worst class pairs based on confusion matrix
    def rocauc_worstn(self, y_score, y_true, y_pred, n=5, worst=True):
        classes = self.classifier.classes_
        cmatrix = confusion_matrix(y_true, y_pred, labels=classes)
        pairs = []
        label = "worst" if worst else "best"
        fig, ax = plt.subplots()

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                pairs.append((cmatrix[i, j] + cmatrix[j, i], i, j))

        pairs.sort(reverse=worst)
        values = []
        for _, i, j in pairs[:n]:
            mask = (y_true == classes[i]) | (y_true == classes[j])
            y_pair = (y_true[mask] == classes[j]).astype(int)
            if len(np.unique(y_pair)) < 2:
                continue
            display = RocCurveDisplay.from_predictions(
                y_pair,
                y_score[mask, j],
                name=f"{classes[i]} vs {classes[j]}",
                ax=ax,
            )
            pair_auc = display.roc_auc
            values.append((classes[i], classes[j], pair_auc))
            print(f"{self.modelname} pair {classes[i]} vs {classes[j]} AUC:", pair_auc)
        ax.set_title(f"ROC Curves for {self.modelname} (OVO {label})")
        fig.savefig(f'figures/{self.modelname}_roc_auc_ovo_{label}.png')
        plt.show()
        return values