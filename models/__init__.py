# Needed to import from the modules directory
# Alternative approach is to add the directory path to sys.path and then reference files directly

from .knn import KNN
from .logistic import LogReg
from .mlp import MLP

__all__ = ["KNN", "LogReg", "MLP"]