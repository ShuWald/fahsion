import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mnist_reader
from sklearn.model_selection import train_test_split
from seed import seed

class DataEvaluator:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.clothing_types = {
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
        self.X_temp, self.y_temp = mnist_reader.load_mnist(dataset_path, kind='train')
        self.X_test, self.y_test = mnist_reader.load_mnist(dataset_path, kind='t10k')
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_temp, self.y_temp, test_size=0.15, random_state=seed)

    def get_data(self):
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

    def plot_clothing_distribution(self, save_path='figures/clothing_distribution.png', show=True):
        df_ytrain = pd.DataFrame(self.y_train, columns=['label'])
        df_yval = pd.DataFrame(self.y_val, columns=['label'])
        df_ytest = pd.DataFrame(self.y_test, columns=['label'])
        train_labels = df_ytrain['label'].value_counts().sort_index()
        val_labels = df_yval['label'].value_counts().sort_index()
        test_labels = df_ytest['label'].value_counts().sort_index()

        plt.figure(figsize=(10, 5))
        plt.bar([self.clothing_types[i] for i in train_labels.index], train_labels.values, alpha=0.65, label='Train')
        plt.bar([self.clothing_types[i] for i in val_labels.index], val_labels.values, alpha=0.65, label='Validation')
        plt.bar([self.clothing_types[i] for i in test_labels.index], test_labels.values, alpha=0.65, label='Test')
        plt.xlabel('Clothing')
        plt.ylabel('Count')
        plt.title('Distribution of Clothing Types in Train, Validation, and Test Sets')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        if show:
            plt.show()
        return train_labels, val_labels, test_labels