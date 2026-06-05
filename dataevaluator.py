import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mnist_reader


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
        self.X_train, self.y_train = mnist_reader.load_mnist(dataset_path, kind='train')
        self.X_test, self.y_test = mnist_reader.load_mnist(dataset_path, kind='t10k')

    def get_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def plot_clothing_distribution(self, save_path='figures/clothing_distribution.png', show=True):
        df_ytrain = pd.DataFrame(self.y_train, columns=['label'])
        df_y_test = pd.DataFrame(self.y_test, columns=['label'])
        train_labels = df_ytrain['label'].value_counts().sort_index()
        test_labels = df_y_test['label'].value_counts().sort_index()

        plt.figure(figsize=(10, 5))
        plt.bar([self.clothing_types[i] for i in train_labels.index], train_labels.values, alpha=0.65, label='Train')
        plt.bar([self.clothing_types[i] for i in test_labels.index], test_labels.values, alpha=0.65, label='Test')
        plt.xlabel('Clothing')
        plt.ylabel('Count')
        plt.title('Distribution of Clothing Types in Train and Test Sets')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        if show:
            plt.show()
        return train_labels, test_labels