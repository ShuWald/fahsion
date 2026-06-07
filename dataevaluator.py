import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mnist_reader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    def get_data(self): 
        self.scaler = StandardScaler()
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_temp, self.y_temp, test_size=0.15, random_state=seed)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

    def plot_clothing_distribution(self, save_path='figures/clothing_distribution.png', show=True):
        df_train, df_val, df_test = [pd.DataFrame(y, columns=['label']) for y in (self.y_train, self.y_val, self.y_test)]
        train_labels, val_labels, test_labels = [df['label'].value_counts().sort_index() for df in (df_train, df_val, df_test)]
        labels = [self.clothing_types[i] for i in train_labels.index]

        plt.figure(figsize=(10, 5))
        plt.bar(labels, train_labels.values, alpha=0.65, label='Train')
        plt.bar(labels, val_labels.values, bottom=train_labels.values, alpha=0.65, label='Validation')
        plt.bar(labels, test_labels.values, bottom=train_labels.values + val_labels.values, alpha=0.65, label='Test')
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