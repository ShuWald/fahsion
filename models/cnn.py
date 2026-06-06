from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

class CNN:
    def __init__(self):
        self.classifier = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        
        self.classifier.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def fit(self, X, y):
        self.history = self.classifier.fit(
            X,
            y,
            epochs=10,
            batch_size=64,
            validation_split=0.1,
            verbose=1
        )

    def predict(self, X):
        probs = self.classifier.predict(X)
        return probs.argmax(axis=1)