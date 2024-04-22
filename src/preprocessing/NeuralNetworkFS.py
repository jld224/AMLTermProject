import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

class NeuralNetworkFS:
    def __init__(self, input_shape, layers=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001):
        self.model = self.build_model(input_shape, layers, dropout_rate, learning_rate)
        self.history = None
        self.selected_features = None

    def build_model(self, input_shape, layers, dropout_rate, learning_rate):
        model = Sequential()
        model.add(Dense(layers[0], activation='relu', input_shape=(input_shape,)))
        for layer_size in layers[1:]:
            model.add(Dropout(dropout_rate))
            model.add(Dense(layer_size, activation='relu'))
        model.add(Dense(1))  # Output layer for regression
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def fit(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                      validation_split=validation_split, callbacks=[early_stopping])

        # Feature importance based on the weights (just a placeholder, more sophisticated methods can be applied)
        weights = self.model.layers[0].get_weights()[0]
        importance = np.mean(np.abs(weights), axis=1)
        self.selected_features = np.where(importance > np.percentile(importance, 75))[0]  # Selecting the top 25% features

    def transform(self, X):
        if self.selected_features is None:
            raise RuntimeError("Neural network has not been fitted before transformation.")
        return X[:, self.selected_features]

    def plot_training_history(self):
        if self.history is None:
            raise RuntimeError("Model has not been trained yet.")
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()

        # Ensure the 'images' directory exists
        os.makedirs('images', exist_ok=True)
        plt.savefig('images/training_history.svg')
        plt.show()
