import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from typing import Optional
from sklearn.datasets import load_iris

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, max_depth: Optional[int] = None, random_state: Optional[int] = None) -> DecisionTreeClassifier:
   
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise ValueError("X_train must be a pandas DataFrame and y_train must be a pandas Series.")
    
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier

def train_neural_network(X_train: pd.DataFrame, y_train: pd.Series, input_shape: int, epochs: int = 10, batch_size: int = 32, verbose: int = 1, validation_split: float = 0.2) -> tf.keras.Model:
    
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise ValueError("X_train must be a pandas DataFrame and y_train must be a pandas Series.")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    try:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)
    except Exception as e:
        print(f"Error during training: {e}")
    
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    
    if isinstance(model, DecisionTreeClassifier):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
    elif isinstance(model, tf.keras.Model):
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    else:
        raise ValueError("Model must be a DecisionTreeClassifier or tf.keras.Model.")
    
    print(f"Accuracy: {accuracy}")
    return accuracy

def compare_models(y_test: np.ndarray, y_pred_dt: np.ndarray, y_pred_nn: np.ndarray):
    """Compare model performance."""
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    nn_accuracy = accuracy_score(y_test, y_pred_nn)
    
    print(f"Decision Tree Accuracy: {dt_accuracy}")
    print(f"Neural Network Accuracy: {nn_accuracy}")


if __name__ == "__main__":
    
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    
    X_binary = X[y != 2]
    y_binary = y[y != 2]


    X_binary = np.array(X_binary)
    y_binary = np.array(y_binary)

    
    X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

   
    print("Training Decision Tree...")
    dt_model = train_decision_tree(X_train_dt, y_train_dt, max_depth=5, random_state=42)
    dt_accuracy = evaluate_model(dt_model, X_test_dt, y_test_dt)
    print(f"Decision Tree Accuracy: {dt_accuracy}\n")
    
    
    print("Training Neural Network...")
    input_shape = X_train_nn.shape[1]
    nn_model = train_neural_network(X_train_nn, y_train_nn, input_shape, epochs=10, batch_size=32, verbose=1, validation_split=0.2)
    nn_accuracy = evaluate_model(nn_model, X_test_nn, y_test_nn)
    print(f"Neural Network Accuracy: {nn_accuracy}")
    
    
    print("\nComparing Models...")
    y_pred_dt = dt_model.predict(X_test_dt)
    y_pred_nn = (nn_model.predict(X_test_nn) > 0.5).astype(int)  
    compare_models(y_test_dt, y_pred_dt, y_pred_nn)
