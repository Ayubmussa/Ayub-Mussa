import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf

def load_data(filepath):
    """Load dataset from a CSV file."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """Preprocess the data by handling missing values and normalizing."""
    scaler = StandardScaler()
    features = data.drop(columns=['target'])
    target = data['target']
    
    
    features_scaled = scaler.fit_transform(features)
    
    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train, max_depth=None, random_state=None):
    """Train a decision tree classifier."""
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier

def train_neural_network(X_train, y_train, input_shape):
    """Train a neural network model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    return model

def evaluate_model(y_true, y_pred, average='binary'):
    """Evaluate the model using various metrics."""
   
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    return metrics

def compare_models(y_true, y_pred_dt, y_pred_nn, average='binary'):
    """Compare performance between two models."""
    print("Decision Tree Model Evaluation:")
    metrics_dt = evaluate_model(y_true, y_pred_dt, average)
    
    print("\nNeural Network Model Evaluation:")
    metrics_nn = evaluate_model(y_true, y_pred_nn, average)
    
    return metrics_dt, metrics_nn


if __name__ == "__main__":
   
    filepath = 'c:\\Users\\ayoub\\Desktop\\Ayub Mussa\\data\\heart.csv'
    data = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    
    dt_classifier = train_decision_tree(X_train, y_train, max_depth=5, random_state=42)
    nn_model = train_neural_network(X_train, y_train, input_shape=X_train.shape[1])
    
   
    y_pred_dt = dt_classifier.predict(X_test)
    y_pred_nn = (nn_model.predict(X_test) > 0.5).astype("int32").flatten() 
    
    
    metrics_dt, metrics_nn = compare_models(y_test, y_pred_dt, y_pred_nn)
