import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np 

def train_decision_tree(X_train, y_train, max_depth=None, random_state=None):
   
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier


data = pd.read_csv('data/heart.csv')


features = data.drop(columns=['target'])
target = data['target']


scaler = StandardScaler()
features = scaler.fit_transform(features)


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


dt_classifier = train_decision_tree(X_train, y_train, max_depth=5, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy (Neural Network): {accuracy:.4f}')


y_pred_nn = (model.predict(X_test) > 0.5).astype("int32")
print(f'Sklearn Accuracy (Neural Network): {accuracy_score(y_test, y_pred_nn):.4f}')


y_pred_dt = dt_classifier.predict(X_test)
print(f'Sklearn Accuracy (Decision Tree): {accuracy_score(y_test, y_pred_dt):.4f}')
