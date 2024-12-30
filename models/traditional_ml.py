import pandas as pd
import numpy as np  
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('data/heart.csv')


features = data.drop(columns=['target'])
target = data['target']


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


features_scaled = np.array(features_scaled)  
target = np.array(target)  


X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)


dt_classifier = DecisionTreeClassifier(random_state=42)


param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}


grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")


best_dt_classifier = grid_search.best_estimator_
best_dt_classifier.fit(X_train, y_train)


y_pred = best_dt_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.2f}")


print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


feature_importance = best_dt_classifier.feature_importances_


for name, importance in zip(features.columns, feature_importance):
    print(f"{name}: {importance:.4f}")
