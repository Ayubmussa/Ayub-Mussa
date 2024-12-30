import numpy as np
from utils.data_preprocessing import load_data, preprocess_data
from utils.dimensionality_reduction import apply_pca, feature_importance, correlation_analysis
from models.model_implementation import train_decision_tree, train_neural_network
from utils.evaluation import compare_models
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    try:
       
        data = load_data('data/heart.csv')
        if data is None:
            print("Failed to load data.")
            return
    except FileNotFoundError:
        print("Dataset not found. Please check the path to the dataset.")
        return
    except Exception as e:
        print(f"Unexpected error during data loading: {e}")
        return

    try:
        processed_data = preprocess_data(data)
        if processed_data is None or processed_data.empty:
            print("Data preprocessing failed.")
            return
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return

    
    try:
        features = processed_data.drop(columns=['target'])
        target = processed_data['target']
    except KeyError:
        print("Target column 'target' not found in the dataset.")
        return

    
    try:
        pca_components = apply_pca(features)
        feature_importances = feature_importance(features, target)
        correlations = correlation_analysis(features)
    except Exception as e:
        print(f"Error in dimensionality reduction: {e}")
        return

    print("PCA Components Head:")
    print(pca_components.head())
    
    print("\nFeature Importances:")
    print(feature_importances)
    
    print("\nCorrelation Matrix:")
    print(correlations)

   
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    try:
        
        dt_model = train_decision_tree(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)

        
        input_shape = X_train.shape[1]  
        nn_model = train_neural_network(X_train, y_train, input_shape=input_shape)
        y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(np.int32)  
    except Exception as e:
        print(f"Error in model training or prediction: {e}")
        return

    
    try:
        compare_models(y_test, y_pred_dt, y_pred_nn)
    except Exception as e:
        print(f"Error in model evaluation: {e}")

if __name__ == "__main__":
    main()
