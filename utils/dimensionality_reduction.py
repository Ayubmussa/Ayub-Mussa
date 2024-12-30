import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_pca(data, n_components=2):
    """
    Apply PCA to reduce dimensionality of the dataset.

    Parameters:
    data (pd.DataFrame or np.ndarray): The input data to be reduced.
    n_components (int): The number of principal components to keep.

    Returns:
    pd.DataFrame: The data transformed to the principal components.
    """
    if data is None or data.empty:
        logging.error("Input data is None or empty.")
        return None
    
    logging.info(f"Applying PCA with {n_components} components.")
    pca = PCA(n_components=n_components)
    
    # Ensure data is in numpy format for PCA (PCA expects numpy arrays)
    data_array = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    components = pca.fit_transform(data_array)
    
    logging.info("PCA transformation completed.")
    return pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(n_components)])

def feature_importance(data, target):
    """
    Determine feature importance using a Random Forest classifier.

    Parameters:
    data (pd.DataFrame or np.ndarray): The input features.
    target (pd.Series or np.ndarray): The target variable.

    Returns:
    pd.Series: The feature importances sorted in descending order.
    """
    if data is None or data.empty or target is None or target.empty:
        logging.error("Input data or target is None or empty.")
        return None
    
    logging.info("Training Random Forest to determine feature importances.")
    
    
    data_array = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    target_array = target.to_numpy() if isinstance(target, pd.Series) else target
    
    model = RandomForestClassifier(random_state=42)
    model.fit(data_array, target_array)
    importances = model.feature_importances_
    
    logging.info("Random Forest training completed.")
    return pd.Series(importances, index=data.columns).sort_values(ascending=False)

def correlation_analysis(data):
    """
    Perform correlation analysis on the dataset.

    Parameters:
    data (pd.DataFrame or np.ndarray): The input data.

    Returns:
    pd.DataFrame: The correlation matrix of the data.
    """
    if data is None or data.empty:
        logging.error("Input data is None or empty.")
        return None
    
    logging.info("Performing correlation analysis.")
    
   
    data_array = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    correlation_matrix = np.corrcoef(data_array, rowvar=False)  
    
    logging.info("Correlation analysis completed.")
    
    
    if isinstance(data, pd.DataFrame):
        correlation_matrix = pd.DataFrame(correlation_matrix, columns=data.columns, index=data.columns)
    return correlation_matrix

if __name__ == "__main__":
    
    filepath = 'c:\\Users\\ayoub\\Desktop\\Ayub Mussa\\data\\heart.csv'
    data = pd.read_csv(filepath)

    
    if data is not None:
       
        features = data.drop(columns=['target'])
        target = data['target']

        
        pca_data = apply_pca(features, n_components=2)
        if pca_data is not None:
            print("PCA Data:")
            print(pca_data.head())

        
        importance = feature_importance(features, target)
        if importance is not None:
            print("Feature Importances:")
            print(importance)

        
        correlation_matrix = correlation_analysis(features)
        if correlation_matrix is not None:
            print("Correlation Matrix:")
            print(correlation_matrix)
    else:
        logging.error("Failed to load data from the provided filepath.")
