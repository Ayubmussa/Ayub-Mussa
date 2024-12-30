import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"The file at {filepath} was not found.")
        return None

def handle_missing_values(data):
    """Handle missing values in the dataset."""
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    logging.info("Missing values handled.")
    return data_imputed

def create_new_features(data):
    """Create new features for the dataset."""
    if 'age' in data.columns and 'chol' in data.columns:
       
        data['age_chol'] = np.multiply(data['age'], data['chol'])  
        logging.info("New feature 'age_chol' created.")
    else:
        logging.warning("'age' or 'chol' columns are missing. New feature not created.")
    return data

def normalize_features(data):
    """Normalize the features in the dataset."""
    feature_columns = data.drop(columns=['target']).columns if 'target' in data.columns else data.columns
    scaler = StandardScaler()

    
    data[feature_columns] = scaler.fit_transform(data[feature_columns].values)  
    logging.info("Features normalized.")
    return data

def preprocess_data(data):
    """Preprocess the data: handle missing values, feature engineering, and normalization."""
    if data is None:
        logging.error("No data provided.")
        return None

    logging.info("Starting data preprocessing...")

    
    data = handle_missing_values(data)

    
    data = create_new_features(data)

    
    data = normalize_features(data)

    logging.info("Data preprocessing completed.")
    return data


filepath = 'c:\\Users\\ayoub\\Desktop\\Ayub Mussa\\data\\heart.csv'  
data = load_data(filepath)
processed_data = preprocess_data(data)

if processed_data is not None:
    print(processed_data.head())
