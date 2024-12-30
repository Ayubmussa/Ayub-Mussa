# AI Term Project

## Overview
This project applies AI techniques to solve a real-world problem in [healthcare], involves data preprocessing, feature selection, model implementation, and performance analysis.

## Problem Description
The problem involves predicting patients with heart diseases using AI techniques

## Data Source
- **Source**: [https://www.kaggle.com/code/desalegngeb/heart-disease-predictions/input]

## Installation Instructions
To set up the environment and run the project, follow these steps:

1. Clone the repository:
      bash
   git clone [https://console.cloud.google.com/artifacts/docker/kaggle-images/us/gcr.io/python/sha256:a6241bd0d98e81ea21f4cfba26ef48fc159646b112fc4986113d5ab219c6c26f?pli=1]
   

2. Install the required dependencies:
      bash
   pip install -r requirements.txt

## Required dependencies
- **Python**
- **TensorFlow**
- **Pandas**
- **Scikit-learn**
- **numpy**,
 whhich are located in requirements.txt file

## Usage
1. Ensure the dataset is in the `data/` directory.
2. Run the main script to execute the project workflow:
      bash
   python main.py
   

## Project Structure
- **data/**: Contains data files or instructions for downloading them.
- **models/**: Includes scripts for traditional ML, neural network models and model implementation.
- **utils/**: Contains utilities for data preprocessing, dimensionality reduction, and evaluation.
- **main.py**: The main script orchestrating data processing, model training, and evaluation.
- **requirements.txt**: Lists all the required Python packages.
- **README.md**: This document.

## Results
1. Data Preprocessing:
Data Loading: The dataset was successfully loaded from data/your_dataset.csv.
Data Cleaning and Transformation: The preprocessing step ensured that the data was cleaned and transformed into a suitable format for model training. This included handling missing values, encoding categorical variables, and normalizing numerical features.

2. Dimensionality Reduction:
PCA (Principal Component Analysis):

PCA was applied to the features to reduce dimensionality and capture the most important components.
Key Insight: The first few principal components explained a significant portion of the variance in the data, indicating that dimensionality reduction was effective in summarizing the dataset.
Feature Importance:

Feature importance analysis was conducted to identify the most influential features for the target variable.
Key Insight: Certain features were found to be more important than others, guiding us on which features to focus on for model improvement.
Correlation Analysis:

Correlation analysis was performed to understand the relationships between features.
Key Insight: Some features were highly correlated, indicating potential redundancy. This insight was useful for further feature selection and engineering.

3. Model Training and Evaluation:
Decision Tree Model:

The Decision Tree model was trained and evaluated on the dataset.
Performance Metrics: Accuracy, precision, recall, and F1-score were calculated to assess the model’s performance.
Key Insight: The Decision Tree model provided a clear understanding of feature splits and decision rules, but it might be prone to overfitting.
Neural Network Model:

A Neural Network model was also trained and evaluated.
Performance Metrics: Similar metrics (accuracy, precision, recall, F1-score) were used to evaluate the neural network.
Key Insight: The Neural Network model, while potentially more powerful, required careful tuning of hyperparameters and was more computationally intensive.

4. Model Comparison:
Comparison of Decision Tree and Neural Network:
Both models were compared based on their performance metrics.
Key Insight: The comparison revealed the strengths and weaknesses of each model. For instance, the Decision Tree might be more interpretable but less flexible, while the Neural Network might offer better performance at the cost of interpretability and computational resources.

5. Visualizations:
PCA Components Visualization:
Visualizations of the first few principal components helped in understanding the data distribution and the effectiveness of dimensionality reduction.
Feature Importance Visualization:
Bar charts or other visual tools were used to illustrate the importance of different features.
Correlation Matrix Heatmap:
A heatmap of the correlation matrix provided a visual representation of feature correlations.

## Future Work
 Implementing Additional Models:
Ensemble Methods: Implementing ensemble methods like Random Forest, Gradient Boosting Machines (GBM), or XGBoost could potentially improve predictive performance by combining the strengths of multiple models.
Support Vector Machines (SVM): Exploring SVMs, especially for classification tasks, could provide robust performance, particularly in high-dimensional spaces.
K-Nearest Neighbors (KNN): For some datasets, KNN can be an effective and simple algorithm to implement and compare.
 Exploring Other Data Sources:
Additional Datasets: Integrating additional datasets that are related to the current dataset to enrich the feature space and improve model accuracy.
Data Augmentation: For image or text data, applying data augmentation techniques to artificially increase the size and variability of the dataset.


## Acknowledgments
- [GHAZAAL SHEIKHI, FUNDAMENTALS OF ARTIFICIAL INTELLIGENCE]

