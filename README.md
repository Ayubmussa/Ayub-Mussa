# AI Term Project

## Overview
The following project solves an existing problem in the domain of [healthcare], applying AI techniques in pre-processing data, feature selection, implementation of a model, and analysis of results.

## Problem Description
The problem to be solved will be to predict patients suffering from heart diseases using AI techniques

## Data Source
- **Source**: [https://www.kaggle.com/code/desalegngeb/heart-disease-predictions/input]

## Installation Instructions
How to set up the environment and run the project.

1. Clone the repository:
      bash
   git clone https://github.com/Ayubmussa/Ayub-Mussa.git
2. Install the needed packages by running:
      bash
   pip install -r requirements.txt

## Dependencies
- **Python**
- **TensorFlow**
- **Pandas**
- **Scikit-learn**
- **numpy,**
which are placed in requirements.txt file

## Usage
1. Place the dataset into the `data/` directory.
2. Run the main script to execute the project workflow:
      bash
   python main.py
   

## Project Structure
- **data/**: Data files or instructions to download the data.
- **models/**: Scripts containing traditional ML, neural network model implementation.
- **utils/**: Utilities for data preprocessing, dimensionality reduction, and evaluation.
- **main.py**: The main script that runs data processing, trains the models, and evaluates the performance.
- **requirements.txt**: List of all Python packages needed.
- **README.md**: This file.

## Results
1. Data Preprocessing:
Data Loading: Dataset was loaded from data/your_dataset.csv.
Data Cleaning and Transformation: The preprocessing step cleaned and transformed the data into an appropriate format for model training. This includes handling missing values, encoding categorical variables, and scaling numerical features.

2. Dimensionality Reduction:
PCA- Principal Component Analysis:

Applied PCA to the features to reduce the dimension and retain the most important principal components.
Key Insight: The first few principal components explained the majority of the variance in the data, hence dimensionality reduction was effective in summarizing the dataset.
Feature Importance:

Feature importance analysis was done to identify those features that are most influential for the target variable.
Key Insight: Some features were found to be more important when compared with others, hence guiding us on which features to focus more on for model improvement.
Correlation Analysis:

Feature correlation analysis was done to understand the relationships between features.
Key Insight: A few of the features were highly correlated and hence redundant. This insight has been useful for further feature selection and engineering.

3. Model Training and Evaluation:
Decision Tree Model:

The Decision Tree model was trained and evaluated on the dataset.
Performance Metrics: Accuracy, precision, recall, and F1-score were calculated to assess the model's performance.
Key Insight: Though the Decision Tree model clearly explained feature splits and decision rules, it might also be prone to over-fitting. Neural Network Model: A Neural Network model was also trained and tested. Performance Metrics: It will be evaluated on similar metrics: accuracy, precision, recall, F1-score. Key Insight: The more powerful Neural Network model comes at the price of painful hyperparameter tunning and higher computational consumption. 

4. Comparing Models
Decision Tree vs Neural Network Comparison:
Performance metrics for both models were compared.
Key Insight: Comparing the two models provided strengths and weaknesses of each model. For example, a Decision Tree can be more interpretable but less flexible, while a Neural Network might give better performance at the cost of interpretability and computational resources.

5. Visualizations:
PCA Components Visualization:
Visualizing the first few principal components provided valuable insights into the distribution of data and how well the dimensionality reduction was done.
Feature Importance Visualization:
This can be visualized with a bar chart or any other visualization tool that can display feature importance.
Correlation Matrix Heatmap:
One may use a correlation matrix heatmap to view the features' correlations.
Conclusion
Future Work
More Model Implementation:
Ensemble Methods: Using ensemble methods like Random Forest, GBM, or XGBoost could provide better predictive performance by leveraging the strengths of multiple models.
SVMs: Exploration of SVMs, especially for classification tasks, may yield strong performance, especially in high-dimensional spaces.
KNN: For some datasets, KNN can be effective and easy to implement for comparison.
 Other Data Sources Exploration:
Additional datasets - Integration of more datasets about the one that already exists to enrich the feature space for the model and improve the accuracy.
Data augmentation - if the data is either images or text, perform some data augmentation techniques so as to increase artificially both size and variability of the dataset.

## Acknowledgments
-[GHAZAAL SHEIKHI, FUNDAMENTALS OF ARTIFICIAL INTELLIGENCE]
