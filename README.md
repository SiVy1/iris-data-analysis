# Iris Data Analysis

This project contains a set of data analysis and machine learning exercises based on synthetic Iris-like data. It includes exploratory data analysis, correlation analysis, K-Means clustering and KNN classification.

The original academic datasets are not included. Publicly available example files were replaced with synthetic data that preserves the same structure required by the scripts.

## Project structure

- `data/` - synthetic example datasets
- `src/` - Python scripts
- `plots/` - generated charts and visualizations
- `requirements.txt` - required Python packages

## Technologies

Python, Pandas, NumPy, Matplotlib, Scikit-learn

## Included analyses

- Exploratory data analysis
- Descriptive statistics
- Pearson correlation and linear regression
- K-Means clustering
- Elbow method / WCSS analysis
- KNN classification
- Accuracy comparison and confusion matrix

## How to run

```bash
pip install -r requirements.txt
python src/eda_iris_analysis.py
python src/kmeans_clustering.py
python src/knn_classification.py
