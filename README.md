# Iris Flower Classification

## Project Overview

This project builds a machine learning classification model to predict Iris flower species using sepal and petal measurements.

## Dataset

The dataset contains 150 Iris flower samples and 5 columns.

### Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Target

- Species

## Exploratory Data Analysis (EDA)

The following exploratory data analysis was performed:

- Dataset information was examined.
- Missing values were checked.
- Statistical summary was generated.
- Species distribution was visualized.
- Scatter plots were used to study class separability.
- Pair plot was used to visualize relationships between features.
- Box plots were used to examine feature distributions.

## Machine Learning Algorithms

The following classification algorithms were trained and compared:

1. k-Nearest Neighbors (k-NN)
2. Logistic Regression
3. Decision Tree

## Model Evaluation

The models were evaluated using the following metrics:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score

## Results

| Model | Accuracy |
|---|---:|
| k-NN | 100.00% |
| Logistic Regression | 96.67% |
| Decision Tree | 93.33% |

The k-NN model achieved the highest accuracy on the selected test split.

## Saved Model

The best-performing model on the selected test split was k-NN.

The trained model was saved using Joblib as:

`iris_knn_model.joblib`

## Example Inference

The saved k-NN model can be loaded and used to predict the species of a new Iris flower.

```python
import joblib

loaded_model = joblib.load('iris_knn_model.joblib')

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = loaded_model.predict(new_flower)

print("Predicted class:", prediction[0])
