# Elastic Net Regression

## Overview

Elastic Net Regression is a regularized linear regression technique that combines **L1 (Lasso)** and **L2 (Ridge)** regularization.

It is useful when the dataset contains multiple features and there may be multicollinearity between them. Elastic Net can both shrink model coefficients and perform feature selection.

---

## Dataset

The project uses a **California Housing-style dataset** for predicting house values.

### Features

* `MedInc`
* `HouseAge`
* `AveRooms`
* `AveBedrms`
* `Population`
* `AveOccup`
* `Latitude`
* `Longitude`

### Target

* `Median House Value`

The unnecessary `Unnamed: 0` column was removed during preprocessing.

---

## Objective

The objective of this project is to:

* Perform data preprocessing
* Identify and remove outliers using the IQR method
* Prepare features and target variables
* Split the data into training and testing sets
* Apply feature scaling
* Train an Elastic Net Regression model
* Predict house values
* Evaluate model performance using regression metrics

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Removed the unnecessary `Unnamed: 0` column.
3. Checked the dataset structure and missing values.
4. Identified outliers using the **Interquartile Range (IQR)** method.
5. Considered outliers in:

   * `MedInc`
   * `AveRooms`
   * `Population`
   * `AveOccup`
   * `Median House Value`
6. Removed the identified outliers.
7. Separated independent variables (`X`) and the target variable (`y`).

---

## Train-Test Split

The cleaned dataset was divided into training and testing sets.

* **Training data:** 80%
* **Testing data:** 20%
* **Random state:** 42

This allows the model to learn from the training data and evaluate its performance on unseen test data.

---

## Feature Scaling

Since Elastic Net uses regularization, feature scaling is important.

`StandardScaler` was used to standardize the features before training the model.

The scaler was fitted only on the training data and then applied to both training and testing data.

---

## Elastic Net Regression

Elastic Net combines two types of regularization:

### L1 Regularization

L1 regularization is used by Lasso Regression.

It can reduce some coefficients to exactly zero, which can help with feature selection.

### L2 Regularization

L2 regularization is used by Ridge Regression.

It shrinks coefficients toward zero and helps control model complexity.

### Elastic Net

Elastic Net combines both:

**Elastic Net = L1 + L2 Regularization**

The model was implemented using:

```python
ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,
    random_state=42
)
```

### Important Parameters

#### `alpha`

Controls the overall strength of regularization.

* Smaller `alpha` → weaker regularization
* Larger `alpha` → stronger regularization

#### `l1_ratio`

Controls the balance between L1 and L2 regularization.

* `l1_ratio = 0` → Ridge-like behavior
* `l1_ratio = 1` → Lasso-like behavior
* `l1_ratio = 0.5` → Balanced combination of L1 and L2

---

## Model Training

The Elastic Net model was trained using the scaled training data.

```python
elastic_model.fit(X_train_scaled, y_train)
```

The trained model was then used to predict house values for the test dataset.

```python
y_pred_elastic = elastic_model.predict(X_test_scaled)
```

---

## Model Evaluation

The model was evaluated using the following regression metrics:

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

Lower MSE indicates better performance.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

Lower MAE indicates better performance.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and represents the error in the same units as the target.

Lower RMSE indicates better performance.

### R² Score

R² measures how much of the variation in the target variable is explained by the model.

A higher R² generally indicates better predictive fit.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook / VS Code

---

## Machine Learning Concepts

This project helped practice:

* Regression
* Regularization
* L1 Regularization
* L2 Regularization
* Elastic Net
* Outlier Detection
* IQR Method
* Train-Test Split
* Feature Scaling
* Model Training
* Model Prediction
* Regression Evaluation Metrics

---

## Key Learning

Through this practical, I learned how Elastic Net combines the advantages of Ridge and Lasso Regression.

I also learned why feature scaling is important for regularized regression models and how outlier handling can affect the training dataset and model performance.

---

## Conclusion

Elastic Net Regression was implemented on a California Housing-style dataset after preprocessing and outlier handling.

The model was trained using standardized features and evaluated using MSE, MAE, RMSE, and R² Score.

This practical provided hands-on understanding of **regularized regression and the combination of L1 and L2 penalties**.

---


⭐ Part of my Machine Learning Practical Series
