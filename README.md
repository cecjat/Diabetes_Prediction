# Diabetes Prediction Model

## Author
Cecille Jatulan

## Introduction
This project aims to build a predictive model to predict the likelihood of a patient having diabetes based on certain features. The dataset used contains information about the medical history of patients, including features like Glucose level, Blood Pressure, BMI, etc., and a target variable indicating whether the patient has diabetes (1) or not (0).

## Requirements
- Python 3.x
- Libraries:
  - numpy
  - pandas
  - seaborn
  - matplotlib
  - scikit-learn

## Usage
1. Install the required libraries by running: `pip install -r requirements.txt`
2. Run the Jupyter notebook `Diabetes_Prediction.ipynb` to train the model, perform data analysis, and evaluate the model's performance.
3. After training and tuning the model, you can save the final model using pickle for later use.

## Contents
- **Diabetes_Prediction.ipynb**: Jupyter notebook containing the code for building, training, and evaluating the diabetes prediction model.
- **datasets_228_482_diabetes.csv**: Dataset containing the medical history of patients.
- **logistic_regression_model.pkl**: Pickle file containing the trained logistic regression model.
- **tuned_logistic_regression_model.pkl**: Pickle file containing the tuned logistic regression model after hyperparameter tuning.

## Instructions
1. Explore the dataset to understand its structure and contents.
2. Perform necessary data preprocessing steps such as handling missing values, encoding categorical variables, and scaling numerical features.
3. Split the dataset into training and testing sets.
4. Build a Logistic Regression model to predict the likelihood of diabetes based on the features provided.
5. Evaluate the model using appropriate metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
6. Tune the model's hyperparameters using techniques like GridSearchCV.
7. Interpret the model coefficients to understand the impact of different features on the likelihood of diabetes.
8. Save the trained and tuned model for future use.

