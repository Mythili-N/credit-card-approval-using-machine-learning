
# Credit Card Approval Using Machine Learning

This project uses machine learning techniques to predict credit card approval based on customer features such as age, income, loan amount, credit score, marital status, and gender. The dataset used for training and testing is synthetically generated, and various classification models are trained and evaluated for their predictive performance.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
  - [Data Creation](#data-creation)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Handling Class Imbalance](#handling-class-imbalance)
  - [Feature Importance Analysis](#feature-importance-analysis)
- [Models Used](#models-used)
- [License](#license)

## Overview

This project demonstrates how to predict credit card approval using machine learning models. It generates a synthetic dataset and applies several machine learning models such as Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, and XGBoost to classify customers as either approved or rejected for credit cards.

### Steps:
1. **Data Creation**: If the dataset is not already present, a synthetic dataset is created and saved in CSV format.
2. **Data Preprocessing**: The dataset is cleaned, categorical features are encoded, and numerical features are scaled.
3. **Handling Class Imbalance**: Synthetic data is generated using SMOTE to balance the class distribution.
4. **Model Training**: Multiple machine learning models are trained on the processed dataset.
5. **Model Evaluation**: Models are evaluated using accuracy, confusion matrix, and classification report.
6. **Feature Importance Analysis**: Feature importance is visualized for tree-based models.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/credit-card-approval-ml.git
   ```

2. Navigate to the project directory:

   ```bash
   cd credit-card-approval-ml
   ```

3. Install the required dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   You can create the `requirements.txt` file with the following content:

   ```
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   imbalanced-learn
   xgboost
   ```

## Usage

1. Run the `credit_card_approval.py` script:

   ```bash
   python credit_card_approval.py
   ```

2. The script will:
   - Check if the dataset file (`credit_card_data.csv`) exists and create it if not.
   - Load the dataset and preprocess it.
   - Handle class imbalance using SMOTE.
   - Split the data into training and testing sets.
   - Train multiple machine learning models and evaluate them.
   - Print classification reports, confusion matrices, and accuracy scores for each model.
   - Analyze and plot the feature importance for tree-based models.

## Code Explanation

### Data Creation

The `create_sample_data(file_name)` function generates a synthetic dataset containing the following features:
- `Age`: Age of the customer (18–70 years)
- `Income`: Annual income of the customer (20,000–120,000)
- `LoanAmount`: Loan amount taken by the customer (1,000–50,000)
- `CreditScore`: Customer's credit score (300–850)
- `MaritalStatus`: Customer's marital status (Single, Married, Divorced)
- `Gender`: Customer's gender (Male, Female)
- `Approval`: Target variable (0 = Rejected, 1 = Approved)

### Data Preprocessing

The `preprocess_data(data)` function handles missing values by filling them with the median for numerical columns and the mode for categorical columns. It encodes categorical columns (`MaritalStatus`, `Gender`) using `LabelEncoder` and scales numerical features using `StandardScaler`.

### Model Training

The `train_models(X_train, y_train)` function trains five different machine learning models:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. XGBoost Classifier

These models are trained using the training data and are stored in a dictionary for easy access.

### Model Evaluation

The `evaluate_models(models, X_test, y_test)` function evaluates the performance of each trained model using the test data. It prints out the classification report, confusion matrix, and accuracy score for each model.

### Handling Class Imbalance

The `handle_imbalance(X, y)` function uses **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance in the dataset. SMOTE generates synthetic data points for the minority class to balance the target variable.

### Feature Importance Analysis

The `analyze_feature_importance(models, feature_names)` function analyzes feature importance for tree-based models (Random Forest, Gradient Boosting, XGBoost) by displaying and plotting the importance of each feature.

## Models Used

- **Logistic Regression**: A simple linear model used for binary classification.
- **Decision Tree Classifier**: A tree-based model that makes decisions based on splitting the data into subsets.
- **Random Forest Classifier**: An ensemble of decision trees that improves accuracy by reducing overfitting.
- **Gradient Boosting Classifier**: An ensemble technique that builds models sequentially, correcting errors made by previous models.
- **XGBoost Classifier**: A highly efficient and scalable implementation of gradient boosting.

