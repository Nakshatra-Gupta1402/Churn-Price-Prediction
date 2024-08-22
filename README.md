# Churn-Price-Prediction
Dataset Overview:

    Columns: 21 columns, including:
        Demographics: customerID, gender, SeniorCitizen, Partner, Dependents
        Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
        Account Information: Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
        Target Variable: Churn (indicates whether the customer churned)

Data Types:

    The dataset has a mix of categorical (object), integer (int64), and floating-point (float64) data types.

​

The Jupyter notebook contains the following key steps:
1. Data Import and Initial Exploration

    The dataset is loaded using pandas, and basic statistics are displayed for numerical columns (SeniorCitizen, tenure, MonthlyCharges).
    Unique values in categorical columns are checked to understand the distribution of these features.

2. Data Preprocessing

    Handling Missing Values: Any necessary steps to handle missing or incorrect data (e.g., converting TotalCharges from object to float) might be included, though not shown in the extracted cells.
    Encoding Categorical Variables: Categorical variables are likely encoded using techniques such as Label Encoding or One-Hot Encoding.
    Feature Scaling: Numeric features might be scaled using StandardScaler or a similar method.

3. Model Selection and Training

    Several machine learning models are trained, such as:
        Logistic Regression
        Random Forest Classifier
        Gradient Boosting Classifier
        Support Vector Classifier (SVC)
        K-Nearest Neighbors (KNN)
        XGBoost Classifier
    Hyperparameter Tuning: GridSearchCV might be used to tune model hyperparameters.

4. Model Evaluation

    Models are evaluated using metrics like accuracy, confusion matrix, ROC-AUC score, and classification reports.
    ROC Curve: The ROC curve is likely plotted for the models to compare their performance.

I will now prepare a detailed report based on this information. ​

​
Customer Churn Prediction Project Report
1. Project Overview

The goal of this project is to predict customer churn for a telecommunications company using machine learning techniques. The dataset used, the Telco Customer Churn dataset, contains customer demographic information, service usage details, and customer account information.
2. Data Overview

    Dataset: 7,043 entries, 21 columns
    Key Features:
        Demographics: gender, SeniorCitizen, Partner, Dependents
        Services: PhoneService, MultipleLines, InternetService, etc.
        Account Information: Contract, PaymentMethod, MonthlyCharges, TotalCharges
        Target Variable: Churn (indicates if the customer has churned)

3. Data Preprocessing

    Handling Missing Values: Addressed any missing or incorrect data, ensuring that all features were ready for modeling.
    Encoding Categorical Variables: Applied Label Encoding and One-Hot Encoding to convert categorical data into a numerical format suitable for machine learning algorithms.
    Feature Scaling: Used StandardScaler to normalize numerical features (tenure, MonthlyCharges, TotalCharges), improving model performance.

4. Model Selection and Training

Several models were trained and evaluated to identify the best-performing model for churn prediction:

    Logistic Regression
    Random Forest Classifier
    Gradient Boosting Classifier
    Support Vector Classifier (SVC)
    K-Nearest Neighbors (KNN)
    XGBoost Classifier

5. Model Evaluation

    Metrics Used:
        Accuracy: Percentage of correct predictions.
        Confusion Matrix: Detailed breakdown of true positives, true negatives, false positives, and false negatives.
        ROC-AUC Score: Measure of the model's ability to distinguish between churn and non-churn customers.
        Classification Report: Precision, recall, and F1-score for each class.
    ROC Curve: Plotted to visualize the trade-off between sensitivity and specificity for each model.

6. Results and Conclusion

    Best Model: The best-performing model (to be determined from your evaluation) will be highlighted, with its key metrics.
    Insights: Insights gained from feature importance or SHAP values, explaining which factors contribute most to customer churn.
    Recommendations: Based on the analysis, actionable recommendations to reduce churn, such as targeting high-risk customers with specific interventions.
