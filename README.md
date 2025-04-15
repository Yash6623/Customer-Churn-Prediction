# 📊 Customer Churn Prediction

This project uses machine learning techniques to predict customer churn in a telecom company. By analyzing customer attributes, we aim to classify whether a customer is likely to churn or stay, helping businesses take proactive steps in customer retention.

## 🔍 Problem Statement

Customer churn is a major concern in the telecom industry. Predicting which customers are likely to leave allows companies to implement targeted retention strategies. This project implements both a **Random Forest Classifier** and an **Artificial Neural Network (ANN)** to build accurate churn prediction models.

## 🧠 Models Used

- **Random Forest Classifier** (with GridSearchCV for hyperparameter tuning)
- **Artificial Neural Network (ANN)** using TensorFlow/Keras

## 🧹 Data Preprocessing

- Dropped irrelevant columns (e.g., `customerID`)
- Handled missing values (e.g., converted `TotalCharges` to numeric and dropped nulls)
- Encoded categorical variables using `pd.get_dummies`
- Scaled features using `StandardScaler` for ANN compatibility
- Split data into training and testing sets (80/20)

## 🧪 Model Evaluation

### Random Forest
- **Accuracy**: 80%
- **AUC Score**: 0.85
- **Key Features**: TotalCharges, MonthlyCharges, tenure, Contract
- **Evaluation**: Confusion matrix, classification report, ROC curve

### ANN (Artificial Neural Network)
- **Accuracy**: 78%
- **Architecture**: 3 hidden layers (128, 64, 32 units) with dropout regularization
- **EarlyStopping** applied to prevent overfitting
- **Evaluation**: Confusion matrix, classification report

## 📈 Results

- Random Forest slightly outperformed ANN in accuracy and interpretability.
- ROC Curve was plotted for Random Forest showing strong class separation (AUC = 0.85).
- ANN showed competitive performance with deep learning techniques.

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn

## 📌 Future Improvements

- Try additional models (XGBoost, SVM)
- Apply SMOTE for class imbalance
- Model deployment using Flask and AWS (in progress)


