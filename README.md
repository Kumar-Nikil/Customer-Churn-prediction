# <div align="center">Telecom Customer Churn Prediction</div>

![Intro](https://github.com/Kumar-Nikil/Customer-Churn-prediction.git)

# Customer Churn Prediction

## Overview

Customer churn occurs when a customer discontinues their relationship with a business. In highly competitive sectors like telecom, where churn rates range between 15% to 25% annually, retaining customers is more cost-effective than acquiring new ones. This project aims to predict potential churners using customer behavioral and account data to support proactive retention strategies.

## Objectives

- Determine the churn rate and distinguish between retained and lost customers.
- Analyze key features that contribute to customer churn.
- Identify the most effective machine learning model for predicting churn.

## Dataset

Source: [Telco Customer Churn - Kaggle](https://www.kaggle.com/bhartiprasad17/customer-churn-prediction/data)

Features include:
- **Customer info**: tenure, contract type, payment method, billing method.
- **Services**: internet, phone, streaming, tech support, online security.
- **Demographics**: gender, senior citizen status, dependents.
- **Churn**: whether a customer left in the last month.

## Tools & Libraries

- Python, pandas, numpy, seaborn, matplotlib
- Scikit-learn (Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Voting Classifier)

## Exploratory Data Analysis (EDA)

- **Contract Type**: ~75% of monthly contract users churned.
- **Payment Method**: Electronic check users showed higher churn.
- **Service Usage**: Lack of online security and tech support correlated with higher churn.
- **Tenure**: Newer customers were more likely to churn.
- **Charges**: Higher monthly charges increased churn probability.

![Churn Distribution](https://github.com/Pradnya1208/Telecom-Customer-Churn-prediction/blob/main/output/Churn%20Distribution.png?raw=true)

## Model Building

Trained and evaluated multiple models:
- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting

### Final Model: Voting Classifier

Combined Logistic Regression, Gradient Boosting, and AdaBoost using soft voting.  
Achieved an accuracy of **~84.6%**, outperforming individual models.

```python
from sklearn.ensemble import VotingClassifier
clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = AdaBoostClassifier()

eclf1 = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
eclf1.fit(X_train, y_train)
predictions = eclf1.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
