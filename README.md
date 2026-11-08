# Machine Learning Classification Models using Dry Bean Dataset

## Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on a real-world dataset and deploy the solution as an interactive Streamlit web application. The application allows users to upload test data, select a classifier, and view performance metrics and classification results without retraining the model.

---

## Dataset Description

**Dataset Name:** Dry Bean Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset

**Problem Type:** Multi-Class Classification

**Number of Instances:** 13,611

**Number of Features:** 16 numerical features

**Number of Classes:** 7 bean varieties

**Target Variable:** Class

The dataset contains measurements related to the shape and size characteristics of different dry bean varieties. The objective is to classify each bean sample into its correct class based on the provided physical attributes.

---

## GitHub Repository Link

https://github.com/kkrishnapriya1103-droid/ml-classification-assignment

---

## Streamlit App Link

https://ml-classification-assignment-5ldqbjuc9yjqxxmvvli8mj.streamlit.app/

---

## Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Support Vector Machine (SVM)

---

## Evaluation Metrics Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
|---------------|----------|----------|----------|----------|----------|----------|
| Logistic Regression | 0.920676 | 0.994768 | 0.921533 | 0.920676 | 0.920873 | 0.904167 |
| Decision Tree | 0.892031 | 0.944996 | 0.891693 | 0.892031 | 0.891630 | 0.869569 |
| KNN | 0.916636 | 0.983317 | 0.917409 | 0.916636 | 0.916805 | 0.899204 |
| Naive Bayes | 0.763863 | 0.967193 | 0.765371 | 0.763863 | 0.761474 | 0.715406 |
| Support Vector Machine | 0.922145 | 0.994926 | 0.922426 | 0.922145 | 0.922177 | 0.905870 |

---

## Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Logistic Regression achieved strong performance with an accuracy of 92.07% and an MCC score of 0.9042. The model demonstrated good generalization and produced consistent results across all evaluation metrics. |
| Decision Tree | Decision Tree achieved an accuracy of 89.20%. Although it performed reasonably well, its results were lower than Logistic Regression, KNN, and SVM, indicating comparatively weaker generalization on this dataset. |
| KNN | KNN achieved an accuracy of 91.66% and delivered competitive results across all evaluation metrics. It effectively classified samples with similar feature patterns but slightly underperformed compared to SVM. |
| Naive Bayes | Naive Bayes achieved the lowest accuracy of 76.39%. Its assumption of feature independence is not fully suitable for this dataset, resulting in lower performance compared to the other models. |
| Support Vector Machine | Support Vector Machine achieved the highest accuracy (92.21%), highest AUC (0.994926), and highest MCC score (0.905870). It demonstrated the best overall classification performance among all evaluated models. |

---

## Overall Winner for the Dataset

### Support Vector Machine (SVM)

SVM was identified as the best-performing model because it achieved the highest values for Accuracy, AUC, Precision, F1 Score, and MCC Score. Therefore, SVM is the most suitable classifier for the Dry Bean Dataset among the five models evaluated.

---

## Streamlit Application Features

The deployed Streamlit application provides the following features:

- Upload test dataset in CSV format
- Select any one of the five trained classification models
- Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC Score
- Display Confusion Matrix
- Display Classification Report
- Perform predictions using pre-trained models without retraining

---

## Repository Structure

```text
project-folder/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── scaler.pkl
│
└── source_code/
    └── Assignment_2_Classification.ipynb
```

---

## Requirements

```text
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

---

## Author

Koduri Krishna Priya, 
BITS Pilani WILP

