import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

st.title("Machine Learning Assignment 2")

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

selected_model = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]
)

model_paths = {
    "Logistic Regression":
        "model/logistic_regression.pkl",

    "Decision Tree":
        "model/decision_tree.pkl",

    "KNN":
        "model/knn.pkl",

    "Naive Bayes":
        "model/naive_bayes.pkl",

    "SVM":
        "model/svm.pkl"
}

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    model = joblib.load(
        model_paths[selected_model]
    )

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    if selected_model in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:

        X = scaler.transform(X)

    preds = model.predict(X)

    probs = model.predict_proba(X)

    accuracy = accuracy_score(y, preds)

    precision = precision_score(
        y,
        preds,
        average="weighted"
    )

    recall = recall_score(
        y,
        preds,
        average="weighted"
    )

    f1 = f1_score(
        y,
        preds,
        average="weighted"
    )

    mcc = matthews_corrcoef(
        y,
        preds
    )

    auc = roc_auc_score(
        y,
        probs,
        multi_class="ovr"
    )

    st.subheader("Evaluation Metrics")

    st.write("Accuracy:", round(accuracy,4))
    st.write("AUC:", round(auc,4))
    st.write("Precision:", round(precision,4))
    st.write("Recall:", round(recall,4))
    st.write("F1 Score:", round(f1,4))
    st.write("MCC:", round(mcc,4))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, preds)

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    st.pyplot(fig)

    st.subheader("Classification Report")

    st.text(
        classification_report(y, preds)
    )