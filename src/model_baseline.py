from typing import Tuple
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def train_logistic_regression(
    X_train: csr_matrix,
    y_train: np.ndarray
) -> LogisticRegression:
    """
    Train a baseline logistic regression classifier.

    Args:
        X_train (csr_matrix): Feature matrix for training data.
        y_train (np.ndarray): Binary class labels for training data.

    Returns:
        LogisticRegression: Trained scikit-learn model.
    """
    clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(
    model: LogisticRegression,
    X_test: csr_matrix,
    y_test: np.ndarray
) -> None:
    """
    Evaluate the classifier and print accuracy, precision, recall, and F1 score.

    Args:
        model (LogisticRegression): Trained classifier.
        X_test (csr_matrix): Feature matrix for test data.
        y_test (np.ndarray): Binary class labels for test data.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nBaseline Logistic Regression Evaluation:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
