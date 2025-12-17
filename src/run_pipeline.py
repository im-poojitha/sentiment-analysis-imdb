# src/run_pipeline.py

import os

from src.data_loader import load_imdb_directory
from src.preprocessing import normalize_reviews
from src.features import extract_tfidf_features
from src.model_baseline import train_logistic_regression, evaluate_classifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "aclImdb")

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

print("Loading IMDB training data...")
train_df = load_imdb_directory(TRAIN_DIR)

print("Loading IMDB test data...")
test_df = load_imdb_directory(TEST_DIR)

print("Normalizing text...")
train_df = normalize_reviews(train_df)
test_df = normalize_reviews(test_df)

print("Extracting TF-IDF features (fit on train only)...")
X_train, vectorizer = extract_tfidf_features(train_df["review"])
y_train = train_df["label"].values

print("Transforming test data...")
X_test = vectorizer.transform(test_df["review"].tolist())
y_test = test_df["label"].values

print("Training Logistic Regression model...")
model = train_logistic_regression(X_train, y_train)

# Save model and vectorizer
import pickle
import os

os.makedirs("models", exist_ok=True)

with open("models/log_reg_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


print("Evaluating model...")
evaluate_classifier(model, X_test, y_test)

print("Pipeline complete.")
