# Sentiment Analysis for Real-World Movie Review Analytics

## 1. Project Overview

In today’s data-driven entertainment industry, organizations rely heavily on user-generated reviews to understand audience sentiment. Film studios, streaming platforms, and review aggregators use sentiment analysis to guide marketing strategies, content decisions, and reputation management.

This project implements an **end-to-end sentiment analysis system** that classifies movie reviews as **positive or negative**. It covers the full machine-learning lifecycle: raw data ingestion, text preprocessing, feature extraction, model training and evaluation, model persistence, and a lightweight web interface for real-time inference.

The system is intentionally designed as a **clean, interpretable baseline**, prioritizing correctness, reproducibility, and extensibility over unnecessary complexity.

You can view it here: https://sentiment-analysis-imdb-app.streamlit.app/

---

## 2. Dataset Description

### IMDB Movie Reviews Dataset

The project uses the **IMDB Large Movie Review Dataset**, a widely adopted benchmark for sentiment analysis. It contains **50,000 labeled movie reviews**, evenly split between positive and negative sentiment, with a predefined train/test split.

#### Why IMDB?
- Credible and widely used in research and industry
- Balanced positive and negative classes
- Real-world, noisy user-generated text
- Fixed train/test split ensures reproducible evaluation

#### Source & Licensing
- Source: IMDB dataset curated by Stanford University researchers  
- Dataset link: https://ai.stanford.edu/~amaas/data/sentiment/  
- License: Academic and research use (refer to official documentation)

---

## 3. Folder Structure
```
sentiment-analysis/
│
├── app/
│ └── streamlit_app.py # Streamlit UI for sentiment prediction
│
├── data/
│ └── aclImdb/ # IMDB dataset (train/test, pos/neg)
│
├── models/
│ ├── log_reg_model.pkl # Trained Logistic Regression model
│ └── tfidf_vectorizer.pkl # Trained TF-IDF vectorizer
│
├── src/
│ ├── data_loader.py # IMDB data ingestion
│ ├── preprocessing.py # Text normalization pipeline
│ ├── features.py # TF-IDF feature extraction
│ ├── model_baseline.py # Model training & evaluation
│ ├── split.py # Train/test splitting
│ └── run_pipeline.py # End-to-end training pipeline
│
├── README.md
└── requirements.txt
```

---

## 4. Project Phases

### Phase 1: Problem Definition & Setup
- Defined real-world sentiment analysis use case
- Selected and documented dataset
- Designed scalable project structure

### Phase 2: Data Preparation & Baseline Modeling
- Robust data ingestion from raw IMDB directories
- Text normalization with configurable preprocessing
- TF-IDF feature extraction (unigrams + bigrams)
- Logistic Regression baseline model
- Proper train/test evaluation (Accuracy, Precision, Recall, F1)

### Phase 3: Streamlit App & Inference
- Persisted trained model and vectorizer
- Built Streamlit UI for interactive sentiment prediction
- Reused preprocessing and feature pipeline for inference

---

## 5. How to Run the Project

### 1. Download the Dataset

Download the IMDB dataset from:  
https://ai.stanford.edu/~amaas/data/sentiment/

Extract it into the `data/` directory.

Expected structure:
```
data/
└── aclImdb/
├── train/
│ ├── pos/
│ └── neg/
└── test/
├── pos/
└── neg/
```
---

### 2. Install Dependencies

pip install -r requirements.txt

import nltk
nltk.download('stopwords')
nltk.download('wordnet')


### 3. Run the Training Pipeline

From the project root:

python -m src.run_pipeline


This will:
- Load and preprocess the dataset
- Train the baseline model
- Evaluate performance on the test set
- Save the trained model and TF-IDF vectorizer to models/

### 4. Run the Streamlit App

After training, run:
streamlit run app/streamlit_app.py

Enter a movie review and receive a real-time sentiment prediction.

## 6. Model Performance

Baseline Logistic Regression using TF-IDF features:
- Accuracy: ~89%
- F1-Score: ~89%

These results are consistent with strong classical machine-learning baselines on the IMDB dataset.

## 7. Future Improvements

- Neutral or multi-class sentiment classification
- Hyperparameter tuning and feature selection
- Error analysis and interpretability dashboards
- Deployment using Docker or cloud platforms
- Extension to other text domains (e-commerce, social media)

## 8. Acknowledgements

- IMDB Large Movie Review Dataset (Maas et al.)
- scikit-learn, pandas, NLTK, Streamlit
