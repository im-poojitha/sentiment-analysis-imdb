from typing import Tuple, Union, List
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_tfidf_features(
    texts: Union[pd.Series, List[str]]
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Extract baseline TF-IDF features from preprocessed text using unigram and bigram tokens.
    ngram_range and max_features are hard-coded for baseline reproducibility and comparison.

    Args:
        texts (Union[pd.Series, List[str]]): Iterable of preprocessed documents (strings).

    Returns:
        Tuple[csr_matrix, TfidfVectorizer]:
            X: TF-IDF document-term sparse matrix shape [n_samples, n_features].
            vectorizer: The fitted TfidfVectorizer instance (for later use on test/production data).
    """
    if isinstance(texts, pd.Series):
        docs = texts.tolist()
    else:
        docs = list(texts)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        dtype=float
    )
    X = vectorizer.fit_transform(docs)
    return X, vectorizer
