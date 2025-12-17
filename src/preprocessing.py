import pandas as pd
import logging
import re
from nltk.corpus import stopwords as nltk_stopwords


logger = logging.getLogger(__name__)

# Negation words that must always be preserved
_NEGATIONS = {"not", "no", "never", "n't", "cannot"}

# Consolidated dependency check at module level
missing_deps = []
try:
    from bs4 import BeautifulSoup
except ImportError:
    missing_deps.append("beautifulsoup4 (bs4)")
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except ImportError:
    missing_deps.append("nltk")
if missing_deps:
    raise ImportError(
        f"Missing dependencies: {', '.join(missing_deps)}. Install with: pip install nltk beautifulsoup4"
    )

def normalize_reviews(
    df: pd.DataFrame,
    text_column: str = "review",
    remove_stopwords: bool = False,
    stemming: bool = False,
    lemmatization: bool = False
) -> pd.DataFrame:
    """
    Apply text normalization to a DataFrame of reviews.

    Args:
        df (pd.DataFrame): DataFrame with a text column to clean.
        text_column (str): Name of the column containing review text. Defaults to 'review'.
        remove_stopwords (bool): Removes stopwords if True (default False).
        stemming (bool): Applies stemming if True (default False).
        lemmatization (bool): Applies lemmatization if True (default False).

    Returns:
        pd.DataFrame: A COPY of the input DataFrame with normalized text.

    Notes:
        - This function does NOT mutate the input DataFrame.
        - Negation words (not, no, never, etc.) are always preserved.
        - If stopword removal is enabled, negation terms are explicitly excluded.
        - Stemming and lemmatization are mutually exclusive and OFF by default.
        - Excessive cleaning can remove sentiment cues; clean sparingly and document decisions.

    Raises:
        ValueError: If the required text column is missing.
        ValueError: If both stemming and lemmatization are enabled simultaneously.
        ImportError: If NLTK stopwords/wordnet data is missing.
    """
    if text_column not in df.columns:
        logger.error(f"DataFrame missing text column: '{text_column}'")
        raise ValueError(f"DataFrame must contain column '{text_column}'")

    if stemming and lemmatization:
        logger.error("Stemming and lemmatization cannot both be enabled.")
        raise ValueError("Choose either stemming or lemmatization, not both.")

    # Check for required NLTK resources
    try:
        stopword_set = set(nltk_stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords")
        stopword_set = set(nltk_stopwords.words("english"))

    if lemmatization:
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            import nltk
            nltk.download("wordnet")


    def clean_text(text: str) -> str:
        # All values in the text column are coerced to string before processing.
        # 1. Lowercase
        text = text.lower()
        # 2. Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        # 3. Remove non-alphabetic characters but keep whitespace + apostrophe + negations
        text = re.sub(r"[^a-zA-Z\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Prepare resources for optional steps
    do_stopwords = remove_stopwords
    stopword_set = set()
    if do_stopwords:
        stopword_set = set(nltk_stopwords.words('english')) - _NEGATIONS
    do_stemming = stemming
    do_lemmatization = lemmatization
    stemmer = PorterStemmer() if do_stemming else None
    lemmatizer = WordNetLemmatizer() if do_lemmatization else None

    def process(text: str) -> str:
        text = clean_text(text)
        tokens = text.split()
        # 4. Stopword removal (except negations)
        if do_stopwords:
            tokens = [t for t in tokens if t not in stopword_set]
        # 5. Stemming or lemmatization
        if do_stemming:
            tokens = [stemmer.stem(t) for t in tokens]
        elif do_lemmatization:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    df_clean = df.copy()
    # Apply processing
    num_rows = len(df_clean)
    logger.info(
        f"Applying text normalization on {num_rows} rows... "
        f"Options: stopwords={do_stopwords}, stemming={do_stemming}, lemmatization={do_lemmatization}"
    )
    df_clean[text_column] = df_clean[text_column].astype(str).apply(process)
    logger.info("Text normalization complete.")
    return df_clean
