"""
text_features.py
------------------
TF-IDF vectorization for the phishing detection pipeline.

Kept as a thin, single-purpose module so both train.py and
phishing_detector.py import the exact same fit/transform logic — avoids the
classic bug where training and inference vectorizers drift apart.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import get_logger

logger = get_logger("text_features")

TFIDF_PARAMS = dict(
    max_features=5000,
    ngram_range=(1, 2),   # unigrams + bigrams — phishing signal often lives in short phrases
                            # ("verify your", "click here", "account suspended")
    min_df=2,               # drop hapax legomena — pure noise/typos, and there's plenty in
                            # randomized synthetic text (order numbers, IPs, etc.)
    sublinear_tf=True,      # log-scale term frequency; avoids one repeated token dominating
    stop_words="english",
)


def fit_vectorizer(texts) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    vectorizer.fit(texts)
    logger.info(f"Fitted TF-IDF vectorizer: vocabulary size = {len(vectorizer.vocabulary_)}")
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts):
    return vectorizer.transform(texts)
