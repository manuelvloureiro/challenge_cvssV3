from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class Vectorizer(TfidfVectorizer):

    def __init__(
            self,
            analyzer='word',
            stop_words='english',
            ngram_range=(1, 3),
            max_features=1000,
            **kwargs
    ):
        super().__init__(
            analyzer=analyzer,
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_features=max_features,
            **kwargs
        )

    def to_df(self, raw_documents):
        tfidf_encodings = self.transform(raw_documents)
        return pd.DataFrame(
            tfidf_encodings.toarray(),
            columns=self.get_feature_names()
        )
