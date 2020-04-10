from sklearn.base import BaseEstimator, TransformerMixin


class DTypeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtypes):
        self.dtypes = dtypes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(self.dtypes)
