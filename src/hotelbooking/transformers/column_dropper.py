from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    The ColumnSelector transformer selects certain specified columns in order to supply them to other transformers.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X.drop(columns=self.columns)
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)