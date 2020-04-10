from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
import pandas as pd


class ChiSquareFeatureDropper(BaseEstimator, TransformerMixin):
    """
    This chi-square feature dropper aims to determine the relationship between the independent
    category feature (predictor) and dependent category feature(response).
    In feature selection, we aim to select the features which are highly dependent on the response.
    When two features are independent, the observed count is close to the expected count,
    thus we will have smaller Chi-Square value. So high Chi-Square value indicates that the hypothesis of independence is incorrect.

    In simple words, higher the Chi-Square value the feature is more dependent on the response and it can be selected for model training.

    - Null Hypothesis (H0): Two variables are independent.
    - Alternate Hypothesis (H1): Two variables are not independent.

    Steps to perform the Chi-Square Test:
    1. Define Hypothesis.
    2. Build a Contingency table.
    3. Find the expected values.
    4. Calculate the Chi-Square statistic.
    5. Accept or Reject the Null Hypothesis.
    """

    def fit(self, X, y):
        self.columns_to_drop_ = self.get_columns_to_drop(X, y)
        return self


    def transform(self, X, y=None):
        return X.drop(columns=self.columns_to_drop_ )


    @staticmethod
    def get_categorical_columns(X):
        return list(X.select_dtypes(include=['category', 'object']).columns)


    def get_columns_to_drop(self, X, y):
        cols_to_drop = []
        columns = self.get_categorical_columns(X)
        for col in columns:
            print(col)
            chi2 = stats.chi2_contingency(pd.crosstab(X[col], y))
            assert len(chi2[3][chi2
                                   [3] < 5]) == 0, f"The Chi-square test expected value (>5) assumption is violated for column {col}."

            if chi2[1] > 0.05:
                cols_to_drop.append(col)

        return cols_to_drop

