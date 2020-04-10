from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

class CorrelationFilter(TransformerMixin, BaseEstimator):
    """
    This transformer removes highly correlated features. Hence, it prevents multi-collinearity.
    """
    def __init__(self, threshold=0.5):
        """
        The init method of the CorrelationFilter has only one parameter; threshold of the correlation.
        :param threshold: the threshold value for multi-collinearity.
        """
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        The fit method estimates, based on the training set, which columns need to be dropped.
        The columns_to_drop_ will be calculated by the get_columns_to_drop method.
        :param X: X dataframe
        :param y: y series
        :return: self
        """
        self.columns_to_drop_ = self.get_columns_to_drop(X, y)
        return self

    def transform(self, X):
        """
        This transform method will drop the columns that are specified in the fit method.
        :param X: X dataframe
        :return: X dataframe without the columsn_to_drop
        """
        return X.drop(columns=self.columns_to_drop_)

    @staticmethod
    def _get_abs_corr(X):
        """
        This static method returns the absolute correlation matrix of X.
        By subtracting a identity matrix of X, the correlation values with a column itself will be removed since
        we are only interested in correlations between different columns.
        :param X: X dataframe
        :return: absolute correlation matrix
        """
        return X.corr().abs() - np.eye(X.shape[1])

    def get_columns_to_drop(self, X, y):
        """
        The get_columns_to_drop method returns a list of columns to drop.
        Columns are added to the list while the max value of the absolute correlation matrix is higher then the threshold.
        The max value of each column of the correlation matrix is taken by the first .max(), followed by a second .max()
        to get the maximum correlation value of the rows.
        Subsequently, the highest_corr_cols are taken by unstacking the correlation matrix and taking the index of
        the max value and putted into a list.
        Which of the correlated columns is dropped, is decided by the choose_from_two method.

        That column is added to the cols_to_drop list.
        A new X dataframe is created by dropping the column, and a correlation matrix is subsequently constructed.
        The while loop starts over again, until there are no highly correlated columns left.
        :param X: X dataframe
        :param y: y series
        :return: columns to be dropped
        """

        cols_to_drop = []
        abs_corr = self._get_abs_corr(X)

        while abs_corr.max().max() > self.threshold:
            highest_corr_cols = list(abs_corr.unstack().idxmax())
            col_to_drop = self.choose_from_two(X, y, highest_corr_cols)

            cols_to_drop.append(col_to_drop)
            X = X.drop(columns=col_to_drop)
            abs_corr = self._get_abs_corr(X)

        return cols_to_drop

    def choose_from_two(self, X, y, cols):
        """
        This method chooses which of the two correlated columns is dropped.
        In this case, the second column is being dropped.
        :param X: X dataframe
        :param y: y series
        :param cols: correlated columns
        :return: a column to be dropped
        """
        return cols[-1]


class CorrFilterHighTotalCorrelation(CorrelationFilter):
    """
    This CorrFilterHighTotalCorrelation class is based upon its CorrelationFilter superclass.
    The method choose_from_two is overridden.
    """
    def __init__(self):
        super(CorrFilterHighTotalCorrelation, self).__init__()

    def choose_from_two(self, X, y, cols):
        """
        This method chooses which of the two correlated columns is dropped.
        In this case, the column with the highest sum of correlations with other columns is dropped.
        :param X: X dataframe
        :param y: y dataframe
        :param cols: correlated columns
        :return: a column to be dropped
        """
        return X.corr().loc[:, cols].sum(axis=0).idxmax()