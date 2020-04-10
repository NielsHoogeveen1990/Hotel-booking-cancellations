from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from hotelbooking.transformers.dtype_selector import DTypeSelector


class CustomCategoricalImputer(BaseEstimator, TransformerMixin):
    """
    A custom machine learning imputer for categorical features.
    This categorical imputer has a Random Forest classifier as the default imputer model.
    """

    def __init__(self,
                 column,
                 model=RandomForestClassifier(),
                 num_imputer='mean',
                 cat_imputer='most_frequent',
                 num_scaler=StandardScaler(),
                 cat_encoder=OneHotEncoder()):

        self.column = column
        self.model = model
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer
        self.num_scaler = num_scaler
        self.cat_encoder = cat_encoder

    def fit(self, X, y=None):
        X = X.copy()
        # Get train data
        X_train, y_train = self.get_train_data(X)
        # Create pipeline object
        self.clf_ = self.pipeline()
        # Train model
        self.clf_.fit(X_train, y_train)

        return self

    def transform(self, X):
        # Get prediction data
        X_predict = self.get_prediction_data(X)
        # Make predictions
        X_predict[self.column + '_predict'] = self.clf_.predict(X_predict)
        # Merge the original X dataframe with the prediction Series
        X = X.merge(X_predict[self.column + '_predict'].to_frame(), how='left', left_index=True, right_index=True)
        # Fill na with predictions
        X[self.column].fillna(X[self.column + '_predict'], inplace=True)
        X = X.drop(columns=self.column + '_predict')

        return X

    @staticmethod
    def _drop_nan_columns(X):
        # Drop all columns that contain at least one NaN value
        return X[X.columns[~X.isna().any()]]

    def get_train_data(self, X):
        try:
            X_temp = self._drop_nan_columns(X)
            y = X[self.column]
            Xy = pd.concat([X_temp, y], axis=1)

            Xy_train = Xy.loc[lambda d: ~d[self.column].isna()]
            X_train = Xy_train.drop(columns=self.column)
            y_train = Xy_train[self.column]

            return X_train, y_train

        except KeyError:
            raise KeyError(f"The DataFrame does not include the column: {self.column}")

    def get_prediction_data(self, X):
        X_temp = self._drop_nan_columns(X)
        y = X[self.column]
        Xy = pd.concat([X_temp, y], axis=1)

        Xy_predict = Xy.loc[lambda d: d[self.column].isna()]
        X_predict = Xy_predict.drop(columns=self.column)

        return X_predict

    def pipeline(self):

        numerical_pipeline = make_pipeline(
            DTypeSelector('number'),
            SimpleImputer(strategy=self.num_imputer),
            self.num_scaler
        )

        object_pipeline = make_pipeline(
            DTypeSelector('object'),
            SimpleImputer(strategy=self.cat_imputer),
            self.cat_encoder
        )

        return make_pipeline(
            make_union(
                numerical_pipeline,
                object_pipeline
            ),
            self.model
        )
