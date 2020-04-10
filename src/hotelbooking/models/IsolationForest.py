from sklearn.ensemble import IsolationForest

from hotelbooking.transformers.dtype_selector import DTypeSelector
from hotelbooking.transformers.correlationfilter import CorrFilterHighTotalCorrelation
from hotelbooking.transformers.column_selector import ColumnSelector

from category_encoders import HashingEncoder

from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer


def pipeline():

    numerical_pipeline = make_pipeline(
        DTypeSelector('number'),
        CorrFilterHighTotalCorrelation(),
        KNNImputer(n_neighbors=5),
        RobustScaler()
    )

    object_pipeline = make_pipeline(
        DTypeSelector('object'),
        SimpleImputer(strategy='most_frequent'),
        HashingEncoder(n_components=50)
    )


    return make_pipeline(
        make_union(
            numerical_pipeline,
            object_pipeline,
        ),
        IsolationForest(n_jobs=-1,
                        random_state=42,
                        verbose=0)
    )


def hyperparams():
    return {
        'isolationforest__n_estimators': [100,500,1000],
        'isolationforest__contamination': [0.01, 0.1,0.2],
        'isolationforest__max_samples': [100,500,100],
        'isolationforest__max_features': [5, 10, 15],
        'isolationforest__bootstrap': [True, False]
    }






