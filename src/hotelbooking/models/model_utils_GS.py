from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from hotelbooking.preprocessing import get_df
from hotelbooking.models import IsolationForest
import pickle
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
import pandas as pd


def split_data(df):
    X = df.drop(columns='show_up')
    y = df['show_up']

    return train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)


def fit(model, X_train, y_train):
    clf = model.pipeline()
    f1sc = make_scorer(f1_score, pos_label=-1)

    skf = StratifiedKFold(n_splits=5)
    folds = list(skf.split(X_train, y_train))

    gridsearch = GridSearchCV(clf, model.hyperparams(),
                              cv=folds,
                              refit=True,
                              verbose=1,
                              n_jobs=-1,
                              scoring=f1sc)

    gridsearch.fit(X_train, y_train)

    # returns best model (with best parameters)
    return gridsearch.best_estimator_


def evaluate(y_hat, y_true):
    print(classification_report(y_true, y_hat))


def run(datapath, model_version):
    df = get_df(datapath)

    X_train, X_test, y_train, y_test = split_data(df)

    fitted_model = fit(IsolationForest, X_train, y_train)

    y_hat = fitted_model.predict(X_test)

    evaluate(y_hat, y_test)

    print(fitted_model.get_params())

    with open(f'src/hotelbooking/trained_models/model_{model_version}.pkl', 'wb') as file:
        pickle.dump(fitted_model, file)


