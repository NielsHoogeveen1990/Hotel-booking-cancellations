from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from hotelbooking.preprocessing import get_df
from hotelbooking.models import IsolationForest
import pickle


def split_data(df):
    X = df.drop(columns='show_up')
    y = df['show_up']

    return train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)


def fit(model, X_train):
    model = model.pipeline()
    # Train only on X_train, since anomaly detection methods are unsupervised
    model.fit(X_train)

    return model


def evaluate(y_hat, y_true):
    print(classification_report(y_true, y_hat))


def run(datapath, model_version):
    df = get_df(datapath)

    X_train, X_test, y_train, y_test = split_data(df)

    fitted_model = fit(IsolationForest, X_train)

    y_hat = fitted_model.predict(X_test)

    evaluate(y_hat, y_test)

    with open(f'src/hotelbooking/trained_models/model_{model_version}.pkl', 'wb') as file:
        pickle.dump(fitted_model, file)