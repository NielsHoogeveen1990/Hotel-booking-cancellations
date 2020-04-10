import pandas as pd
import numpy as np

from hotelbooking.utils import log_step


def read_data(data_path):
    """
    Get data by specifying a datapath where the data is stored.
    :param data_path: data path of the CSV file
    :return: dataframe
    """
    return pd.read_csv(data_path)


def change_dtypes(df):
    """
    This function changes the data types of specific columns.
    :param df: dataframe
    :return: dataframe
    """
    return df.assign(
        agent=lambda d: d['agent'].astype('object')
    )


def drop_irrelevant_features(df, *cols):
    """
    This function drops irrevalant features that are not required in the model.
    :param df: dataframe
    :param cols: irrelevant columns
    :return: dataframe
    """
    return df.drop(columns=list(cols))


def replace_months(df):
    """
    This function replaces the months as strings to integers (e.g. January --> 1, February --> 2)
    :param df: dataframe
    :return: dataframe
    """
    import calendar
    months = dict((v, k) for k, v in enumerate(calendar.month_name))

    return df.assign(
        arrival_date_month=lambda d: df['arrival_date_month'].map(months)
    )


def encode_cyclical_features(df):
    """
    This function encodes cyclical date features, such as day of the month, week of the year, et cetera.
    :param df: dataframe
    :return: dataframe
    """
    return df.assign(
        arrival_date_month_sin = lambda d: np.sin(2 * np.pi * d['arrival_date_month']/12),
        arrival_date_month_cos = lambda d: np.cos(2 * np.pi * d['arrival_date_month']/12),
        arrival_date_week_number_sin = lambda d: np.sin(2 * np.pi * d['arrival_date_week_number']/52),
        arrival_date_week_number_cos = lambda d: np.cos(2 * np.pi * d['arrival_date_week_number']/52),
        arrival_date_day_of_month_sin = lambda d: np.sin(2 * np.pi * d['arrival_date_day_of_month']/31),
        arrival_date_day_of_month_cos = lambda d: np.cos(2 * np.pi * d['arrival_date_day_of_month']/31)
    )


def change_labels(df):
    """
    This function changes the labels to either 1 (for non-anomaly) or -1 (anomalies).
    :param df: dataframe
    :return: dataframe
    """
    conditions = [
        (df['reservation_status'] == 'Check-Out') | (df['reservation_status'] == 'Canceled'),
        df['reservation_status'] == 'No-Show'
    ]

    choices = [1, -1]

    df['show_up'] = np.select(conditions, choices, 99)

    return df


@log_step
def get_df(data_path):
    return (read_data(data_path)
            .pipe(change_dtypes)
            .pipe(drop_irrelevant_features,
              'is_canceled',
              'reservation_status_date',
              'assigned_room_type',
              'required_car_parking_spaces',
              'company')
            .drop_duplicates(subset=None, keep='first')
            .pipe(replace_months)
            .pipe(encode_cyclical_features)
            .pipe(change_labels)
            ).drop(columns=['arrival_date_month',
                            'arrival_date_week_number',
                            'arrival_date_day_of_month',
                            'reservation_status'])