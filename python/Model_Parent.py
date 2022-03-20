import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score


def get_air_quality_df():
    parent_path = os.path.dirname(os.getcwd())
    df = pd.read_csv(parent_path + '/cleaned_data/AirQualityUCI_fixed_cleaned.csv', index_col=0)
    return df


def get_auto_mpg_df():
    parent_path = os.path.dirname(os.getcwd())
    df = pd.read_csv(parent_path + '/cleaned_data/auto_mpg_fixed_cleaned.csv', index_col=0)
    return df


def get_forest_fires_df():
    parent_path = os.path.dirname(os.getcwd())
    df = pd.read_csv(parent_path + '/cleaned_data/forestfires_cleaned.csv', index_col=0)
    return df

def get_ccpp_df():
    parent_path = os.path.dirname(os.getcwd())
    df = pd.read_csv(parent_path + '/cleaned_data/CCPP.csv', index_col=0)
    return df


def get_bike_sharing_df():
    parent_path = os.path.dirname(os.getcwd())
    df = pd.read_csv(parent_path + '/cleaned_data/bike_sharing_hour.csv', index_col=0)
    return df


def get_wine_quality_df():
    parent_path = os.path.dirname(os.getcwd())
    df = pd.read_csv(parent_path + '/cleaned_data/winequality-white_fixed.csv', index_col=0)
    return df


def forward_selection(model, feature_df: pd.DataFrame, response_series: pd.Series):
    f_df = feature_df.copy(deep=True)
    features = pd.DataFrame()
    featstrings = []
    r2_cvs = []
    r2_bars = []
    while not f_df.empty:
        best_f = None
        best_r2_bar = -1000
        best_r2_cv = -1000
        print("Base Features: ", ','.join([feat for feat in features]))
        for f in f_df:
            print("Testing feature: ", f)
            temp_features = pd.concat([features, pd.Series(f_df[f])], axis=1)
            r2_bar = get_r2_bar(model, temp_features, response_series)
            cv = cross_validate(model, temp_features, response_series, scoring='r2', cv=5)  # 80-20 split
            r2_cv = np.mean(cv['test_score'])
            print("R^2 Bar: ", r2_bar)
            print("R^2 CV: ", r2_cv)
            if r2_cv > best_r2_cv:
                best_r2_cv = r2_cv
                best_f = f
                best_r2_bar = r2_bar
        features = pd.concat([features, pd.Series(f_df[best_f])], axis=1)
        featstrings.append(','.join([feat for feat in features]))
        r2_cvs.append(best_r2_cv)
        r2_bars.append(best_r2_bar)
        print('Best feature: ', best_f)
        print(featstrings)
        print(r2_cvs)
        print(r2_bars)
        f_df.drop(columns=[best_f], inplace=True)
    plt.plot(featstrings, r2_cvs, label="r2 cv")
    plt.plot(featstrings, r2_bars, label="r2 bar")
    plt.legend()
    plt.show()
    return


def backward_selection(model, feature_df: pd.DataFrame, response_series: pd.Series):
    f_df = feature_df.copy(deep=True)
    features = feature_df.copy(deep=True)
    featstrings = []
    r2_cvs = []
    r2_bars = []
    while len(f_df.columns) > 1:
        best_f = None
        best_r2_bar = -1000
        best_r2_cv = -1000
        print("Base Features: ", ','.join([feat for feat in features]))
        for f in f_df:
            print("Testing feature: ", f)
            temp_features = features.drop(columns=[f])
            r2_bar = get_r2_bar(model, temp_features, response_series)
            r2_cv = np.mean(
                cross_validate(model, temp_features, response_series, scoring='r2', cv=5)['test_score'])  # 80-20 split
            print("R^2 Bar: ", r2_bar)
            print("R^2 CV: ", r2_cv)
            if r2_cv > best_r2_cv:
                best_r2_cv = r2_cv
                best_f = f
                best_r2_bar = r2_bar
        features = features.drop(columns=[best_f])
        featstrings.append(','.join([feat for feat in features]))
        r2_cvs.append(best_r2_cv)
        r2_bars.append(best_r2_bar)
        print('Best feature: ', best_f)
        print(featstrings)
        print(r2_cvs)
        print(r2_bars)
        f_df.drop(columns=[best_f], inplace=True)
    plt.plot(featstrings, r2_cvs, label="r2 cv")
    plt.plot(featstrings, r2_bars, label="r2 bar")
    plt.legend()
    plt.show()
    return


def get_r2_bar(model, data, y_true):
    model.fit(data, y_true)
    y_pred = model.predict(data)
    r2 = r2_score(y_true, y_pred)
    n = len(data)
    p = len(data.columns)
    r2_bar = calc_r2_bar(n, p, r2)
    # r2_bar = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    return r2_bar


def calc_r2_bar(m, n, r2):
    # m = number of data points
    # n = number of features

    dfr = n - 1
    df = m - n

    rdf = (dfr + df) / df  # ratio of total degrees of freedom to degrees of freedom for error
    r2_bar = 1 - rdf * (1 - r2)

    return r2_bar

