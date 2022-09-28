import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime


def preprocessing(dataframe: pd.DataFrame,
                  selected_columns: list,
                  missing_index: list,
                  encode_index: list,
                  need_extract_labels: bool = True):
    selected_features = dataframe[selected_columns]

    if 'Stage' in selected_columns:
        selected_features = selected_features[selected_features.Stage != 'In Progress']

    if 'Created Date' in selected_columns and 'Close Date' in selected_columns:
        dateDiff = [(datetime.strptime(row['Close Date'], '%Y-%m-%d %H:%M:%S') -
                     datetime.strptime(row['Created Date'], '%Y-%m-%d %H:%M:%S')).days
                    for row in selected_features.iloc]
        selected_features.drop(columns=['Close Date', 'Created Date'], inplace=True)
        selected_features.insert(len(selected_features.columns) - 1, 'Date Diff', dateDiff, True)

    if need_extract_labels:
        X = selected_features.iloc[:, :-1].values
        Y = selected_features.iloc[:, -1].values
        Y = list(map(lambda stage: 1 if stage == 'Won' else 0, Y))
    else:
        X = selected_features.values

    ct = ColumnTransformer(transformers=[
        ("encoder", OneHotEncoder(sparse=False), encode_index),
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean"), missing_index)
    ], remainder="passthrough")
    X = np.array(ct.fit_transform(X))

    if need_extract_labels:
        return X, Y
    else:
        return X
