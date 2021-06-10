import random

import pandas as pd
from PIL import Image
import numpy as np
import os
import cv2
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_images(dataframe, directory, size, grey_scale):
    images = []
    for file_name in dataframe['Image_path']:
        path = os.path.join(directory, file_name)

        try:
            image = cv2.imread(path)
            image = cv2.resize(image, size)
            if grey_scale:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = image[:, :, np.newaxis]
            images.append(image)
        except FileNotFoundError:
            images.append(np.nan)

    return images


class OutlierNullifier(TransformerMixin):
    def __init__(self, **kwargs):
        """
        Create a transformer to remove outliers.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """

        self.quantiles = {}

    def fit(self, X, y=None, **fit_params):
        if isinstance(X, np.ndarray):
            for i in range(X.shape[1]):
                self.quantiles[i] = np.quantile(X[i], [0.25, 0.75])
        else:
            for column in X.columns:
                self.quantiles[column] = X[column].quantile(0.25), X[column].quantile(0.75)

        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            for i in range(X.shape[1]):
                q1, q3 = self.quantiles[i]
                iqr = q3 - q1
                X[i] = np.where((X[i] < q1 - 1.5 * iqr) | (X[i] > q3 + 1.5 * iqr), np.nan, X[i])
        else:
            for column in X.columns:
                q1, q3 = self.quantiles[column]
                iqr = q3 - q1
                X[column] = np.where((X[column] < q1 - 1.5 * iqr) | (X[column] > q3 + 1.5 * iqr), np.nan, X[column])

        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)


class Preprocessor(TransformerMixin):
    def __init__(self):
        # self.encoder = OneHotEncoder(use_cat_names=True)
        # self.imputer = SimpleImputer
        self.pipeline = make_pipeline(
            ColumnTransformer(transformers=[
                ('numerical', make_pipeline(
                    OutlierNullifier(),
                    SimpleImputer(strategy="median"),
                    StandardScaler()
                ), selector(dtype_include=np.number)),
                ('categorical', make_pipeline(
                    SimpleImputer(strategy='most_frequent'),
                    OneHotEncoder()
                ), selector(dtype_include=['category', object, 'bool'])),
            ]),
        )

    def _engineer(self, df):
        df['Expiry_date'] = pd.to_numeric(pd.to_datetime(df['Expiry_date']))
        df = df.drop(columns=['Image_path', 'Cost_of_vehicle'])
        # df['Days_until_expiry'] = df['Expiry_date'].apply(lambda x: (x - pd.Timestamp('today')).days)
        df['Coverage_range'] = df['Max_coverage'] - df['Min_coverage']

        return df

    def fit(self, df):
        self.pipeline.fit(self._engineer(df))
        return self

    def transform(self, df):
        return self.pipeline.transform(self._engineer(df))

    def fit_transform(self, df):
        return self.pipeline.fit_transform(self._engineer(df))
