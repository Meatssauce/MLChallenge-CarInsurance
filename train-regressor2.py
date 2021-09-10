import joblib
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from tools import load_images, OutlierNullifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd
import xgboost as xgb


def feature_engineer(df):
    df['Expiry_date'] = pd.to_numeric(pd.to_datetime(df['Expiry_date']))
    df = df.drop(columns=['Image_path', 'Cost_of_vehicle'])
    # df['Days_until_expiry'] = df['Expiry_date'].apply(lambda x: (x - pd.Timestamp('today')).days)
    df['Coverage_range'] = df['Max_coverage'] - df['Min_coverage']
    return df


def main():
    seed = 42
    img_height, img_width = 128, 128
    batch_size = 32

    df = pd.read_csv("dataset/train.csv")
    df = df.dropna(axis=0, how='any', subset=['Amount', 'Condition'])

    X, y = df.drop(columns=['Amount']), df['Amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = make_pipeline(
        FunctionTransformer(feature_engineer),
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
        xgb.XGBRegressor(objective='reg:squarederror')
    )

    model.fit(X_train, y_train)

    scores = 100 * cross_val_score(model, X, y, scoring='r2', cv=KFold(10, shuffle=True, random_state=seed),
                                   verbose=1, n_jobs=-1)
    scores = np.asarray([max(0, i) for i in scores])
    print("%0.4f score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    # 0.8795 score with a standard deviation of 2.34
    # neg_mean_squared_error 0

    joblib.dump(model, 'xgbRegressor.joblib')

    return


if __name__ == "__main__":
    main()
