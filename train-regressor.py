import tensorflow as tf
import autokeras as ak
from tools import load_images
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tools import load_images, Preprocessor

def main():
    seed = 42
    img_height, img_width = 128, 128
    batch_size = 32

    # df = load_data('dataset/train.csv', 'dataset/trainImages/', (128, 128), grey_scale=True)
    # X, y = df['Images'].to_numpy(), df['Condition'].to_numpy()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=seed)
    # train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    df = pd.read_csv("dataset/train.csv")
    df = df.dropna(axis=0, how='any', subset=['Amount', 'Condition'])
    df['Images'] = load_images(df, directory='dataset/trainImages', size=(img_height, img_width), grey_scale=False)

    train_df, test_df = train_test_split(df, test_size=0.2)
    train_images = np.asarray(train_df.pop('Images').to_list())
    test_images = np.asarray(test_df.pop('Images').to_list())
    train_amount = np.asarray(train_df.pop('Amount'))
    test_amount = np.asarray(test_df.pop('Amount'))

    model = ak.ImageRegressor(overwrite=True, max_trials=5)
    model.fit(train_images, train_amount, validation_split=0.2)
    model.evaluate(test_images, test_amount)
    # loss: 8428122.0000 - mean_squared_error: 8428122.0000

    try:
        model.export_model().save("regressor_model_autokeras", save_format="tf")
    except Exception:
        model.export_model().save("regressor_model_autokeras.h5")

    return


if __name__ == "__main__":
    main()
