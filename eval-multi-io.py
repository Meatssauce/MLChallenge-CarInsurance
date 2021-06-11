import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.python.keras.backend import int_shape
from sklearn.metrics import f1_score, r2_score

from tools import load_images, Preprocessor


def main():
    seed = 42
    img_height, img_width = 150, 150
    batch_size = 32

    df = pd.read_csv("dataset/train.csv")
    df = df.dropna(axis=0, how='any', subset=['Amount', 'Condition'])
    df['Images'] = load_images(df, directory='dataset/trainImages', size=(img_height, img_width), grey_scale=False)

    train_df, test_df = train_test_split(df, test_size=0.2)
    train_images = np.asarray(train_df.pop('Images').to_list())
    test_images = np.asarray(test_df.pop('Images').to_list())
    train_conditions, train_amount = np.asarray(train_df.pop('Condition')), np.asarray(train_df.pop('Amount'))
    test_conditions, test_amount = np.asarray(test_df.pop('Condition')), np.asarray(test_df.pop('Amount'))

    preprocessor = Preprocessor()
    train_df = preprocessor.fit_transform(train_df)
    test_df = preprocessor.transform(test_df)

    model = tf.keras.models.load_model('car-Insurance-tabular-resNet2')

    model.evaluate(
        {'image': test_images, 'metadata': test_df},
        {'condition': test_conditions, 'amount': test_amount}
    )

    predicted_conditions, predicted_amount = model.predict(
        {'image': test_images, 'metadata': test_df}
    )

    condition_score = max(0, 100 * f1_score(test_conditions, predicted_conditions, average='macro'))
    amount_score = max(0, 100 * r2_score(test_amount, predicted_amount))

    print(f'Classifier score: {condition_score:.4f}')
    print(f'Regressor score: {amount_score:.4f}')
    print(f'Final score: {np.mean([condition_score, amount_score])}')

    return


if __name__ == "__main__":
    main()
