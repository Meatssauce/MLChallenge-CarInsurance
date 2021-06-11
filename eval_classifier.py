import tensorflow as tf
import autokeras as ak
from tools import load_images
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd


def main():
    seed = 42
    img_height, img_width = 128, 128
    batch_size = 32

    df = pd.read_csv("dataset/train.csv")
    df = df.dropna(axis=0, how='any', subset=['Amount', 'Condition'])
    df['Images'] = load_images(df, directory='dataset/trainImages', size=(img_height, img_width), grey_scale=False)

    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=seed)
    train_images = np.asarray(train_df.pop('Images').to_list())
    test_images = np.asarray(test_df.pop('Images').to_list())
    train_conditions = np.asarray(train_df.pop('Condition'))
    test_conditions = np.asarray(test_df.pop('Condition'))

    model = tf.keras.models.load_model('classifier_model_autokeras2')
    # model.evaluate(test_ds)
    # loss: 0.2085 - accuracy: 0.9355

    predicted_conditions = list(map(lambda x: 1 if x > 0.5 else 0, model.predict(test_images)))
    score = max(0, 100 * f1_score(test_conditions, predicted_conditions, average='macro'))
    print(f'Classifier score: {score:.4f}')
    # 48.0374

    return


if __name__ == "__main__":
    main()
