import tensorflow as tf
import autokeras as ak
from tools import load_images
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    seed = 42
    img_height, img_width = 128, 128
    batch_size = 32

    # df = load_data('dataset/train.csv', 'dataset/trainImages/', (128, 128), grey_scale=True)
    # X, y = df['Images'].to_numpy(), df['Condition'].to_numpy()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=seed)
    # train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='dataset/trainImages/',
        validation_split=0.2,
        subset='training',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='dataset/trainImages/',
        validation_split=0.2,
        subset='validation',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    model = ak.ImageClassifier(overwrite=True, max_trials=5)
    model.fit(train_ds, validation_split=0.2)
    model.evaluate(test_ds)

    try:
        model.export_model().save("model_autokeras", save_format="tf")
    except Exception:
        model.export_model().save("model_autokeras.h5")

    return


if __name__ == "__main__":
    main()
