import tensorflow as tf
import autokeras as ak
from tools import load_images
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    seed = 42
    img_height, img_width = 128, 128
    batch_size = 32

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
    model = tf.keras.models.load_model('model_autokeras')
    model.evaluate(test_ds)
    # loss: 0.2085 - accuracy: 0.9355

    return


if __name__ == "__main__":
    main()
