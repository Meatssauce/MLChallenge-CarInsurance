import tensorflow as tf
import autokeras as ak
from tools import load_images
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score


# def main():
#     seed = 42
#     img_height, img_width = 128, 128
#     batch_size = 32
#
#     # df = load_data('dataset/train.csv', 'dataset/trainImages/', (128, 128), grey_scale=True)
#     # X, y = df['Images'].to_numpy(), df['Condition'].to_numpy()
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=seed)
#     # train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#     # test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
#     train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#         directory='dataset/trainImages/',
#         validation_split=0.2,
#         subset='training',
#         seed=seed,
#         image_size=(img_height, img_width),
#         batch_size=batch_size
#     )
#     test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#         directory='dataset/trainImages/',
#         validation_split=0.2,
#         subset='validation',
#         seed=seed,
#         image_size=(img_height, img_width),
#         batch_size=batch_size
#     )
#
#     AUTOTUNE = tf.data.AUTOTUNE
#
#     train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#     test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
#     model = ak.ImageClassifier(overwrite=True, max_trials=5)
#     model.fit(train_ds, validation_split=0.2)
#     model.evaluate(test_ds)
#
#     try:
#         model.export_model().save("classifier_model_autokeras", save_format="tf")
#     except Exception:
#         model.export_model().save("classifier_model_autokeras.h5")
#
#     return


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

    class_weights = compute_class_weight('balanced', classes=np.unique(train_conditions), y=np.ravel(train_conditions))
    class_weights = dict(enumerate(class_weights))
    print(class_weights)

    model = ak.ImageClassifier(overwrite=True, max_trials=5)
    model.fit(train_images, train_conditions, validation_split=0.2, class_weight=class_weights)

    model.evaluate(test_images, test_conditions)
    # loss: 0.2412 - accuracy: 0.9245

    predicted_conditions = list(map(lambda x: 1 if x > 0.5 else 0, model.predict(test_images)))
    score = max(0, 100 * f1_score(test_conditions, predicted_conditions, average='macro'))
    print(f'Classifier score: {score:.4f}')

    try:
        model.export_model().save("classifier_model_autokeras2", save_format="tf")
    except Exception:
        model.export_model().save("classifier_model_autokeras2.h5")

    return


if __name__ == "__main__":
    main()
