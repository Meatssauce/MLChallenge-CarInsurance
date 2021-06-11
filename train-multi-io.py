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


def make_model(img_height, img_width, metadata_dim):
    image_input = Input(shape=(img_height, img_width, 3), name='image')
    tabular_input = Input(metadata_dim, name='metadata')

    # experimental.preprocessing.Rescaling(1./225)
    experimental.preprocessing.RandomFlip()
    experimental.preprocessing.RandomZoom(0.2)
    experimental.preprocessing.RandomRotation(0.2)

    resnet_output = ResNet50(weights=None, include_top=False)(image_input)
    image_features = GlobalAveragePooling2D()(resnet_output)  # ?
    image_features = Dropout(0.5)(image_features)  # ?

    metadata_features = Dense(metadata_dim // 2)(tabular_input)
    metadata_features = BatchNormalization()(metadata_features)
    metadata_features = Activation(activations.relu)(metadata_features)

    metadata_features = Dense(metadata_dim // 3)(metadata_features)
    metadata_features = BatchNormalization()(metadata_features)
    metadata_features = Activation(activations.relu)(metadata_features)

    x = concatenate([image_features, metadata_features])

    x = Dense(int_shape(x)[1] // 2, kernel_regularizer='l1')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = Dropout(0.2)(x)

    condition_pred = Dense(1, activation='softmax', name='condition')(x)
    amount_pred = Dense(1, name='amount')(x)

    model = Model(inputs=[image_input, tabular_input],
                  outputs=[condition_pred, amount_pred],
                  name='car_insurance_tabular_resnet')
    model.summary()

    # keras.utils.plot_model(model, "car_insurance_tabular_resnet.png", show_shapes=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss={
                      'condition': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      'amount': tf.keras.losses.MeanSquaredError(),
                  },
                  loss_weights=[1.0, 1.0],
                  )

    return model


def main():
    seed = 42
    img_height, img_width = 128, 128
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

    model = make_model(img_height, img_width, train_df.shape[1])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=100,  restore_best_weights=True
    )
    model.fit(
        {'image': train_images, 'metadata': train_df},
        {'condition': train_conditions, 'amount': train_amount},
        validation_split=0.2,
        epochs=1000,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )
    # val_condition_loss: 0.1842 - val_amount_loss: 6983358.5000

    try:
        model.save("car-Insurance-tabular-resNet", save_format="tf")
    except Exception:
        model.save('car-Insurance-tabular-resNet.h5')

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
