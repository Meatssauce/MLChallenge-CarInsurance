import tensorflow as tf
from sklearn.utils import compute_class_weight
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, elu, softmax
from tensorflow import identity
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score
from tools import load_images, Preprocessor


def get_typeNet(n_table_cols, image_shape=(128, 128, 1), output_size=(1, 1), num_top_features=128):
    assert len(image_shape) == 3 and image_shape[0] == image_shape[1]

    table = Input(n_table_cols, name='table')
    image = Input(shape=image_shape, name='image')

    x1 = Dense(n_table_cols // 2)(table)
    x1 = BatchNormalization()(x1)
    x1 = Activation(relu)(x1)

    x1 = Dense(n_table_cols // 3)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation(relu)(x1)

    x2 = experimental.preprocessing.RandomFlip()(image)
    x2 = experimental.preprocessing.RandomZoom(0.2)(x2)
    x2 = experimental.preprocessing.RandomRotation(0.2)(x2)

    last_conv_size = 128
    x2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation=elu, use_bias=False)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=elu, use_bias=False)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=elu, use_bias=False)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(filters=last_conv_size, kernel_size=3, strides=1, padding='same', activation=elu, use_bias=False)(x2)
    x2 = BatchNormalization()(x2)

    nt1 = Conv2D(filters=num_top_features, kernel_size=1, strides=1, padding='valid', activation=identity,
                 use_bias=False)(x2)
    nt2 = Conv2D(filters=num_top_features, kernel_size=1, strides=1, padding='valid', activation=identity,
                 use_bias=False)(x2)

    x2 = nt1 + nt2

    ns1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x2)
    ns2 = identity(x2)
    ns3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x2)

    x2 = concatenate([ns1, ns2, ns3])

    in_size = 3 * num_top_features
    x2 = Flatten()(x2)
    x2 = Dense(in_size, activation='elu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dense(in_size // 2, activation='elu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dense(in_size // 4, activation='elu')(x2)
    x2 = BatchNormalization()(x2)

    x3 = concatenate([x1, x2])

    condition = Dense(output_size[0], activation='softmax', name='condition')(x3)
    amount = Dense(output_size[1], activation='relu', name='amount')(x3)

    model = Model(inputs=[table, image], outputs=[condition, amount])
    # tf.keras.utils.plot_model(model, "typeNet.png", show_shapes=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
          'condition': 'binary_crossentropy',
          'amount': 'mse',
        },
        loss_weights=[0.5, 0.5],
        # metrics=['accuracy']
    )

    return model


def main():
    seed = 42
    img_height, img_width = 128, 128
    batch_size = 32

    df = pd.read_csv("dataset/train.csv")
    df = df.dropna(axis=0, how='any', subset=['Amount', 'Condition'])
    df['Images'] = load_images(df, directory='dataset/trainImages', size=(img_height, img_width), grey_scale=True)

    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=seed)
    train_images = np.asarray(train_df.pop('Images').to_list())
    test_images = np.asarray(test_df.pop('Images').to_list())
    train_conditions, train_amount = np.asarray(train_df.pop('Condition')), np.asarray(train_df.pop('Amount'))
    test_conditions, test_amount = np.asarray(test_df.pop('Condition')), np.asarray(test_df.pop('Amount'))

    preprocessor = Preprocessor()
    train_df = preprocessor.fit_transform(train_df)
    test_df = preprocessor.transform(test_df)

    # class_weights = compute_class_weight('balanced', classes=np.unique(train_conditions), y=np.ravel(train_conditions))
    # class_weights = dict(enumerate(class_weights))

    model = get_typeNet(train_df.shape[1])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=100, restore_best_weights=True
    )

    model.fit(
        {'image': train_images, 'table': train_df},
        {'condition': train_conditions, 'amount': train_amount},
        validation_split=0.2,
        epochs=1000,
        callbacks=[early_stopping],
        # class_weight=class_weights
    )
    # - loss: 5756750.5000 - condition_loss: 0.2311 - amount_loss: 11513501.0000 - val_loss: 3263970.5000
    # - val_condition_loss: 0.2366 - val_amount_loss: 6527941.0000

    try:
        model.save('TypeNetClassifierRegressor', save_format="tf")
    except Exception:
        model.save('TypeNetClassifierRegressor.h5')

    model.evaluate(
        {'image': test_images, 'table': test_df},
        {'condition': test_conditions, 'amount': test_amount}
    )

    predicted_conditions, predicted_amount = model.predict(
        {'image': test_images, 'table': test_df}
    )

    condition_score = max(0, 100 * f1_score(test_conditions, predicted_conditions, average='macro'))
    amount_score = max(0, 100 * r2_score(test_amount, predicted_amount))

    print(f'Classifier score: {condition_score:.4f}')
    print(f'Regressor score: {amount_score:.4f}')
    print(f'Final score: {np.mean([condition_score, amount_score])}')
    # - loss: 3540382.2500 - condition_loss: 0.2490 - amount_loss: 7080763.5000
    # Classifier score: 48.0374
    # Regressor score: 0.0000

    return


if __name__ == "__main__":
    main()
