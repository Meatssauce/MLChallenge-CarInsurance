import tensorflow as tf
from sklearn.utils import compute_class_weight
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, concatenate, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.activations import elu, softmax
from tensorflow import identity
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tools import load_images, Preprocessor


def get_typeNet(input_shape=(96, 96, 1), output_size=1, num_top_features=128):
    assert len(input_shape) == 3 and input_shape[0] == input_shape[1]

    last_conv_size = 128

    image = Input(shape=input_shape)
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation=elu, use_bias=False)(image)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=elu, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=elu, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=last_conv_size, kernel_size=3, strides=1, padding='same', activation=elu, use_bias=False)(x)
    x = BatchNormalization()(x)

    nt1 = Conv2D(filters=num_top_features, kernel_size=1, strides=1, padding='valid', activation=identity,
                 use_bias=False)(x)
    nt2 = Conv2D(filters=num_top_features, kernel_size=1, strides=1, padding='valid', activation=identity,
                 use_bias=False)(x)

    x = nt1 + nt2

    ns1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    ns2 = identity(x)
    ns3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)

    x = concatenate([ns1, ns2, ns3])

    in_size = 3 * num_top_features
    x = Flatten()(x)
    x = Dense(in_size, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(in_size // 2, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(in_size // 4, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(output_size, activation='softmax')(x)

    model = Model(inputs=image, outputs=x)
    # tf.keras.utils.plot_model(model, "typeNet.png", show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    return model


def main():
    seed = 42
    img_height, img_width = 96, 96
    batch_size = 32

    df = pd.read_csv("dataset/train.csv")
    df = df.dropna(axis=0, how='any', subset=['Amount', 'Condition'])
    df['Images'] = load_images(df, directory='dataset/trainImages', size=(img_height, img_width), grey_scale=True)

    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=seed)
    train_images = np.asarray(train_df.pop('Images').to_list())
    test_images = np.asarray(test_df.pop('Images').to_list())
    train_conditions, train_amount = np.asarray(train_df.pop('Condition')), np.asarray(train_df.pop('Amount'))
    test_conditions, test_amount = np.asarray(test_df.pop('Condition')), np.asarray(test_df.pop('Amount'))

    # preprocessor = Preprocessor()
    # train_df = preprocessor.fit_transform(train_df)
    # test_df = preprocessor.transform(test_df)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_conditions), y=np.ravel(train_conditions))
    class_weights = dict(enumerate(class_weights))

    model = get_typeNet()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=20, restore_best_weights=True
    )
    # from sklearn.preprocessing import OneHotEncoder
    # train_conditions = np.asarray(OneHotEncoder().fit_transform(train_conditions.reshape(-1, 1)))
    model.fit(
        train_images,
        train_conditions,
        validation_split=0.2,
        epochs=1000,
        callbacks=[early_stopping],
        class_weight=class_weights
    )
    # loss: 0.0022 - accuracy: 0.9257 - val_loss: 0.3291 - val_accuracy: 0.9459

    try:
        model.save("car-Insurance-tabular-resNet", save_format="tf")
    except Exception:
        model.save('car-Insurance-tabular-resNet.h5')

    model.evaluate(test_images, test_conditions)

    predicted_conditions = model.predict(test_images)

    condition_score = max(0, 100 * f1_score(test_conditions, predicted_conditions, average='macro'))
    print(f'Classifier score: {condition_score:.4f}')

    return


if __name__ == "__main__":
    main()
