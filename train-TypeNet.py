import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.activations import elu, softmax
from tensorflow import identity


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

    ns1 = MaxPooling2D(pool_size=(3, 3))(x)
    ns2 = identity(x)
    ns3 = MaxPooling2D(pool_size=(5, 5))(x)

    x = concatenate([ns1, ns2, ns3])

    in_size = 3 * num_top_features
    x = Dense(in_size, activation='elu')(x)
    x = BatchNormalization(x)
    x = Dense(in_size // 2, activation='elu')(x)
    x = BatchNormalization(x)
    x = Dense(in_size // 4, activation='elu')(x)
    x = BatchNormalization(x)
    x = Dense(output_size, activation='elu')(x)
    x = BatchNormalization(x)

    model = Model(inputs=[image], outputs=[x])
    tf.keras.utils.plot_model(model, "typeNet.png", show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    return model


def main():
    model = get_typeNet()
    print('done')
    return


if __name__ == "__main__":
    main()
