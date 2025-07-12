import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Flatten, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

import tensorflow as tf


def add_pos_2(input, nb):
    input_pos_encoding = tf.constant(nb, shape=[input.shape[1]], dtype="int32") / input.shape[1]
    input_pos_encoding = tf.cast(tf.reshape(input_pos_encoding, [1, 10]), tf.float32)
    input = tf.add(input, input_pos_encoding)
    return input

def stack_block_transformer(num_transformer_blocks):
    input1 = keras.Input(shape=(100, 1))
    x = input1
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, 100, 2)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dense(10, activation='selu')(x)
    return input1, x

def stack_block_transformer_spatial(num_transformer_blocks, x):
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, 10 * 18, 2)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    return x


def transformer_encoder(inputs, key_dim, num_heads):
    dropout = 0.1
    # Normalization and Attention
    # print("transformer_encoder",inputs.shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=key_dim, num_heads=num_heads
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(key_dim, activation='softmax')(x)
    return x + res


def multiple_transformer(nb):
    '''

    :param nb: number of features ( indicates the number of parallel branches)
    :return:
    '''
    # initialise with the first input

    num_transformer_blocks = 2  # hyperparameter
    input_, transformer_ = stack_block_transformer(num_transformer_blocks)
    transformers = []
    inputs = []
    transformers.append(transformer_)
    inputs.append(input_)
    for i in range(1, nb):
        input_i, transformer_i = stack_block_transformer(num_transformer_blocks)
        inputs.append(input_i)
        transformer_i = add_pos_2(transformer_i, i)
        transformers.append(transformer_i)

    x = layers.concatenate(transformers, axis=-1)
    x = tf.expand_dims(x, -1)  # -1 denotes the last dimension
    x = stack_block_transformer_spatial(num_transformer_blocks, x)
    x = Dropout(0.1)(x)
    x = layers.Dense(100, activation='selu')(x)
    x = Dropout(0.1)(x)
    x = layers.Dense(20, activation='selu')(x)
    x = Dropout(0.1)(x)
    answer = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, answer)
    opt = optimizers.RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], experimental_run_tf_function=False)
    # print(model.summary())
    return model


def multiple_transformer_5_level(nb):
    '''
    Model for severity prediction , 5 classes output
    :param nb:  number of parallel branch
    :return:
    '''

    # initialise with the first input

    num_transformer_blocks = 2  # hyperparameter
    input_, transformer_ = stack_block_transformer(num_transformer_blocks)
    transformers = []
    inputs = []
    transformers.append(transformer_)
    inputs.append(input_)
    for i in range(1, nb):
        input_i, transformer_i = stack_block_transformer(num_transformer_blocks)
        inputs.append(input_i)
        transformer_i = add_pos_2(transformer_i, i)
        transformers.append(transformer_i)

    x = layers.concatenate(transformers, axis=-1)
    x = tf.expand_dims(x, -1)  # -1 denotes the last dimension
    x = stack_block_transformer_spatial(num_transformer_blocks, x)
    x = Dropout(0.1)(x)
    x = layers.Dense(100, activation='selu')(x)
    x = Dropout(0.1)(x)
    x = layers.Dense(20, activation='selu')(x)
    x = Dropout(0.1)(x)
    answer = layers.Dense(5, activation='sigmoid')(x)

    model = Model(inputs, answer)
    opt = optimizers.RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], experimental_run_tf_function=False)
    # print(model.summary())
    return model