import os
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, LayerNormalization, Add, Softmax, GlobalAveragePooling2D, Dropout

# CNN Model

def cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# SHMSA Block

def shmsa_block(inputs, num_heads, d_model):
    depth = d_model // num_heads

    def split_heads(x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(x):
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, d_model))

    query = Dense(d_model)(inputs)
    key = Dense(d_model)(inputs)
    value = Dense(d_model)(inputs)

    key_down = Dense(d_model // 2)(key)
    query_down = Dense(d_model // 2)(query)

    key_transposed = tf.transpose(key_down, perm=[0, 2, 1])
    attn_s = tf.matmul(query_down, key_transposed)
    attn_s = Softmax()(attn_s)
    context_s = tf.matmul(attn_s, value)

    query_transposed = tf.transpose(query_down, perm=[0, 2, 1])
    attn_c = tf.matmul(key_down, query_transposed)
    attn_c = Softmax()(attn_c)
    context_c = tf.matmul(attn_c, value)

    context = Add()([context_s, context_c])
    output = combine_heads(context)
    output = Add()([inputs, output])

    return output

# Bayesian Optimization for CNN Hyperparameters

def black_box_function(params):
    num_filters, dropout_rate = int(params[0]), params[1]
    input_shape = (32, 32, 3)
    num_classes = 10

    model = cnn_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Example dataset (CIFAR-10 for simplicity)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)

    return -accuracy

# Bayesian Optimization Setup

def run_bayesian_optimization():
    search_space = [
        Real(16, 64, name='num_filters'),
        Real(0.1, 0.5, name='dropout_rate')
    ]

    result = gp_minimize(
        func=black_box_function,
        dimensions=search_space,
        acq_func='EI',
        n_calls=10,
        n_initial_points=5,
        random_state=42
    )

    print("Best parameters found:", result.x)
    print("Best accuracy achieved:", -result.fun)

# Main Script
if __name__ == "__main__":
    print("Running CNN Model...")
    input_shape = (32, 32, 3)
    num_classes = 10
    model = cnn_model(input_shape, num_classes)
    model.summary()

    print("Running SHMSA Block...")
    inputs = Input(shape=(32, 32, 3))
    shmsa_output = shmsa_block(inputs, num_heads=4, d_model=64)

    print("Running Bayesian Optimization...")
    run_bayesian_optimization()
