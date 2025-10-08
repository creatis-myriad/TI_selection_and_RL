
import sys
import tensorflow as tf

sys.path.append('../../') 

from tensorflow import keras
from keras import layers
from nn_libr.layers.SamplingLayer import SamplingLayer



def Encoder(latent_dim,shape_input, summary=False) :
    encoder_inputs = tf.keras.Input(shape=shape_input)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(encoder_inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim)(x)


    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z         = SamplingLayer()([z_mean, z_log_var])

    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    if summary : encoder.summary()

    return encoder


def Encoder_unwrap(latent_dim,shape_input, convX=9, summary=False) :
    encoder_inputs = tf.keras.Input(shape=shape_input)

    x = layers.Conv2D(32, (convX, 5), activation="relu", padding="same")(encoder_inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = layers.Conv2D(64, (convX, 5), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), padding='same')(x)

    x = layers.Conv2D(128, (convX, 5), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim)(x)


    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z         = SamplingLayer()([z_mean, z_log_var])

    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    if summary : encoder.summary()

    return encoder

