import tensorflow as tf
from tensorflow import keras
from keras import layers



def Decoder(latent_dim, shape_input, summary=False) :
    shape_x = int(shape_input[0]/8)
    shape_y = int(shape_input[1]/8)
    latent_inputs = tf.keras.Input(shape=(latent_dim,))

    x = layers.Dense(shape_x * shape_y * 128, activation="relu")(latent_inputs)
    x = layers.Reshape((shape_x, shape_y, 128))(x)
    x = layers.UpSampling2D(size=(2, 2))(x)

    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)

    x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)


    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
    decoder_outputs = tf.keras.activations.relu(decoder_outputs, max_value=1.0)

    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    if summary : decoder.summary()

    return decoder


def Decoder_unwrap(latent_dim, shape_input, convX=9, summary=False) :
    shape_x = int(shape_input[0]/8)
    shape_y = int(shape_input[1]/4)
    latent_inputs = tf.keras.Input(shape=(latent_dim,))

    x = layers.Dense(shape_x * shape_y * 128, activation="relu")(latent_inputs)
    x = layers.Reshape((shape_x, shape_y, 128))(x)
    x = layers.UpSampling2D(size=(2, 2))(x)

    x = layers.Conv2DTranspose(64, (convX, 5), activation="relu", padding="same")(x)
    x = layers.UpSampling2D(size=(2, 1))(x)

    x = layers.Conv2DTranspose(32, (convX, 5), activation="relu", padding="same")(x)
    x = layers.UpSampling2D(size=(2, 2))(x)


    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="relu", padding="same")(x)
    decoder_outputs = tf.keras.activations.relu(decoder_outputs, max_value=1.0)

    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    if summary : decoder.summary()

    return decoder