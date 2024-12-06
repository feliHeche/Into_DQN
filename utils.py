from tensorflow import keras
from keras import layers
import tensorflow as tf
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw


def create_mlp(num_units, output_activation, dropout_rate, output_dim, name=None):
    """
    Build and return a multilayer perceptron (MLP).

    :param num_units: Number of units per layers. The length of the sequence, determine the number of layers.
    :param output_activation: Last activation function.
    :param dropout_rate: Dropout rate used for all layers.
    :param output_dim: Output dimension.
    :param name: Name of the MLP.
    :return: a MLP built with above hyperparameters.
    """
    fnn_layers = []
    for units in num_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation='relu'))

    # Add the output layer
    fnn_layers.append(layers.Dense(output_dim, activation=output_activation))

    return keras.Sequential(fnn_layers, name=name)


def soft_update_from_to(source_model, target_model, tau):
    """
    Soft update from the source_model to the target.

    :param source_model: Source model where the weight are coming from.
    :param target_model: Model whose weights need to be updated.
    """
    for params, new_params in zip(target_model.weights, source_model.weights):
        new_param_ = tau * tf.stop_gradient(new_params) + (1.0 - tau)*tf.stop_gradient(params)
        params.assign(new_param_)


def label_with_episode_number(frame, episode_num):
    """
    Small utils function used to save frames into a gif file.
    """
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text((im.size[0]/20, im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)

    return im

