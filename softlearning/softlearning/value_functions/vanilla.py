import tensorflow as tf
from tensorflow.python.keras.engine import training_utils

from softlearning.models.feedforward import feedforward_model
from softlearning.models.utils import flatten_input_structure, create_inputs
from softlearning.utils.keras import PicklableModel


def create_feedforward_Q_function(input_shapes,
                                  *args,
                                  latent_dim=0,
                                  preprocessors=None,
                                  observation_keys=None,
                                  name='feedforward_Q',
                                  **kwargs):
    if latent_dim > 0:
        input_shapes['observations']['env_latents'] = tf.TensorShape(latent_dim)
        preprocessors['observations']['env_latents'] = None
    else:
        1/0
    inputs_flat = create_inputs(input_shapes)
    preprocessors_flat = (
        flatten_input_structure(preprocessors)
        if preprocessors is not None
        else tuple(None for _ in inputs_flat))

    assert len(inputs_flat) == len(preprocessors_flat), (
        inputs_flat, preprocessors_flat)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_
        in zip(preprocessors_flat, inputs_flat)
    ]

    Q_function = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs)

    Q_function = PicklableModel(inputs_flat, Q_function(preprocessed_inputs))
    Q_function.observation_keys = observation_keys

    return Q_function
