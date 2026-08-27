"""
=============================================================================
FUMD-AI Training Workflow -- shared library: forecasting model
=============================================================================
Used by: notebooks/step_2_train_model.ipynb, step_3_evaluate_model.ipynb

Author(s):
  - Cristina Bernad (ORCID: 0000-0001-9537-415X)
  - Sonja Filiposka <sonja.filiposka@finki.ukim.mk> (ORCID: 0000-0003-0034-2855)
  - Katja Gilly (ORCID: 0000-0002-8985-0639)

Funding: This work has been funded by the FUMD-AI project, an EOSC GRAVITY -
Inter Project with Grant Number 25-EOSC-GRV-INTER-013.

SPDX-License-Identifier: MIT
-----------------------------------------------------------------------------

The multi-step servingCell forecaster: a 2x bidirectional-LSTM encoder
feeding a per-step Bahdanau-attention decoder that autoregressively emits
one softmax classification per future second (`len(future_steps)` outputs,
each over `num_base_stations` classes). This is the same architecture as
the original exploratory notebook
(combined_TimeSequence_*_BahdanauAtention_encoder_2BLSTM.ipynb) -- it's a
validated design (~95% top-1 / ~99% top-2 accuracy across 8 real datasets in
the original study) -- rebuilt as a single parameterized, importable
function instead of one-off inline notebook cells.

Only `tensorflow.keras` imports are used throughout (never a bare `import
keras`) so behaviour doesn't depend on which of the two the interpreter
happens to resolve first -- both are pinned to the same 2.14.0 release in
this project's execution environment, but there's no reason to rely on that.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    Layer,
)
from tensorflow.keras.optimizers import Adam


class BahdanauAttention(Layer):
    """
    Additive (Bahdanau) attention over the encoder's output sequence,
    scored against the decoder's current hidden state.

    inputs: (encoder_output, decoder_hidden_state)
        encoder_output:      (batch, seq_len, encoder_dim)
        decoder_hidden_state: (batch, decoder_dim)
    returns: (context_vector, attention_weights)
        context_vector:    (batch, encoder_dim)
        attention_weights: (batch, seq_len)
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        encoder_output_shape, decoder_hidden_state_shape = input_shape
        self.W1 = self.add_weight(
            name="W1", shape=(encoder_output_shape[-1], self.units),
            initializer="glorot_uniform", trainable=True,
        )
        self.W2 = self.add_weight(
            name="W2", shape=(decoder_hidden_state_shape[-1], self.units),
            initializer="glorot_uniform", trainable=True,
        )
        self.V = self.add_weight(
            name="V", shape=(self.units, 1),
            initializer="glorot_uniform", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        encoder_output, decoder_hidden_state = inputs
        query = tf.tensordot(encoder_output, self.W1, axes=[[2], [0]])
        key = tf.matmul(decoder_hidden_state, self.W2)
        score = tf.nn.tanh(query + key[:, tf.newaxis, :])
        attention_score = tf.squeeze(tf.tensordot(score, self.V, axes=[[2], [0]]), axis=-1)
        attention_weights = tf.nn.softmax(attention_score, axis=-1)
        context_vector = tf.reduce_sum(attention_weights[:, :, tf.newaxis] * encoder_output, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class ZeroInitialState(Layer):
    """
    Returns a batch of zero vectors shaped `(batch_size, units)`, with
    `batch_size` inferred from `inputs` (any tensor sharing the model's
    batch dimension) -- used as the decoder's initial state before the
    first forecast step.

    This used to be `Lambda(lambda x: tf.zeros((tf.shape(x)[0], units)))`,
    which works fine for training/inference within one process, but the
    `.keras` format's "safe mode" deserialization refuses by default to
    load a Lambda layer's embedded Python bytecode when reopening a saved
    model (`ValueError: Requested the deserialization of a Lambda layer
    with a Python \`lambda\` inside it...`) -- exactly what
    step_3_evaluate_model.ipynb does. A proper subclassed Layer with
    `get_config()` (same pattern as `BahdanauAttention` above) serializes
    as ordinary, safe layer config instead of bytecode, so the saved model
    reloads without needing `safe_mode=False`.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.zeros((batch_size, self.units))

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_model(
    sequence_length: int,
    num_features: int,
    future_steps: list[int],
    num_base_stations: int,
    *,
    lstm_units: int = 128,
    attention_units: int = 128,
    embedding_dim: int = 128,
    dense_units: int = 128,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> Model:
    """
    Build (and compile) the encoder-decoder forecasting model.

    `num_base_stations` should cover the full range of serving-cell ids
    that appear anywhere in the *training* data (current, lagged, and
    target values) -- see `evaluate.infer_num_base_stations`. Every output
    head is a softmax over exactly this many classes.
    """
    n_steps = len(future_steps)

    encoder_input = Input(shape=(sequence_length, num_features), name="encoder_input")
    encoder_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bi_lstm_1")(encoder_input)
    encoder_out = Dropout(dropout_rate)(encoder_out)
    encoder_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bi_lstm_2")(encoder_out)
    encoder_out = Dropout(dropout_rate)(encoder_out)

    embedding_layer = Embedding(input_dim=num_base_stations, output_dim=embedding_dim, name="embedding")

    # Pre-instantiate per-step layers (rather than inside the loop below) so
    # each forecast step gets its own trainable weights instead of reusing
    # -- and silently overwriting -- one shared layer instance.
    dense_shared_layers = [Dense(dense_units, activation="relu", name=f"shared_dense_{t + 1}") for t in range(n_steps)]
    dropout_layers1 = [Dropout(dropout_rate, name=f"dropout1_{t + 1}") for t in range(n_steps)]
    dropout_layers2 = [Dropout(dropout_rate, name=f"dropout2_{t + 1}") for t in range(n_steps)]
    dense_output_layers = [Dense(num_base_stations, activation="softmax", name=f"output_{t + 1}") for t in range(n_steps)]
    attention_layers = [BahdanauAttention(units=attention_units, name=f"attention_{t + 1}") for t in range(n_steps)]

    outputs = []
    # Step 0's decoder state is a zero vector; each later step conditions on
    # the embedding of the *previous* step's predicted (argmax) class.
    decoder_state = ZeroInitialState(units=embedding_dim, name="zero_initial_decoder_state")(encoder_input)

    for t in range(n_steps):
        context_vector, _attn_weights = attention_layers[t]([encoder_out, decoder_state])
        context_vector = dropout_layers1[t](context_vector)
        shared_rep = dense_shared_layers[t](context_vector)
        shared_rep = dropout_layers2[t](shared_rep)
        output_t = dense_output_layers[t](shared_rep)
        outputs.append(output_t)

        predicted_token = tf.argmax(output_t, axis=-1)
        decoder_state = embedding_layer(predicted_token)

    model = Model(inputs=encoder_input, outputs=outputs, name="fumd_migration_forecaster")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipvalue=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
