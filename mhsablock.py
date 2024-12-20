import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Softmax, Add

def emhsa_block(inputs, num_heads, d_model):
    """
    Efficient Multi-Head Self-Attention (EMHSA) block.

    Args:
        inputs: Input tensor of shape (batch_size, seq_length, d_model).
        num_heads: Number of attention heads.
        d_model: Dimensionality of the model.

    Returns:
        Output tensor of shape (batch_size, seq_length, d_model).
    """
    depth = d_model // num_heads

    def split_heads(x):
        "Split the last dimension into (num_heads, depth)."
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(x):
        "Combine the heads back into (batch_size, seq_length, d_model)."
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, d_model))

    # Linear projections
    query = Dense(d_model)(inputs)
    key = Dense(d_model)(inputs)
    value = Dense(d_model)(inputs)

    # Downscaling keys and queries
    key_down = Dense(d_model // 2)(key)
    query_down = Dense(d_model // 2)(query)

    # Query-less Attention
    key_transposed = tf.transpose(key_down, perm=[0, 2, 1])
    attn_s = tf.matmul(query_down, key_transposed)
    attn_s = Softmax()(attn_s)
    context_s = tf.matmul(attn_s, value)

    # Key-less Attention
    query_transposed = tf.transpose(query_down, perm=[0, 2, 1])
    attn_c = tf.matmul(key_down, query_transposed)
    attn_c = Softmax()(attn_c)
    context_c = tf.matmul(attn_c, value)

    # Combine contexts
    context = Add()([context_s, context_c])

    # Combine heads
    output = combine_heads(context)

    # Residual connection
    output = Add()([inputs, output])

    return output

def crosshead_interaction(inputs, d_model):
    """
    Crosshead interaction layer.

    Args:
        inputs: Input tensor of shape (batch_size, seq_length, d_model).
        d_model: Dimensionality of the model.

    Returns:
        Output tensor after crosshead interaction.
    """
    x = LayerNormalization()(inputs)
    x = Dense(d_model, activation='relu')(x)
    x = Dense(d_model)(x)
    return Add()([inputs, x])

def build_emhsa_model(input_shape, num_heads, d_model):
    """
    Build a model using the EMHSA block and Crosshead Interaction.

    Args:
        input_shape: Shape of the input tensor (seq_length, d_model).
        num_heads: Number of attention heads.
        d_model: Dimensionality of the model.

    Returns:
        Keras Model.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # EMHSA block
    x = emhsa_block(inputs, num_heads, d_model)

    # Crosshead Interaction
    x = crosshead_interaction(x, d_model)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# Define parameters
input_shape = (128, 64)  # Example: sequence length = 128, embedding dim = 64
num_heads = 4
d_model = 64

# Build and summarize the model
model = build_emhsa_model(input_shape, num_heads, d_model)
model.summary()
