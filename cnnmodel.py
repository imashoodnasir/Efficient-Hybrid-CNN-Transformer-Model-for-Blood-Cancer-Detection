import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, SeparableConv2D, Dense, Dropout, GlobalAveragePooling2D,
    LayerNormalization, Add, MultiHeadAttention, Embedding
)
from tensorflow.keras.models import Model

def efficient_attention_block(inputs, num_heads):
    """Efficient Attention Block."""
    x = LayerNormalization()(inputs)
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(x, x)
    x = Add()([inputs, attention_output])
    return x

def efficient_mlp_block(inputs, hidden_units, dropout_rate):
    """Efficient MLP Block."""
    x = LayerNormalization()(inputs)
    x = Dense(hidden_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Dropout(dropout_rate)(x)
    x = Add()([inputs, x])
    return x

def cnn_backbone(inputs):
    """CNN Backbone with separable convolutions."""
    x = Conv2D(20, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for _ in range(3):
        x = SeparableConv2D(20, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    return x

def transformer_block(inputs, num_heads, mlp_units, dropout_rate):
    """Transformer Block with Efficient Attention and MLP."""
    x = efficient_attention_block(inputs, num_heads)
    x = efficient_mlp_block(x, mlp_units, dropout_rate)
    return x

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # CNN Backbone
    cnn_features = cnn_backbone(inputs)

    # Flatten spatial dimensions for transformer input
    transformer_input = tf.reshape(cnn_features, shape=(-1, cnn_features.shape[1] * cnn_features.shape[2], cnn_features.shape[3]))

    # Add position embeddings
    position_embedding = Embedding(input_dim=transformer_input.shape[1], output_dim=transformer_input.shape[-1])(tf.range(start=0, limit=transformer_input.shape[1]))
    transformer_input = transformer_input + position_embedding

    # Transformer Block
    transformer_output = transformer_block(transformer_input, num_heads=4, mlp_units=64, dropout_rate=0.1)

    # Global Average Pooling and Dense Layers
    x = LayerNormalization()(transformer_output)
    x = GlobalAveragePooling2D()(cnn_features)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Define the model
input_shape = (224, 224, 3)  # Example input shape
num_classes = 10             # Example number of classes
model = build_model(input_shape, num_classes)

# Model Summary
model.summary()
