from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization, Add,
                                     Conv2DTranspose, concatenate, Dropout, Activation)

def residual_block(x, filters):
    """Residual Block dengan penyesuaian channel pada shortcut"""
    shortcut = Conv2D(filters, (1, 1), padding="same")(x)  # Menyesuaikan channel shortcut

    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])  # Residual connection setelah penyesuaian channel
    x = Activation("relu")(x)

    return x

def multi_unet_model(n_classes=5, image_height=256, image_width=256, image_channels=3):
    inputs = Input((image_height, image_width, image_channels)) 

    # Encoder
    c1 = residual_block(inputs, 16)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = residual_block(p1, 32)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = residual_block(p2, 64)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = residual_block(p3, 128)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = residual_block(p4, 256)

    # Decoder
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = residual_block(u6, 128)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = residual_block(u7, 64)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = residual_block(u8, 32)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = residual_block(u9, 16)

    outputs = Conv2D(n_classes, (1, 1), activation="softmax")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model