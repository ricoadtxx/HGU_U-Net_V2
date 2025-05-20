from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization, Add,
                                     Conv2DTranspose, concatenate, Dropout, Activation,
                                     Multiply, Reshape, Permute, Lambda)
from tensorflow.keras import backend as K

def attention_gate(x, g, filters):
    """
    Attention Gate untuk fokus pada fitur penting
    x: feature map dari encoder (skip connection)
    g: gating signal dari decoder
    """
    theta_x = Conv2D(filters, (1, 1), padding='same')(x)  # Transformasi untuk feature map
    phi_g = Conv2D(filters, (1, 1), padding='same')(g)    # Transformasi untuk gating signal
    
    # Menggabungkan feature map dan gating signal
    f = Activation('relu')(Add()([theta_x, phi_g]))
    
    # Komputasi attention coefficients
    psi_f = Conv2D(1, (1, 1), padding='same')(f)
    attention = Activation('sigmoid')(psi_f)
    
    # Aplikasi attention coefficients pada feature map
    y = Multiply()([x, attention])
    
    return y, attention

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

def multi_unet_model_with_attention(n_classes=5, image_height=256, image_width=256, image_channels=3):
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
    
    # Decoder dengan Attention Gates
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    att6, att_map6 = attention_gate(c4, u6, 128)  # Attention gate
    u6 = concatenate([u6, att6])
    c6 = residual_block(u6, 128)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    att7, att_map7 = attention_gate(c3, u7, 64)  # Attention gate
    u7 = concatenate([u7, att7])
    c7 = residual_block(u7, 64)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    att8, att_map8 = attention_gate(c2, u8, 32)  # Attention gate
    u8 = concatenate([u8, att8])
    c8 = residual_block(u8, 32)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    att9, att_map9 = attention_gate(c1, u9, 16)  # Attention gate
    u9 = concatenate([u9, att9])
    c9 = residual_block(u9, 16)
    
    outputs = Conv2D(n_classes, (1, 1), activation="softmax")(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def spatial_attention_module(x, reduction_ratio=16):
    """
    Spatial Attention Module - fokus pada lokasi spasial yang penting
    """
    batch_size, height, width, channel = K.int_shape(x)
    
    # Channel attention
    avg_pool = Lambda(lambda x: K.mean(x, axis=[1, 2], keepdims=True))(x)
    avg_pool = Conv2D(channel // reduction_ratio, (1, 1), padding='same', activation='relu')(avg_pool)
    avg_pool = Conv2D(channel, (1, 1), padding='same')(avg_pool)
    
    max_pool = Lambda(lambda x: K.max(x, axis=[1, 2], keepdims=True))(x)
    max_pool = Conv2D(channel // reduction_ratio, (1, 1), padding='same', activation='relu')(max_pool)
    max_pool = Conv2D(channel, (1, 1), padding='same')(max_pool)
    
    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Activation('sigmoid')(channel_attention)
    
    # Spatial attention
    channel_refined = Multiply()([x, channel_attention])
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined)
    concat = concatenate([avg_pool, max_pool])
    
    spatial_attention = Conv2D(1, (7, 7), padding='same')(concat)
    spatial_attention = Activation('sigmoid')(spatial_attention)
    
    refined = Multiply()([channel_refined, spatial_attention])
    
    return Add()([refined, x])  # Residual connection

def multi_unet_model_with_spatial_attention(n_classes=5, image_height=256, image_width=256, image_channels=3):
    inputs = Input((image_height, image_width, image_channels)) 
    
    # Encoder dengan Spatial Attention
    c1 = residual_block(inputs, 16)
    c1 = spatial_attention_module(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = residual_block(p1, 32)
    c2 = spatial_attention_module(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = residual_block(p2, 64)
    c3 = spatial_attention_module(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = residual_block(p3, 128)
    c4 = spatial_attention_module(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    c5 = residual_block(p4, 256)
    c5 = spatial_attention_module(c5)
    
    # Decoder
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = residual_block(u6, 128)
    c6 = spatial_attention_module(c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = residual_block(u7, 64)
    c7 = spatial_attention_module(c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = residual_block(u8, 32)
    c8 = spatial_attention_module(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = residual_block(u9, 16)
    c9 = spatial_attention_module(c9)
    
    outputs = Conv2D(n_classes, (1, 1), activation="softmax")(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def multi_unet_model_with_combined_attention(n_classes=5, image_height=256, image_width=256, image_channels=3):
    """
    U-Net dengan kombinasi attention gate dan spatial attention module
    """
    inputs = Input((image_height, image_width, image_channels)) 
    
    # Encoder dengan Spatial Attention
    c1 = residual_block(inputs, 16)
    c1 = spatial_attention_module(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = residual_block(p1, 32)
    c2 = spatial_attention_module(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = residual_block(p2, 64)
    c3 = spatial_attention_module(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = residual_block(p3, 128)
    c4 = spatial_attention_module(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    c5 = residual_block(p4, 256)
    c5 = spatial_attention_module(c5)
    
    # Decoder dengan Attention Gates
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    att6, _ = attention_gate(c4, u6, 128)  # Attention gate
    u6 = concatenate([u6, att6])
    c6 = residual_block(u6, 128)
    c6 = spatial_attention_module(c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    att7, _ = attention_gate(c3, u7, 64)  # Attention gate
    u7 = concatenate([u7, att7])
    c7 = residual_block(u7, 64)
    c7 = spatial_attention_module(c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    att8, _ = attention_gate(c2, u8, 32)  # Attention gate
    u8 = concatenate([u8, att8])
    c8 = residual_block(u8, 32)
    c8 = spatial_attention_module(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    att9, _ = attention_gate(c1, u9, 16)  # Attention gate
    u9 = concatenate([u9, att9])
    c9 = residual_block(u9, 16)
    c9 = spatial_attention_module(c9)
    
    outputs = Conv2D(n_classes, (1, 1), activation="softmax")(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model