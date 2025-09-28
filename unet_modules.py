import tensorflow as tf

import keras
from keras import backend as K
from keras import layers, initializers
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda

def ECA(x: tf.Tensor, k_size: int = 3) -> tf.Tensor:
    """
    Efficient Channel Attention (ECA) module as a standalone function.
    
    Args:
        x (tf.Tensor): Input tensor with shape (batch, height, width, channels)
        k_size (int): Kernel size for Conv1D. Defaults to 3.
    
    Returns:
        tf.Tensor: Output tensor with same shape as input
    """
    # Global average pooling: (batch, height, width, channels) -> (batch, 1, 1, channels)
    y = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    
    # Reshape to (batch, channels, 1) for Conv1D
    batch_size = tf.shape(y)[0]
    channels = tf.shape(y)[-1]
    y = tf.reshape(y, [batch_size, channels, 1])
    
    # Apply 1D convolution along channel dimension
    conv = layers.Conv1D(filters=1, kernel_size=k_size, padding='same', use_bias=False)
    y = conv(y)  # (batch, channels, 1)
    
    # Reshape back to (batch, 1, 1, channels)
    y = tf.reshape(y, [batch_size, 1, 1, channels])
    
    # Apply sigmoid and multiply with input
    y = tf.sigmoid(y)
    return x * y


def CBAM(cbam_feature: tf.Tensor, ratio: int = 8):
	"""
    Referenced from https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py

    CBAM enhances feature representations by sequentially applying channel attention
    followed by spatial attention. This module adaptively refines features by
    emphasizing important channels and spatial locations.

    Based on:  
    Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.  
    https://arxiv.org/abs/1807.06521  

    Args:
        cbam_feature (tf.Tensor): 
            Tensor of shape `(batch, channels, height, width)` if 
            `data_format='channels_first'`, or `(batch, height, width, channels)` 
            if `data_format='channels_last'`.
        ratio (int, optional): 
            Reduction ratio for the channel attention sub-module. Must be a positive 
            integer ≤ number of input channels. Defaults to 8.

    Returns:
        tf.Tensor: 
            Output tensor of the same shape as input, after applying both 
            channel and spatial attention mechanisms.
	"""
	    
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature: tf.Tensor, ratio: int = 8):
    """
    Computes channel-wise attention weights and refines the input feature map.

    This function implements the channel attention module of CBAM by:
      1. Performing global average and max pooling across spatial dimensions.
      2. Passing both pooled features through a shared two-layer MLP.
      3. Adding the outputs and applying a sigmoid to generate channel attention weights.
      4. Multiplying the input by these weights to recalibrate channel responses.

    Args:
        input_feature (tf.Tensor): 
            Input 4D tensor (see `CBAM` for shape details).
        ratio (int, optional): 
            Reduction ratio for the bottleneck in the shared MLP. Defaults to 8.

    Returns:
        tf.Tensor: 
            Channel-refined tensor of the same shape as `input_feature`.
    """
	
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature: tf.Tensor):
    """
    Computes spatial attention weights and refines the input feature map.

    This function implements the spatial attention module of CBAM by:
      1. Applying channel-wise average and max pooling to create two 2D maps.
      2. Concatenating these maps along the channel axis.
      3. Applying a convolutional layer (kernel size 7x7) to produce a spatial attention map.
      4. Using a sigmoid activation to normalize weights, then multiplying with input.

    Args:
        input_feature (tf.Tensor): 
            Input 4D tensor (see `CBAM` for shape details).

    Returns:
        tf.Tensor: 
            Spatially refined tensor of the same shape as `input_feature`.

    Note:
        - Kernel size is fixed to 7 as per the original paper.
        - Contains mixed use of `_keras_shape` and `K.int_shape`; for consistency and 
          TF 2.x support, prefer `K.int_shape()`.
        - The function handles both `channels_first` and `channels_last` data formats 
          via permuting, though broadcasting often makes explicit permutation unnecessary.
    """

    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel_axis = 1
        channel = input_feature.shape[channel_axis]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = keras.ops.mean(cbam_feature, axis=-1, keepdims=True)
    max_pool = keras.ops.max(cbam_feature, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])       # (B, H, W, 2)
        
    # concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)	
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])

@keras.saving.register_keras_serializable()
def ResidualBlock(width: int, norm_type: str = "batch"): 
    """
    Creates Residual Blocks for U-Net, main Conv2D. 
    Allows better flow of gradients.  
    Mainly used in Downblock() and UpBlock()

    Args: 
        widths (List[int]): number of output Channels in U-Net

    Returns:
        x (tf.Tensor): Output tensor after forward pass
    """

    # Width specify the number of output channels
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            # Set residual to be the same as x if it matches
            residual = x
        else:
            # Set residual to the desired width
            residual = layers.Conv2D(width, kernel_size=1, kernel_initializer=initializers.HeNormal())(x)     
        if norm_type == "batch":     
            x = layers.BatchNormalization(center=False, scale=False)(x)           
        else: x = layers.GroupNormalization(groups=8, axis=-1)(x) 
             
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=initializers.HeNormal())(x) 
        x = layers.Activation(keras.activations.silu)(x)  
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=initializers.HeNormal())(x)
        x = layers.Add()([x, residual])
        return x
    return apply

@keras.saving.register_keras_serializable()
def DownBlock(width: int, block_depth: int, norm_type: str): 
    """
    Creates a Downsample block for U-Net (Functional API)
    A Block is a sequence of ResidualBlock, followed by a Downsampling (Conv2D with stride of 2)
    
    Args: 
        widths (int]):      Output channels
        block_depth (int):  Number of Residual Blocks to apply (depth of the block)

    Returns: 
        apply (function): A function that takes (tf.Tensor, skip: List[]) and returns downsampled tensor x, skips is modified/appended with tf.Tensors

    Example: 
        skips = []
        x = DownBlock(width, block_depth)([x, skips])
    
    """
    def apply(x: tf.Tensor):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width, norm_type)(x)
            skips.append(x)
            
        # Downsampling: stride of 2
        x = layers.Conv2D(width, kernel_size=3, strides=2, padding="same", activation="swish", kernel_initializer=initializers.HeNormal())(x)
        return x
        
    return apply # returns the function, since we are using functional API

@keras.saving.register_keras_serializable()
def UpBlock(width: int, block_depth: int, norm_type: str, use_attention: bool, attention_type: str = "cbam"):
    """
    Creates a Upsample block for U-Net (Functional API)
    This Block consist of a Conv2DTranspose (upsampling), followed by concatenation
    with skip, and sequence of ResidualBlock
    
    Args: 
        widths (int]):      Output channels
        block_depth (int):  Number of Residual Blocks to apply (depth of the block)
        use_attention (bool): if True, applies attention to incoming skip (similar to Attention U-Net)
        attention_type (str): if "cbam", uses CBAM for attention, else uses ECA 

    Returns: 
        apply (function): A function that takes (tf.Tensor, skip: List[]) and returns upsampled tensor x

    Example: 
        skips = [tf.Tensor_object, tf.Tensor_object, etc...]
        x = UpBlock(width, block_depth, attention_in_up_down_sample)([x, skips])
    
    """

    def apply(x):
        x, skips = x
        x = layers.Conv2DTranspose(width, kernel_size=3, strides=2, padding="same", kernel_initializer=initializers.HeNormal())(x)
        x = layers.Activation(keras.activations.silu)(x)
        for _ in range(block_depth):
            s = skips.pop()
            x = layers.Concatenate()([x, s]) 

            if use_attention: 
                x = CBAM(x) if attention_type == "cbam" else ECA(x)
                
            x = ResidualBlock(width, norm_type)(x)     
             
        return x
        
    return apply # returns the function, since we are using functional API
