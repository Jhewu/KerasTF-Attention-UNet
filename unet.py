# library
import tensorflow as tf
import keras
from keras import layers
from typing import List, Tuple

# local
from unet_modules import ResidualBlock, DownBlock, UpBlock, CBAM, ECA

@keras.saving.register_keras_serializable()
def build_UNET(image_size: Tuple[int, int],
                num_classes: int,
                widths: List[int], 
                block_depth: int, 
                norm_type: str = "batch",
                apply_attention_in_skip: bool = True,
                apply_attention_in_bottleneck: bool = False,
                skip_attention: str = "cbam",
                bottleneck_attention: str = None):
    """
    Builds Attention Res-UNet. Attention Options are CBAM or ECA Blocks, differing
    from traditional Attention U-Net, which uses soft attention. 

    Attention U-Net original paper: https://arxiv.org/pdf/1804.03999
    ResUNet original paper: https://arxiv.org/pdf/1904.00592
    Credit: https://keras.io/examples/generative/ddim/ (This code is strongly influenced by this Keras tutorial)

    Args: 
        image_size (Tuple[int, int]): Image size. Can accept non-square sizes, but must ensure size (height, width)
                                      when halved by len(widths)-1 times, it's a whole number 
                                      
                                      For example:  
                                      if image_size = (200, 600) and widths = [64, 128, 256, 512] 
                                      at each downsampling, image size is (200, 600) -> (100, 300) -> (50, 150) -> (25, 75)

        num_classes (int): Number of classes. If 1, final activation is 'sigmoid', else is 'softmax'
        
        widths (List[int]): Number of output channels in U-Net, per "block." Length of widths also contribute
                            to how "deep" the model is. [128, 256, 512] only 2 downsamples, and [64, 128, 256, 512] is 4 downsamples
                            
        block_depth (int): Number of Residual Blocks to apply (depth of the residual blocks at each "block/layer")
        norm_type (str): Choices are "batch" (BatchNorm) else GroupNorm. Use batch if batch sizes are >= 16, else use group, defaults to batch
        apply_attention_in_skip (bool): if True, applies attention (either CBAM or ECA) on the incoming skip connections on the decoder
        apply_attention_in_bottleneck (bool): if True, applies attention in bottleneck (either CBAM Or ECA)
        skip_attention (str): Choices are 'cbam', else is 'eca' in the skip (apply_attention_in_skip must be True) 
        bottleneck_attention (str): Choices are 'cbam' else is 'eca' in the bottleneck (apply_attention_in_bottleneck must be True)

    Returns: 
        attention_residual_unet (keras.Model): Keras Model for Training or Inference. Use your own custom training loop or Keras

    Examples: 
        Check if __name__ == "__main__"

    """
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3)) 
    x = layers.Conv2D(widths[0], kernel_size=1)(inputs)

    skips = [] # <- stores the skip connections

    ### Encoder
    for width in widths[:-1]:
        """
        # Each downblock halves spatial resolution and increases the number
        # of filters, determined by widths
        """
        x = DownBlock(width, block_depth, norm_type)([x, skips])

    ### Bottleneck
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], norm_type)(x)
        
        # Add attention layer if specified
        if apply_attention_in_bottleneck: 
             x = CBAM(x) if bottleneck_attention == "cbam" else ECA(x)

    ### Decoder
    for width in reversed(widths[:-1]):
        """
        # Each upblock doubles the features and reduces the number 
        # of filters, determined by widths
        """
        x = UpBlock(width, block_depth, norm_type, use_attention = apply_attention_in_skip, attention_type = skip_attention)([x, skips])

    # if one class, sigmoid, else softmax
    if num_classes == 1: output_act = "sigmoid"
    else: output_act = "softmax"

    outputs = layers.Conv2D(num_classes, kernel_size=1, activation=output_act, kernel_initializer="zeros")(x)

    return keras.Model(inputs, outputs, name="attention_residual_unet")

if __name__ == "__main__":
    image_size = (200, 600)
    widths = [64, 128, 256, 512]
    block_depth = 3
    num_classes = 1

    model = build_UNET(
        image_size=image_size,
        num_classes=num_classes, 
        widths=widths,
        block_depth=block_depth, 
        apply_attention_in_skip = True, 
        skip_attention = "cbam")

    model.summary()
