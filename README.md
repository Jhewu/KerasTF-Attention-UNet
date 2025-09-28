# Attention Res-Unet implemented in Keras/Tensorflow (CBAM or ECA Attention)

A customizable **Attention Residual U-Net** with **CBAM** or **ECA** attention—designed for easy prototyping and non-square inputs. 

Instead of soft attention (as in Attention U-Net), this implementation uses CBAM, applied after skip connection concatenation, to jointly leverage channel and spatial attention with minimal overhead. For even greater efficiency, ECA (Efficient Channel Attention) is also supported.

## ✨ Features
- ✅ Plug-and-play CBAM or ECA attention in skip connections or bottleneck
- ✅ Supports non-square images (e.g., 200×600)  
- ✅ Residual blocks with BatchNorm or GroupNorm  

## 📁 Structure
```
├── unet.py           # Main model builder
└── unet_modules.py   # Blocks: CBAM, ECA, Residual, Up/Down
```

## 🚀 Quick Start

```python
from unet import build_UNET

model = build_UNET(
    image_size=(200, 600),
    num_classes=1,
    widths=[64, 128, 256, 512],
    block_depth=3,
    apply_attention_in_skip=True,
    skip_attention="cbam"
)
model.summary()
```

### References: 
- [UNet (2015)](https://arxiv.org/abs/1505.04597?spm=a2ty_o01.29997173.0.0.1987c921B61ggU&file=1505.04597)
- [Attention U-Net (2018)](https://arxiv.org/abs/1804.03999?spm=a2ty_o01.29997173.0.0.1987c921B61ggU&file=1804.03999)
- [ResUNet (2019)](https://arxiv.org/abs/1904.00592?spm=a2ty_o01.29997173.0.0.1987c921B61ggU&file=1904.00592)
- [CBAM (2018)](https://arxiv.org/abs/1807.06521?spm=a2ty_o01.29997173.0.0.1987c921B61ggU&file=1807.06521)
- [ECA-Net (2020)](https://arxiv.org/abs/1910.03151?spm=a2ty_o01.29997173.0.0.1987c921B61ggU&file=1910.03151)

>**Note**: If you use this code, please cite the respective papers. 
     
## ⚠️ Limitations 
- Requires input height/width divisible by 2^(n_blocks−1)  
- No built-in training loop (bring your own)
     

