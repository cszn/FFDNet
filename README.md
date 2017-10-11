# FFDNet

## FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising


## Abstract
Due to the fast inference and good performance, discriminative learning methods have been widely studied in image denoising. However, these methods mostly learn a specific model for each noise level, and require multiple models for denoising images with different noise levels. They also lack flexibility to deal with spatially variant noise, limiting their applications in practical denoising. To address these issues, we present a fast and flexible denoising convolutional neural network, namely FFDNet, with a tunable noise level map as the input. The proposed FFDNet works on downsampled sub-images to speed up the inference, and adopts orthogonal regularization to enhance the generalization ability. In contrast to the existing discriminative denoisers, FFDNet enjoys several desirable properties, including

- the ability to handle a wide range of noise levels (i.e., [0, 75]) effectively with a single network, 
- the ability to remove spatially variant noise by specifying a non-uniform noise level map, and 
- faster speed than benchmark BM3D even on CPU without sacrificing denoising performance. 

Extensive experiments on synthetic and real noisy images are conducted to evaluate FFDNet in comparison with state-of-the-art denoisers. The results show that FFDNet is effective and efficient, making it highly attractive for practical denoising applications.

### Test FFDNet models
- `Demo_AWGN_Gray.m` is the testing demo of FFDNet for denoising gray-level images corrupted by AWGN.
- `Demo_AWGN_Color.m` is the testing demo of FFDNet for denoising color images corrupted by AWGN.
- `Demo_REAL_Gray.m` is the testing demo of FFDNet for denoising real noisy (gray-level) images.
- `Demo_REAL_Color.m` is the testing demo of FFDNet for denoising real noisy (color) images.

### Denoising example
![example](https://github.com/cszn/FFDNet/blob/master/utilities/figs/Frog.gif)

### Requirements and Dependencies
- MATLAB R2015b
- Cuda v-8.0 & [cuDNN](https://developer.nvidia.com/cudnn) v-5.1
- [MatConvNet](http://www.vlfeat.org/matconvnet/)
