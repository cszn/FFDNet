# FFDNet

# [FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising](https://arxiv.org/abs/1710.04026)


# Abstract
Due to the fast inference and good performance, discriminative learning methods have been widely studied in image denoising. However, these methods mostly learn a specific model for each noise level, and require multiple models for denoising images with different noise levels. They also lack flexibility to deal with spatially variant noise, limiting their applications in practical denoising. 
To address these issues, we present a fast and flexible denoising convolutional neural network, namely FFDNet, with a tunable noise level map as the input. 
The proposed FFDNet works on downsampled subimages,achieving a good trade-off between inference speed and
denoising performance. In contrast to the existing discriminative denoisers, FFDNet enjoys several desirable properties, including

- the ability to handle a wide range of noise levels (i.e., [0, 75]) effectively with a single network, 
- the ability to remove spatially variant noise by specifying a non-uniform noise level map, and 
- faster speed than benchmark BM3D even on CPU without sacrificing denoising performance. 

Extensive experiments on synthetic and real noisy images are conducted to evaluate FFDNet in comparison with state-of-the-art denoisers. The results show that FFDNet is effective and efficient, making it highly attractive for practical denoising applications.

# Test FFDNet models
- `Demo_AWGN_Gray.m` is the testing demo of FFDNet for denoising grayscale images corrupted by AWGN.
- `Demo_AWGN_Color.m` is the testing demo of FFDNet for denoising color images corrupted by AWGN.

- `Demo_AWGN_Gray_Clip.m` is the testing demo of FFDNet for denoising grayscale images corrupted by AWGN with clipping setting.
- `Demo_AWGN_Color_Clip.m` is the testing demo of FFDNet for denoising color images corrupted by AWGN with clipping setting.

- `Demo_REAL_Gray.m` is the testing demo of FFDNet for denoising real noisy (grayscale) images.
- `Demo_REAL_Color.m` is the testing demo of FFDNet for denoising real noisy (color) images.


# Image Denoising for AWGN

The left is the noisy image corrupted by AWGN with noise level 75. The right is the denoised image by FFDNet.

<img src="utilities/figs/05_75_75.png" width="321px"/> <img src="utilities/figs/05_75_75_PSNR_2479.png" width="321px"/>

<img src="utilities/figs/102061_75_75.png" width="321px"/> <img src="utilities/figs/102061_75_75_PSNR_2698.png" width="321px"/>


# Real Image Denoising

The left is the real noisy image. The right is the denoised image by FFDNet.

<img src="utilities/figs/David_Hilbert.jpg" width="321px"/> <img src="utilities/figs/David_Hilbert_15.png" width="321px"/>

![example](https://github.com/cszn/FFDNet/blob/master/utilities/figs/Frog.gif)


# Requirements and Dependencies
To run the code, you should install Matconvnet first. 
Alternatively, you can use function [vl_ffdnet_matlab](utilities/vl_ffdnet_matlab.m) to perform denoising without Matconvnet.
- MATLAB R2015b
- [Cuda](https://developer.nvidia.com/cuda-toolkit-archive)-8.0 & [cuDNN](https://developer.nvidia.com/cudnn) v-5.1
- [MatConvNet](http://www.vlfeat.org/matconvnet/)
