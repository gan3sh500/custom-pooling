## Custom Pooling Layers
This repo contains implementations for the following papers in tensorflow or pytorch. 
### Tensorflow
* Convolutional Bottleneck Attention Module ([arxiv](https://arxiv.org/pdf/1807.06521.pdf))(Not pooling I know)
* Stochastic Spatial Sampling ([arxiv](https://arxiv.org/pdf/1611.05138.pdf))
* Detail Preserving Pooling ([arxiv](https://arxiv.org/pdf/1804.04076.pdf))
### Pytorch
* Lossless Pooling ([arxiv](https://arxiv.org/pdf/1812.06023.pdf))

## Notice
* I just picked these layers because they seem simple but the papers claim decent improvments using them. 
* The S3Pool and DPP implementations are hacky approximations which use softmax instead of selecting indices. If you can think of a more efficient but also clean way to do it in Tensorflow please submit a PR. 
