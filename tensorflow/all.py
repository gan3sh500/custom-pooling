import tensorflow as tf


def conv_bottleneck_attention_module(x,
                                dilation_value=4,
                                reduction_ratio=16):
    c = int(x.shape[-1])
    d = dilation_rate
    # Building channel attention
    rc = max(c // reduction_ration, 1)
    l = tf.reduce_mean(x, axis=(1, 2))
    l = tf.layers.dense(l, rc)
    l = tf.layers.dense(l, c)
    l = tf.expand_dims(l, axis=1)
    l = tf.expand_dims(l, axis=1)
    channel_attention = tf.layers.batch_normalization(l)
    # Building spatial attention
    l = tf.layers.conv2d(l, rc, 1)
    for _ in range(2):
        l = tf.layers.conv2d(l, rc, 3,
                             dilation_rate=(d, d),
                             padding='same')
    l = tf.layers.conv2d(l, 1, 1)
    spatial_attention = tf.layers.batch_normalization(l)
    combined_attention = tf.sigmoid(channel_attention + spatial_attention)
    return x + x * combined_attention


class DetailPreservingPooling:
    def __init__(self, kernel, stride, pool_type='lite', name='dpp',
                 symmetric=False, epsilon=1e-3, padding='valid'):
        self.kernel = kernel
        self.stride = stride
        self.pool_type = pool_type
        self.name = name
        self.symmetric = symmetric
        self.epsilon = epsilon
        self.padding = padding

    def __call__(self, input_tensor):
        self.input_tensor = input_tensor
        h, w, c = list(map(int, input_tensor.shape[1:]))
        if self.pool_type == 'lite':
            downsampled = tf.layers.average_pooling2d(self.input_tensor,
                                                      self.kernel,
                                                      self.stride,
                                                      self.padding)
        else:
            linear_w = tf.get_variable('linear_w',
                        shape=[self.kernel, self.kernel, 1, 1],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer)
            linear_w = tf.exp(linear_w)
            full_linear_w = tf.concat([linear_w for _ in range(channels)],
                                      axis=2)
            Z_w = tf.reduce_sum(linear_w)
            full_linear_w = full_linear_w / Z_w
            padding = self.padding.upper()
            downsampled = tf.nn.depthwise_conv2d(self.input_tensor,
                                                 full_linear_w,
                                                 [1,
                                                  self.stride,
                                                  self.stride,
                                                  1],
                                                 padding=padding)
        downsampled = tf.cast(downsampled, tf.float32)
        reupsampled = tf.image.resize_nearest_neighbor(downsampled,
                                                       size=(h, w))
        argument = self.input_tensor - reupsampled
        log_alpha = tf.get_variable('log_alpha',
                                shape=[1, 1, 1, c],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer)
        log_lambda = tf.get_variable('log_lambda',
                                shape=[1, 1, 1, c],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer)
        if not self.symmetric:
            argument = tf.nn.relu(argument)
        reward = tf.pow(argument **2 + self.epsilon**2, tf.exp(log_lambda) / 2)
        w = reward + tf.exp(log_alpha)
        masked = tf.multiply(self.input_tensor, w)
        w_pool = tf.layers.average_pooling2d(w,
                                             self.kernel,
                                             self.stride,
                                             self.padding)
        masked_pool = tf.layers.average_pooling2d(masked,
                                                  self.kernel,
                                                  self.stride,
                                                  self.padding)
        return tf.div(masked_pool, w_pool)


class StochasticDownsampling:
    def __init__(self, kernel, stride, padding='valid',
                 name='s3pool', alpha=7):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.name = name
        self.alpha = alpha

    def __call__(self, input_tensor):
        self.input_tensor = input_tensor
        self.bsp = tf.shape(input_tensor)[0] 
        with tf.variable_scope(self.name):
            h, w = list(map(int, self.input_tensor.shape[1:3]))
            noise_w = tf.random_uniform([1, self.bsp, 1, w], maxval=self.alpha,
                                        dtype=tf.float32)
            noise_h = tf.random_uniform([1, self.bsp, h, 1], maxval=self.alpha,
                                        dtype=tf.float32)
            noise_mat = tf.transpose(tf.matmul(noise_h, noise_w), (1, 2, 3, 0))
            noise_mat = noise_mat - tf.reduce_max(noise_mat)
            noise_mat = tf.exp(noise_mat)
            masked = tf.multiply(self.input_tensor, noise_mat)
            noise_pool = tf.layers.average_pooling2d(noise_mat,
                                                     self.kernel,
                                                     self.stride,
                                                     self.padding)
            masked_pool = tf.layers.average_pooling2d(masked,
                                                      self.kernel,
                                                      self.stride,
                                                      self.padding)
            return tf.div(masked_pool, noise_pool)


def detail_preserving_pooling(input_, kernel, stride,
                              pool_type='lite',
                              name='dpp', symmetric=False,
                              stochastic=False,
                              epsilon=1e-3,
                              padding='valid',
                              alpha=7):
    if stochastic:
        layer = DetailPreservingPooling(kernel, 1, pool_type, name, symmetric,
                                        epsilon, padding)
        pool_layer = StochasticDownsampling(kernel, stride, padding, name, alpha)
        l1 = layer(input_)
        return pool_layer(l1)
    layer = DetailPreservingPooling(kernel, stride, pool_type, name, symmetric,
                                    epsilon, padding)
    return layer(input_)


def s3pool(input_, kernel, stride,
           padding='valid',
           name='s3pool',
           alpha=7):
    l1 = tf.layers.max_pooling2d(input_, kernel, 1,
                                 padding, name=name)
    layer = StochasticDownsampling(kernel, stride, padding, name, alpha)
    return layer(l1)

