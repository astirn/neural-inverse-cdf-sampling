import numpy as np
import tensorflow as tf

# set trainable variable initialization routines
KERNEL_INIT = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32, uniform=False)
WEIGHT_INIT = tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False)
BIAS_INIT = tf.constant_initializer(0.0, dtype=tf.float32)

# set minimum value for all variables that are > 0
MIN_POSITIVE_VAL = 1e-3

# log epsilon
LOG_EPSILON = 1e-6


def convolution_layer(x, dropout_ph, kernel_dim, n_out_chan, name):

    # run convolution
    x = tf.layers.conv2d(inputs=x,
                         filters=n_out_chan,
                         kernel_size=kernel_dim,
                         strides=[1, 1],
                         padding='same',
                         activation=tf.nn.elu,
                         use_bias=True,
                         kernel_initializer=KERNEL_INIT,
                         bias_initializer=BIAS_INIT,
                         name=name)

    # run max pooling
    x = tf.layers.max_pooling2d(inputs=x,
                                pool_size=3,
                                strides=2,
                                padding='same',
                                name=name)

    # apply drop out
    # x = tf.layers.batch_normalization(x, training=tf.greater(dropout_ph, 0))

    return x


def fully_connected_layer(x, dropout_ph, dim_out, name):

    # run fully connected layers
    x = tf.layers.dense(inputs=x,
                        units=dim_out,
                        activation=tf.nn.elu,
                        use_bias=True,
                        kernel_initializer=WEIGHT_INIT,
                        bias_initializer=BIAS_INIT,
                        name=name)

    # apply drop out
    # x = tf.layers.batch_normalization(x, training=tf.greater(dropout_ph, 0))

    return x


def deconvolution_layer(x, dropout_ph, kernel_dim, n_out_chan, name):

    # run convolution transpose layers
    x = tf.layers.Conv2DTranspose(filters=n_out_chan,
                                  kernel_size=kernel_dim,
                                  strides=[1, 1],
                                  padding='SAME',
                                  activation=tf.nn.elu,
                                  use_bias=True,
                                  kernel_initializer=KERNEL_INIT,
                                  bias_initializer=BIAS_INIT,
                                  name=name)(x)

    # up-sample data
    size = [int(x.shape[1] * 2), int(x.shape[2] * 2)]
    x = tf.image.resize_bilinear(x, size=size, align_corners=None, name=None)

    # apply drop out
    # x = tf.layers.batch_normalization(x, training=tf.greater(dropout_ph, 0))

    return x


def base_encoder_network(x, dropout_ph, enc_arch):

    # loop over the convolution layers
    for i in range(len(enc_arch['conv'])):

        # run convolution layer
        x = convolution_layer(x,
                              dropout_ph=dropout_ph,
                              kernel_dim=enc_arch['conv'][i]['k_size'],
                              n_out_chan=enc_arch['conv'][i]['out_chan'],
                              name='conv_layer{:d}'.format(i + 1))

    # flatten features to vector
    x = tf.contrib.layers.flatten(x)

    # loop over fully connected layers
    for i in range(len(enc_arch['full'])):

        # run fully connected layer
        x = fully_connected_layer(x,
                                  dropout_ph=dropout_ph,
                                  dim_out=enc_arch['full'][i],
                                  name='full_layer{:d}'.format(i + 1))

    return x


def base_decoder_network(z, dropout_ph, dim_x, dec_arch, final_activation, name):

    with tf.variable_scope(name) as scope:

        # initialize recursion variable to z
        x = z

        # loop over fully connected layers
        for i in range(len(dec_arch['full'])):

            # run fully connected layer
            x = fully_connected_layer(x,
                                      dropout_ph=dropout_ph,
                                      dim_out=dec_arch['full'][i],
                                      name='full_layer{:d}'.format(i + 1))

        # determine final fully connected layer's output dimensions
        n_conv_layers = len(dec_arch['conv'])
        conv_start_dim1 = int(dim_x[0] / 2 ** n_conv_layers)
        conv_start_dim2 = int(dim_x[1] / 2 ** n_conv_layers)
        total_dims = int(conv_start_dim1 * conv_start_dim2 * dec_arch['conv_start_chans'])

        # run final fully connected layer
        x = fully_connected_layer(x,
                                  dropout_ph=dropout_ph,
                                  dim_out=total_dims,
                                  name='full_layer_final')

        # reshape for convolution layers
        x = tf.reshape(x, shape=(-1, conv_start_dim1, conv_start_dim2, int(dec_arch['conv_start_chans'])))

        # loop over the de-convolution layers
        for i in range(len(dec_arch['conv'])):

            # run de-convolution layer
            x = deconvolution_layer(x,
                                    dropout_ph=dropout_ph,
                                    kernel_dim=dec_arch['conv'][i]['k_size'],
                                    n_out_chan=dec_arch['conv'][i]['out_chan'],
                                    name='deconv_layer{:d}'.format(i + 1))

        # run final convolution layer to ensure requisite number of channels
        x = tf.layers.conv2d(inputs=x,
                             filters=dim_x[-1],
                             kernel_size=[1, 1],
                             strides=[1, 1],
                             padding='same',
                             activation=final_activation,
                             use_bias=True,
                             kernel_initializer=KERNEL_INIT,
                             bias_initializer=BIAS_INIT,
                             name='deconv_layer_final')

        # ensure dimensions match
        dim_out = x.get_shape().as_list()[1:]
        assert dim_x == dim_out

    return x


def dirichlet_encoder_network(x, dropout_ph, K, enc_arch):

    with tf.variable_scope('AlphaEncoder') as scope:

        # construct base network
        x = base_encoder_network(x, dropout_ph, enc_arch)

        # compute alpha
        alpha = tf.layers.dense(inputs=x,
                                units=K,
                                activation=tf.nn.elu,
                                use_bias=True,
                                kernel_initializer=WEIGHT_INIT,
                                bias_initializer=BIAS_INIT,
                                name='alpha')

        # adjust for elu on (-1, inf) and add minimum value
        alpha = alpha + tf.constant(1.0 + MIN_POSITIVE_VAL, dtype=tf.float32)

        return alpha


def gaussian_encoder_network(x, dropout_ph, K, dim_z, enc_arch, name):

    def enc_network(x, dropout_ph, dim_out, enc_arch, name):

        with tf.variable_scope(name) as scope:

            # construct base network
            x = base_encoder_network(x, dropout_ph, enc_arch)

            # compute mean
            mu = tf.layers.dense(inputs=x,
                                 units=dim_out,
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=WEIGHT_INIT,
                                 bias_initializer=BIAS_INIT,
                                 name=(name + '_mu'))

            # compute covariance
            sigma = tf.layers.dense(inputs=x,
                                    units=dim_out,
                                    activation=tf.nn.elu,
                                    use_bias=True,
                                    kernel_initializer=WEIGHT_INIT,
                                    bias_initializer=BIAS_INIT,
                                    name=(name + '_sigma'))

            # adjust for elu on (-1, inf) and add minimum value
            sigma = sigma + tf.constant(1.0 + MIN_POSITIVE_VAL, dtype=tf.float32)

            return mu, sigma

    # do all z_k parameters share a base network?
    if enc_arch['shared']:

        # generate shared network
        mu, sigma = enc_network(x, dropout_ph, (K * dim_z), enc_arch, name)

        # reshape output to [batch, latent dimensions, K clusters]
        mu = tf.reshape(mu, [-1, dim_z, K])
        sigma = tf.reshape(sigma, [-1, dim_z, K])

    # independent networks
    else:

        # loop over the independent networks
        mu = []
        sigma = []
        for i in range(K):

            # generate a network
            m, s = enc_network(x, dropout_ph, dim_z, enc_arch, (name + '_K{:d}'.format(i + 1)))

            # expand the dimensions to [batch, latent dimensions, 1]
            mu.append(tf.expand_dims(m, axis=-1))
            sigma.append(tf.expand_dims(s, axis=-1))

        # concatenate parameters to [batch, latent dimensions, K clusters]
        mu = tf.concat(mu, axis=-1)
        sigma = tf.concat(sigma, axis=-1)

    return mu, sigma


def bernoulli_decoder_network(z, dropout_ph, dim_x, dec_arch, k=1):

    with tf.variable_scope('BernoulliDecoder_k{:d}.'.format(k)) as scope:

        # construct base network to generate mean
        x_hat_mu = base_decoder_network(z, dropout_ph, dim_x, dec_arch, tf.nn.sigmoid, 'mu')

        # no variance for this model
        x_hat_var = None

        return x_hat_mu, x_hat_var


def gaussian_decoder_network(z, dropout_ph, dim_x, dec_arch, variance, k=1):

    with tf.variable_scope('GaussianDecoder_k{:d}.'.format(k)) as scope:

        # construct base network to generate mean
        x_hat_mu = base_decoder_network(z, dropout_ph, dim_x, dec_arch, None, 'mu')

        # construct base network to generate diagonal covariance
        x_hat_var = base_decoder_network(z, dropout_ph, dim_x, dec_arch, tf.nn.elu, 'sigma')

        # diagonal variance
        if variance == 'diag':

            # adjust for elu on (-1, inf) and add minimum value
            x_hat_var = x_hat_var + tf.constant(1.0 + MIN_POSITIVE_VAL, dtype=tf.float32)

        # scalar variance:
        elif variance == 'scalar':

            # run a final fully connected layer that generates a scalar
            x_hat_var = tf.layers.flatten(x_hat_var)
            x_hat_var = fully_connected_layer(x_hat_var, dropout_ph, 1, 'sigma_scalar')

            # adjust for elu on (-1, inf) and add minimum value
            x_hat_var = x_hat_var + tf.constant(1.0 + MIN_POSITIVE_VAL, dtype=tf.float32)

        # not supported
        else:
            assert False

        return x_hat_mu, x_hat_var


def bernoulli_log_likelihood(x, x_hat_mu):

    with tf.variable_scope('BernoulliLogLikelihood') as scope:

        # flatten input and reconstruction
        x = tf.layers.flatten(x)
        x_hat_mu = tf.layers.flatten(x_hat_mu)

        # compute reconstruction loss: E[ln p(x|z)]
        ll = tf.reduce_sum(x * tf.log(x_hat_mu + 1e-6) + (1 - x) * tf.log(1 - x_hat_mu + 1e-6), axis=1)

    return ll


def gaussian_log_likelihood(x, x_hat_mu, x_hat_var, variance):

    with tf.variable_scope('GaussianLogLikelihood') as scope:

        # flatten input and reconstruction
        x = tf.layers.flatten(x)
        x_hat_mu = tf.layers.flatten(x_hat_mu)

        # compute log determinant portion of reconstruction loss: -E[ln p(x|z)]
        if variance == 'diag':
            x_hat_var = tf.layers.flatten(x_hat_var)
            log_det = -0.5 * tf.log(2 * np.pi * x_hat_var)
        elif variance == 'scalar':
            log_det = -0.5 * tf.log(2 * np.pi * x_hat_var) * tf.constant(x.get_shape().as_list()[1], dtype=tf.float32)
        else:
            assert False

        # sum over the dimensions
        log_det = tf.reduce_sum(log_det, axis=1)

        # compute log exponential portion of reconstruction loss: -E[ln p(x|z)]
        log_exp = -0.5 * tf.reduce_sum(tf.squared_difference(x, x_hat_mu) / x_hat_var, axis=1)

        # combine loss terms
        ll = log_exp + log_det

    return ll


def reconstruction_mse(x, x_hat_mu):

    with tf.variable_scope('MSE') as scope:

        # flatten input and reconstruction
        x = tf.layers.flatten(x)
        x_hat_mu = tf.layers.flatten(x_hat_mu)

        # compute mse
        mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(x, x_hat_mu), axis=1))

    return mse


def kl_discrete(q, p):

    # compute kl
    kl = tf.reduce_sum(q * (tf.log(q) - tf.log(p)), axis=1)

    # take mean across batch
    kl = tf.reduce_mean(kl)

    return kl


def kl_dirichlet(alpha, beta):

    # compute convenient terms
    alpha_0 = tf.reduce_sum(alpha, axis=-1, keepdims=True)
    beta_0 = tf.reduce_sum(beta, axis=-1, keepdims=True)

    # compute KL(q || p)
    kl = \
        tf.lgamma(alpha_0) - \
        tf.reduce_sum(tf.lgamma(alpha), axis=-1) - \
        tf.lgamma(beta_0) + \
        tf.reduce_sum(tf.lgamma(beta)) + \
        tf.reduce_sum((alpha - beta) * (tf.digamma(alpha) - tf.digamma(alpha_0)), axis=-1)

    # take mean across batch
    kl = tf.reduce_mean(kl)

    return kl


def kl_gaussian(q_mu, q_sigma, p_sigma2):

    # square mu to arrive at mean square
    q_mu2 = tf.square(q_mu)

    # square q sigma to arrive at variance
    q_sigma2 = tf.square(q_sigma)

    # compute latent loss per sample: KL(q || N(0,I))
    kl = tf.reduce_sum(0.5 * tf.log(p_sigma2) - tf.log(q_sigma) + (q_sigma2 + q_mu2) / (2 * p_sigma2) - 0.5, axis=1)

    # take mean across batch
    kl = tf.reduce_mean(kl)

    return kl


def softmax(x):

    # softmax with max shift to avoid under/over flow
    s = np.exp(x - np.max(x, axis=1, keepdims=True))
    s = s / np.sum(s, axis=1, keepdims=True)

    return s