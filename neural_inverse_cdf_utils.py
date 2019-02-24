import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def inn_network_parameters(theta, out_dim, n_layers, name, trainable=True):
    """
    Invertible Neural Network Parameter Functions
    :param theta: distribution parameters
    :param out_dim: invertible neural network parameter shape
    :param n_layers: number of layers
    :param name: a name
    :param trainable: whether variables are trainable
    :return: a non-linear mapping from theta-->out_dim
    """
    # assign feed-forward networks' input
    x = theta

    # loop over the network layers
    for i in range(n_layers):
        # layer name
        layer_name = name + '_L{:d}'.format(i + 1)

        # run fully connected layers
        x = tf.layers.dense(inputs=x,
                            units=30,
                            activation=tf.nn.elu,
                            use_bias=True,
                            kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1, dtype=tf.float32),
                            bias_initializer=tf.constant_initializer(0.0),
                            trainable=trainable,
                            reuse=tf.AUTO_REUSE,
                            name=layer_name)

    # output layer name
    layer_name = name + '_Out'

    # output layer
    x = tf.layers.dense(inputs=x,
                        units=np.product(out_dim),
                        activation=None,
                        use_bias=True,
                        kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1, dtype=tf.float32),
                        bias_initializer=tf.constant_initializer(0.0),
                        trainable=trainable,
                        reuse=tf.AUTO_REUSE,
                        name=layer_name)

    # reshape to target dimensions
    shape = tf.concat((tf.expand_dims(tf.shape(x)[0], axis=0), tf.constant(out_dim, dtype=tf.int32)), axis=0)
    x = tf.reshape(x, shape)

    return x


class InvertibleNeuralNetworkLayer(object):
    """
    Invertible Neural Network Layer
    """
    def __init__(self, dim, name, trainable=True):

        # save name
        self.name = name

        # save whether variable is trainable
        self.trainable = trainable

        # must have even dimension
        assert np.mod(dim, 2) == 0

        # save dimension
        self.dim = dim
        self.dim_half = int(dim / 2)

        # set non-linear base power
        self.b = tf.constant(1.1, dtype=tf.float32)

        # set activation type
        self.activation = tf.nn.elu

        # parameter architecture
        self.n_layers = 1

    @staticmethod
    def _batch_matmul(w, x):
        return tf.einsum('bij,bj->bi', w, x)

    def _s1(self, x, theta):
        """
        :param x:
        :return: s1 non-linearity
        """
        with tf.variable_scope(self.name + '_s1', reuse=True) as scope:

            # weight parameter
            ws1 = inn_network_parameters(theta=theta,
                                         out_dim=[self.dim_half, self.dim_half],
                                         n_layers=self.n_layers,
                                         name=scope.name + '_W',
                                         trainable=self.trainable)

            # bias parameter
            bs1 = inn_network_parameters(theta=theta,
                                         out_dim=[self.dim_half],
                                         n_layers=self.n_layers,
                                         name=scope.name + '_b',
                                         trainable=self.trainable)

            # non-linearity
            s1 = self.activation(self._batch_matmul(ws1, x) + bs1)

        return s1

    def _s2(self, x, theta):
        """
        :param x:
        :return: s2 non-linearity
        """
        with tf.variable_scope(self.name + '_s2', reuse=True) as scope:

            # weight parameter
            ws2 = inn_network_parameters(theta=theta,
                                         out_dim=[self.dim_half, self.dim_half],
                                         n_layers=self.n_layers,
                                         name=scope.name + '_W',
                                         trainable=self.trainable)

            # bias parameter
            bs2 = inn_network_parameters(theta=theta,
                                         out_dim=[self.dim_half],
                                         n_layers=self.n_layers,
                                         name=scope.name + '_b',
                                         trainable=self.trainable)

            # non-linearity
            s2 = self.activation(self._batch_matmul(ws2, x) + bs2)

        return s2

    def _t1(self, x, theta):
        """
        :param x:
        :return: t1 non-linearity
        """
        with tf.variable_scope(self.name + '_t1', reuse=True) as scope:

            # weight parameter
            wt1 = inn_network_parameters(theta=theta,
                                         out_dim=[self.dim_half, self.dim_half],
                                         n_layers=self.n_layers,
                                         name=scope.name + '_W',
                                         trainable=self.trainable)

            # bias parameter
            bt1 = inn_network_parameters(theta=theta,
                                         out_dim=[self.dim_half],
                                         n_layers=self.n_layers,
                                         name=scope.name + '_b',
                                         trainable=self.trainable)

            # non-linearity
            t1 = self.activation(self._batch_matmul(wt1, x) + bt1)

        return t1

    def _t2(self, x, theta):
        """
        :param x:
        :return: t2 non-linearity
        """
        with tf.variable_scope(self.name + '_t2', reuse=True) as scope:

            # weight parameter
            wt2 = inn_network_parameters(theta=theta,
                                         out_dim=[self.dim_half, self.dim_half],
                                         n_layers=self.n_layers,
                                         name=scope.name + '_W',
                                         trainable=self.trainable)

            # bias parameter
            bt2 = inn_network_parameters(theta=theta,
                                         out_dim=[self.dim_half],
                                         n_layers=self.n_layers,
                                         name=scope.name + '_b',
                                         trainable=self.trainable)

            # non-linearity
            t2 = self.activation(self._batch_matmul(wt2, x) + bt2)

        return t2

    def forward_evaluation(self, u, theta):
        """
        :param u: input of shape [batch, input/output dimension]
        :return: v output
        """

        # split input
        u1 = u[:, :self.dim_half]
        u2 = u[:, self.dim_half:]

        # compute v1 and v2
        v1 = tf.multiply(u1, tf.pow(self.b, self._s2(u2, theta))) + self._t2(u2, theta)
        v2 = tf.multiply(u2, tf.pow(self.b, self._s1(v1, theta))) + self._t1(v1, theta)

        # recombine to create output
        v = tf.concat((v1, v2), axis=1)

        return v

    def inverse_evaluation(self, v, theta):
        """
        :param v: input of shape [batch, input/output dimension]
        :return: u output
        """

        # split input
        v1 = v[:, :self.dim_half]
        v2 = v[:, self.dim_half:]

        # compute u2 and u1
        u2 = tf.multiply(v2 - self._t1(v1, theta), tf.pow(self.b, -self._s1(v1, theta)))
        u1 = tf.multiply(v1 - self._t2(u2, theta), tf.pow(self.b, -self._s2(u2, theta)))

        # recombine to create output
        u = tf.concat((u1, u2), axis=1)

        return u


def train(mdl, sess, plot_dir=None):

    # get losses
    loss_op, u_loss_op, z_loss_op = mdl.loss()

    # configure training operation
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tf.contrib.layers.optimize_loss(loss=loss_op,
                                               global_step=global_step,
                                               learning_rate=mdl.learning_rate,
                                               optimizer=mdl.optimizer,
                                               summaries=['loss', 'gradients'])

    # tensorFlow saver
    tf_saver = tf.train.Saver()
    tf_summary = tf.summary.FileWriter(logdir=mdl.target.log_dir)
    tf_summary.add_graph(tf.get_default_graph())

    # run initialization
    sess.run(tf.global_variables_initializer())

    # figure init
    fig = plt.figure(figsize=(16, 9))
    if plot_dir is not None:
        plt.ion()

    # loop over the epochs
    t = 0
    loss = np.zeros(mdl.num_epochs)
    u_loss = np.zeros(mdl.num_epochs)
    z_loss = np.zeros(mdl.num_epochs)
    while t < mdl.num_epochs:

        # load feed dictionary
        feed_dict = mdl.load_feed_dict_train()

        # run training and loss
        _, loss[t], u_loss[t], z_loss[t] = sess.run([train_op, loss_op, u_loss_op, z_loss_op], feed_dict=feed_dict)

        # if we hit a NaN, restart training
        if np.isnan(loss[t]):
            print('\nWhelp! NaN loss... Restarting training!')
            sess.run(tf.global_variables_initializer())
            t = 0
            continue

        # print update
        print('\r' +
              'Epoch {:d}/{:d} | '.format((t + 1), mdl.num_epochs) +
              'u Loss = {:.5f} | '.format(u_loss[t]) +
              'z Loss = {:.5f}'.format(z_loss[t]), end='')

        # plot update
        if np.mod(t + 1, 1) == 0:
            fig.clf()
            mdl.train_plot(fig, loss[:t + 1], sess)

        # draw the plot if not saving
        if plot_dir is None:
            plt.pause(0.01)

        # otherwise, save the plot
        else:
            fig.savefig(os.path.join(plot_dir, 'learning_curve'))

        # increment t
        t += 1

    # close the plot if saving
    if plot_dir is not None:
        plt.close(fig)

    # save the model if it has a specified save directory
    if mdl.target.mdl_dir is not None:
        tf_saver.save(sess, mdl.target.mdl_dir)


if __name__ == '__main__':

    # begin test session
    tf.reset_default_graph()
    with tf.Session() as sess:

        # set dimensions
        batch_size = 100
        dim_theta = 2
        dim_inn = 30

        # declare placeholders
        theta_ph = tf.placeholder(dtype=tf.float32, shape=[None, dim_theta])
        u_ph = tf.placeholder(dtype=tf.float32, shape=[None, dim_inn])

        # declare invertible neural network
        inn = InvertibleNeuralNetworkLayer(dim_inn, 'hi')
        v = inn.forward_evaluation(u_ph, theta_ph)
        num_vars = len(tf.global_variables())
        u_recovered = inn.inverse_evaluation(v, theta_ph)
        num_vars_reuse = len(tf.global_variables())
        assert num_vars == num_vars_reuse

        # run initialization
        sess.run(tf.global_variables_initializer())

        # load feed dict
        u = np.random.random(size=[batch_size, dim_inn])
        feed_dict = {theta_ph: np.ones([batch_size, dim_theta]),
                     u_ph: u}

        # run the model
        u_r, v = sess.run([u_recovered, v], feed_dict=feed_dict)

        # check maximum absolute difference is acceptable
        max_diff = np.max(np.abs(u - u_r))
        assert max_diff < 1e-6

        # test einstein summation batch matrix multiplication
        A = np.random.normal(size=[batch_size, dim_theta, dim_theta])
        x = np.random.normal(size=[batch_size, dim_theta])
        y = np.zeros([batch_size, dim_theta])
        for i in range(y.shape[0]):
            y[i] = A[i] @ x[i]
        y_tf = sess.run(inn._batch_matmul(tf.constant(A), tf.constant(x)), feed_dict={})

        # check maximum absolute difference is acceptable
        max_diff = np.max(np.abs(y - y_tf))
        assert max_diff < 1e-6

        print('tests complete!')