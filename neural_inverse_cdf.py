import os
import shutil
import numpy as np
import tensorflow as tf
from scipy.stats import gamma
from matplotlib import pyplot as plt

from neural_inverse_cdf_utils import InvertibleNeuralNetworkLayer, train, result_plot


class GammaCDF(object):

    def __init__(self, theta_max=15, base_dir=os.getcwd()):

        # define theta range
        self.theta_min = 0
        self.theta_max = theta_max

        # log directory
        self.log_dir = os.path.join(base_dir, 'InverseCDF', 'Gamma', 'logdir')
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

        # checkpoint directory
        self.mdl_dir = os.path.join(base_dir, 'InverseCDF', 'Gamma', 'checkpoint', 'gamma')
        if os.path.exists(self.mdl_dir):
            shutil.rmtree(self.mdl_dir)

    def sample_training_points(self, thetas_per_batch, samples_per_theta):

        # sample thetas
        thetas = np.random.random(thetas_per_batch) * 2 * self.theta_max - self.theta_max
        thetas[thetas < 0] = np.exp(thetas[thetas < 0])

        # loop over theta samples
        z = []
        u = []
        theta = []
        for i in range(len(thetas)):

            # sample z
            z.append(np.random.gamma(shape=thetas[i], size=samples_per_theta))

            # compute target u
            u.append(gamma.cdf(x=z[-1], a=thetas[i]))

            # up-sample theta
            theta.append(thetas[i] * np.ones(samples_per_theta))

        # convert to arrays
        z = np.concatenate(z)
        u = np.concatenate(u)
        theta = np.concatenate(theta)

        return z, u, theta

    def sample_test_points(self, theta, num_points=100):

        # compute target theta quantile
        theta = theta * np.ones(num_points)

        # compute evaluation points
        # z = np.linspace(0, self.z_max, num_points)
        z = np.random.gamma(shape=theta, size=num_points)
        z = np.sort(z)

        # compute target
        u = gamma.cdf(z, theta)

        return z, u, theta

    @staticmethod
    def u_clamp(u):
        # return clamped value--THIS CLAMP MUST BE INVERTIBLE!
        return tf.nn.sigmoid(u)

    @staticmethod
    def z_clamp(z):
        # return clamped value--THIS CLAMP MUST BE INVERTIBLE!
        return tf.nn.elu(z) + 1


class NeuralInverseCDF(object):
    """
    Neural CDF Forward: F(z; theta) --> u
    Neural CDF Reverse: F_inv(u; theta) --> z
    """
    def __init__(self, target, fwd_direction='cdf', trainable=True):

        # save target object
        self.target = target

        # check and save learning direction
        assert fwd_direction == 'cdf' or fwd_direction == 'inv_cdf'
        self.fwd_direction = fwd_direction

        # configure dimensions
        self.inn_dim = 2
        self.inn_layers = 8

        # declare the Invertible Neural Network Blocks
        self.inn = []
        for i in range(self.inn_layers):
            self.inn.append(InvertibleNeuralNetworkLayer(self.inn_dim, 'inn{:d}'.format(i+1), trainable=trainable))

        # training placeholders
        self.z_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='z')
        self.u_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='u')
        self.theta_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='theta')

        # training outputs (None by default--will be overwritten by self.loss(*))
        self.u_hat = None
        self.z_hat = None

        # configure training
        self.thetas_per_batch = 100
        self.samples_per_theta = 100
        self.learning_rate = 5e-4
        self.num_epochs = 1000
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def _forward_eval(self, x, theta):

        # run the forward direction
        for i in range(self.inn_layers):
            x = self.inn[i].forward_evaluation(x, theta)

        return x

    def _inverse_eval(self, x, theta):

        # run the inverse direction
        for i in range(self.inn_layers):
            x = self.inn[self.inn_layers - 1 - i].inverse_evaluation(x, theta)

        return x

    def _load_feed_dict(self, z, u, theta):

        # expand dimensions if needed
        if len(np.shape(z)) < 2:
            z = np.expand_dims(z, axis=-1)
        if len(np.shape(u)) < 2:
            u = np.expand_dims(u, axis=-1)
        if len(np.shape(theta)) < 2:
            theta = np.expand_dims(theta, axis=-1)

        # initialize feed dictionary
        feed_dict = dict()

        # load dictionary
        feed_dict.update({self.z_ph: z, self.u_ph: u, self.theta_ph: theta})

        return feed_dict

    def loss(self):

        # build the forward model (double input to achieve even dimensions)
        if self.fwd_direction == 'cdf':
            x_fwd = self._forward_eval(tf.concat((self.z_ph, self.z_ph), axis=1), self.theta_ph)
        else:
            x_fwd = self._forward_eval(tf.concat((self.u_ph, self.u_ph), axis=1), self.theta_ph)

        # apply forward model clamps and build backward model
        if self.fwd_direction == 'cdf':
            self.u_hat = self.target.u_clamp(x_fwd)
            self.z_hat = self._inverse_eval(x_fwd, self.theta_ph)
        else:
            self.z_hat = self.target.z_clamp(x_fwd)
            self.u_hat = self._inverse_eval(x_fwd, self.theta_ph)

        # compute both losses
        u_loss = tf.reduce_mean(tf.abs(self.u_ph - self.u_hat))
        z_loss = tf.reduce_mean(tf.abs(self.z_ph - self.z_hat))

        # if cdf direction, loss is w.r.t. u
        if self.fwd_direction == 'cdf':
            loss = u_loss

        # otherwise it is w.r.t. z
        else:
            loss = z_loss

        return loss, u_loss, z_loss

    def load_feed_dict_train(self):

        # sample training points
        z, u, theta = self.target.sample_training_points(self.thetas_per_batch, self.samples_per_theta)

        return self._load_feed_dict(z, u, theta)

    def sample_operation(self, u, theta):

        # expand dimensions if needed
        if len(u.get_shape().as_list()) < 2:
            u = tf.expand_dims(u, axis=-1)
        if len(theta.get_shape().as_list()) < 2:
            theta = tf.expand_dims(theta, axis=-1)

        # cdf is the forward direction
        if self.fwd_direction == 'cdf':

            # run the inverse direction (double input to achieve even dimensions)
            z = self._inverse_eval(tf.concat((u, u), axis=1), theta)

        # inverse cdf is the forward direction
        else:

            # run the forward direction (double input to achieve even dimensions)
            z = self._forward_eval(tf.concat((u, u), axis=1), theta)

        # apply the z clamp and take the mean of the two dimensions
        return tf.reduce_mean(self.target.z_clamp(z), axis=-1)

    def restore(self, sess, var_list=None):

        # variable list not provide
        if var_list is None:

            # restore all checkpoint variables
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess, self.target.mdl_dir)

        # variable list provided
        else:

            # restore all checkpoint variables
            tf_saver = tf.train.Saver(var_list=var_list)
            tf_saver.restore(sess, self.target.mdl_dir)


if __name__ == '__main__':

    # set random seeds
    np.random.seed(123)
    tf.set_random_seed(123)

    # set training parameters
    theta_max = 15

    # begin test session
    tf.reset_default_graph()
    with tf.Session() as sess:

        # declare model
        mdl = NeuralInverseCDF(target=GammaCDF(theta_max=theta_max))

        # train the model
        train(mdl, sess, show_plots=True, save_results=True)

    # test loading
    tf.reset_default_graph()
    with tf.Session() as sess:

        # declare model
        mdl = NeuralInverseCDF(target=GammaCDF(theta_max=theta_max), trainable=False)
        _ = mdl.loss()

        # restore variables (order of operations matter)
        sess.run(tf.global_variables_initializer())
        mdl.restore(sess)

        # test it
        fig_results, _ = result_plot(mdl, sess)

    # keep plots open
    plt.ioff()
    plt.show()
