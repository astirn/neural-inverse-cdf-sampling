import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# import neural reparameterization functions
from neural_inverse_cdf import NeuralInverseCDF, GammaCDF
from neural_inverse_cdf_utils import train


# set seed
tf.set_random_seed(0)


def gamma_variance_test(x, alphas, alpha_prior, inn_layers, N_trials):

    # get useful numbers
    K = x.shape[0]

    # compute optimal posterior parameters
    alpha_star = alpha_prior + x
    print('alpha max = {:.2f}'.format(np.max(alpha_star)))

    # establish training session
    tf.reset_default_graph()
    with tf.Session() as sess:

        # declare model
        mdl = NeuralInverseCDF(target=GammaCDF(theta_max=np.max(alpha_star)), inn_layers=inn_layers)

        # train the model
        train(mdl, sess)

    # initialize gradients
    gradients = np.zeros([len(alphas), N_trials, K])

    # reset graph with new session
    tf.reset_default_graph()
    with tf.Session() as sess:

        # declare training variable
        alpha_ph = tf.placeholder(tf.float32, [1, K])

        # declare noise variable
        epsilon_ph = tf.placeholder(tf.float32, [1, K])

        # declare Gamma sampler
        sampler = NeuralInverseCDF(target=GammaCDF(), inn_layers=inn_layers, trainable=False)

        # clamp alpha to supported value
        alpha_ph = tf.minimum(alpha_ph, sampler.target.theta_max)
        alpha_ph = tf.maximum(alpha_ph, sampler.target.theta_min)

        # route each alpha dimension through a shared neural gamma sampler
        num_vars = len(tf.global_variables())
        pi = []
        for k in range(K):
            pi.append(sampler.sample_operation(epsilon_ph[:, k], alpha_ph[:, k]))

        # normalize Gamma samples so they form a Dirichlet sample
        pi = tf.stack(pi, axis=-1)
        pi = pi / tf.reduce_sum(pi, axis=-1, keepdims=True)

        # compute the expected log likelihood
        ll = tf.reduce_sum(x * tf.log(pi))

        # compute the ELBO
        elbo = ll - kl_dirichlet(alpha_ph, tf.constant(alpha_prior, dtype=tf.float32))

        # compute gradient
        grad = tf.gradients(xs=[alpha_ph], ys=elbo)

        # save the list of gamma sampler variables that will need to be loaded as constants
        sampler_vars = tf.global_variables()[num_vars:]

        # restore variables (order of operations matter)
        sess.run(tf.global_variables_initializer())
        sampler.restore(sess, sampler_vars)

        # loop over the alphas
        for i in range(len(alphas)):

            # set alpha for this test
            alpha = alpha_star * np.ones([1, K])
            alpha[0, 0] = alphas[i]

            # compute the gradient over the specified number of trials
            for j in range(N_trials):
                gradients[i, j] = sess.run(grad,
                                           feed_dict={alpha_ph: alpha, epsilon_ph: np.random.random(size=(1, K))})[0]

                # print update
                a_per = 100 * (i + 1) / len(alphas)
                n_per = 100 * (j + 1) / N_trials
                update_str = 'Alphas done: {:.2f}%, Trials done: {:.2f}%'.format(a_per, n_per)
                print('\r' + update_str, end='')

        print('')

    # return the gradients
    return gradients


if __name__ == '__main__':

    K = 100
    N = 100
    p_true = np.random.dirichlet(np.ones(K))
    x = np.random.multinomial(n=N, pvals=p_true)
    alphas = np.linspace(1.01, 3.0, 100)
    grads = gamma_variance_test(x=x, alphas=alphas, alpha_prior=np.ones(K), inn_layers=8, N_trials=100)

    # take the variance across samples
    grad_var = np.var(grads, axis=1)

    plt.figure()
    plt.plot(alphas, grad_var[:, 0])
    plt.show()





