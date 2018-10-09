import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from config.config import Config
import numpy as np
from utils.preprocessor import get_batch_data_iterator
import math


def bivar_gauss_prob(y, means, variance, co_rel):
    delta = 1e-7
    y_x, y_y = tf.split(y, 2, axis=-1)
    mux, muy = tf.split(means, 2, axis=-1)
    co_rel = tf.expand_dims(co_rel, axis=-1)
    varx, vary = tf.split(variance, 2, axis=-1)
    z = tf.square(tf.divide((y_x - mux), varx)) + tf.square(tf.divide((y_y - muy), vary))
    z -= tf.divide((2 * co_rel * (y_x - mux) * (y_y - muy)), tf.multiply(varx, vary))
    prob = tf.exp(-z / (2 * (1 - tf.square(co_rel) + delta)))
    prob /= 2 * np.pi * varx * vary * tf.sqrt(1 + delta - tf.square(co_rel))
    return tf.clip_by_value(tf.squeeze(prob, axis=-1), 0, 1.0)


class HandWritingGenerator:
    def __init__(self, random_seed=1):
        self.config = Config()
        self.outputs = []
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        self.initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.075)
        self.loss = 0.0
        self.add_model()
        self.data_mean = 0.0
        self.data_var = 0.0
        self.n_batches = 0
        self.saver = tf.train.Saver()
        self.random_seed = random_seed

    def add_model(self):
        self.add_placeholders()
        self.assemble_model()
        self.loss /= self.config.batch_size * self.config.seq_length
        self.add_train_op()

    def add_placeholders(self):
        self.x_t = tf.placeholder(dtype=tf.float32, shape=(None, None, self.config.num_input), name='x_t')
        self.y_t = tf.placeholder(dtype=tf.float32, shape=(None, None, self.config.num_input), name='x_t')
        self.rand_sample = tf.placeholder(dtype=tf.float32, shape=(None, None, 2), name='rand_sample')
        self.dropout_keep = tf.placeholder(dtype=tf.float32, shape=(), name='dropout_keep')
        self.is_test = tf.placeholder(dtype=tf.bool, shape=(), name='is_test')

    def assemble_model(self):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            cell = self.create_lstm_multicell(self.config.lstm_layers, self.config.num_hidden)
            states = cell.zero_state(tf.shape(self.x_t)[0], tf.float32)
            xs = tf.split(self.x_t, self.config.seq_length - 1, axis=1)
            ys = tf.split(self.y_t, self.config.seq_length - 1, axis=1)
            rand_samples = tf.split(self.rand_sample, self.config.seq_length - 1, axis=1)
            self.outputs = [xs[0]]
            for i in range(self.config.seq_length - 1):
                x = tf.cond(self.is_test, lambda: self.outputs[-1], lambda: xs[i])
                output, states = self.forward(cell, states, x, rand_samples[i])
                self.add_loss(ys[i])
                self.outputs.append(output)
            return self.outputs

    def create_lstm_multicell(self, n_layers, nstates):
        def lstm_cell():
            cell = tf.contrib.rnn.LSTMCell(nstates, reuse=False,
                                           initializer=self.initializer)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_keep)
            return cell
        return tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)])

    def forward(self, cell, states, x, rand_sample):
        output, states = tf.nn.dynamic_rnn(cell, x, initial_state=states)
        linear_output = self.add_linear(states)
        output = self.add_projection(linear_output, rand_sample)
        return output, states

    # rnn_states = (n_layers, num_hidden, num_hidden)
    def add_linear(self, rnn_states):
        U = tf.get_variable(name="U", shape=[self.config.num_hidden * len(rnn_states), self.config.num_output],
                            initializer=self.initializer)
        B = tf.get_variable(name="B", shape=[self.config.num_output], initializer=self.initializer)
        states = []
        for i in range(len(rnn_states)):
            states.append(rnn_states[i][1])
        return tf.add(tf.matmul(tf.concat(states, axis=-1), U), B, name='scores')

    # output = (batch_size, 1, 3)
    # rnn_output = (batch_size, 121)
    def add_projection(self, rnn_output, rand_sample):
        # e = (?, 1, ),
        # pi, co_rel = (?, 20, )
        # means, vars = (?, 20, 2)
        self.rnn_output = rnn_output
        self.means, self.variance, self.pi, self.co_rel, self.e = self.extract_params(rnn_output)
        bi_var_gauss = tf.reshape(tf.multiply(tf.reshape(self.pi, shape=(-1, 1)), rand_sample),
                                  shape=(-1, self.config.num_models, 2))
        bi_var_gauss = tf.reshape(tf.reduce_sum(bi_var_gauss, axis=1), shape=(-1, 2))
        eos = tf.reshape(tfd.Bernoulli(logits=self.e).sample(), shape=(-1, 1))
        output = tf.concat([tf.cast(eos, tf.float32), bi_var_gauss], axis=-1)
        output = tf.reshape(output, shape=(-1, 1, self.config.num_input))
        return output

    def get_co_var_tensor(self):
        var_x, var_y = tf.split(self.variance, 2, axis=-1)
        var_x2 = tf.square(var_x)
        var_y2 = tf.square(var_y)
        var_xy = tf.multiply(var_x, var_y)
        return tf.reshape(tf.concat([var_x2, var_xy, var_xy, var_y2], axis=-1),
                          shape=(-1, self.config.num_models, 2, 2))

    # y = (batch_size, 2)
    def add_loss(self, y_t):
        delta = 1e-7
        y_e, y_x, y_y = tf.split(y_t, 3, axis=-1)
        y_x = tf.reshape(y_x, shape=(-1, 1))
        y_y = tf.reshape(y_y, shape=(-1, 1))
        y = tf.reshape(tf.concat([y_x, y_y], axis=-1), shape=(-1, 2))
        # y = tf.concat([y for _ in range(self.config.num_models)], axis=1)
        self.prob = bivar_gauss_prob(y, self.means, self.variance, self.co_rel)
        # co_var = self.get_co_var_tensor()
        # self.bi_var_gauss_dist = tfd.MultivariateNormalFullCovariance(loc=self.means, covariance_matrix=co_var)
        # self.prob = self.bi_var_gauss_dist.cdf(y)
        self.bi_var_gauss = tf.multiply(self.prob, self.pi)
        self.loss_gaussian = -tf.log(tf.maximum(tf.reduce_sum(self.bi_var_gauss, axis=1, keepdims=True), delta))
        self.loss_bernoulli = -tf.log((self.e) * y_e + (1 - self.e) * (1 - y_e))
        self.loss += tf.reduce_sum(self.loss_bernoulli + self.loss_gaussian)

    def add_train_op(self):
        with tf.variable_scope("training"):
            grads, variables = zip(*self.optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads, self.config.gradient_clip_norm)
            self.train_op = self.optimizer.apply_gradients(zip(grads, variables))

    # rnn_output = (batch_size, 121)
    def extract_params(self, rnn_output):
        e = 1 / (1 + tf.exp(rnn_output[:, -1]))
        pi, mux, muy, varx, vary, co_rel = tf.split(value=rnn_output[:, :-1], num_or_size_splits=6, axis=-1)
        pi = tf.nn.softmax(pi - tf.reduce_max(pi, 1, keepdims=True))
        means = tf.reshape(tf.concat([mux, muy], axis=-1), shape=(-1, self.config.num_models, 2), name="means")
        varx = tf.exp(tf.abs(varx))
        vary = tf.exp(tf.abs(vary))
        variance = tf.reshape(tf.concat([varx, vary], axis=-1), shape=(-1, self.config.num_models, 2), name="variance")
        co_rel = tf.clip_by_value(tf.tanh(co_rel), -0.999999999, 0.999999999, name="co_rel")
        return means, variance, pi, co_rel, e

    def run_batch(self, sess, step=0, epoch=0, strokes_batch=None, means=None, variance=None, co_rel=None, pi=None):
        is_train = True
        if strokes_batch is None:
            is_train = False
            strokes_batch = np.zeros((self.config.batch_size, self.config.seq_length, self.config.num_input))
        strokes_batch = np.reshape(strokes_batch, newshape=(self.config.batch_size, -1, 3))
        outputs = []
        if is_train:
            strokes_batch, mean, std = self.normalise(strokes_batch)
            self.data_mean += mean
            self.data_var += np.square(std)
            self.n_batches += 1
            feed_dict = {
                self.x_t: strokes_batch[:, :-1, :],
                self.y_t: strokes_batch[:, 1:, :],
                self.rand_sample: strokes_batch[:, 1:, :-1],
                self.dropout_keep: 1.0,
                self.is_test: False
            }
            fetch = [self.loss, self.loss_gaussian, self.loss_bernoulli,
                     self.means, self.variance, self.co_rel,
                     self.bi_var_gauss, self.prob, self.pi, self.rnn_output, self.train_op]
            loss, gauss_loss, bernoulli_loss, means, variance, co_rel, bi_var_gauss, prob, pi, rnn_output, _ = \
                sess.run(fetch, feed_dict)
            print(" loss = ", loss, " bernoulli = ", np.sum(bernoulli_loss), " gauss loss = ", np.sum(gauss_loss),
                  " step = ", step, " epoch = ", epoch)
            if np.sum(gauss_loss) < 0 or math.isnan(loss) or math.isnan(gauss_loss):
                # print(" bi_var_gauss = ", bi_var_gauss)
                print(" prob = ", prob)
                print(" rho = ", co_rel)
                # print(" pi = ", pi)
                # print(" pi sum = ", np.sum(pi))
                # print(" variance = ", variance)
            if loss < 0 or math.isnan(loss):
                exit()
            return loss, means, variance, co_rel, pi
        else:
            if means is None or variance is None or co_rel is None:
                raise Exception("Please provide mean, variance, co-relation coefficients")
            rand_sample = self.sample_rand_nums(means, variance, co_rel, pi)
            outputs.append(np.reshape([[[0., 0., 0]]], newshape=(1, 1, 3)))
            for i in range(self.config.seq_length):
                feed_dict = {
                    self.x_t: np.reshape(strokes_batch[:, :-1, :],
                                         newshape=(1, self.config.seq_length - 1, self.config.num_input)),
                    self.dropout_keep: 1.0,
                    self.rand_sample: rand_sample,
                    self.is_test: True
                }
                fetch = [self.means, self.variance, self.co_rel, self.pi]
                means, variance, co_rel, pi = sess.run(fetch, feed_dict)
                rand_sample = self.sample_rand_nums(means, variance, co_rel, pi)

                feed_dict = {
                    self.x_t: np.reshape(strokes_batch[:, :-1, :],
                                         newshape=(1, self.config.seq_length - 1, self.config.num_input)),
                    self.dropout_keep: 1.0,
                    self.rand_sample: rand_sample,
                    self.is_test: True
                }

                fetch = [self.outputs[1]]
                output = sess.run(fetch, feed_dict)

                new_shape = (self.config.batch_size, 1, self.config.num_input)
                outputs.append(np.squeeze(np.reshape(np.squeeze(output), newshape=new_shape)))
            outputs.pop(0)
            outputs.insert(0, np.array([0, 0, 0]))
            outputs = np.squeeze(self.unnormalise(outputs))
            np.save('saved_models/unconditional_writing_h' + str(self.config.num_hidden) + "_t" +
                    str(self.config.seq_length) + "_b" + str(self.config.batch_size) + "_e" + str(self.config.n_epoch),
                    outputs)
            return np.squeeze(outputs)

    def sample_rand_nums(self, means, variance, co_rel, pi):
        means = np.mean(means, axis=0)
        variance = np.mean(variance, axis=0)
        co_rel = np.mean(co_rel, axis=0)
        co_var = self.get_co_var(variance, co_rel)
        rand_sample = np.zeros((1, self.config.seq_length - 1, 2))
        np.random.seed(self.random_seed)
        r = np.random.random()
        tot_prob = 0.0
        for j in range(self.config.num_models):
            tot_prob += pi[0, j]
            if tot_prob > r:
                rand_sample = np.reshape(np.random.multivariate_normal(
                    means[j], co_var[j], size=(self.config.seq_length - 1)),
                    newshape=(1, self.config.seq_length - 1, 2))
                break
        return rand_sample

    def get_co_var(self, variance, co_rel):
        sigmax2 = np.square(variance[:, 0])
        sigmay2 = np.square(variance[:, 1])
        sigmaxy = variance[:, 0] * variance[:, 1] * co_rel
        co_var = np.zeros((self.config.num_models, 2, 2))
        co_var[:, 0, 0] = sigmax2
        co_var[:, 0, 1] = sigmaxy
        co_var[:, 1, 0] = sigmaxy
        co_var[:, 0, 0] = sigmay2
        return co_var

    def normalise(self, strokes):
        mean = np.expand_dims(np.mean(strokes[:, :, 1:], axis=1), axis=1)
        std = np.std(strokes)
        strokes[:, :, 1:] -= mean
        strokes[:, :, 1:] /= std
        return strokes, mean, std

    def unnormalise(self, output):
        output = np.reshape(output, newshape=(self.config.batch_size, self.config.seq_length + 1, self.config.num_input))
        output[:, :, 1:] *= np.sqrt(self.data_var) / self.n_batches
        output[:, :, 1:] += self.data_mean / self.n_batches
        return output


def test():
    strokes = np.load('../data/strokes.npy')
    with tf.Graph().as_default() as graph:
        hand_writing_generator = HandWritingGenerator()
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            generator = get_batch_data_iterator(1, strokes, 1)
            batch_stroke = generator.next()
            while batch_stroke is not None:
                hand_writing_generator.run_batch(sess, batch_stroke[:, :2, :])
                hand_writing_generator.run_batch(sess)
                break


if __name__ == '__main__':
    test()
