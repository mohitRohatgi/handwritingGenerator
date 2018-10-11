import tensorflow as tf
from config.config import Config
import numpy as np
from utils.preprocessor import get_batch_data_iterator
import math


class HandWritingGenerator:
    def __init__(self, random_seed=1):
        self.config = Config()
        self.outputs = []
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        self.initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.075)
        self.add_model()
        self.data_mean = 0.0
        self.data_var = 0.0
        self.n_batches = 0
        self.scale = self.config.scale
        self.saver = tf.train.Saver()
        # np.random.seed(random_seed)

    def add_model(self):
        self.add_placeholders()
        self.assemble_model()
        # self.loss /= self.config.batch_size
        self.add_train_op()

    def add_placeholders(self):
        self.x_t = tf.placeholder(dtype=tf.float32, shape=(None, None, self.config.num_input), name='x_t')
        self.y_t = tf.placeholder(dtype=tf.float32, shape=(None, None, self.config.num_input), name='x_t')
        self.dropout_keep = tf.placeholder(dtype=tf.float32, shape=(), name='dropout_keep')

    def assemble_model(self):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            output = self.x_t
            rnn_output = []
            for i in range(self.config.lstm_layers):
                if i != 0:
                    # skip connections
                    output = tf.concat([output, self.x_t], axis=-1)
                cell = tf.contrib.rnn.LSTMCell(self.config.num_hidden, reuse=False, initializer=self.initializer)
                states = cell.zero_state(tf.shape(output)[0], tf.float32)
                output, _ = tf.nn.dynamic_rnn(cell, self.x_t, initial_state=states)
                rnn_output.append(output)
            rnn_output = tf.concat(rnn_output, axis=-1)
            self.output = self.add_linear(rnn_output)
            self.extract_params(self.output)
            self.add_loss()

    # rnn_states = (batch_size, time_steps, num_layers * num_hidden)
    def add_linear(self, rnn_states):
        U = tf.get_variable(name="U", shape=[self.config.num_hidden * self.config.lstm_layers, self.config.num_output],
                            initializer=self.initializer)
        B = tf.get_variable(name="B", shape=[self.config.num_output], initializer=self.initializer)
        return tf.add(tf.matmul(rnn_states, tf.expand_dims(U, axis=0)), B, name='scores')

    # rnn_output = (batch_size, time_steps, 121)
    # mux, muy, varx, vary, co_rel, pi = (batch_size, time_steps, 20)
    # e = (batch_size, time_steps, 1)
    def extract_params(self, output):
        self.e = 1 / (1 + tf.exp(output[:, :, -1]))
        pi, mux, muy, varx, vary, co_rel = tf.split(value=output[:, :, :-1], num_or_size_splits=6, axis=-1)
        self.mux = mux
        self.muy = muy
        self.pi = tf.nn.softmax(pi)
        self.varx = tf.exp(tf.abs(varx))
        self.vary = tf.exp(tf.abs(vary))
        min_val = -0.99
        max_val = 0.99
        self.co_rel = tf.clip_by_value(tf.tanh(co_rel), clip_value_min=min_val, clip_value_max=max_val, name="co_rel")

    # y = (batch_size, time_steps, 3)
    # e = (batch_size, time_steps, 1)
    # prob, pi = (batch_size, time_steps, 20)
    def add_loss(self):
        y_e, y_x, y_y = tf.split(self.y_t, 3, axis=-1)
        self.prob = self.bivar_gauss_prob(y_x, y_y)
        self.bi_var_gauss = tf.multiply(self.prob, self.pi)

        # finding the max acc. model to maximise it
        # loss_gauss_max_model = tf.reduce_max(self.bi_var_gauss, axis=-1, keepdims=True)
        self.loss_gaussian = -tf.log(tf.reduce_sum(self.bi_var_gauss, axis=-1))
        self.loss_bernoulli = -tf.log((self.e * y_e + (1 - self.e) * (1 - y_e)))
        self.loss = tf.reduce_sum(self.loss_bernoulli + self.loss_gaussian)

    # prob, pi = (batch_size, time_steps, 20)
    # mux, muy, varx, vary, co_rel, pi = (batch_size, time_steps, 20)
    # y_x, y_y = (batch_size, time_steps, 1)
    def bivar_gauss_prob(self, y_x, y_y):
        z1 = tf.divide((y_x - self.mux), self.varx)
        z2 = tf.divide((y_y - self.muy), self.vary)
        z = tf.square(z1) + tf.square(z2) - 2 * self.co_rel * z1 * z2
        co_rel_sq = 1.0 - tf.square(self.co_rel)
        prob = tf.exp(-z / (2.0 * co_rel_sq))
        prob /= 2.0 * np.pi * self.varx * self.vary * tf.sqrt(co_rel_sq)
        return prob

    def add_train_op(self):
        with tf.variable_scope("training"):
            grads, variables = zip(*self.optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads, self.config.gradient_clip_norm)
            self.train_op = self.optimizer.apply_gradients(zip(grads, variables))

    def run_batch(self, sess, step=0, epoch=0, strokes_batch=None):
        outputs = []
        if strokes_batch is not None:
            strokes_batch = np.reshape(strokes_batch, newshape=(self.config.batch_size, -1, 3)) / self.scale
            # strokes_batch, mean, std = self.normalise(strokes_batch)
            # self.data_mean += mean
            # self.data_var += np.square(std)
            # self.n_batches += 1
            feed_dict = {
                self.x_t: strokes_batch[:, :-1, :],
                self.y_t: strokes_batch[:, 1:, :],
                self.dropout_keep: 1.0
            }
            fetch = [self.loss, self.loss_gaussian, self.loss_bernoulli,
                     self.mux, self.varx, self.muy, self.vary, self.co_rel,
                     self.bi_var_gauss, self.prob, self.pi, self.e, self.train_op]
            loss, gauss_loss, bernoulli_loss, mu_x, var_x, mu_y, var_y, co_rel, bi_var_gauss, prob, pi, e, _ = \
                sess.run(fetch, feed_dict)

            # print(" loss = ", loss, " bernoulli = ", np.sum(bernoulli_loss), " gauss loss = ", np.sum(gauss_loss),
            #       " time step = ", i, " epoch = ", epoch)
            if math.isnan(loss) or math.isnan(gauss_loss):
                print(" bi_var_gauss = ", bi_var_gauss)
                print(" prob = ", prob)
                print(" rho = ", co_rel)
                print(" pi = ", pi)
                print(" pi sum = ", np.sum(pi))
                print(" varx = ", var_x)
                print(" vary = ", var_y)
                print("mux = ", mu_x)
                print("muy = ", mu_y)
            # if math.isnan(loss):
            #     exit()

            print(" loss = ", loss, " bernoulli = ", np.sum(bernoulli_loss), " gauss loss = ", np.sum(gauss_loss),
                  " step = ", step, " epoch = ", epoch)
            return loss, mu_x, mu_y, var_x, var_y, co_rel, pi
        else:
            outputs.append([0, 0, 0])
            for i in range(self.config.seq_length):
                feed_dict = {
                    self.x_t: np.reshape(outputs[-1], newshape=(1, 1, self.config.num_input)),
                    self.dropout_keep: 1.0,
                }
                fetch = [self.mux, self.muy, self.varx, self.vary, self.co_rel, self.pi, self.e]
                mu_x, mu_y, var_x, var_y, co_rel, pi, e = sess.run(fetch, feed_dict)
                e = np.random.binomial(1.0, p=e[0], size=1)[0]
                rand_sample = self.sample_rand_nums(mu_x, mu_y, var_x, var_y, co_rel, pi)
                x, y = rand_sample[0, 0, 0], rand_sample[0, 0, 1]
                output = np.array([e, x, y])
                outputs.append(output)
            outputs[-1][0] = 1
            outputs[:, 1:] *= self.scale
            outputs = np.squeeze(outputs)
            # outputs = np.squeeze(self.unnormalise(outputs))
            np.save('saved_models/unconditional_writing_h' + str(self.config.num_hidden) + "_t" +
                    str(self.config.seq_length) + "_b" + str(self.config.batch_size) + "_e" + str(self.config.n_epoch),
                    outputs)
            return np.squeeze(outputs)

    def sample_rand_nums(self, mu_x, mu_y, var_x, var_y, co_rel, pi):
        means = np.array([np.squeeze(mu_x), np.squeeze(mu_y)])
        pi = np.squeeze(pi)
        co_rel = np.squeeze(co_rel)
        co_var = self.get_co_var(np.squeeze(var_x), np.squeeze(var_y), co_rel)
        rand_sample = np.reshape(np.random.multivariate_normal(
                    means[:, -1], co_var[:, :, -1], size=(self.config.seq_length - 1)),
                    newshape=(1, self.config.seq_length - 1, 2))
        r = np.random.random()
        tot_prob = 0.0
        for i in range(self.config.num_models):
            tot_prob += pi[i]
            if tot_prob > r:
                print i, tot_prob, r, np.sum(pi)
                rand_sample = np.reshape(np.random.multivariate_normal(
                    means[:, i], co_var[:, :, i], size=(self.config.seq_length - 1)),
                    newshape=(1, self.config.seq_length - 1, 2))
                return rand_sample
        return rand_sample

    def get_co_var(self, var_x, var_y, co_rel):
        sigmax2 = np.square(var_x)
        sigmay2 = np.square(var_y)
        sigmaxy = var_x * var_y * co_rel
        co_var = np.array([[sigmax2, sigmaxy], [sigmaxy, sigmay2]])
        return co_var

    def normalise(self, strokes):
        mean = np.expand_dims(np.mean(strokes[:, :, 1:], axis=1), axis=1)
        std = np.std(strokes[:, :, 1:], axis=1)
        strokes[:, :, 1:] -= mean
        strokes[:, :, 1:] /= std
        return strokes, mean, std

    def unnormalise(self, output):
        output = np.reshape(output, newshape=(self.config.batch_size, self.config.seq_length + 1, self.config.num_input))
        output[:, :, 1:] *= np.sqrt(self.data_var) / self.n_batches
        # output[:, :, 1:] += self.data_mean / self.n_batches
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
