from utils.preprocessor import get_batch_data_iterator
from config.config import Config
import tensorflow as tf
from hand_writing_generator import HandWritingGenerator
import os
import numpy as np


model_name = os.path.join(os.getcwd(), 'saved_models')
model_no = 0
save_path = os.path.join(model_name, str(model_no))


def main(random_seed=1):
    strokes = np.load('../data/strokes.npy')
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = HandWritingGenerator(random_seed=random_seed)
            sess.run(tf.global_variables_initializer())
            avg_loss, means, variance, co_rel = None, None, None, None
            if not os.path.exists(save_path):
                avg_loss, means, variance, co_rel, pi = train(strokes, sess, model)
            return model.run_batch(sess=sess, means=means, variance=variance, co_rel=co_rel, pi=pi)


def train(strokes, sess, model):
    config = Config()
    x_t_gen = get_batch_data_iterator(config.n_epoch, strokes, config.batch_size)
    model_path = os.path.join(model_name, "tf_models", '_h' + str(config.num_hidden), "_t" + str(config.seq_length),
                              "_b" + str(config.batch_size), "_e" + str(config.n_epoch))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    model_path = os.path.join(model_path, "model")

    saver = tf.train.Saver(tf.trainable_variables())
    num_batches_per_epoch = int(config.seq_length / config.batch_size) + 1
    avg_loss, means, variance, co_rel, pi = None, None, None, None, None
    for i in range(config.n_epoch):
        avg_loss = 0.0
        for j in range(num_batches_per_epoch):
            x_t_batch = x_t_gen.next()
            loss, means, variance, co_rel, pi = model.run_batch(sess, strokes_batch=x_t_batch, step=j, epoch=i)
            # print("loss = ", loss, " \t step = ", j, " \t epoch = ", i)
            avg_loss += loss
        print("avg_loss = ", avg_loss / num_batches_per_epoch, " epoch = ", i)
    print(" rho = ", co_rel)
    print(" pi = ", pi)

    saver.save(sess=sess, save_path=model_path)

    return avg_loss, means, variance, co_rel, pi

    # saver.save(sess, os.path.join(save_path, "model.ckpt"))


def generate():
    model = HandWritingGenerator()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model_path = os.path.join(model_name, "tf_models", '_h' + str(64), "_t" + str(400),
                                      "_b" + str(1), "_e" + str(200), "model")
            saver = tf.train.import_meta_graph("{}.meta".format(model_path))
            saver.restore(sess, os.path.join(model_path, "model"))
    # write generator code here. Model should be loaded and then used.
    pass


if __name__ == '__main__':
    main(random_seed=1)
    # generate()