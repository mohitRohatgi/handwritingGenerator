import numpy as np
from config.config import Config

config = Config()


def create_batches(strokes_batch):
    new_strokes_batch = np.zeros((len(strokes_batch), config.seq_length, 3))
    for i, stroke in enumerate(strokes_batch):
        for j in range(len(stroke)):
            if j >= config.seq_length:
                break
            else:
                new_strokes_batch[i][j] = stroke[j]
        new_strokes_batch[:, -1, 0] = 1
        # new_strokes_batch = np.minimum(new_strokes_batch, 30)
        # new_strokes_batch = np.maximum(new_strokes_batch, -30)
        new_strokes_batch = np.array(new_strokes_batch, dtype=np.float32)

    return new_strokes_batch


def get_batch_data(strokes, batch_size, random=True, start=0):
    if random:
        indices = np.random.randint(0, len(strokes), batch_size)
    else:
        indices = range(start, min(start + batch_size, len(strokes)))
    return create_batches(np.array(strokes)[indices])


def get_batch_data_iterator(n_epoch, strokes, batch_size):
    num_batches_per_epoch = int(config.seq_length / batch_size) + 1
    for i in range(n_epoch):
        print("epoch: ", i)
        for j in range(num_batches_per_epoch):
            strokes_batch = get_batch_data(strokes, batch_size=batch_size)
            yield strokes_batch


def test():
    strokes = np.load('../data/strokes.npy')
    data = get_batch_data_iterator(1, strokes, 64).next()
    assert len(data[0].shape) == 3


if __name__ == '__main__':
    test()