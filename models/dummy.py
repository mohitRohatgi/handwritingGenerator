import numpy
from train_generator import main


strokes = numpy.load('../data/strokes.npy')
stroke = strokes[0]
trained_strokes = numpy.load('../models/saved_models/unconditional_writing_h256_t300_b1_e30.npy')


def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    trained_strokes[-1, 0] = 1
    return trained_strokes


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'


def test():
    generate_unconditionally()


if __name__ == '__main__':
    test()