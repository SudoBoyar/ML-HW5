import argparse
import sys

from model import ModelConfig, build_model


def main(args):
    config_args = {arg:val for arg, val in vars(args).items() if val is not None}
    configs = ModelConfig(**config_args)

    build_model(in_tensors, configs, is_training)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--nconv', dest='nconv', metavar='n', type=int, help='Number of convolutional layers')
    parser.add_argument('-f', '--filters', dest='nfilters', metavar='n', type=int, nargs='+',
                        help='Number of filters [at each layer]')
    parser.add_argument('-k', '--kernel', dest='kernel', metavar='n', type=int, nargs='+',
                        help='Size of the filters [at each layer]')
    parser.add_argument('-m', '--maxpool', dest='maxpool', metavar='n', type=int, nargs='+',
                        help='Maxpooling samples [at each layer]')
    parser.add_argument('-d', '--dropout', dest='dropout', metavar='d', type=float, nargs='+',
                        help='Dropout rate [at each layer]')
    parser.add_argument('-n', '--neurons', dest='fc_units', metavar='n', type=int,
                        help='Number of neurons in the fully connected layer')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
