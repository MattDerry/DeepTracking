import argparse
import numpy as np
import matplotlib.pyplot as plt

from model import FeedForwardRNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '-gpu', help='use GPU', action='store_true', default=False)
    parser.add_argument('-it', '-iter', default=100000, type=int, action='store', help='the number of training iterations')
    parser.add_argument('-N', default=100, type=int, action='store', help='training sequence length')
    parser.add_argument('-m', '-model', default='model', action='store', help='neural network model')
    parser.add_argument('-d', '-data', default='data.t7', action='store', help='training data file')
    parser.add_argument('-lr', '-learningRate', default=0.01, type=float, action='store', help='learning rate')
    parser.add_argument('-iw', '-initweights', default='', action='store', help='initial weights from file')
    args = parser.parse_args()
    print args

if __name__ == "__main__":
    main()
