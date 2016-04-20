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
    parser.add_argument('-grid_minX', default=-25, type=float, action='store', help='occupancy grid bounds [m]')
    parser.add_argument('-grid_maxX', default=25, type=float, action='store', help='occupancy grid bounds [m]')
    parser.add_argument('-grid_minY', default=-45, type=float, action='store', help='occupancy grid bounds [m]')
    parser.add_argument('-grid_maxY', default=5, type=float, action='store', help='occupancy grid bounds [m]')
    parser.add_argument('-grid_step', default=1, type=float, action='store', help='resolution of the occupancy grid [m]')
    parser.add_argument('-sensor_start', default=-180, type=float, action='store', help='first depth measurement [degrees]')
    parser.add_argument('-sensor_step', default=0.5, type=float, action='store', help='resolution of depth measurements [degrees]')
    args = parser.parse_args()
    print args

if __name__ == "__main__":
    main()
