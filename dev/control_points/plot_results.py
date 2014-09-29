#!/usr/bin/python

import numpy
import matplotlib.pyplot as plt
import sys
from scipy.stats.stats import *


def main(file_name):

    fo = open(file_name)
    r = fo.readlines()
    x = numpy.array([int((float(n[0]))) for n in [line.strip().split() for line in r]])
    y = numpy.array([(float(n[1])) for n in [line.strip().split() for line in r]])
    #z=numpy.array([int(n[1]) for n in [line.strip().split() for line in r]])
    fo.close()

    print spearmanr(x, y)
    print pearsonr(x,y)

    print y
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    #ax1.set_title("Plot title...")
    ax1.set_xlabel('length (mm)')
    ax1.set_ylabel('number of control points')

    ax1.plot(x,y, 'ro:', linestyle='None')

    leg = ax1.legend()

    plt.show()


def mainBis(file_name):

    fo = open(file_name)
    r = fo.readlines()
    x = numpy.array([int((float(n[1]))) for n in [line.strip().split() for line in r]])
    y = numpy.array([(float(n[5])) for n in [line.strip().split() for line in r]])
    #z=numpy.array([int(n[1]) for n in [line.strip().split() for line in r]])
    fo.close()

    print y
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    #ax1.set_title("Plot title...")
    ax1.set_xlabel('number of control points')
    ax1.set_ylabel('MSE')

    ax1.plot(x,y, 'ro:', label='point 3 (262 mm)', linestyle='None')

    plt.show()


if __name__ == "__main__":
    file_name = sys.argv[1]
    print file_name
    #main(file_name)
    mainBis(file_name)