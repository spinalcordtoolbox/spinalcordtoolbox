#!/usr/bin/python

import numpy
import matplotlib.pyplot as plt
import sys
from scipy.stats.stats import *
import math


def main(file_name):

    fo = open(file_name)
    r = fo.readlines()
    x = numpy.array([int((float(n[0]))) for n in [line.strip().split() for line in r]])
    y = numpy.array([(float(n[1])) for n in [line.strip().split() for line in r]])
    #z=numpy.array([int(n[1]) for n in [line.strip().split() for line in r]])
    fo.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)


    # Tendance
    x_fit = []
    y_fit = []
    for i in range(20,400):
        x_fit.append(i)

    for i in x_fit:
        #y_fit.append(math.log(i*i/2+i*i*i))
        y_fit.append(30*(math.log(i-15, 13)) - 25)

    ax1.plot(x_fit, y_fit, color='yellow', label='30*log(x-15,13)-25')

    y_fit = []

    for i in x_fit:
        #y_fit.append(math.log(i*i/2+i*i*i))
        y_fit.append(30*(math.log(i, 10)) - 42)

    ax1.plot(x_fit, y_fit, color='red', label='30*log(x,10)-42')

    y_fit = []

    for i in x_fit:
        #y_fit.append(math.log(i*i/2+i*i*i))
        y_fit.append(30*(math.log(i, 11)) - 55)

    ax1.plot(x_fit, y_fit, color='blue', label='30*log(x,11)-55')

    print spearmanr(x, y)
    print pearsonr(x,y)

    print y




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
    main(file_name)
    #mainBis(file_name)