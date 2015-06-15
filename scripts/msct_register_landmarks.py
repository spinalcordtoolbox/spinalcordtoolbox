#!/usr/bin/env python
#########################################################################################
#
# msct_icp
# This file contains an implementation of the iterative closest point algorithm.
# This algo registers two sets of points (3D coordinates) together.
#
# Adapted from: http://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Created: 2015-06-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from msct_types import Point
from numpy import array, sin, cos, matrix, sum, mean
from math import pow, sqrt
from operator import itemgetter


def getNeighbors(point, set_points, k=1):
    '''
    Locate most similar neighbours
    :param point: the point for which we want to compute the neighbors
    :param trainingSet: list of other Points
    :param k: number of neighbors wanted
    :return: k nearest neighbors of input point
    '''
    distances = []
    for other_point in set_points:
        dist = point.euclideanDistance(other_point)
        distances.append((other_point, dist))
    distances.sort(key=itemgetter(1))
    return [distances[x][0] for x in range(k)]


def SSE(pointsA, pointsB):
    return sum(array(pointsA[:, 0:3]-pointsB[:, 0:3])**2.0)


def minRigidTransform(params, points_fixed, points_moving):
    alpha, beta, gamma, tx, ty, tz = params[0], params[1], params[2], params[3], params[4], params[5]

    rotation_matrix = matrix([[cos(alpha)*cos(beta), cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma), cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma)],
                              [sin(alpha)*cos(beta), sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma), sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma)],
                              [-sin(beta), cos(beta)*sin(gamma), cos(beta)*cos(gamma)]])

    points_moving_barycenter = mean(points_moving, axis=0)

    points_moving_reg = ((rotation_matrix * (matrix(points_moving) - points_moving_barycenter).T).T + points_moving_barycenter) + matrix([tx, ty, tz])

    return SSE(matrix(points_fixed), points_moving_reg)


def minRigid_xy_Transform(params, points_fixed, points_moving):
    gamma, tx, ty = params[0], params[1], params[2]

    rotation_matrix = matrix([[cos(gamma), - sin(gamma), 0],
                              [sin(gamma), cos(gamma), 0],
                              [0, 0, 1]])

    points_moving_barycenter = mean(points_moving, axis=0)

    points_moving_reg = ((rotation_matrix * (
        matrix(points_moving) - points_moving_barycenter).T).T + points_moving_barycenter) + matrix([tx, ty, 0])

    return SSE(matrix(points_fixed), points_moving_reg)


def minTranslation_Transform(params, points_fixed, points_moving):
    return SSE(matrix(points_fixed), matrix(points_moving) + matrix([params[0], params[1], params[2]]))


def getRigidTransformFromLandmarks(points_fixed, points_moving, constraints=None, show=False):
    list_constraints = [None, 'xy', 'translation']
    if constraints not in list_constraints:
        raise 'ERROR: the constraints must be one of those: '+', '.join(list_constraints)

    points = (points_fixed, points_moving)
    points_moving_reg = points_moving

    from scipy.optimize import minimize

    rotation_matrix = matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    translation_array = matrix([0.0, 0.0, 0.0])

    if constraints is None:
        initial_parameters = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        res = minimize(minRigidTransform, x0=initial_parameters, args=points, method='Nelder-Mead', tol=1e-6, options={'maxiter': 10000, 'disp': True})

        alpha, beta, gamma, tx, ty, tz = res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5]
        rotation_matrix = matrix([[cos(alpha) * cos(beta), cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
                               cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)],
                              [sin(alpha) * cos(beta), sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
                               sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma)],
                              [-sin(beta), cos(beta) * sin(gamma), cos(beta) * cos(gamma)]])
        translation_array = matrix([tx, ty, tz])

        points_moving_barycenter = mean(points_moving, axis=0)
        points_moving_reg = ((rotation_matrix * (
            matrix(points_moving) - points_moving_barycenter).T).T + points_moving_barycenter) + translation_array

    elif constraints is 'xy':
        initial_parameters = [0.0, 0.0, 0.0]
        res = minimize(minRigid_xy_Transform, x0=initial_parameters, args=points, method='Nelder-Mead', tol=1e-6,
                       options={'maxiter': 10000, 'disp': True})

        gamma, tx, ty = res.x[0], res.x[1], res.x[2]

        rotation_matrix = matrix([[cos(gamma), - sin(gamma), 0],
                                  [sin(gamma), cos(gamma), 0],
                                  [0, 0, 1]])
        translation_array = matrix([tx, ty, 0])

        points_moving_barycenter = mean(points_moving, axis=0)
        points_moving_reg = ((rotation_matrix * (
            matrix(points_moving) - points_moving_barycenter).T).T + points_moving_barycenter) + translation_array

    elif constraints is 'translation':
        initial_parameters = [0.0, 0.0, 0.0]
        res = minimize(minTranslation_Transform, x0=initial_parameters, args=points, method='Nelder-Mead', tol=1e-6,
                       options={'maxiter': 10000, 'disp': True})

        translation_array = matrix([res.x[0], res.x[1], res.x[2]])
        points_moving_reg = matrix(points_moving) + translation_array

    if show:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        points_moving_matrix = matrix(points_moving)
        points_fixed_matrix = matrix(points_fixed)

        number_points = len(points_fixed)

        ax.scatter([points_moving_matrix[i, 0] for i in range(0, number_points)],
                   [points_moving_matrix[i, 1] for i in range(0, number_points)],
                   [points_moving_matrix[i, 2] for i in range(0, number_points)], c='r')
        ax.scatter([points_fixed_matrix[i, 0] for i in range(0, number_points)],
                   [points_fixed_matrix[i, 1] for i in range(0, number_points)],
                   [points_fixed_matrix[i, 2] for i in range(0, number_points)], c='g')
        ax.scatter([points_moving_reg[i, 0] for i in range(0, number_points)],
                   [points_moving_reg[i, 1] for i in range(0, number_points)],
                   [points_moving_reg[i, 2] for i in range(0, number_points)], c='b')
        plt.show()

    return rotation_matrix, translation_array
