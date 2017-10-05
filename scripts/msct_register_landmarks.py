#!/usr/bin/env python
#########################################################################################
#
# This file contains an implementation of the iterative closest point algorithm.
# This algo registers two sets of points (3D coordinates) together.
#
# Adapted from:
# http://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python
#
# NOTES ON ITK Transform Files:
# http://www.neuro.polymtl.ca/tips_and_tricks/how_to_use_ants#itk_transform_file
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Created: 2015-06-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: homogeneize input parameters: (src=src, dest=dest), instead of (dest, src).
# TODO: add full affine transfo
# TODO: normalize SSE: currently, it depends on the number of landmarks

# from msct_types import Point
from numpy import array, sin, cos, matrix, sum, mean, absolute
from math import pow, sqrt
from operator import itemgetter
# from msct_register_regularized import generate_warping_field
import sct_utils as sct
from nibabel import load

sse_results = []
ini_param_rotation = 0.5  # rad
ini_param_trans_x = 270.0  # pix
ini_param_trans_y = -150.0  # pix
initial_step = 2


def register_landmarks(fname_src, fname_dest, dof, fname_affine='affine.txt', verbose=1, path_qc='./'):
    """
    Register two NIFTI volumes containing landmarks
    :param fname_src: fname of source landmarks
    :param fname_dest: fname of destination landmarks
    :param dof: degree of freedom. Separate with "_". Example: Tx_Ty_Tz_Rx_Ry_Sz
    :param fname_affine: output affine transformation
    :param verbose: 0, 1, 2
    :return:
    """
    from msct_image import Image
    # open src label
    im_src = Image(fname_src)
    # coord_src = im_src.getNonZeroCoordinates(sorting='value')  # landmarks are sorted by value
    coord_src = im_src.getCoordinatesAveragedByValue()  # landmarks are sorted by value
    # open dest labels
    im_dest = Image(fname_dest)
    # coord_dest = im_dest.getNonZeroCoordinates(sorting='value')
    coord_dest = im_dest.getCoordinatesAveragedByValue()
    # Reorganize landmarks

    points_src, points_dest = [], []
    for coord in coord_src:
        point_src = im_src.transfo_pix2phys([[coord.x, coord.y, coord.z]])
        # convert NIFTI to ITK world coordinate
        # points_src.append([point_src[0][0], point_src[0][1], point_src[0][2]])
        points_src.append([-point_src[0][0], -point_src[0][1], point_src[0][2]])
    for coord in coord_dest:
        point_dest = im_dest.transfo_pix2phys([[coord.x, coord.y, coord.z]])
        # convert NIFTI to ITK world coordinate
        # points_dest.append([point_dest[0][0], point_dest[0][1], point_dest[0][2]])
        points_dest.append([-point_dest[0][0], -point_dest[0][1], point_dest[0][2]])

    # display
    sct.printv('Labels src: ' + str(points_src), verbose)
    sct.printv('Labels dest: ' + str(points_dest), verbose)
    sct.printv('Degrees of freedom (dof): ' + dof, verbose)

    if len(coord_src) != len(coord_dest):
        raise Exception('Error: number of source and destination landmarks are not the same, so landmarks cannot be paired.')

    # estimate transformation
    # N.B. points_src and points_dest are inverted below, because ITK uses inverted transformation matrices, i.e., src->dest is defined in dest instead of src.
    # (rotation_matrix, translation_array, points_moving_reg, points_moving_barycenter) = getRigidTransformFromLandmarks(points_dest, points_src, constraints=dof, verbose=verbose, path_qc=path_qc)
    (rotation_matrix, translation_array, points_moving_reg, points_moving_barycenter) = getRigidTransformFromLandmarks(points_src, points_dest, constraints=dof, verbose=verbose, path_qc=path_qc)
    # writing rigid transformation file
    # N.B. x and y dimensions have a negative sign to ensure compatibility between Python and ITK transfo
    text_file = open(fname_affine, 'w')
    text_file.write("#Insight Transform File V1.0\n")
    text_file.write("#Transform 0\n")
    text_file.write("Transform: AffineTransform_double_3_3\n")
    text_file.write("Parameters: %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % (
        rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
        rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
        rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2],
        translation_array[0, 0], translation_array[0, 1], translation_array[0, 2]))
    text_file.write("FixedParameters: %.9f %.9f %.9f\n" % (points_moving_barycenter[0],
                                                           points_moving_barycenter[1],
                                                           points_moving_barycenter[2]))
    text_file.close()


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
    """
    Compute sum of squared error between pair-wise landmarks
    :param pointsA:
    :param pointsB:
    :return:
    """
    return sum(array(pointsA[:, 0:3] - pointsB[:, 0:3])**2.0)


def real_optimization_parameters(param_from_optimizer, initial_param = 0, initial_step = 10):
    # The initial step for the Nelder-Mead algorithm is of (initial_param * 5e-2) which is too small when we want initial_param = 30 pix and step = 5 or 10.
    # This function allows to choose the scale of the steps after the first movement
    step_factor = float(initial_step) / float(initial_param * 5e-2)
    real_param = initial_param + (param_from_optimizer - initial_param) * step_factor
    return real_param


def Metric_Images(imageA, imageB, type=''):

    data_A_list = load(imageA).get_data().tolist()
    data_B_list = load(imageB).get_data().tolist()

    # Define both list of intensity
    list_A = []
    list_B = []
    for i in range(len(data_A_list)):
        list_A = list_A + data_A_list[i]
        list_B = list_B + data_B_list[i]
    # Calculate metric depending on the type
    if type == 'MeanSquares':
        result_metric = 1.0 / (len(list_A)) * sum(array([list_A[i][0] - list_B[i][0] for i in range(len(list_A))])**2)
        #result_metric = 1/(len(list_A)) * sum(array(list_A - list_B)**2)

    if type == 'Correlation':
        result_metric = 1.0 / (len(list_A)) * sum(absolute(array([list_A[i][0] - list_B[i][0] for i in range(len(list_A))])))

    if type == 'MI':
        sct.printv('\nto do: MI')

    # Return results
    sct.printv('\nResult of metric is: ' + str(result_metric))
    return result_metric


def minimize_transform(params, points_dest, points_src, constraints):
    """
    Cost function to minimize
    :param params:
    :param points_dest:
    :param points_src:
    :param constraints:
    :return: sum of squared error between pair-wise landmarks
    """
    # initialize dof
    dof = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    # initialize dictionary to relate constraints index to dof
    dict_dof = {'Tx': 0, 'Ty': 1, 'Tz': 2, 'Rx': 3, 'Ry': 4, 'Rz': 5, 'Sx': 6, 'Sy': 7, 'Sz': 8}
    # extract constraints
    list_constraints = constraints.split('_')
    # loop across constraints and update dof
    for i in range(len(list_constraints)):
        dof[dict_dof[list_constraints[i]]] = params[i]
    # convert dof to more intuitive variables
    tx, ty, tz, alpha, beta, gamma, scx, scy, scz = dof[0], dof[1], dof[2], dof[3], dof[4], dof[5], dof[6], dof[7], dof[8]
    # build rotation matrix
    rotation_matrix = matrix([[cos(alpha) * cos(beta), cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma), cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)],
                              [sin(alpha) * cos(beta), sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma), sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma)],
                              [-sin(beta), cos(beta) * sin(gamma), cos(beta) * cos(gamma)]])
    # build scaling matrix
    scaling_matrix = matrix([[scx, 0.0, 0.0], [0.0, scy, 0.0], [0.0, 0.0, scz]])
    # compute rotation+scaling matrix
    rotsc_matrix = scaling_matrix * rotation_matrix
    # compute center of mass from moving points (src)
    points_src_barycenter = mean(points_src, axis=0)
    # apply transformation to moving points (src)
    points_src_reg = ((rotsc_matrix * (matrix(points_src) - points_src_barycenter).T).T + points_src_barycenter) + matrix([tx, ty, tz])
    # record SSE for later display
    sse_results.append(SSE(matrix(points_dest), points_src_reg))
    # return SSE
    return SSE(matrix(points_dest), points_src_reg)


def getRigidTransformFromImages(img_dest, img_src, constraints='none', metric = 'MeanSquares', center_rotation=None):
    list_constraints = [None, 'none', 'xy', 'translation', 'translation-xy', 'rotation', 'rotation-xy']
    list_center_rotation = [None, 'BarycenterImage']
    if constraints not in list_constraints:
        raise 'ERROR: the constraints must be one of those: ' + ', '.join(list_constraints)
    if center_rotation not in list_center_rotation:
        raise 'ERROR: the center_rotation must be one of those: ' + ', '.join(list_center_rotation)

    from scipy.optimize import minimize

    rotation_matrix = matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    translation_array = matrix([0.0, 0.0, 0.0])

    # Get barycenter of the images if specified
    if center_rotation == 'BarycenterImage':
        sct.printv('\nEvaluating barycenters of images...')
        # Importing data
        from nibabel import load
        from numpy import amax, cross, dot
        from math import acos, pi
        data_moving = load(img_src).get_data()
        data_fixed = load(img_dest).get_data()
        data_moving_10percent = data_moving > amax(data_moving) * 0.1
        data_fixed_10percent = data_fixed > amax(data_fixed) * 0.1
        # Calculating position of barycenters
        coord_barycenter_moving = (1.0 / (sum(data_moving))) * sum(array([[data_moving[i, j, k] * i, data_moving[i, j, k] * j, data_moving[i, j, k] * k] for i in range(data_moving.shape[0]) for j in range(data_moving.shape[1]) for k in range(data_moving.shape[2])]), axis=0)
        coord_barycenter_fixed = (1.0 / (sum(data_fixed))) * sum(array([[data_fixed[i, j, k] * i, data_fixed[i, j, k] * j, data_fixed[i, j, k] * k] for i in range(data_fixed.shape[0]) for j in range(data_fixed.shape[1]) for k in range(data_fixed.shape[2])]), axis=0)
        coord_barycenter_moving_10percent = (1.0 / (sum(data_moving_10percent))) * sum(array([[data_moving_10percent[i, j, k] * i, data_moving_10percent[i, j, k] * j, data_moving_10percent[i, j, k] * k] for i in range(data_moving_10percent.shape[0]) for j in range(data_moving_10percent.shape[1]) for k in range(data_moving_10percent.shape[2])]), axis=0)
        coord_barycenter_fixed_10percent = (1.0 / (sum(data_fixed_10percent))) * sum(array([[data_fixed_10percent[i, j, k] * i, data_fixed_10percent[i, j, k] * j, data_fixed_10percent[i, j, k] * k] for i in range(data_fixed_10percent.shape[0]) for j in range(data_fixed_10percent.shape[1]) for k in range(data_fixed_10percent.shape[2])]), axis=0)

        sct.printv('\nPosition of the barycenters:' 
                   '\n\t-moving image : ' + str(coord_barycenter_moving) +
                   '\n\t-fixed image: ' + str(coord_barycenter_fixed))
        # Evaluating initial translations to match the barycenters
        ini_param_trans_x_real = int(round(coord_barycenter_fixed[0] - coord_barycenter_moving[0]))
        ini_param_trans_y_real = int(round(coord_barycenter_fixed[1] - coord_barycenter_moving[1]))

        # Defining new center of rotation
        coord_center_rotation = [int(round(coord_barycenter_fixed[0])), int(round(coord_barycenter_fixed[1])), int(round(coord_barycenter_fixed[2]))]

        # Evaluating the initial rotation to match the 10 percent barycenters
        # We have calculated two relevant points to evaluate the best initial registration for the algorithm so that it may converge more quickly
        # First a translation to match the barycenters and then a rotation (of center: the barycenter of the fixed image) to match the 10_percent barycenters
        vector_bar_fix_2_bar_10p_moving = coord_barycenter_moving_10percent - coord_barycenter_fixed
        vector_bar_fix_2_bar_10p_fixed = coord_barycenter_fixed_10percent - coord_barycenter_fixed
        vector_bar_10p_fix_2_10p_moving = coord_barycenter_moving_10percent - coord_barycenter_fixed_10percent
        a = dot(vector_bar_fix_2_bar_10p_moving, vector_bar_fix_2_bar_10p_moving)  # OAm
        b = dot(vector_bar_fix_2_bar_10p_fixed, vector_bar_fix_2_bar_10p_fixed)  # OAf
        c = dot(vector_bar_10p_fix_2_10p_moving, vector_bar_10p_fix_2_10p_moving)  # AfAm
        e = cross(vector_bar_fix_2_bar_10p_moving, vector_bar_fix_2_bar_10p_fixed)
        if e[2] >= 0:
            ini_param_rotation_real = acos((a + b - c) / (2.0 * sqrt(a) * sqrt(b)))   # theorem of Al-Kashi
        else:
            ini_param_rotation_real = -acos((a + b - c) / (2.0 * sqrt(a) * sqrt(b)))    # theorem of Al-Kashi

    else:
        coord_center_rotation = None
        ini_param_trans_x_real = ini_param_trans_x
        ini_param_trans_y_real = ini_param_trans_y
        ini_param_rotation_real = ini_param_rotation

    if constraints == 'rotation-xy':
        initial_parameters = [ini_param_rotation]
        res = minimize(minRotation_xy_Transform_for_Images, x0=initial_parameters, args=(img_dest, img_src, metric), method='Nelder-Mead', tol=1e-2,
                       options={'maxiter': 1000, 'disp': True})

        gamma = res.x[0]
        rotation_matrix = matrix([[cos(gamma), - sin(gamma), 0],
                                  [sin(gamma), cos(gamma), 0],
                                  [0, 0, 1]])

    elif constraints == 'xy':
        initial_parameters = [ini_param_rotation_real, ini_param_trans_x_real, ini_param_trans_y_real]
        res = minimize(minRigid_xy_Transform_for_Images, x0=initial_parameters, args=(img_dest, img_src, coord_center_rotation, metric), method='Nelder-Mead', tol=1e-2,
                       options={'maxiter': 1000, 'disp': True})

        # change result if input parameters are changed
        # tx_real = ini_param_trans + (ini_param_trans - tx) * 10
        gamma, tx, ty = res.x[0], real_optimization_parameters(res.x[1], initial_param=ini_param_trans_x, initial_step=initial_step), real_optimization_parameters(res.x[2], initial_param=ini_param_trans_y, initial_step=initial_step)
        rotation_matrix = matrix([[cos(gamma), - sin(gamma), 0],
                                  [sin(gamma), cos(gamma), 0],
                                  [0, 0, 1]])
        translation_array = matrix([tx, ty, 0])

    return rotation_matrix, translation_array


def getRigidTransformFromLandmarks(points_dest, points_src, constraints='Tx_Ty_Tz_Rx_Ry_Rz', verbose=0, path_qc='./'):
    """
    Compute affine transformation to register landmarks
    :param points_src:
    :param points_dest:
    :param constraints:
    :param verbose: 0, 1, 2
    :return: rotsc_matrix, translation_array, points_src_reg, points_src_barycenter
    """
    # TODO: check input constraints
    from scipy.optimize import minimize

    # initialize default parameters
    init_param = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    # initialize parameters for optimizer
    init_param_optimizer = []
    # initialize dictionary to relate constraints index to dof
    dict_dof = {'Tx': 0, 'Ty': 1, 'Tz': 2, 'Rx': 3, 'Ry': 4, 'Rz': 5, 'Sx': 6, 'Sy': 7, 'Sz': 8}
    # extract constraints
    list_constraints = constraints.split('_')
    # loop across constraints and build initial_parameters
    for i in range(len(list_constraints)):
        init_param_optimizer.append(init_param[dict_dof[list_constraints[i]]])

    # launch optimizer
    # res = minimize(minimize_transform, x0=init_param_optimizer, args=(points_src, points_dest, constraints), method='Nelder-Mead', tol=1e-8, options={'xtol': 1e-8, 'ftol': 1e-8, 'maxiter': 10000, 'maxfev': 10000, 'disp': show})
    res = minimize(minimize_transform, x0=init_param_optimizer, args=(points_dest, points_src, constraints), method='Powell', tol=1e-8, options={'xtol': 1e-8, 'ftol': 1e-8, 'maxiter': 100000, 'maxfev': 100000, 'disp': verbose})
    # res = minimize(minAffineTransform, x0=initial_parameters, args=points, method='COBYLA', tol=1e-8, options={'tol': 1e-8, 'rhobeg': 0.1, 'maxiter': 100000, 'catol': 0, 'disp': show})
    # loop across constraints and update dof
    dof = init_param
    for i in range(len(list_constraints)):
        dof[dict_dof[list_constraints[i]]] = res.x[i]
    # convert dof to more intuitive variables
    tx, ty, tz, alpha, beta, gamma, scx, scy, scz = dof[0], dof[1], dof[2], dof[3], dof[4], dof[5], dof[6], dof[7], dof[8]
    # convert results to intuitive variables
    # tx, ty, tz, alpha, beta, gamma, scx, scy, scz = res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], res.x[6], res.x[7], res.x[8]
    # build translation matrix
    translation_array = matrix([tx, ty, tz])
    # build rotation matrix
    rotation_matrix = matrix([[cos(alpha) * cos(beta), cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma), cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)],
                              [sin(alpha) * cos(beta), sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma), sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma)],
                              [-sin(beta), cos(beta) * sin(gamma), cos(beta) * cos(gamma)]])
    # build scaling matrix
    scaling_matrix = matrix([[scx, 0.0, 0.0], [0.0, scy, 0.0], [0.0, 0.0, scz]])
    # compute rotation+scaling matrix
    rotsc_matrix = scaling_matrix * rotation_matrix
    # compute center of mass from moving points (src)
    points_src_barycenter = mean(points_src, axis=0)
    # apply transformation to moving points (src)
    points_src_reg = ((rotsc_matrix * (matrix(points_src) - points_src_barycenter).T).T + points_src_barycenter) + translation_array
    # display results
    sct.printv('Matrix:\n' + str(rotation_matrix))
    sct.printv('Center:\n' + str(points_src_barycenter))
    sct.printv('Translation:\n' + str(translation_array))

    if verbose == 2:
        import matplotlib
        # use Agg to prevent display
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        points_src_matrix = matrix(points_src)
        points_dest_matrix = matrix(points_dest)

        number_points = len(points_dest)

        ax.scatter([points_dest_matrix[i, 0] for i in range(0, number_points)],
                   [points_dest_matrix[i, 1] for i in range(0, number_points)],
                   [points_dest_matrix[i, 2] for i in range(0, number_points)], c='g', marker='+', s=500, label='dest')
        ax.scatter([points_src_matrix[i, 0] for i in range(0, number_points)],
                   [points_src_matrix[i, 1] for i in range(0, number_points)],
                   [points_src_matrix[i, 2] for i in range(0, number_points)], c='r', label='src')
        ax.scatter([points_src_reg[i, 0] for i in range(0, number_points)],
                   [points_src_reg[i, 1] for i in range(0, number_points)],
                   [points_src_reg[i, 2] for i in range(0, number_points)], c='b', label='src_reg')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('auto')
        plt.legend()
        # plt.show()
        plt.savefig(path_qc + 'getRigidTransformFromLandmarks_plot.png')

        fig2 = plt.figure()
        plt.plot(sse_results)
        plt.grid()
        plt.title('#Iterations: ' + str(res.nit) + ', #FuncEval: ' + str(res.nfev) + ', Error: ' + str(res.fun))
        plt.show()
        plt.savefig(path_qc + 'getRigidTransformFromLandmarks_iterations.png')

    # transform numpy matrix to list structure because it is easier to handle
    points_src_reg = points_src_reg.tolist()

    return rotsc_matrix, translation_array, points_src_reg, points_src_barycenter
