#!/usr/bin/env python

# @package sct_nurbs
#
# - python class. Approximate or interpolate a 3D curve with a B-Spline curve from either a set of data points or a set of control points
#
#
# Description about how the function works:
#
# If a set of data points is given, it generates a B-spline that either approximates the curve in the least square sens, or interpolates the curve.
# It also computes the derivative of the 3D curve.
# getCourbe3D() returns the 3D fitted curve. The fitted z coordonate corresponds to the initial z, and the x and y are averaged for a given z
# getCourbe3D_deriv() returns the derivative of the 3D fitted curve also averaged along z-axis
#
# USAGE
# ---------------------------------------------------------------------------------------
# from sct_nurbs import *
# nurbs=NURBS(degree,precision,data)
#
# MANDATORY ARGUMENTS
# ---------------------------------------------------------------------------------------
#   degree          the degree of the fitting B-spline curve
#   precision       number of points before averaging data
#   data            3D list [x,y,z] of the data requiring fitting
#
# OPTIONAL ARGUMENTS
# ---------------------------------------------------------------------------------------
#
#
#
# EXAMPLES
# ---------------------------------------------------------------------------------------
#   from sct_nurbs import *
#   nurbs = NURBS(3,1000,[[x_centerline[n],y_centerline[n],z_centerline[n]] for n in range(len(x_centerline))])
#   P = nurbs.getCourbe3D()
#   x_centerline_fit = P[0]
#   y_centerline_fit = P[1]
#   z_centerline_fit = P[2]
#   D = nurbs.getCourbe3D_deriv()
#   x_centerline_fit_der = D[0]
#   y_centerline_fit_der = D[1]
#   z_centerline_fit_der = D[2]

#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# - scipy: <http://www.scipy.org>
# - numpy: <http://www.numpy.org>
#
# EXTERNAL SOFTWARE
#
# none
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Authors: Benjamin De Leener, Julien Touati
# Modified: 2014-07-01
#
# License: see the LICENSE.TXT
# ============================================================================================================
# check if needed Python libraries are already installed or not

import math
from sys import exit

import numpy as np
import sct_utils as sct

class NURBS():
    def __init__(self, degre=3, precision=1000, liste=None, sens=False, nbControl=None, verbose=1, tolerance=0.01, maxControlPoints=50, all_slices=True, twodim=False, weights=True):
        """
        Ce constructeur initialise une NURBS et la construit.
        Si la variable sens est True : On construit la courbe en fonction des points de controle
        Si la variable sens est False : On reconstruit les points de controle en fonction de la courbe
        """
        self.degre = degre + 1
        self.sens = sens
        self.pointsControle = []
        self.pointsControleRelatif = []
        self.courbe3D, self.courbe3D_deriv = [], []
        self.courbe2D, self.courbe2D_deriv = [], []
        self.nbControle = 10  # correspond au nombre de points de controle calcules.
        self.precision = precision
        self.tolerance = tolerance  # in mm
        self.maxControlPoints = maxControlPoints
        self.verbose = verbose
        self.all_slices = all_slices
        self.twodim = twodim

        if sens:                  # si on donne les points de controle#####
            if type(liste[0][0]).__name__ == 'list':
                self.pointsControle = liste
            else:
                self.pointsControle.append(liste)
            for li in self.pointsControle:
                if twodim:
                    [[P_x, P_y], [P_x_d, P_y_d]] = self.construct2D(li, degre, self.precision)
                    self.courbe2D.append([[P_x[i], P_y[i]] for i in len(P_x)])
                    self.courbe2D_deriv.append([[P_x_d[i], P_y_d[i]] for i in len(P_x_d)])
                else:
                    [[P_x, P_y, P_z], [P_x_d, P_y_d, P_z_d]] = self.construct3D(li, degre, self.precision)
                    self.courbe3D.append([[P_x[i], P_y[i], P_z[i]] for i in len(P_x)])
                    self.courbe3D_deriv.append([[P_x_d[i], P_y_d[i], P_z_d[i]] for i in len(P_x_d)])
        else:
            # La liste est sous la forme d'une liste de points
            P_x = [x[0] for x in liste]
            P_y = [x[1] for x in liste]
            if not twodim:
                P_z = [x[2] for x in liste]
                self.P_z = P_z

            if nbControl is None:
                # self.nbControl = len(P_z)/5  ## ordre 3 -> len(P_z)/10, 4 -> len/7, 5-> len/5   permet d'obtenir une bonne approximation sans trop "interpoler" la courbe
                # compute the ideal number of control points based on tolerance
                error_curve = 1000.0
                self.nbControle = self.degre + 1
                nb_points = len(P_x)
                if self.nbControle > nb_points - 1:
                    sct.printv('ERROR : There are too few points to compute. The number of points of the curve must be strictly superior to degre +2 which is: ', self.nbControle, '. Either change degre to a lower value, either add points to the curve.')
                    exit(2)

                # compute weights based on curve density
                w = [1.0] * len(P_x)
                if weights:
                    if not twodim:
                        for i in range(1, len(P_x) - 1):
                            dist_before = math.sqrt((P_x[i - 1] - P_x[i])**2 + (P_y[i - 1] - P_y[i])**2 + (P_z[i - 1] - P_z[i])**2)
                            dist_after = math.sqrt((P_x[i] - P_x[i + 1])**2 + (P_y[i] - P_y[i + 1])**2 + (P_z[i] - P_z[i + 1])**2)
                            w[i] = (dist_before + dist_after) / 2.0
                    else:
                        for i in range(1, len(P_x) - 1):
                            dist_before = math.sqrt((P_x[i - 1] - P_x[i])**2 + (P_y[i - 1] - P_y[i])**2)
                            dist_after = math.sqrt((P_x[i] - P_x[i + 1])**2 + (P_y[i] - P_y[i + 1])**2)
                            w[i] = (dist_before + dist_after) / 2.0
                    w[0], w[-1] = w[1], w[-2]

                list_param_that_worked = []
                last_error_curve = 0.0
                second_last_error_curve = 0.0
                while self.nbControle < len(P_x) and self.nbControle <= self.maxControlPoints:
                    if abs(error_curve - last_error_curve) <= self.tolerance and abs(error_curve - second_last_error_curve) <= self.tolerance and error_curve <= last_error_curve and error_curve <= last_error_curve:
                        break

                    second_last_error_curve = last_error_curve
                    last_error_curve = error_curve

                    # compute the nurbs based on input data and number of controle points
                    if verbose >= 1:
                        sct.printv('Test: # of control points = ' + str(self.nbControle))
                    try:
                        if not twodim:
                            self.pointsControle = self.reconstructGlobalApproximation(P_x, P_y, P_z, self.degre, self.nbControle, w)
                            self.courbe3D, self.courbe3D_deriv = self.construct3D(self.pointsControle, self.degre, self.precision / 3)  # generate curve with low resolution
                        else:
                            self.pointsControle = self.reconstructGlobalApproximation2D(P_x, P_y, self.degre, self.nbControle, w)
                            self.courbe2D, self.courbe2D_deriv = self.construct2D(self.pointsControle, self.degre, self.precision / 3)

                        # compute error between the input data and the nurbs
                        error_curve = 0.0
                        if not twodim:
                            for i in range(0, len(P_x)):
                                min_dist = 10000.0
                                for k in range(0, len(self.courbe3D[0])):
                                    dist = (self.courbe3D[0][k] - P_x[i])**2 + (self.courbe3D[1][k] - P_y[i])**2 + (self.courbe3D[2][k] - P_z[i])**2
                                    if dist < min_dist:
                                        min_dist = dist
                                error_curve += min_dist
                        else:
                            for i in range(0, len(P_x)):
                                min_dist = 10000.0
                                for k in range(0, len(self.courbe2D[0])):
                                    dist = (self.courbe2D[0][k] - P_x[i])**2 + (self.courbe2D[1][k] - P_y[i])**2
                                    if dist < min_dist:
                                        min_dist = dist
                                error_curve += min_dist
                        error_curve /= float(len(P_x))

                        if verbose >= 1:
                            sct.printv('Error on approximation = ' + str(round(error_curve, 2)) + ' mm')

                        # Create a list of parameters that have worked in order to call back the last one that has worked
                        list_param_that_worked.append([self.nbControle, self.pointsControle, error_curve])

                    except Exception as ex:
                        if verbose >= 1:
                            sct.log.error(ex)
                        error_curve = last_error_curve + 10000.0

                    # prepare for next iteration
                    self.nbControle += 1
                self.nbControle -= 1  # last addition does not count

                # self.courbe3D, self.courbe3D_deriv = self.construct3D(self.pointsControle, self.degre, self.precision)  # generate curve with hig resolution
                # select number of control points that gives the best results
                list_param_that_worked_sorted = sorted(list_param_that_worked, key=lambda list_param_that_worked: list_param_that_worked[2])
                nbControle_that_last_worked = list_param_that_worked_sorted[0][0]
                pointsControle_that_last_worked = list_param_that_worked_sorted[0][1]
                self.error_curve_that_last_worked = list_param_that_worked_sorted[0][2]
                if not twodim:
                    self.courbe3D, self.courbe3D_deriv = self.construct3D_uniform(pointsControle_that_last_worked, self.degre, self.precision)  # generate curve with hig resolution
                else:
                    self.courbe2D, self.courbe2D_deriv = self.construct2D(pointsControle_that_last_worked, self.degre, self.precision)
                self.pointsControle = pointsControle_that_last_worked

                if verbose >= 1:
                    if self.nbControle != nbControle_that_last_worked:
                        sct.printv("The fitting of the curve was done using {} control points: the number that gave the best results. \nError on approximation = {} mm".format(nbControle_that_last_worked, round(self.error_curve_that_last_worked, 2)))
                    else:
                        sct.printv('Number of control points of the optimal NURBS = {}'.format(self.nbControle))
            else:
                if verbose >= 1:
                    sct.printv('In NURBS we get nurbs_ctl_points = ', nbControl)
                w = [1.0] * len(P_x)
                self.nbControl = nbControl  # increase nbeControle if "short data"
                if not twodim:
                    self.pointsControle = self.reconstructGlobalApproximation(P_x, P_y, P_z, self.degre, self.nbControle, w)
                    self.courbe3D, self.courbe3D_deriv = self.construct3D(self.pointsControle, self.degre, self.precision)
                else:
                    self.pointsControle = self.reconstructGlobalApproximation2D(P_x, P_y, self.degre, self.nbControle, w)
                    self.courbe2D, self.courbe2D_deriv = self.construct2D(self.pointsControle, self.degre, self.precision)

    def getControle(self):
        return self.pointsControle

    def setControle(self, pointsControle):
        self.pointsControle = pointsControle

    def getCourbe3D(self):
        return self.courbe3D

    def getCourbe3D_deriv(self):
        return self.courbe3D_deriv

    def getCourbe2D(self):
        return self.courbe2D

    def getCourbe2D_deriv(self):
        return self.courbe2D_deriv

    # Multiplie deux polynomes
    def multipolynome(self, polyA, polyB):
        result = []
        for r in polyB:
            temp = polyA * r[0]
            result.append([temp, r[-1]])
        return result

    def N(self, i, k, x):
        global Nik_temp
        if k == 1:
            tab = [[np.poly1d(1), i + 1]]
        else:
            tab = []
            den_g = x[i + k - 1] - x[i]
            den_d = x[i + k] - x[i + 1]
            if den_g != 0:
                if Nik_temp[i][k - 1] == -1:
                    Nik_temp[i][k - 1] = self.N(i, k - 1, x)
                tab_b = self.multipolynome(np.poly1d([1 / den_g, -x[i] / den_g]), Nik_temp[i][k - 1])
                tab.extend(tab_b)
            if den_d != 0:
                if Nik_temp[i + 1][k - 1] == -1:
                    Nik_temp[i + 1][k - 1] = self.N(i + 1, k - 1, x)
                tab_d = self.multipolynome(np.poly1d([-1 / den_d, x[i + k] / den_d]), Nik_temp[i + 1][k - 1])
                tab.extend(tab_d)

        return tab

    def Np(self, i, k, x):
        global Nik_temp_deriv, Nik_temp
        if k == 1:
            tab = [[np.poly1d(0), i + 1]]
        else:
            tab = []
            den_g = x[i + k - 1] - x[i]
            den_d = x[i + k] - x[i + 1]
            if den_g != 0:
                if Nik_temp_deriv[i][-1] == -1:
                    Nik_temp_deriv[i][-1] = self.N(i, k - 1, x)
                tab_b = self.multipolynome(np.poly1d([k / den_g]), Nik_temp_deriv[i][-1])
                tab.extend(tab_b)
            if den_d != 0:
                if Nik_temp_deriv[i + 1][-1] == -1:
                    Nik_temp_deriv[i + 1][-1] = self.N(i + 1, k - 1, x)
                tab_d = self.multipolynome(np.poly1d([-k / den_d]), Nik_temp_deriv[i + 1][-1])
                tab.extend(tab_d)

        return tab

    def evaluateN(self, Ni, t, x):
        result = 0
        for Ni_temp in Ni:
            if x[Ni_temp[-1] - 1] <= t <= x[Ni_temp[-1]]:
                result += Ni_temp[0](t)
        return result

    def calculX3D(self, P, k):
        n = len(P) - 1
        c = []
        sumC = 0
        for i in range(n):
            dist = math.sqrt((P[i + 1][0] - P[i][0])**2 + (P[i + 1][1] - P[i][1])**2 + (P[i + 1][2] - P[i][2])**2)
            c.append(dist)
            sumC += dist

        x = [0] * k
        sumCI = 0
        for i in range(n - k + 1):
            sumCI += c[i + 1]
            x.append((n - k + 2) / sumC * ((i + 1) * c[i + 1] / (n - k + 2) + sumCI))

        x.extend([n - k + 2] * k)

        return x

    def calculX2D(self, P, k):
        n = len(P) - 1
        c = []
        sumC = 0
        for i in range(n):
            dist = math.sqrt((P[i + 1][0] - P[i][0])**2 + (P[i + 1][1] - P[i][1])**2)
            c.append(dist)
            sumC += dist

        x = [0] * k
        sumCI = 0
        for i in range(n - k + 1):
            sumCI += c[i + 1]
            x.append((n - k + 2) / sumC * ((i + 1) * c[i + 1] / (n - k + 2) + sumCI))

        x.extend([n - k + 2] * k)

        return x

    def construct3D(self, P, k, prec):  # P point de controles
        global Nik_temp, Nik_temp_deriv
        n = len(P)  # Nombre de points de controle - 1

        # Calcul des xi
        x = self.calculX3D(P, k)

        # Calcul des coefficients N(i,k)
        Nik_temp = [[-1 for j in range(k)] for i in range(n)]
        for i in range(n):
            Nik_temp[i][-1] = self.N(i, k, x)
        Nik = []
        for i in range(n):
            Nik.append(Nik_temp[i][-1])

        # Calcul des Nik,p'
        Nik_temp_deriv = [[-1] for i in range(n)]
        for i in range(n):
            Nik_temp_deriv[i][-1] = self.Np(i, k, x)
        Nikp = []
        for i in range(n):
            Nikp.append(Nik_temp_deriv[i][-1])

        # Calcul de la courbe
        param = np.linspace(x[0], x[-1], prec)
        P_x, P_y, P_z = [], [], []  # coord fitees
        P_x_d, P_y_d, P_z_d = [], [], []  # derivees
        for i in range(len(param)):
            sum_num_x, sum_num_y, sum_num_z, sum_den = 0, 0, 0, 0
            sum_num_x_der, sum_num_y_der, sum_num_z_der, sum_den_der = 0, 0, 0, 0

            for l in range(n - k + 1):  # utilisation que des points non nuls
                if x[l + k - 1] <= param[i] < x[l + k]:
                    debut = l
            fin = debut + k - 1

            for j, point in enumerate(P[debut:fin + 1]):
                j = j + debut
                N_temp = self.evaluateN(Nik[j], param[i], x)
                N_temp_deriv = self.evaluateN(Nikp[j], param[i], x)
                sum_num_x += N_temp * point[0]
                sum_num_y += N_temp * point[1]
                sum_num_z += N_temp * point[2]
                sum_den += N_temp
                sum_num_x_der += N_temp_deriv * point[0]
                sum_num_y_der += N_temp_deriv * point[1]
                sum_num_z_der += N_temp_deriv * point[2]
                sum_den_der += N_temp_deriv

            P_x.append(sum_num_x / sum_den)  # sum_den = 1 !
            P_y.append(sum_num_y / sum_den)
            P_z.append(sum_num_z / sum_den)
            P_x_d.append(sum_num_x_der)
            P_y_d.append(sum_num_y_der)
            P_z_d.append(sum_num_z_der)

            if sum_den <= 0.05:
                raise Exception('WARNING: NURBS instability -> wrong reconstruction')

        P_x = [P_x[i] for i in np.argsort(P_z)]
        P_y = [P_y[i] for i in np.argsort(P_z)]
        P_x_d = [P_x_d[i] for i in np.argsort(P_z)]
        P_y_d = [P_y_d[i] for i in np.argsort(P_z)]
        P_z_d = [P_z_d[i] for i in np.argsort(P_z)]
        P_z = np.sort(P_z)

        # on veut que les coordonnees fittees aient le meme z que les coordonnes de depart. on se ramene donc a des entiers et on moyenne en x et y  .
        P_x = np.array(P_x)
        P_y = np.array(P_y)
        P_x_d = np.array(P_x_d)
        P_y_d = np.array(P_y_d)
        P_z_d = np.array(P_z_d)

        if self.all_slices:
            P_z = np.array([int(round(P_z[i])) for i in range(0, len(P_z))])

            # not perfect but works (if "enough" points), in order to deal with missing z slices
            for i in range(min(P_z), max(P_z) + 1, 1):
                if i not in P_z:
                    # sct.printv(' Missing z slice ')
                    # sct.printv(i)
                    P_z_temp = np.insert(P_z, np.where(P_z == i - 1)[-1][-1] + 1, i)
                    P_x_temp = np.insert(P_x, np.where(P_z == i - 1)[-1][-1] + 1, (P_x[np.where(P_z == i - 1)[-1][-1]] + P_x[np.where(P_z == i - 1)[-1][-1] + 1]) / 2)
                    P_y_temp = np.insert(P_y, np.where(P_z == i - 1)[-1][-1] + 1, (P_y[np.where(P_z == i - 1)[-1][-1]] + P_y[np.where(P_z == i - 1)[-1][-1] + 1]) / 2)
                    P_x_d_temp = np.insert(P_x_d, np.where(P_z == i - 1)[-1][-1] + 1, (P_x_d[np.where(P_z == i - 1)[-1][-1]] + P_x_d[np.where(P_z == i - 1)[-1][-1] + 1]) / 2)
                    P_y_d_temp = np.insert(P_y_d, np.where(P_z == i - 1)[-1][-1] + 1, (P_y_d[np.where(P_z == i - 1)[-1][-1]] + P_y_d[np.where(P_z == i - 1)[-1][-1] + 1]) / 2)
                    P_z_d_temp = np.insert(P_z_d, np.where(P_z == i - 1)[-1][-1] + 1, (P_z_d[np.where(P_z == i - 1)[-1][-1]] + P_z_d[np.where(P_z == i - 1)[-1][-1] + 1]) / 2)
                    P_x, P_y, P_z, P_x_d, P_y_d, P_z_d = P_x_temp, P_y_temp, P_z_temp, P_x_d_temp, P_y_d_temp, P_z_d_temp

            coord_mean = np.array([[np.mean(P_x[P_z == i]), np.mean(P_y[P_z == i]), i] for i in range(min(P_z), max(P_z) + 1, 1)])

            P_x = coord_mean[:, :][:, 0]
            P_y = coord_mean[:, :][:, 1]

            coord_mean_d = np.array([[np.mean(P_x_d[P_z == i]), np.mean(P_y_d[P_z == i]), np.mean(P_z_d[P_z == i])] for i in range(min(P_z), max(P_z) + 1, 1)])

            P_z = coord_mean[:, :][:, 2]

            P_x_d = coord_mean_d[:, :][:, 0]
            P_y_d = coord_mean_d[:, :][:, 1]
            P_z_d = coord_mean_d[:, :][:, 2]

        return [P_x, P_y, P_z], [P_x_d, P_y_d, P_z_d]

    def construct2D(self, P, k, prec):  # P point de controles
        global Nik_temp, Nik_temp_deriv
        n = len(P)  # Nombre de points de controle - 1

        # Calcul des xi
        x = self.calculX2D(P, k)

        # Calcul des coefficients N(i,k)
        Nik_temp = [[-1 for j in range(k)] for i in range(n)]
        for i in range(n):
            Nik_temp[i][-1] = self.N(i, k, x)
        Nik = []
        for i in range(n):
            Nik.append(Nik_temp[i][-1])

        # Calcul des Nik,p'
        Nik_temp_deriv = [[-1] for i in range(n)]
        for i in range(n):
            Nik_temp_deriv[i][-1] = self.Np(i, k, x)
        Nikp = []
        for i in range(n):
            Nikp.append(Nik_temp_deriv[i][-1])

        # Calcul de la courbe
        param = np.linspace(x[0], x[-1], prec)
        P_x, P_y = [], []  # coord fitees
        P_x_d, P_y_d = [], []  # derivees
        for i in range(len(param)):
            sum_num_x, sum_num_y, sum_den = 0, 0, 0
            sum_num_x_der, sum_num_y_der, sum_den_der = 0, 0, 0

            for l in range(n - k + 1):  # utilisation que des points non nuls
                if x[l + k - 1] <= param[i] < x[l + k]:
                    debut = l
            fin = debut + k - 1

            for j, point in enumerate(P[debut:fin + 1]):
                j = j + debut
                N_temp = self.evaluateN(Nik[j], param[i], x)
                N_temp_deriv = self.evaluateN(Nikp[j], param[i], x)
                sum_num_x += N_temp * point[0]
                sum_num_y += N_temp * point[1]
                sum_den += N_temp
                sum_num_x_der += N_temp_deriv * point[0]
                sum_num_y_der += N_temp_deriv * point[1]
                sum_den_der += N_temp_deriv

            P_x.append(sum_num_x / sum_den)  # sum_den = 1 !
            P_y.append(sum_num_y / sum_den)
            P_x_d.append(sum_num_x_der)
            P_y_d.append(sum_num_y_der)

            if sum_den <= 0.05:
                raise Exception('WARNING: NURBS instability -> wrong reconstruction')

        P_x = [P_x[i] for i in np.argsort(P_y)]
        P_x_d = [P_x_d[i] for i in np.argsort(P_y)]
        P_y_d = [P_y_d[i] for i in np.argsort(P_y)]
        P_y = np.sort(P_y)

        # on veut que les coordonnees fittees aient le meme z que les coordonnes de depart. on se ramene donc a des entiers et on moyenne en x et y  .
        P_x = np.array(P_x)
        P_y = np.array(P_y)
        P_x_d = np.array(P_x_d)
        P_y_d = np.array(P_y_d)

        if self.all_slices:
            P_y = np.array([int(round(P_y[i])) for i in range(0, len(P_y))])

            # not perfect but works (if "enough" points), in order to deal with missing z slices
            for i in range(min(P_y), max(P_y) + 1, 1):
                if i not in P_y:
                    P_y_temp = np.insert(P_y, np.where(P_y == i - 1)[-1][-1] + 1, i)
                    P_x_temp = np.insert(P_x, np.where(P_y == i - 1)[-1][-1] + 1, (P_x[np.where(P_y == i - 1)[-1][-1]] + P_x[np.where(P_y == i - 1)[-1][-1] + 1]) / 2)
                    P_x_d_temp = np.insert(P_x_d, np.where(P_y == i - 1)[-1][-1] + 1, (P_x_d[np.where(P_y == i - 1)[-1][-1]] + P_x_d[np.where(P_y == i - 1)[-1][-1] + 1]) / 2)
                    P_y_d_temp = np.insert(P_y_d, np.where(P_y == i - 1)[-1][-1] + 1, (P_y_d[np.where(P_y == i - 1)[-1][-1]] + P_y_d[np.where(P_y == i - 1)[-1][-1] + 1]) / 2)
                    P_x, P_y, P_x_d, P_y_d = P_x_temp, P_y_temp, P_x_d_temp, P_y_d_temp

            coord_mean = np.array([[np.mean(P_x[P_y == i]), i] for i in range(min(P_y), max(P_y) + 1, 1)])

            P_x = coord_mean[:, :][:, 0]

            coord_mean_d = np.array([[np.mean(P_x_d[P_y == i]), np.mean(P_y_d[P_y == i])] for i in range(min(P_y), max(P_y) + 1, 1)])

            P_y = coord_mean[:, :][:, 1]

            P_x_d = coord_mean_d[:, :][:, 0]
            P_y_d = coord_mean_d[:, :][:, 1]

        return [P_x, P_y], [P_x_d, P_y_d]

    def Tk(self, k, Q, Nik, ubar, u):
        return Q[k] - self.evaluateN(Nik[-1], ubar, u) * Q[-1] - self.evaluateN(Nik[0], ubar, u) * Q[0]

    def isXinY(self, y, x):
        result = True
        for i in range(0, len(y) - 1):
            if y[i] - y[i + 1] != 0.0:
                result_temp = False
                for j in range(0, len(x)):
                    if y[i] - y[i + 1] != 0.0 and y[i] <= x[j] <= y[i + 1]:
                        result_temp = True
                        break
                result = result and result_temp
        return result

    def reconstructGlobalApproximation(self, P_x, P_y, P_z, p, n, w):
        # p = degre de la NURBS
        # n = nombre de points de controle desires
        # w is the weigth on each point P
        global Nik_temp
        m = len(P_x)

        # Calcul des chords
        di = 0.0
        for k in range(m - 1):
            di += math.sqrt((P_x[k + 1] - P_x[k])**2 + (P_y[k + 1] - P_y[k])**2 + (P_z[k + 1] - P_z[k])**2)
        ubar = [0]
        for k in range(m - 1):
            # ubar.append((k+1)/float(m))  # uniform method
            # ubar.append(ubar[-1]+abs((P_x[k+1]-P_x[k])**2 + (P_y[k+1]-P_y[k])**2 + (P_z[k+1]-P_z[k])**2)/di)  # chord length method
            ubar.append(ubar[-1] + math.sqrt((P_x[k + 1] - P_x[k])**2 + (P_y[k + 1] - P_y[k])**2 + (P_z[k + 1] - P_z[k])**2) / di)  # centripetal method

        # the knot vector should reflect the distribution of ubar
        d = (m + 1) / (n - p + 1)
        u_nonuniform = [0.0] * p
        for j in range(n - p):
            i = int((j + 1) * d)
            alpha = (j + 1) * d - i
            u_nonuniform.append((1 - alpha) * ubar[i - 1] + alpha * ubar[i])
        u_nonuniform.extend([1.0] * p)

        # the knot vector can also is uniformly distributed
        u_uniform = [0.0] * p
        for j in range(n - p):
            u_uniform.append((float(j) + 1) / float(n - p))
        u_uniform.extend([1.0] * p)

        # The only condition for NURBS to work here is that there is at least one point P_.. in each knot space.
        # The uniform knot vector does not ensure this condition while the nonuniform knot vector ensure it but lack of uniformity in case of variable density of points.
        # We need a compromise between the two methods: the knot vector must be as uniform as possible, with at least one point between each pair of knots.
        # New algo:
        # knotVector = uniformKnotVector
        # while isKnotSpaceEmpty:
        #     knotVector += gamma * (nonuniformKnotVector - nonuniformKnotVector)
        #     # where gamma is a ratio [0,1] multiplier of an integer: 1/gamma = int
        u_uniform = np.array(u_uniform)
        u_nonuniform = np.array(u_nonuniform)
        u = np.array(u_uniform, copy=True)
        gamma = 1.0 / 10.0
        n_iter = 0
        while not self.isXinY(y=u, x=ubar) and n_iter <= 10000:
            u += gamma * (u_nonuniform - u_uniform)
            n_iter += 1

        Nik_temp = [[-1 for j in range(p)] for i in range(n)]  # noqa
        for i in range(n):
            Nik_temp[i][-1] = self.N(i, p, u)
        Nik = []
        for i in range(n):
            Nik.append(Nik_temp[i][-1])

        R = []
        for k in range(m - 1):
            Rtemp = []
            den = 0
            for Ni in Nik:
                den += self.evaluateN(Ni, ubar[k], u)
            for i in range(n - 1):
                Rtemp.append(self.evaluateN(Nik[i], ubar[k], u) / den)
            R.append(Rtemp)
        R = np.matrix(R)

        # create W diagonal matrix
        W = np.diag(w[0:-1])

        # calcul des denominateurs par ubar
        denU = []
        for k in range(m - 1):
            temp = 0
            for Ni in Nik:
                temp += self.evaluateN(Ni, ubar[k], u)
            denU.append(temp)
        Tx = []
        for i in range(n - 1):
            somme = 0
            for k in range(m - 1):
                somme += w[k] * self.evaluateN(Nik[i], ubar[k], u) * self.Tk(k, P_x, Nik, ubar[k], u) / denU[k]
            Tx.append(somme)
        Tx = np.matrix(Tx)

        Ty = []
        for i in range(n - 1):
            somme = 0
            for k in range(m - 1):
                somme += w[k] * self.evaluateN(Nik[i], ubar[k], u) * self.Tk(k, P_y, Nik, ubar[k], u) / denU[k]
            Ty.append(somme)
        Ty = np.matrix(Ty)

        Tz = []
        for i in range(n - 1):
            somme = 0
            for k in range(m - 1):
                somme += w[k] * self.evaluateN(Nik[i], ubar[k], u) * self.Tk(k, P_z, Nik, ubar[k], u) / denU[k]
            Tz.append(somme)
        Tz = np.matrix(Tz)

        P_xb = (R.T * W * R).I * Tx.T
        P_yb = (R.T * W * R).I * Ty.T
        P_zb = (R.T * W * R).I * Tz.T

        # Modification of first and last control points
        P_xb[0], P_yb[0], P_zb[0] = P_x[0], P_y[0], P_z[0]
        P_xb[-1], P_yb[-1], P_zb[-1] = P_x[-1], P_y[-1], P_z[-1]

        # At this point, we need to check if the control points are in a correct range or if there were instability.
        # Typically, control points should be far from the data points. One way to do so is to ensure that the
        from numpy import std
        std_factor = 10.0
        std_Px, std_Py, std_Pz, std_x, std_y, std_z = std(P_xb), std(P_yb), std(P_zb), std(np.array(P_x)), std(np.array(P_y)), std(np.array(P_z))
        if std_x >= 0.1 and std_y >= 0.1 and std_z >= 0.1 and (std_Px > std_factor * std_x or std_Py > std_factor * std_y or std_Pz > std_factor * std_z):
            raise Exception('WARNING: NURBS instability -> wrong control points')

        P = [[P_xb[i, 0], P_yb[i, 0], P_zb[i, 0]] for i in range(len(P_xb))]

        return P

    def reconstructGlobalApproximation2D(self, P_x, P_y, p, n, w):
        # p = degre de la NURBS
        # n = nombre de points de controle desires
        # w is the weigth on each point P
        global Nik_temp
        m = len(P_x)

        # Calcul des chords
        di = 0.0
        for k in range(m - 1):
            di += math.sqrt((P_x[k + 1] - P_x[k])**2 + (P_y[k + 1] - P_y[k])**2)
        ubar = [0]
        for k in range(m - 1):
            # ubar.append((k+1)/float(m))  # uniform method
            # ubar.append(ubar[-1]+abs((P_x[k+1]-P_x[k])**2 + (P_y[k+1]-P_y[k])**2 + (P_z[k+1]-P_z[k])**2)/di)  # chord length method
            ubar.append(ubar[-1] + math.sqrt((P_x[k + 1] - P_x[k])**2 + (P_y[k + 1] - P_y[k])**2) / di)  # centripetal method

        # the knot vector should reflect the distribution of ubar
        d = (m + 1) / (n - p + 1)
        u_nonuniform = [0.0] * p
        for j in range(n - p):
            i = int((j + 1) * d)
            alpha = (j + 1) * d - i
            u_nonuniform.append((1 - alpha) * ubar[i - 1] + alpha * ubar[i])
        u_nonuniform.extend([1.0] * p)

        # the knot vector can also is uniformly distributed
        u_uniform = [0.0] * p
        for j in range(n - p):
            u_uniform.append((float(j) + 1) / float(n - p))
        u_uniform.extend([1.0] * p)

        # The only condition for NURBS to work here is that there is at least one point P_.. in each knot space.
        # The uniform knot vector does not ensure this condition while the nonuniform knot vector ensure it but lack of uniformity in case of variable density of points.
        # We need a compromise between the two methods: the knot vector must be as uniform as possible, with at least one point between each pair of knots.
        # New algo:
        # knotVector = uniformKnotVector
        # while isKnotSpaceEmpty:
        #     knotVector += gamma * (nonuniformKnotVector - nonuniformKnotVector)
        #     # where gamma is a ratio [0,1] multiplier of an integer: 1/gamma = int
        u_uniform = np.array(u_uniform)
        u_nonuniform = np.array(u_nonuniform)
        u = np.array(u_uniform, copy=True)
        gamma = 1.0 / 10.0
        n_iter = 0
        while not self.isXinY(y=u, x=ubar) and n_iter <= 10000:
            u += gamma * (u_nonuniform - u_uniform)
            n_iter += 1

        Nik_temp = [[-1 for j in range(p)] for i in range(n)]  # noqa
        for i in range(n):
            Nik_temp[i][-1] = self.N(i, p, u)
        Nik = []
        for i in range(n):
            Nik.append(Nik_temp[i][-1])

        R = []
        for k in range(m - 1):
            Rtemp = []
            den = 0
            for Ni in Nik:
                den += self.evaluateN(Ni, ubar[k], u)
            for i in range(n - 1):
                Rtemp.append(self.evaluateN(Nik[i], ubar[k], u) / den)
            R.append(Rtemp)
        R = np.matrix(R)

        # create W diagonal matrix
        W = np.diag(w[0:-1])

        # calcul des denominateurs par ubar
        denU = []
        for k in range(m - 1):
            temp = 0
            for Ni in Nik:
                temp += self.evaluateN(Ni, ubar[k], u)
            denU.append(temp)
        Tx = []
        for i in range(n - 1):
            somme = 0
            for k in range(m - 1):
                somme += w[k] * self.evaluateN(Nik[i], ubar[k], u) * self.Tk(k, P_x, Nik, ubar[k], u) / denU[k]
            Tx.append(somme)
        Tx = np.matrix(Tx)

        Ty = []
        for i in range(n - 1):
            somme = 0
            for k in range(m - 1):
                somme += w[k] * self.evaluateN(Nik[i], ubar[k], u) * self.Tk(k, P_y, Nik, ubar[k], u) / denU[k]
            Ty.append(somme)
        Ty = np.matrix(Ty)

        P_xb = (R.T * W * R).I * Tx.T
        P_yb = (R.T * W * R).I * Ty.T

        # Modification of first and last control points
        P_xb[0], P_yb[0] = P_x[0], P_y[0]
        P_xb[-1], P_yb[-1] = P_x[-1], P_y[-1]

        # At this point, we need to check if the control points are in a correct range or if there were instability.
        # Typically, control points should be far from the data points. One way to do so is to ensure that the
        from numpy import std
        std_factor = 10.0
        std_Px, std_Py, std_x, std_y = std(P_xb), std(P_yb), std(np.array(P_x)), std(np.array(P_y))
        if std_x >= 0.1 and std_y >= 0.1 and (std_Px > std_factor * std_x or std_Py > std_factor * std_y):
            raise Exception('WARNING: NURBS instability -> wrong control points')

        P = [[P_xb[i, 0], P_yb[i, 0]] for i in range(len(P_xb))]

        return P

    def reconstructGlobalInterpolation(self, P_x, P_y, P_z, p):  # now in 3D
        global Nik_temp
        n = 13
        l = len(P_x)
        newPx = P_x[::int(round(l / (n - 1)))]
        newPy = P_y[::int(round(l / (n - 1)))]
        newPz = P_y[::int(round(l / (n - 1)))]
        newPx.append(P_x[-1])
        newPy.append(P_y[-1])
        newPz.append(P_z[-1])
        n = len(newPx)

        # Calcul du vecteur de noeuds
        di = 0
        for k in range(n - 1):
            di += math.sqrt((newPx[k + 1] - newPx[k])**2 + (newPy[k + 1] - newPy[k])**2 + (newPz[k + 1] - newPz[k])**2)
        u = [0] * p
        ubar = [0]
        for k in range(n - 1):
            ubar.append(ubar[-1] + math.sqrt((newPx[k + 1] - newPx[k])**2 + (newPy[k + 1] - newPy[k])**2 + (newPz[k + 1] - newPz[k])**2) / di)
        for j in range(n - p):
            sumU = 0
            for i in range(p):
                sumU = sumU + ubar[j + i]
            u.append(sumU / p)
        u.extend([1] * p)

        # Construction des fonctions basiques
        Nik_temp = [[-1 for j in range(p)] for i in range(n)]
        for i in range(n):
            Nik_temp[i][-1] = self.N(i, p, u)
        Nik = []
        for i in range(n):
            Nik.append(Nik_temp[i][-1])

        # Construction des matrices
        M = []
        for i in range(n):
            ligneM = []
            for j in range(n):
                ligneM.append(self.evaluateN(Nik[j], ubar[i], u))
            M.append(ligneM)
        M = np.matrix(M)

        # Matrice des points interpoles
        Qx = np.matrix(newPx).T
        Qy = np.matrix(newPy).T
        Qz = np.matrix(newPz).T

        # Calcul des points de controle
        P_xb = M.I * Qx
        P_yb = M.I * Qy
        P_zb = M.I * Qz

        return [[P_xb[i, 0], P_yb[i, 0], P_zb[i, 0]] for i in range(len(P_xb))]

    def compute_curve_from_parametrization(self, P, k, x, Nik, Nikp, param):
        n = len(P)  # Nombre de points de controle - 1
        P_x, P_y, P_z = [], [], []  # coord fitees
        P_x_d, P_y_d, P_z_d = [], [], []  # derivees
        for i in range(len(param)):
            sum_num_x, sum_num_y, sum_num_z, sum_den = 0, 0, 0, 0
            sum_num_x_der, sum_num_y_der, sum_num_z_der, sum_den_der = 0, 0, 0, 0

            for l in range(n - k + 1):  # utilisation que des points non nuls
                if x[l + k - 1] <= param[i] < x[l + k]:
                    debut = l
            fin = debut + k - 1

            for j, point in enumerate(P[debut:fin + 1]):
                j = j + debut
                N_temp = self.evaluateN(Nik[j], param[i], x)
                N_temp_deriv = self.evaluateN(Nikp[j], param[i], x)
                sum_num_x += N_temp * point[0]
                sum_num_y += N_temp * point[1]
                sum_num_z += N_temp * point[2]
                sum_den += N_temp
                sum_num_x_der += N_temp_deriv * point[0]
                sum_num_y_der += N_temp_deriv * point[1]
                sum_num_z_der += N_temp_deriv * point[2]
                sum_den_der += N_temp_deriv

            P_x.append(sum_num_x / sum_den)  # sum_den = 1 !
            P_y.append(sum_num_y / sum_den)
            P_z.append(sum_num_z / sum_den)
            P_x_d.append(sum_num_x_der)
            P_y_d.append(sum_num_y_der)
            P_z_d.append(sum_num_z_der)

            if sum_den <= 0.05:
                raise Exception('WARNING: NURBS instability -> wrong reconstruction')

        P_x = [P_x[i] for i in np.argsort(P_z)]
        P_y = [P_y[i] for i in np.argsort(P_z)]
        P_x_d = [P_x_d[i] for i in np.argsort(P_z)]
        P_y_d = [P_y_d[i] for i in np.argsort(P_z)]
        P_z_d = [P_z_d[i] for i in np.argsort(P_z)]
        P_z = np.sort(P_z)

        # on veut que les coordonnees fittees aient le meme z que les coordonnes de depart. on se ramene donc a des entiers et on moyenne en x et y  .
        P_x = np.array(P_x)
        P_y = np.array(P_y)
        P_x_d = np.array(P_x_d)
        P_y_d = np.array(P_y_d)
        P_z_d = np.array(P_z_d)
        return P_x, P_y, P_z, P_x_d, P_y_d, P_z_d

    def construct3D_uniform(self, P, k, prec):  # P point de controles
        global Nik_temp, Nik_temp_deriv
        n = len(P)  # Nombre de points de controle - 1

        # Calcul des xi
        x = self.calculX3D(P, k)

        # Calcul des coefficients N(i,k)
        Nik_temp = [[-1 for j in range(k)] for i in range(n)]
        for i in range(n):
            Nik_temp[i][-1] = self.N(i, k, x)
        Nik = []
        for i in range(n):
            Nik.append(Nik_temp[i][-1])

        # Calcul des Nik,p'
        Nik_temp_deriv = [[-1] for i in range(n)]
        for i in range(n):
            Nik_temp_deriv[i][-1] = self.Np(i, k, x)
        Nikp = []
        for i in range(n):
            Nikp.append(Nik_temp_deriv[i][-1])

        # Calcul de la courbe
        # reparametrization of the curve
        import numpy as np
        param = np.linspace(x[0], x[-1], prec)
        P_x, P_y, P_z, P_x_d, P_y_d, P_z_d = self.compute_curve_from_parametrization(P, k, x, Nik, Nikp, param)
        from msct_types import Centerline
        centerline = Centerline(P_x, P_y, P_z, P_x_d, P_y_d, P_z_d)
        distances_between_points = centerline.progressive_length
        range_points = np.linspace(0.0, 1.0, prec)
        dist_curved = np.zeros(prec)
        for i in range(1, prec):
            dist_curved[i] = dist_curved[i - 1] + distances_between_points[i - 1] / centerline.length
        param = x[0] + (x[-1] - x[0]) * np.interp(range_points, dist_curved, range_points)
        P_x, P_y, P_z, P_x_d, P_y_d, P_z_d = self.compute_curve_from_parametrization(P, k, x, Nik, Nikp, param)

        if self.all_slices:
            P_z = np.array([int(round(P_z[i])) for i in range(0, len(P_z))])

            # not perfect but works (if "enough" points), in order to deal with missing z slices
            for i in range(min(P_z), max(P_z) + 1, 1):
                if i not in P_z:
                    # sct.printv(' Missing z slice ')
                    # sct.printv(i)
                    P_z_temp = np.insert(P_z, np.where(P_z == i - 1)[-1][-1] + 1, i)
                    P_x_temp = np.insert(P_x, np.where(P_z == i - 1)[-1][-1] + 1,
                                         (P_x[np.where(P_z == i - 1)[-1][-1]] + P_x[np.where(P_z == i - 1)[-1][-1] + 1]) / 2)
                    P_y_temp = np.insert(P_y, np.where(P_z == i - 1)[-1][-1] + 1,
                                         (P_y[np.where(P_z == i - 1)[-1][-1]] + P_y[np.where(P_z == i - 1)[-1][-1] + 1]) / 2)
                    P_x_d_temp = np.insert(P_x_d, np.where(P_z == i - 1)[-1][-1] + 1, (
                        P_x_d[np.where(P_z == i - 1)[-1][-1]] + P_x_d[np.where(P_z == i - 1)[-1][-1] + 1]) / 2)
                    P_y_d_temp = np.insert(P_y_d, np.where(P_z == i - 1)[-1][-1] + 1, (
                        P_y_d[np.where(P_z == i - 1)[-1][-1]] + P_y_d[np.where(P_z == i - 1)[-1][-1] + 1]) / 2)
                    P_z_d_temp = np.insert(P_z_d, np.where(P_z == i - 1)[-1][-1] + 1, (
                        P_z_d[np.where(P_z == i - 1)[-1][-1]] + P_z_d[np.where(P_z == i - 1)[-1][-1] + 1]) / 2)
                    P_x, P_y, P_z, P_x_d, P_y_d, P_z_d = P_x_temp, P_y_temp, P_z_temp, P_x_d_temp, P_y_d_temp, P_z_d_temp

            coord_mean = np.array(
                [[np.mean(P_x[P_z == i]), np.mean(P_y[P_z == i]), i] for i in range(min(P_z), max(P_z) + 1, 1)])

            P_x = coord_mean[:, :][:, 0]
            P_y = coord_mean[:, :][:, 1]

            coord_mean_d = np.array([[np.mean(P_x_d[P_z == i]), np.mean(P_y_d[P_z == i]), np.mean(P_z_d[P_z == i])] for i in
                                     range(min(P_z), max(P_z) + 1, 1)])

            P_z = coord_mean[:, :][:, 2]

            P_x_d = coord_mean_d[:, :][:, 0]
            P_y_d = coord_mean_d[:, :][:, 1]
            P_z_d = coord_mean_d[:, :][:, 2]

            # check if slice should be in the result, based on self.P_z
            indexes_to_remove = []
            for i, ind_z in enumerate(P_z):
                if ind_z not in self.P_z:
                    indexes_to_remove.append(i)

            import numpy as np
            P_x = np.delete(P_x, indexes_to_remove)
            P_y = np.delete(P_y, indexes_to_remove)
            P_z = np.delete(P_z, indexes_to_remove)
            P_x_d = np.delete(P_x_d, indexes_to_remove)
            P_y_d = np.delete(P_y_d, indexes_to_remove)
            P_z_d = np.delete(P_z_d, indexes_to_remove)

        return [P_x, P_y, P_z], [P_x_d, P_y_d, P_z_d]
