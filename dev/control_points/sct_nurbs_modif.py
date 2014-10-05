#!/usr/bin/env python

## @package sct_nurbs
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
#=======================================================================================================================
import sys
import math
# check if needed Python libraries are already installed or not
try:
    from numpy import *
except ImportError:
    print '--- numpy not installed! ---'
    sys.exit(2)
try:
    from scipy.interpolate import interp1d
except ImportError:
    print '--- scipy not installed! ---'
    sys.exit(2)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class NURBS():
    def __init__(self, size, div=5, degre=3, precision=1000, liste=None, sens=False):
        """
        Ce constructeur initialise une NURBS et la construit.
        Si la variable sens est True : On construit la courbe en fonction des points de controle
        Si la variable sens est False : On reconstruit les points de controle en fonction de la courbe
        """
        self.error = 0
        self.degre = degre+1
        self.sens = sens
        self.div = div
        self.pointsControle = []
        self.pointsControleRelatif = []
        self.courbe3D = []
        self.courbe3D_deriv = []
        self.nbControle = 10  ### correspond au nombre de points de controle calcules.
        self.precision = precision

        if sens:                  #### si on donne les points de controle#####
            if type(liste[0][0]).__name__ == 'list':
                self.pointsControle = liste
            else:
                self.pointsControle.append(liste)
            for li in self.pointsControle:
                [[P_x,P_y,P_z],[P_x_d,P_y_d,P_z_d]] = self.construct3D(li,degre)
                self.courbe3D.append([[P_x[i],P_y[i],P_z[i]] for i in len(P_x)])
                self.courbe3D_deriv.append([[P_x_d[i],P_y_d[i],P_z_d[i]] for i in len(P_x_d)])
        else:
            # La liste est sous la forme d'une liste de points
            P_x = [x[0] for x in liste]
            P_y = [x[1] for x in liste]
            P_z = [x[2] for x in liste]
            
            self.nbControle = size/div
            print self.nbControle
            #self.nbControle = len(P_z)/5  ## ordre 3 -> len(P_z)/10, 4 -> len/7, 5-> len/5   permet d'obtenir une bonne approximation sans trop "interpoler" la courbe
                                          #   increase nbeControle if "short data"
            self.pointsControle = self.reconstructGlobalApproximation(P_x,P_y,P_z,self.degre,self.nbControle)
            print self.pointsControle
            self.courbe3D, self.courbe3D_deriv= self.construct3D(self.pointsControle,self.degre,self.precision)
            # if self.error == 1:
            #     return 1

    def getControle(self):
        return self.pointsControle

    def setControle(self,pointsControle):
        self.pointsControle = pointsControle


    def getCourbe3D(self):
        if self.error == 1:
            return 1
        return self.courbe3D

    def getCourbe3D_deriv(self):
        return self.courbe3D_deriv

    # Multiplie deux polynomes
    def multipolynome(self,polyA,polyB):
        result = [];
        for r in polyB:
            temp = polyA*r[0]
            result.append([temp, r[-1]])
        return result

    def N(self,i,k,x):
        global Nik_temp
        if k==1:
            tab = [[poly1d(1),i+1]]
        else:
            tab = []
            den_g = x[i+k-1]-x[i]
            den_d = x[i+k]-x[i+1]
            if den_g != 0:
                if Nik_temp[i][k-1] == -1:
                    Nik_temp[i][k-1] = self.N(i,k-1,x)
                tab_b = self.multipolynome(poly1d([1/den_g,-x[i]/den_g]),Nik_temp[i][k-1])
                tab.extend(tab_b)
            if den_d != 0:
                if Nik_temp[i+1][k-1] == -1:
                    Nik_temp[i+1][k-1] = self.N(i+1,k-1,x)
                tab_d = self.multipolynome(poly1d([-1/den_d,x[i+k]/den_d]),Nik_temp[i+1][k-1])
                tab.extend(tab_d)

        return tab

    def Np(self,i,k,x):
        global Nik_temp_deriv, Nik_temp
        if k==1:
            tab = [[poly1d(0),i+1]]
        else:
            tab = []
            den_g = x[i+k-1]-x[i]
            den_d = x[i+k]-x[i+1]
            if den_g != 0:
                if Nik_temp_deriv[i][-1] == -1:
                    Nik_temp_deriv[i][-1] = self.N(i,k-1,x)
                tab_b = self.multipolynome(poly1d([k/den_g]),Nik_temp_deriv[i][-1])
                tab.extend(tab_b)
            if den_d != 0:
                if Nik_temp_deriv[i+1][-1] == -1 :
                    Nik_temp_deriv[i+1][-1] = self.N(i+1,k-1,x)
                tab_d = self.multipolynome(poly1d([-k/den_d]),Nik_temp_deriv[i+1][-1])
                tab.extend(tab_d)

        return tab

    def evaluateN(self,Ni,t,x):
        result = 0;
        for Ni_temp in Ni:
            if x[Ni_temp[-1]-1] <= t <= x[Ni_temp[-1]]:
                result += Ni_temp[0](t)
        return result


    def calculX3D(self,P,k):
        n = len(P)-1
        c = []
        sumC = 0
        for i in xrange(n):
            dist = math.sqrt((P[i+1][0]-P[i][0])**2 + (P[i+1][1]-P[i][1])**2 + (P[i+1][2]-P[i][2])**2)
            c.append(dist)
            sumC += dist

        x = [0]*k
        sumCI = 0
        for i in xrange(n-k+1):
            sumCI += c[i+1]
            value = (n-k+2)/sumC*((i+1)*c[i+1]/(n-k+2) + sumCI)
            x.append(value)

        x.extend([n-k+2]*k)
        return x

    def construct3D(self,P,k,prec): # P point de controles
        global Nik_temp, Nik_temp_deriv
        n = len(P) # Nombre de points de controle - 1

        # Calcul des xi
        x = self.calculX3D(P,k)

        # Calcul des coefficients N(i,k)
        Nik_temp = [[-1 for j in xrange(k)] for i in xrange(n)]
        for i in xrange(n):
            Nik_temp[i][-1] = self.N(i,k,x)
        Nik = []
        for i in xrange(n):
            Nik.append(Nik_temp[i][-1])


        #Calcul des Nik,p'
        Nik_temp_deriv = [[-1] for i in xrange(n)]
        for i in xrange(n):
            Nik_temp_deriv[i][-1]=self.Np(i,k,x)
        Nikp=[]
        for i in xrange(n):
            Nikp.append(Nik_temp_deriv[i][-1])


        # Calcul de la courbe
        param = linspace(x[0],x[-1],prec)
        P_x,P_y,P_z = [],[],[] # coord fitees
        P_x_d,P_y_d,P_z_d=[],[],[] #derivees
        for i in xrange(len(param)):
            sum_num_x,sum_num_y,sum_num_z,sum_den = 0,0,0,0
            sum_num_x_der,sum_num_y_der,sum_num_z_der,sum_den_der = 0,0,0,0

            for l in xrange(n-k+1): # utilisation que des points non nuls
                if x[l+k-1]<=param[i]<x[l+k]:
                    debut = l
            fin = debut+k-1

            for j,point in enumerate(P[debut:fin+1]):
                j = j+debut
                N_temp = self.evaluateN(Nik[j],param[i],x)
                N_temp_deriv = self.evaluateN(Nikp[j],param[i],x)
                sum_num_x += N_temp*point[0]
                sum_num_y += N_temp*point[1]
                sum_num_z += N_temp*point[2]
                sum_den += N_temp
                sum_num_x_der += N_temp_deriv*point[0]
                sum_num_y_der += N_temp_deriv*point[1]
                sum_num_z_der += N_temp_deriv*point[2]
                sum_den_der += N_temp_deriv
            P_x.append(sum_num_x/sum_den) # sum_den = 1 !
            P_y.append(sum_num_y/sum_den)
            P_z.append(sum_num_z/sum_den)
            P_x_d.append(sum_num_x_der)
            P_y_d.append(sum_num_y_der)
            P_z_d.append(sum_num_z_der)

        #on veut que les coordonnees fittees aient le meme z que les coordonnes de depart. on se ramene donc a des entiers et on moyenne en x et y  .
        P_x=array(P_x)
        P_y=array(P_y)
        P_x_d=array(P_x_d)
        P_y_d=array(P_y_d)
        P_z_d=array(P_z_d)
        P_z=array([int(round(P_z[i])) for i in range(0,len(P_z))])
    
        #not perfect but works (if "enough" points), in order to deal with missing z slices

        if max(P_z)-min(P_z)+1<=2000:
            for i in range (min(P_z),max(P_z)+1,1):
                if (i in P_z) is False :
                    print ' Missing z slice '
                    print i
                    P_z = insert(P_z,where(P_z==i-1)[-1][-1]+1,i)
                    P_x = insert(P_x,where(P_z==i-1)[-1][-1]+1,(P_x[where(P_z==i-1)[-1][-1]+1-1]+P_x[where(P_z==i-1)[-1][-1]+1+1])/2)
                    P_y = insert(P_y,where(P_z==i-1)[-1][-1]+1,(P_y[where(P_z==i-1)[-1][-1]+1-1]+P_y[where(P_z==i-1)[-1][-1]+1+1])/2)
                    P_x_d = insert(P_x_d,where(P_z==i-1)[-1][-1]+1,(P_x_d[where(P_z==i-1)[-1][-1]+1-1]+P_x_d[where(P_z==i-1)[-1][-1]+1+1])/2)
                    P_y_d = insert(P_y_d,where(P_z==i-1)[-1][-1]+1,(P_y_d[where(P_z==i-1)[-1][-1]+1-1]+P_y_d[where(P_z==i-1)[-1][-1]+1+1])/2)
                    P_z_d = insert(P_z_d,where(P_z==i-1)[-1][-1]+1,(P_z_d[where(P_z==i-1)[-1][-1]+1-1]+P_z_d[where(P_z==i-1)[-1][-1]+1+1])/2)
        else: 
            self.error = 1
            return 0, 0

        coord_mean = array([[mean(P_x[P_z==i]),mean(P_y[P_z==i]),i] for i in range(min(P_z),max(P_z)+1,1)])

        P_x=coord_mean[:,:][:,0]
        P_y=coord_mean[:,:][:,1]

        coord_mean_d = array([[mean(P_x_d[P_z==i]),mean(P_y_d[P_z==i]),mean(P_z_d[P_z==i])] for i in range(min(P_z),max(P_z)+1,1)])

        P_z=coord_mean[:,:][:,2]
    
        P_x_d=coord_mean_d[:,:][:,0]
        P_y_d=coord_mean_d[:,:][:,1]
        P_z_d=coord_mean_d[:,:][:,2]
        #print P_x_d,P_y_d,P_z_d

        # p=len(P_x)/3
        # n=1
        # #plotting a tangent
        # p1 = [P_x[p],P_y[p],P_z[p]]
        # p2 = [P_x[p]+n*P_x_d[p],P_y[p]+n*P_y_d[p],P_z[p]+n*P_z_d[p]]
               #### 3D plot
        # fig1 = plt.figure()
        # ax = Axes3D(fig1)
        # #ax.plot(x_centerline,y_centerline,z_centerline,zdir='z')
        # ax.plot(P_x,P_y,P_z,zdir='z')
        # #ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],zdir='z')
        # #ax.plot(x_centerline_fit_der,y_centerline_fit_der,z_centerline_fit_der,zdir='z')
        # plt.show()


        #print 'Construction effectuee'
        return [P_x,P_y,P_z], [P_x_d,P_y_d,P_z_d]

    def Tk(self,k,Q,Nik,ubar,u):
        return Q[k] - self.evaluateN(Nik[-1],ubar,u)*Q[-1] - self.evaluateN(Nik[0],ubar,u)*Q[0]

    def reconstructGlobalApproximation(self,P_x,P_y,P_z,p,n):
        # p = degre de la NURBS
        # n = nombre de points de controle desires
        global Nik_temp
        m = len(P_x)

        # Calcul des chords
        di = 0
        for k in xrange(m-1):
            di += math.sqrt((P_x[k+1]-P_x[k])**2 + (P_y[k+1]-P_y[k])**2 + (P_z[k+1]-P_z[k])**2)
        u = [0]*p
        ubar = [0]
        for k in xrange(m-1):
            ubar.append(ubar[-1]+math.sqrt((P_x[k+1]-P_x[k])**2 + (P_y[k+1]-P_y[k])**2 + (P_z[k+1]-P_z[k])**2)/di)
        d = (m+1)/(n-p+1)
        for j in xrange(n-p):
            i = int((j+1)*d)
            alpha = (j+1)*d-i
            u.append((1-alpha)*ubar[i-1]+alpha*ubar[i])
        u.extend([1]*p)

        Nik_temp = [[-1 for j in xrange(p)] for i in xrange(n)]
        for i in xrange(n):
            Nik_temp[i][-1] = self.N(i,p,u)
        Nik = []
        for i in xrange(n):
            Nik.append(Nik_temp[i][-1])

        R = []
        for k in xrange(m-1):
            Rtemp = []
            den = 0
            for Ni in Nik:
                den += self.evaluateN(Ni,ubar[k],u)
            for i in xrange(n-1):
                Rtemp.append(self.evaluateN(Nik[i],ubar[k],u)/den)
            R.append(Rtemp)
        R = matrix(R)

        # calcul des denominateurs par ubar
        denU = []
        for k in xrange(m-1):
            temp = 0
            for Ni in Nik:
                temp += self.evaluateN(Ni,ubar[k],u)
            denU.append(temp)
        Tx = []
        for i in xrange(n-1):
            somme = 0
            for k in xrange(m-1):
                somme += self.evaluateN(Nik[i],ubar[k],u)*self.Tk(k,P_x,Nik,ubar[k],u)/denU[k]
            Tx.append(somme)
        Tx = matrix(Tx)

        Ty = []
        for i in xrange(n-1):
            somme = 0
            for k in xrange(m-1):
                somme += self.evaluateN(Nik[i],ubar[k],u)*self.Tk(k,P_y,Nik,ubar[k],u)/denU[k]
            Ty.append(somme)
        Ty = matrix(Ty)

        Tz = []
        for i in xrange(n-1):
            somme = 0
            for k in xrange(m-1):
                somme += self.evaluateN(Nik[i],ubar[k],u)*self.Tk(k,P_z,Nik,ubar[k],u)/denU[k]
            Tz.append(somme)
        Tz = matrix(Tz)

        P_xb = (R.T*R).I*Tx.T
        P_yb = (R.T*R).I*Ty.T
        P_zb = (R.T*R).I*Tz.T
        P = [[P_xb[i,0],P_yb[i,0],P_zb[i,0]] for i in range(len(P_xb))]
        # On modifie les premiers et derniers points
        P[0][0],P[0][1],P[0][2] = P_x[0],P_y[0],P_z[0]
        P[-1][0],P[-1][1],P[-1][2] = P_x[-1],P_y[-1],P_z[-1]

        #print 'Reconstruction effectuee'
        return P

    def reconstructGlobalInterpolation(self,P_x,P_y,P_z,p):  ### now in 3D
        global Nik_temp
        n = 13
        l = len(P_x)
        newPx = P_x[::int(round(l/(n-1)))]
        newPy = P_y[::int(round(l/(n-1)))]
        newPz = P_y[::int(round(l/(n-1)))]
        newPx.append(P_x[-1])
        newPy.append(P_y[-1])
        newPz.append(P_z[-1])
        n = len(newPx)

        # Calcul du vecteur de noeuds
        di = 0
        for k in xrange(n-1):
            di += math.sqrt((newPx[k+1]-newPx[k])**2 + (newPy[k+1]-newPy[k])**2 +(newPz[k+1]-newPz[k])**2)
        u = [0]*p
        ubar = [0]
        for k in xrange(n-1):
            ubar.append(ubar[-1]+math.sqrt((newPx[k+1]-newPx[k])**2 + (newPy[k+1]-newPy[k])**2 + (newPz[k+1]-newPz[k])**2)/di)
        for j in xrange(n-p):
            sumU = 0
            for i in xrange(p):
                sumU = sumU + ubar[j+i]
            u.append(sumU/p)
        u.extend([1]*p)

        # Construction des fonctions basiques
        Nik_temp = [[-1 for j in xrange(p)] for i in xrange(n)]
        for i in xrange(n):
            Nik_temp[i][-1] = self.N(i,p,u)
        Nik = []
        for i in xrange(n):
            Nik.append(Nik_temp[i][-1])

        # Construction des matrices
        M = []
        for i in xrange(n):
            ligneM = []
            for j in xrange(n):
                ligneM.append(self.evaluateN(Nik[j],ubar[i],u))
            M.append(ligneM)
        M = matrix(M)

        # Matrice des points interpoles
        Qx = matrix(newPx).T
        Qy = matrix(newPy).T
        Qz = matrix(newPz).T

        # Calcul des points de controle
        P_xb = M.I*Qx
        P_yb = M.I*Qy
        P_zb = M.I*Qz

        return [[P_xb[i,0],P_yb[i,0],P_zb[i,0]] for i in range(len(P_xb))]
