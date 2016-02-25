#ifndef __DeformableModelBasicAdaptator__
#define __DeformableModelBasicAdaptator__

/*!
 * \file DeformableModelBasicAdaptator.h
 *
 * \brief Deformation of triangular mesh to gradient feature in an image.
 *
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include "Image3D.h"
#include <itkPoint.h>
#include <vector>
#include "Mesh.h"
#include "Vertex.h"
#include "util/Matrix3x3.h"
#include "util/MatrixNxM.h"
#include "SpinalCord.h"
#include <itkImageAlgorithm.h>
#include <itkImageFileReader.h>
#include <itkSingleValuedCostFunction.h>

#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>
#include <vtkPoints.h>

typedef itk::CovariantVector<double,3> PixelType;
typedef itk::Image< PixelType, 3 > ImageVectorType;
typedef ImageVectorType::IndexType Index;
typedef itk::Point< double, 3 > PointType;

/*!
 * \class FoncteurDeformableBasicLocalAdaptation
 * \brief Equation class for deformable model optimization
 *
 * This class compute the value and the derivatives of the energy equation used for the optimization of the deformable model.
 */
class FoncteurDeformableBasicLocalAdaptation: public itk::SingleValuedCostFunction
{
public:
	typedef itk::SingleValuedCostFunction	Superclass;
	typedef Superclass::ParametersType		ParametersType;
	typedef Superclass::DerivativeType		DerivativeType;

	FoncteurDeformableBasicLocalAdaptation(Image3D* image, Mesh* m, ParametersType &pointsInitiaux, int nbPoints) :
		image_(image), mesh_(m), pointsInitiaux_(pointsInitiaux), nbParametres_(3*nbPoints)
	{
        verbose_ = false;
        
		listeTriangles_ = m->getListTriangles();

		type_image_factor = image->getTypeImageFactor();

		InitParameters();

		trianglesBarycentre_.resize(listeTriangles_.size()/3);

		computeOptimalPoints(pointsInitiaux);

		listeTrianglesContenantPoint = mesh_->getConnectiviteTriangles();
		pointsVoisins = mesh_->getNeighbors();
	}

	virtual void GetDerivative (const ParametersType &parameters, DerivativeType &derivative) const
	{
		//vector<CVector3> der;
		unsigned int nbTrianglesInt = listeTriangles_.size(), nbTriangles = nbTrianglesInt/3, nbPoints = nbParametres_/3;
		vector<CVector3> trianglesBarycentre(nbTriangles);
		for (unsigned int i=0; i<nbTrianglesInt; i+=3)
		{
			trianglesBarycentre[(i+1)/3] = CVector3((parameters[3*listeTriangles_[i]]+parameters[3*listeTriangles_[i+1]]+parameters[3*listeTriangles_[i+2]])/3,
													(parameters[3*listeTriangles_[i]+1]+parameters[3*listeTriangles_[i+1]+1]+parameters[3*listeTriangles_[i+2]+1])/3,
													(parameters[3*listeTriangles_[i]+2]+parameters[3*listeTriangles_[i+1]+2]+parameters[3*listeTriangles_[i+2]+2])/3);
		}

		derivative = DerivativeType(nbParametres_);

		CVector3 index, gradient, laplacien, distancePoint, expect, ci1, c1, ci2, c2, point, voisin;
		for (int i=0; i<nbPoints; i++)
		{
			point(parameters[3*i],parameters[3*i+1],parameters[3*i+2]);
			ci1 = point - transformation_*CVector3(pointsInitiaux_[3*i],pointsInitiaux_[3*i+1],pointsInitiaux_[3*i+2]);
			c1 = CVector3::ZERO; c2 = CVector3::ZERO;
			for (unsigned int j=0; j<pointsVoisins[i].size(); j++) {
				voisin(parameters[3*pointsVoisins[i][j]],parameters[3*pointsVoisins[i][j]+1],parameters[3*pointsVoisins[i][j]+2]);
				c2 += point - voisin;
				c1 += ci1 - voisin + transformation_*CVector3(pointsInitiaux_[3*pointsVoisins[i][j]],pointsInitiaux_[3*pointsVoisins[i][j]+1],pointsInitiaux_[3*pointsVoisins[i][j]+2]);
			}
			derivative[3*i] = 2*alpha*c1[0] + 2*beta*c2[0]; // Internal energy
			derivative[3*i+1] = 2*alpha*c1[1] + 2*beta*c2[1];
			derivative[3*i+2] = 2*alpha*c1[2] + 2*beta*c2[2];
			for (int k=0; k<3; k++)
			{
				for (unsigned int j=0; j<listeTrianglesContenantPoint[i].size(); j++)
				{
					expect = listeXiOpt[listeTrianglesContenantPoint[i][j]];
					if (image_->TransformPhysicalPointToContinuousIndex(expect,index))
					{
						gradient = type_image_factor*image_->GetContinuousPixelVector(index).Normalize();
						distancePoint = listeXiOpt[listeTrianglesContenantPoint[i][j]]-trianglesBarycentre[listeTrianglesContenantPoint[i][j]];
						derivative[3*i+k] += -(2.0/3.0)*listeWi[listeTrianglesContenantPoint[i][j]]*gradient[k]*(gradient*distancePoint);
					}
				}
			}
			//der.push_back(CVector3(derivative[3*indexParam[i]],derivative[3*indexParam[i]+1],derivative[3*indexParam[i]+2]));
			//cout << derivative[i] << " " << derivative[i+1] << " " << derivative[i+2] << " " << endl;
		}
	}
 
	virtual MeasureType GetValue (const ParametersType &parameters) const
	{
		double result = 0.0, interne1 = 0.0, interne2 = 0.0, externe = 0.0;
		
		unsigned int nbTrianglesInt = listeTriangles_.size(), nbTriangles = nbTrianglesInt/3, nbPoints = nbParametres_/3;
		vector<CVector3> trianglesBarycentre(nbTriangles);
		for (unsigned int i=0; i<nbTrianglesInt; i+=3)
		{
			trianglesBarycentre[(i+1)/3] = CVector3((parameters[3*listeTriangles_[i]]+parameters[3*listeTriangles_[i+1]]+parameters[3*listeTriangles_[i+2]])/3,
													(parameters[3*listeTriangles_[i]+1]+parameters[3*listeTriangles_[i+1]+1]+parameters[3*listeTriangles_[i+2]+1])/3,
													(parameters[3*listeTriangles_[i]+2]+parameters[3*listeTriangles_[i+1]+2]+parameters[3*listeTriangles_[i+2]+2])/3);
		}

		CVector3 index, gradient, expect;
		for (unsigned int i=0; i<nbTriangles; i++)
		{
			expect = listeXiOpt[i];
			if (image_->TransformPhysicalPointToContinuousIndex(expect,index))
			{
				gradient = type_image_factor*image_->GetContinuousPixelVector(index).Normalize();
				externe += listeWi[i]*pow(gradient*(listeXiOpt[i]-trianglesBarycentre[i]),2);
			}
		}
		
		CVector3 c1, c2;

		for (int i=0; i<nbPoints; i++)
		{
			for (unsigned int j=0; j<pointsVoisins[i].size(); j++)
			{
				c2 = CVector3(parameters[3*i]-parameters[3*pointsVoisins[i][j]],parameters[3*i+1]-parameters[3*pointsVoisins[i][j]+1],parameters[3*i+2]-parameters[3*pointsVoisins[i][j]+2]);//point - voisin;
				c1 = c2 - transformation_*CVector3(pointsInitiaux_[3*i]-pointsInitiaux_[3*pointsVoisins[i][j]],pointsInitiaux_[3*i+1]-pointsInitiaux_[3*pointsVoisins[i][j]+1],pointsInitiaux_[3*i+2]-pointsInitiaux_[3*pointsVoisins[i][j]+2]);
				interne1 += pow(c1[0],2)+pow(c1[1],2)+pow(c1[2],2);
				interne2 += pow(c2[0],2)+pow(c2[1],2)+pow(c2[2],2);
			}
		}

		result = externe + alpha*interne1 + beta*interne2;
		//cout << "Energie Externe = " << externe << endl << "Energie Interne = " << interne1 << endl << "Energie Totale : " << result << endl;
		return result;
	}

	virtual unsigned int GetNumberOfParameters (void) const { return nbParametres_; }

	MeasureType GetInitialValue ()
	{
		return GetValue(pointsInitiaux_);
	}

	double getInitialNormDerivatives ()
	{
		DerivativeType derivative;
		GetDerivative (pointsInitiaux_, derivative);
		double normeGradient = 0.0;
		for (int i=0; i<nbParametres_; i++)
			normeGradient += pow(derivative[i],2);
		return sqrt(normeGradient);
	}
	
	void setInitialParameters(const ParametersType &pointsInitiaux)
	{
		listeXiOpt.clear();
		listeWi.clear();

		computeOptimalPoints(pointsInitiaux);
	}

	void computeOptimalPoints(const ParametersType &pointsInitiaux)
	{
		listeXiOpt.clear();
		listeWi.clear();

		unsigned int sizeBary = trianglesBarycentre_.size(), nbTrianglesInt = listeTriangles_.size();
		CVector3 point1, normale1, point2, normale2, point3, normale3;
		for (unsigned int i=0; i<nbTrianglesInt; i+=3) {
			point1(pointsInitiaux[3*listeTriangles_[i]],pointsInitiaux[3*listeTriangles_[i]+1],pointsInitiaux[3*listeTriangles_[i]+2]);
			point2(pointsInitiaux[3*listeTriangles_[i+1]],pointsInitiaux[3*listeTriangles_[i+1]+1],pointsInitiaux[3*listeTriangles_[i+1]+2]);
			point3(pointsInitiaux[3*listeTriangles_[i+2]],pointsInitiaux[3*listeTriangles_[i+2]+1],pointsInitiaux[3*listeTriangles_[i+2]+2]);
			trianglesBarycentre_[(i+1)/3] = Vertex((point1+point2+point3)/3,-((point1-point2)^(point1-point3)).Normalize());
		}
		
        // define startPos as meanRadius of the mesh
        int startPos = line_search;
        
		double resultCkMax = 0.0, resultCkMax_mask = 0.0, resultCk;
		CVector3 xi, ni, ck, ci, index, gradient;
		int k;
        
        Matrice imageDistance = Matrice(sizeBary,2*line_search+1);

		listeXiOpt.resize(sizeBary);
		listeWi.resize(sizeBary);
		vector<double> listeDistancePointsOpt(sizeBary);

		double threshold_distance_mask = 2.0;
		CVector3 point_position;
		double distance_point_from_mask = 0.0;
        
		for (unsigned int i=0; i<sizeBary; i++)
		{
			xi = trianglesBarycentre_[i].getPosition();
			ni = trianglesBarycentre_[i].getNormal();
			k = 0;
			resultCkMax = 0.0;

			bool has_point_mask = false;
			int k_point_mask = 0;
			double distance_min_mask = 100000.0;

			for (int j=-startPos; j<=line_search; j++)
			{
			    point_position = xi + j*deltaNormale*ni;
				if (image_->TransformPhysicalPointToContinuousIndex(point_position,index)) {
					resultCk = type_image_factor*ni*image_->GetContinuousPixelVector(index) - tradeOff*deltaNormale*deltaNormale*j*j;
                    //imageDistance(i,j+startPos) = resultCk;
					if (resultCk >= resultCkMax) {
						k = j;
						resultCkMax = resultCk;
					}

					// including information from correction mask
					for (int ind_mask=0; ind_mask<points_mask_correction_.size(); ind_mask++)
					{
					    distance_point_from_mask = sqrt((points_mask_correction_[ind_mask][0]-point_position[0])*(points_mask_correction_[ind_mask][0]-point_position[0]) + (points_mask_correction_[ind_mask][1]-point_position[1])*(points_mask_correction_[ind_mask][1]-point_position[1]) + (points_mask_correction_[ind_mask][2]-point_position[2])*(points_mask_correction_[ind_mask][2]-point_position[2]));
                        if (distance_point_from_mask <= threshold_distance_mask && distance_point_from_mask < distance_min_mask)
                        {
                            k_point_mask = j;
                            distance_min_mask = distance_point_from_mask;
                            resultCkMax_mask = 1000;
                            has_point_mask = true;
                        }
					}
				}
			}

			if (has_point_mask)
			{
                listeXiOpt[i] = xi + k_point_mask*deltaNormale*ni;
			    listeDistancePointsOpt[i] = k_point_mask*deltaNormale;
                listeWi[i] = max(0.0,resultCkMax_mask);
			}
			else
			{
			    listeXiOpt[i] = xi + k*deltaNormale*ni;
			    listeDistancePointsOpt[i] = k*deltaNormale;
                listeWi[i] = max(0.0,resultCkMax);
            }
		}
        
        /*vector<int> indexOptPoints = minimalPath(imageDistance, true);
        double posDist;
        for (unsigned int i=0; i<indexOptPoints.size(); i++)
        {
            xi = trianglesBarycentre_[i].getPosition();
            ni = trianglesBarycentre_[i].getNormal();
            posDist = indexOptPoints[i]-startPos;
            listeXiOpt[i] = xi + posDist*deltaNormale*ni;
            listeDistancePointsOpt[i] = posDist*deltaNormale;
            listeWi[i] = max(0.0,imageDistance(i,indexOptPoints[i]));
        }*/

		meanDistance = 0.0;
		meanAbsoluteDistance = 0.0;
		for (unsigned int i=0; i<sizeBary; i++) {
			meanDistance += listeDistancePointsOpt[i];
			meanAbsoluteDistance += abs(listeDistancePointsOpt[i]);
		}
		meanDistance /= (double)sizeBary;
		meanAbsoluteDistance /= (double)sizeBary;
		if (verbose_) {
            cout << "Most promising points mean distance [mm] = " << meanDistance << endl;
            cout << "Most promising points absolute mean distance [mm] = " << meanAbsoluteDistance << endl;
        }
	}
    
    /*void computeOptimalPoints(const ParametersType &pointsInitiaux)
    {
        listeXiOpt.clear();
        listeWi.clear();
        
        unsigned int sizeBary = trianglesBarycentre_.size(), nbTrianglesInt = listeTriangles_.size();
        CVector3 point1, normale1, point2, normale2, point3, normale3;
        for (unsigned int i=0; i<nbTrianglesInt; i+=3) {
            point1(pointsInitiaux[3*listeTriangles_[i]],pointsInitiaux[3*listeTriangles_[i]+1],pointsInitiaux[3*listeTriangles_[i]+2]);
            point2(pointsInitiaux[3*listeTriangles_[i+1]],pointsInitiaux[3*listeTriangles_[i+1]+1],pointsInitiaux[3*listeTriangles_[i+1]+2]);
            point3(pointsInitiaux[3*listeTriangles_[i+2]],pointsInitiaux[3*listeTriangles_[i+2]+1],pointsInitiaux[3*listeTriangles_[i+2]+2]);
            trianglesBarycentre_[(i+1)/3] = Vertex((point1+point2+point3)/3,-((point1-point2)^(point1-point3)).Normalize());
        }
        
        double resultCkMax = 0.0, resultCk;
        CVector3 xi, ni, ck, ci, index, gradient;
        int k;
        
        Matrice imageDistance = Matrice(sizeBary,2*line_search+1);
        
        listeXiOpt.resize(sizeBary);
        listeWi.resize(sizeBary);
        vector<double> listeDistancePointsOpt(sizeBary);
        
        for (unsigned int i=0; i<sizeBary; i++)
        {
            xi = trianglesBarycentre_[i].getPosition();
            ni = trianglesBarycentre_[i].getNormal();
            k = 0;
            resultCkMax = 0.0;
            for (int j=-line_search; j<=line_search; j++) {
                if (image_->TransformPhysicalPointToContinuousIndex(xi + j*deltaNormale*ni,index)) {
                    resultCk = type_image_factor*ni*image_->GetContinuousPixelVector(index) - tradeOff*deltaNormale*deltaNormale*j*j;
                    imageDistance(i,j+line_search) = resultCk;
                    if (resultCk >= resultCkMax) {
                        k = j;
                        resultCkMax = resultCk;
                    }
                }
            }
            listeXiOpt[i] = xi + k*deltaNormale*ni;
            listeDistancePointsOpt[i] = k*deltaNormale;
            listeWi[i] = max(0.0,resultCkMax);
        }
        
        vector<int> indexOptPoints = minimalPath(imageDistance, true);
        double posDist;
        for (unsigned int i=0; i<indexOptPoints.size(); i++)
        {
            xi = trianglesBarycentre_[i].getPosition();
            ni = trianglesBarycentre_[i].getNormal();
            posDist = indexOptPoints[i]-line_search;
            listeXiOpt[i] = xi + posDist*deltaNormale*ni;
            listeDistancePointsOpt[i] = posDist*deltaNormale;
            listeWi[i] = max(0.0,imageDistance(i,indexOptPoints[i]));
        }
        
        meanDistance = 0.0;
        meanAbsoluteDistance = 0.0;
        for (unsigned int i=0; i<sizeBary; i++) {
            meanDistance += listeDistancePointsOpt[i];
            meanAbsoluteDistance += abs(listeDistancePointsOpt[i]);
        }
        meanDistance /= (double)sizeBary;
        meanAbsoluteDistance /= (double)sizeBary;
        if (verbose_) {
            cout << "Most promising points mean distance [mm] = " << meanDistance << endl;
            cout << "Most promising points absolute mean distance [mm] = " << meanAbsoluteDistance << endl;
        }
    }*/
    
    vector<int> minimalPath(Matrice image, bool invert=true, double factx=sqrt(2))
    {
        /*% MINIMALPATH Recherche du chemin minimum de Haut vers le bas et de
         % bas vers le haut tel que dÈcrit par Luc Vincent 1998
         % [sR,sC,S] = MinimalPath(I,factx)
         %
         %   I     : Image d'entrÔøΩe dans laquelle on doit trouver le
         %           chemin minimal
         %   factx : Poids de linearite [1 10]
         %
         % Programme par : Ramnada Chav
         % Date : 22 fÈvrier 2007
         % ModifiÈ le 16 novembre 2007*/
        
        int m = image.getNombreLignes(); // x
        int n = image.getNombreColonnes(); // y
        
        if (invert)
        {
            double max_value = -1000000000;
            // compute max value in the matrix
            for (int x=0; x<m; x++)
            {
                for (int y=0; y<n; y++)
                {
                    if (image(x,y) > max_value)
                        max_value = image(x,y);
                }
            }
            // invert the matrix
            for (int x=0; x<m; x++)
            {
                for (int y=0; y<n; y++)
                    image(x,y) = max_value - image(x,y);
            }
        }
        
        // create image with high values J1
        // IMPORTANT: first slice of J1 and last slice of J2 must be set to 0...
        Matrice J1 = image, J2 = image, cPixel = image;
        for (int x=0; x<m; x++)
        {
            for (int y=0; y<n; y++)
            {
                if (x==0)
                    J1(x,y) = 0.0;
                else
                    J1(x,y) = 100000000.0;
                if (x==m-1)
                    J2(x,y) = 0.0;
                else
                    J2(x,y) = 100000000.0;
            }
        }
        
        // iterate on slice from slice 1 (start=0) to slice p-2. Basically, we avoid first and last slices.
        for (int row=1; row<m; row++)
        {
            // 1. extract pJ = the (slice-1)th slice of the image J1
            Matrice pJ = Matrice(1,n);
            for (int y=0; y<n; y++)
                pJ(0,y) = J1(row-1,y);
            
            // 2. extract cP = the (slice)th slice of the image cPixel
            Matrice cP = Matrice(1,n);
            for (int y=0; y<n; y++)
                cP(0,y) = cPixel(row,y);
            
            // 3. Create a matrix VI with 5 slices, that are exactly a repetition of cP without borders
            // multiply all elements of all slices of VI except the middle one by factx
            Matrice VI = Matrice(3,n-2);
            for (int i=0; i<3; i++)
            {
                for (int y=0; y<n-2; y++)
                {
                    if (i!=1)
                        VI(i,y) = cP(0,y+1)*factx;
                    else
                        VI(i,y) = cP(0,y+1);
                }
            }
            
            // 4. create a matrix of 5 slices, containing pJ(vectx-1,vecty),pJ(vectx,vecty-1),pJ(vectx,vecty),pJ(vectx,vecty+1),pJ(vectx+1,vecty) where vectx=2:m-1; and vecty=2:n-1;
            Matrice Jq = Matrice(3,n-2);
            for (int y=0; y<n-2; y++)
                Jq(0,y) = pJ(0,y);
            for (int y=0; y<n-2; y++)
                Jq(1,y) = pJ(0,y+1);
            for (int y=0; y<n-2; y++)
                Jq(2,y) = pJ(0,y+2);
            
            // 5. sum Jq and Vi voxel by voxel to produce JV
            Matrice JV = VI + Jq;
            
            // 6. replace each pixel of the (slice)th slice of J1 with the minimum value of the corresponding column in JV
            for (int y=0; y<n-2; y++)
            {
                double min_value = 100000000;
                for (int i=0; i<3; i++)
                {
                    if (JV(i,y) < min_value)
                        min_value = JV(i,y);
                }
                J1(row,y+1) = min_value;
            }
        }
        
        // iterate on slice from slice n-1 to slice 1. Basically, we avoid first and last slices.
        for (int row=m-2; row>=0; row--)
        {
            // 1. extract pJ = the (slice-1)th slice of the image J1
            Matrice pJ = Matrice(1,n);
            for (int y=0; y<n; y++)
                pJ(0,y) = J2(row+1,y);
            
            // 2. extract cP = the (slice)th slice of the image cPixel
            Matrice cP = Matrice(1,n);
            for (int y=0; y<n; y++)
                cP(0,y) = cPixel(row,y);
            
            // 3. Create a matrix VI with 5 slices, that are exactly a repetition of cP without borders
            // multiply all elements of all slices of VI except the middle one by factx
            Matrice VI = Matrice(3,n-2);
            for (int i=0; i<3; i++)
            {
                for (int y=0; y<n-2; y++)
                {
                    if (i!=1)
                        VI(i,y) = cP(0,y+1)*factx;
                    else
                        VI(i,y) = cP(0,y+1);
                }
            }
            
            // 4. create a matrix of 5 slices, containing pJ(vectx-1,vecty),pJ(vectx,vecty-1),pJ(vectx,vecty),pJ(vectx,vecty+1),pJ(vectx+1,vecty) where vectx=2:m-1; and vecty=2:n-1;
            Matrice Jq = Matrice(3,n-2);
            for (int y=0; y<n-2; y++)
                Jq(0,y) = pJ(0,y);
            for (int y=0; y<n-2; y++)
                Jq(1,y) = pJ(0,y+1);
            for (int y=0; y<n-2; y++)
                Jq(2,y) = pJ(0,y+2);
            
            // 5. sum Jq and Vi voxel by voxel to produce JV
            Matrice JV = VI + Jq;
            
            // 6. replace each pixel of the (slice)th slice of J1 with the minimum value of the corresponding column in JV
            for (int y=0; y<n-2; y++)
            {
                double min_value = 100000000;
                for (int i=0; i<3; i++)
                {
                    if (JV(i,y) < min_value)
                        min_value = JV(i,y);
                }
                J2(row,y+1) = min_value;
            }
        }
        
        // add J1 and J2 to produce "S" which is actually J1 here.
        Matrice S = J1 + J2;
        
        // Find the minimal value of S for each slice and create a binary image with all the coordinates
        // TO DO: the minimal path shouldn't be a pixelar path. It should be a continuous spline that is minimum.
        double val_temp;
        vector<int> list_index;
        for (int row=0; row<m; row++)
        {
            double min_value_S = 1000000000;
            int index_min = 0;
            for (int y=1; y<n-1; y++)
            {
                val_temp = S(row,y);
                if (val_temp < min_value_S)
                {
                    min_value_S = val_temp;
                    index_min = y;
                }
            }
            list_index.push_back(index_min);
        }
        
        return list_index;
    }

	void setTransformation(CMatrix4x4 m) { transformation_ = m; };

	vector<CVector3> getMostPromisingPoints() { return listeXiOpt; };

	void setDeltaNormale(double deltaNormale) { this->deltaNormale = deltaNormale; };
	void setTradeOff(double tradeOff) { this->tradeOff = tradeOff; };
    double getTradeOff() { return tradeOff; };
	void setLineSearchLength(double line_search) { this->line_search = line_search; };
    double getLineSearchLength() { return line_search; };
	void setAlpha(double alpha) { this->alpha = alpha; };
	void setBeta(double beta) { this->beta = beta; };

	double getMeanDistance() { return meanDistance; };
	double getAbsoluteMeanDistance() { return meanAbsoluteDistance; };
    
    void setMeanRadius(double meanRadius) { meanRadius_ = meanRadius; };
    
    void setVerbose(bool verbose) { verbose_ = verbose; };
    bool getVerbose() { return verbose_; };

    void addCorrectionPoints(vector<CVector3> points_mask_correction) { points_mask_correction_ = points_mask_correction; };

private:
	void InitParameters()
	{
        line_search = 15; //15;
		alpha = 25; // 25
		beta = 0.0;
		deltaNormale = 0.2;
        tradeOff = 10;
	};

	Image3D* image_;
	Mesh* mesh_;
	vector<int> listeTriangles_;
	ParametersType pointsInitiaux_;
	CMatrix4x4 transformation_;

	int nbParametres_;
	vector<CVector3> listeXiOpt;
	vector<double> listeWi;
	vector< vector<int> > listeTrianglesContenantPoint;
	vector< vector<int> > pointsVoisins;
	vector<Vertex> trianglesBarycentre_;

	double line_search, deltaNormale, tradeOff, alpha, beta;
	double type_image_factor;

	double meanDistance, meanAbsoluteDistance;
    
    double meanRadius_;
    
    bool verbose_;

    vector<CVector3> points_mask_correction_;
};

/*!
 * \class DeformableModelBasicAdaptator
 * \brief Deformation of a mesh to the gradient in an image.
 *
 * This class deform  a ttriangular mesh using deformable model based energy equation towards gradient vector in the input image. The deformation equation is quadratic and minimized using conjugate gradient.
 */
class DeformableModelBasicAdaptator
{
public:
	DeformableModelBasicAdaptator(Image3D* image, Mesh* m);
	DeformableModelBasicAdaptator(Image3D* image, Mesh* m, int nbIteration, double contrast, bool computeFinalMesh=true);
    DeformableModelBasicAdaptator(Image3D* image, Mesh* m, int nbIteration, vector<pair<CVector3,double> > contrast, bool computeFinalMesh=true);
	~DeformableModelBasicAdaptator();

	void setInput(Mesh* m) { mesh_ = m; };
	void setNumberOfIteration(int nbIteration) { numberOfIteration_ = nbIteration; };
	double adaptation();
	Mesh* getOutput() { return meshOutput_; };
	SpinalCord* getSpinalCordOutput() {
		return new SpinalCord(*meshOutput_);
	};

	void setFinalMeshBool(bool f) { meshBool_ = f; };
	void changedParameters() { this->changedParameters_ = true; };
	void setDeltaNormale(double deltaNormale) { this->deltaNormale = deltaNormale; };
	void setTradeOff(double tradeOff) { this->tradeOff = tradeOff; tradeoff_bool = true; };
	void setContrast(double contrast) { this->contrast = contrast; };

	void setLineSearch(double line_search) { this->line_search = line_search; };
	void setAlpha(double alpha) { this->alpha = alpha; };
	void setBeta(double beta) { this->beta = beta; };
	void setStopCondition(double s) { stopCondition = s; };
	void setNumberOptimizerIteration(int nbIt) { numberOptimizerIteration = nbIt; };
    void setProgressiveLineSearchLength(bool value) { progressiveLineSearchLength = value; };
    
    void setVerbose(bool verbose) { verbose_ = verbose; };
    bool getVerbose() { return verbose_; };

    void addCorrectionPoints(vector<CVector3> points_mask_correction) { points_mask_correction_ = points_mask_correction; };

private:
	Image3D* image_;
	int numberOfIteration_;
	Mesh *mesh_, *meshOutput_;
	bool meshBool_;

	bool changedParameters_;
	int line_search;
	double deltaNormale, tradeOff, alpha, beta, contrast;
    vector<pair<CVector3,double> > contrastvector;
	double stopCondition;
	int numberOptimizerIteration;
    bool progressiveLineSearchLength, tradeoff_bool;
    
    bool verbose_;

    vector<CVector3> points_mask_correction_;
};

#endif