#ifndef __GLOBAL_ADAPTATION__
#define __GLOBAL_ADAPTATION__

/*!
 * \file GlobalAdaptation.h
 * \brief Compute the plan minimizing the distance between points and the plan
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include <vector>
#include <string>

#include <itkPoint.h>
#include <itkImageAlgorithm.h>
#include <itkImageFileReader.h>
#include <itkSingleValuedCostFunction.h>

#include "Image3D.h"
#include "SpinalCord.h"
#include "Vertex.h"
#include "util/Matrix3x3.h"
#include "SCRegion.h"


typedef itk::CovariantVector<double,3> PixelType;
typedef itk::Image< PixelType, 3 > ImageVectorType;
typedef ImageVectorType::IndexType IndexType;
typedef itk::Point< double, 3 > PointType;

/*!
 * \class FoncteurGlobalAdaptation
 * \brief Equation class for spinal cord orientation computing
 */
class FoncteurGlobalAdaptation: public itk::SingleValuedCostFunction
{
public:
	FoncteurGlobalAdaptation(Image3D* image, std::vector<Vertex*>* tri, unsigned int numberOfParameters=6) : image_(image), listeTriangles_(tri), iterateur(0), numberOfParameters_(numberOfParameters)
	{
		sizePoints = listeTriangles_->size();
		points = new CVector3[sizePoints];
		normales = new CVector3[sizePoints];
		for (unsigned int i=0; i<sizePoints; i++) {
			points[i] = (*listeTriangles_)[i]->getPosition();
			normales[i] = (*listeTriangles_)[i]->getNormal();
		}
		type_image_factor = image_->getTypeImageFactor();
	}

	// Method used by ITK - Derivative can only be use with unique rotation, not translation
	virtual void GetDerivative (const ParametersType &parameters, DerivativeType &derivative) const
	{
		CMatrix3x3 rotationP0, rotationP1, rotationP2;
		rotationP0[0] = -sin(parameters[0])*cos(parameters[1]),	rotationP0[3] = sin(parameters[2])*cos(parameters[0])*cos(parameters[1]),											rotationP0[6] = cos(parameters[2])*cos(parameters[0])*cos(parameters[1]),
		rotationP0[1] = -sin(parameters[0])*sin(parameters[1]),	rotationP0[4] = sin(parameters[2])*cos(parameters[0])*sin(parameters[1]),											rotationP0[7] = cos(parameters[2])*cos(parameters[0])*sin(parameters[1]),
		rotationP0[2] = -cos(parameters[0]),					rotationP0[5] = -sin(parameters[2])*sin(parameters[0]),																rotationP0[8] = -cos(parameters[2])*sin(parameters[0]);
		rotationP1[0] = -cos(parameters[0])*sin(parameters[1]),	rotationP1[3] = -cos(parameters[2])*cos(parameters[1]) - sin(parameters[2])*sin(parameters[0])*sin(parameters[1]),	rotationP1[6] = sin(parameters[2])*cos(parameters[1]) - cos(parameters[2])*sin(parameters[0])*sin(parameters[1]),
		rotationP1[1] = cos(parameters[0])*cos(parameters[1]),	rotationP1[4] = -cos(parameters[2])*sin(parameters[1]) + sin(parameters[2])*sin(parameters[0])*cos(parameters[1]),	rotationP1[7] = sin(parameters[2])*sin(parameters[1]) + cos(parameters[2])*sin(parameters[0])*cos(parameters[1]),
		rotationP1[2] = 0,										rotationP1[5] = 0,																									rotationP1[8] = 0;
		rotationP2[0] = 0,										rotationP2[3] = sin(parameters[2])*sin(parameters[1]) + cos(parameters[2])*sin(parameters[0])*cos(parameters[1]),	rotationP2[6] = cos(parameters[2])*sin(parameters[1]) - sin(parameters[2])*sin(parameters[0])*cos(parameters[1]),
		rotationP2[1] = 0,										rotationP2[4] = -sin(parameters[2])*cos(parameters[1]) + cos(parameters[2])*sin(parameters[0])*sin(parameters[1]),	rotationP2[7] = -cos(parameters[2])*cos(parameters[1]) - sin(parameters[2])*sin(parameters[0])*sin(parameters[1]),
		rotationP2[2] = 0,										rotationP2[5] = cos(parameters[2])*cos(parameters[0]),																rotationP2[8] = -sin(parameters[2])*cos(parameters[0]);
		
		derivative = DerivativeType(numberOfParameters_); // numberOfParameters_ have to be equal to 3 (just rotation)
		derivative.Fill(0.0);

		CVector3 pnt, index;
		for (unsigned int i=0; i<sizePoints; i++) {
			pnt = rotationP0*(points[i]-pointRotation) + pointRotation;
			if (image_->TransformPhysicalPointToContinuousIndex(pnt,index))
				derivative[0] -= image_->GetContinuousPixelMagnitudeGradient(index)*region_->GetContinuousPixelMagnitudeGradient(index);
			pnt = rotationP1*(points[i]-pointRotation) + pointRotation;
			if (image_->TransformPhysicalPointToContinuousIndex(pnt,index))
				derivative[1] -= image_->GetContinuousPixelMagnitudeGradient(index)*region_->GetContinuousPixelMagnitudeGradient(index);
			pnt = rotationP2*(points[i]-pointRotation) + pointRotation;
			if (image_->TransformPhysicalPointToContinuousIndex(pnt,index))
				derivative[2] -= image_->GetContinuousPixelMagnitudeGradient(index)*region_->GetContinuousPixelMagnitudeGradient(index);
		}
		//cout << "Derivative : " << derivative << endl;
	}
 
	virtual MeasureType GetValue (const ParametersType &parameters) const
	{
		double result = 0.0;
		CMatrix3x3 rotation;
		CVector3 pnt, translation, index;
		rotation[0] = cos(parameters[0])*cos(parameters[1]),	rotation[3] = -cos(parameters[2])*sin(parameters[1]) + sin(parameters[2])*sin(parameters[0])*cos(parameters[1]),	rotation[6] = sin(parameters[2])*sin(parameters[1]) + cos(parameters[2])*sin(parameters[0])*cos(parameters[1]),
		rotation[1] = cos(parameters[0])*sin(parameters[1]),	rotation[4] = cos(parameters[2])*cos(parameters[1]) + sin(parameters[2])*sin(parameters[0])*sin(parameters[1]),		rotation[7] = -sin(parameters[2])*cos(parameters[1]) + cos(parameters[2])*sin(parameters[0])*sin(parameters[1]),
		rotation[2] = -sin(parameters[0]),						rotation[5] = sin(parameters[2])*cos(parameters[0]),																rotation[8] = cos(parameters[2])*cos(parameters[0]);
		if (numberOfParameters_ == 6)
			translation[0] = parameters[3], translation[1] = parameters[4], translation[2] = parameters[5];
		else translation = CVector3::ZERO;
		int nb = 0;
		for (unsigned int i=0; i<sizePoints; i++) {
			pnt = rotation*(points[i]-pointRotation) + pointRotation + translation;
			if (image_->TransformPhysicalPointToContinuousIndex(pnt,index)) {
                //result -= image_->GetContinuousPixelMagnitudeGradient(index)*region_->GetContinuousPixelMagnitudeGradient(index);
                result -= image_->GetContinuousPixelMagnitudeGradient(index);
            }
				
			else nb++;
		}
		//if (nb > 0) cout << "Nombre de points du maillage en dehors de l'image : " << nb << endl;
		//cout << "Result : " << result << endl;
		return result;
	}
	virtual unsigned int GetNumberOfParameters (void) const { return numberOfParameters_; }
	void setPointRotation(CVector3 point) { pointRotation = point; };

	void setGaussianRegion(SCRegion* gaussianRegion) { region_ = gaussianRegion; };

private:
	Image3D* image_;
	std::vector<Vertex*>* listeTriangles_;

	unsigned int sizePoints;
	CVector3 *points, *normales;

	CMatrix3x3 rotation;
	CVector3 pnt, gradient, translation, index;
	CVector3 pointRotation;
	PixelType pixel;

	double type_image_factor;

	unsigned int numberOfParameters_;

	int iterateur;

	SCRegion* region_;
};

/*!
 * \class GlobalAdaptation
 * \brief Compute the orientation of the spinal cord.
 * 
 * This class compute the orientation of the spinal cord by optimizing intensity values at the mesh vertices positions.
 * The transformation is apply directly on the mesh.
 */
class GlobalAdaptation
{
public:
	GlobalAdaptation(Image3D* image, Mesh* v, std::string mode="rotation+translation");
	GlobalAdaptation(Image3D* image, Mesh* v, CVector3 pointRotation, std::string mode="rotation+translation");
	~GlobalAdaptation() {};

	double getInitialValue();
	CMatrix4x4 adaptation(bool itkAmoeba=true);
	bool getBadOrientation() { return badOrientation_; };

	void setNormalMesh(CVector3 normal) { normal_mesh_ = normal; };
    
    void setVerbose(bool verbose) { verbose_ = verbose; };
    bool getVerbose() { return verbose_; };

private:
	Image3D* image_;
	Mesh* mesh_;

	CVector3 pointRotation_, normal_mesh_;
	std::string mode_;

	bool badOrientation_;
    
    bool verbose_;
};

#endif
