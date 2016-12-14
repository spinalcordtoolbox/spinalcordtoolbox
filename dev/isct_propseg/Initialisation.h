#ifndef __INITIALIZATION__
#define __INITIALIZATION__

/*!
 * \file Initialisation.h
 * \brief Spinal cord detection module
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include "util/Vector3.h"
#include <vector>
#include <itkImage.h>
using namespace std;

typedef itk::Image< double, 3 >	ImageType;
typedef itk::Image< double, 2 >	ImageType2D;
typedef itk::SpatialOrientation::ValidCoordinateOrientationFlags OrientationType;

/*!
 * \class Initialisation
 * \brief Spinal cord detection module
 * 
 * This class detects the spinal cord in a MR volume. It use the circular Hough transform on axial slices.
 * 
 * TO DO: save detection as PNG image!
 */

class Initialisation
{
public:
    Initialisation();
	Initialisation(ImageType::Pointer, double imageFactor=1, double gap=1.0);
	~Initialisation() {};
    
    void setInputImage(ImageType::Pointer image);
    void setImageFactor(double imageFactor) { typeImageFactor_ = imageFactor; };
    void setGap(double gap=4.0) { gap_ = gap; };
    void setStartSlice(int slice) { startSlice_ = slice; };
    void setNumberOfSlices(int nbSlice) { numberOfSlices_ = nbSlice-(1-nbSlice%2); }; //need to be impair
    void setRadius(double radius) { radius_ = radius; };
    
    vector<CVector3> getCenterlineUsingMinimalPath(vector<int> middle_slices, double alpha=0.15, double beta=1.0, double gamma=5.0, double sigmaMinimum=1.5, double sigmaMaximum=4.5, unsigned int numberOfSigmaSteps=5, double sigmaDistance=10.0);
    ImageType::Pointer minimalPath3d(ImageType::Pointer image, vector<CVector3> &centerline, bool homoInt=false, bool invert=true, double factx=sqrt(2));
    
	bool computeInitialParameters(float startFactor=-1.0);
    
	void getPoints(CVector3 &point, CVector3 &normal1, CVector3 &normal2, double &radius, double &stretchingFactor);
    vector<CVector3> getPoints() { return points_; };
    double getMeanRadius() { return initialRadius_; };
    void savePointAsBinaryImage(ImageType::Pointer initialImage, string filename, OrientationType orientation);
    void savePointAsAxialImage(ImageType::Pointer initialImage, string filename);
    
    void setVerbose(bool verbose) { verbose_ = verbose; };
    bool getVerbose() { return verbose_; };
    
private:
	void searchCenters(ImageType2D::Pointer im, vector<CVector3> &vecCenter, vector<double> &vecRadii, vector<double> &vecAccumulator, float startZ);
	unsigned int houghTransformCircles(ImageType2D* im, unsigned int numberOfCircles, double** center_result, double* radius_result, double* accumulator_result, double meanRadius, double valPrint=255);
    ImageType::Pointer vesselnessFilter(vector<int> middle_slices, ImageType::Pointer im, double alpha=0.15, double beta=1.0, double gamma=5.0, double sigmaMinimum=1.5, double sigmaMaximum=4.5, unsigned int numberOfSigmaSteps=10, double sigmaDistance=30.0);
    ImageType::Pointer vesselnessFilter2(ImageType::Pointer im);
    int symmetryDetection(ImageType2D::Pointer im, double cropWidth_, double bandWidth_);
    int symmetryDetection3D(ImageType::Pointer im, double cropWidth_, double bandWidth_);
    
    vector<CVector3> points_;
	CVector3 initialPoint_, initialNormal1_, initialNormal2_;
	double initialRadius_, stretchingFactor_;
    
	ImageType::Pointer inputImage_;
	double typeImageFactor_, gap_, radius_;
    unsigned int numberOfSlices_;
    float startSlice_;
    double mean_resolution_;
    
    OrientationType orientation_;

    bool verbose_;
};

#endif
