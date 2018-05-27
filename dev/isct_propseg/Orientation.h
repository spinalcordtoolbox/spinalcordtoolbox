#ifndef __Orientation__
#define __Orientation__

/*!
 * \file Orientation.h
 * \brief Compute the orientation of the spinal cord using circular Hough transform.
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include <vector>

#include "Image3D.h"
#include "SpinalCord.h"
#include "util/Vector3.h"
#include "util/Matrix4x4.h"

typedef itk::Image< double, 3 >	ImageType;
typedef itk::Image< double, 2 >	ImageType2D;

/*!
 * \class Orientation
 * \brief Compute the orientation of the spinal cord using circular Hough transform and apply the rotation around a point on the mesh.
 *
 * This class compute the orientation of the spinal cord by extracting a 2D image (5cm X 5cm) in the orientation of the last iteration of segmentation mesh and performing a circular Hough transform on it.
 * The center of the first detected circle is assume to be the center of the spinal cord.
 * 
 * As an error metric, the distance between the detected position and the center of the 2D image is computed and return as reference. If this distance is larger than 3 cm, the return value is false, true if smaller.
 * The transformation (rotation around the center of mass of the first disk of the tubular mesh) is apply on the mesh if the angle of rotation is between 1 and 3 degrees. Small values don't change anything valuable on the mesh orientation and large values are assumed to be false.
 */
class Orientation {
public:
    Orientation(Image3D* image, SpinalCord* s);
    ~Orientation() {};
    
	bool computeOrientation(double &distance);
	bool getBadOrientation() { return badOrientation_; };
    
    void setVerbose(bool verbose) { verbose_ = verbose; };
    bool getVerbose() { return verbose_; };
    
private:
    void searchCenters(ImageType2D::Pointer im, std::vector<CVector3> &center, std::vector<double> &radius, std::vector<double> &accumulator, float z, CVector3 c);
	unsigned int houghTransformCircles(ImageType2D* im, unsigned int numberOfCircles, double** center_result, double* radius_result, double* accumulator_result, double meanRadius, double valPrint=255);
    
	Image3D* image_;
	SpinalCord* mesh_;
    
	double typeImageFactor_;
	bool badOrientation_;
    
    bool verbose_;
    
};

#endif /* defined(__Test__Orientation__) */
