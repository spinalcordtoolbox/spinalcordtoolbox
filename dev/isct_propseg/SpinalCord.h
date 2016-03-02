#ifndef __SPINAL_CORD__
#define __SPINAL_CORD__

/*!
 * \file SpinalCord.h
 * \brief Contains structure and methods for spinal cord mesh
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include "Mesh.h"
#include "util/Vector3.h"
#include "Image3D.h"
#include <vector>
#include <itkImage.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

typedef itk::Image< double, 3 >	ImageType;
typedef itk::SpatialOrientation::ValidCoordinateOrientationFlags OrientationType;

/*!
 * \class SpinalCord
 * \brief Contains structure and methods for spinal cord mesh
 * 
 * This class represents spinal cord mesh, heriting from Mesh.
 * A spinal cord mesh is a triangular mesh with a specific structure : stack of disks. Each disks have the same number of points and triangles are formed between two adjacent disks.
 * 
 * Methods for saving mesh, centerline and computing cross-sectional areas are available.
 */
class SpinalCord: public Mesh
{
public:
	SpinalCord();
	~SpinalCord();
	SpinalCord(const SpinalCord& sp);
	SpinalCord(const Mesh& m);

	void Initialize(int radialResolution);

	vector<CVector3> getCenterline() { return *centerline_; };
	double getLength() { return length_; };
	vector<CVector3> computeCenterline(bool saveFile=false, string filename="", bool spline=false);
	void saveCenterlineAsBinaryImage(ImageType::Pointer im, string filename, OrientationType orient);
	vector<double> computeApproximateCircleRadius();

	virtual void subdivision();
	void subdivisionAxiale();
	void subdivisionRadiale();

	int getRadialResolution() { return radialResolution_; };
	void setRadialResolution(int radialResolution) { radialResolution_ = radialResolution; };

	vector<double> computeCrossSectionalArea(bool saveFile=false, string filename="", bool spline=false, Image3D* im=0);
	double computeLastCrossSectionalArea();

	virtual vtkSmartPointer<vtkPolyData> reduceMeshUpAndDown(CVector3 upperSlicePoint, CVector3 upperSliceNormal, CVector3 downSlicePoint, CVector3 downSliceNormal, string filename="");
    
    void setCompleteCenterline(bool centerline) { completeCenterline_ = centerline; };
    bool getCompleteCenterline() { return completeCenterline_; };

    vector< vector<CVector3> > extractLastDisks(int numberOfDisks);
    CVector3 computeGravityCenterLastDisk(int numberOfDisks);
    CVector3 computeGravityCenterFirstDisk(int numberOfDisks);
    CVector3 computeGravityCenterSecondDisk();
    CVector3 computeLastDiskNormal(int numberOfDisks);
    SpinalCord* extractLastDiskOfMesh(bool moving);
    SpinalCord* extractPartOfMesh(int numberOfDisk, bool moving1, bool moving2);
    void assembleMeshes(SpinalCord* partOfMesh, int numberOfDisk, int radial_resolution_part);
    
private:
	void saveCenterline(string filename="");
	void saveCrossSectionalArea(string filename="", Image3D* im=0);

	int radialResolution_;
	vector<CVector3>* centerline_, *centerline_derivative_;
	double length_;
	vector<double>* crossSectionalArea_;
    
    bool completeCenterline_;
};

#endif
