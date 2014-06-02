#pragma once

#include "Mesh.h"
#include "Vector3.h"
#include <vector>
#include <itkImage.h>

typedef itk::Image< double, 3 >	ImageType;
typedef itk::SpatialOrientation::ValidCoordinateOrientationFlags OrientationType;

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
	vector<CVector3> computeCenterline(bool saveFile=false, string filename="");
	void saveCenterlineAsBinaryImage(ImageType::Pointer im, string filename, OrientationType orient);
	vector<double> computeApproximateCircleRadius();

	virtual void subdivision();
	void subdivisionAxiale();
	void subdivisionRadiale();

	int getRadialResolution() { return radialResolution_; };
	void setRadialResolution(int radialResolution) { radialResolution_ = radialResolution; };

	vector<double> computeCrossSectionalArea(bool saveFile=false, string filename="");
	double computeLastCrossSectionalArea();

	void reduceMeshUpAndDown(CVector3 upperSlicePoint, CVector3 upperSliceNormal, CVector3 downSlicePoint, CVector3 downSliceNormal, string filename);

private:
	void saveCenterline(string filename="");
	void saveCrossSectionalArea(string filename="");

	int radialResolution_;
	vector<CVector3>* centerline_;
	double length_;
	vector<double>* crossSectionalArea_;
};

