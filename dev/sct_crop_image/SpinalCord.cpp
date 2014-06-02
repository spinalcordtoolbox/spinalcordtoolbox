#include "SpinalCord.h"
#include "OrientImage.h"
#include <fstream>
#include <string>
#include <vtkPlane.h>
#include <vtkCutter.h>
#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>
#include <vtkPolyData.h>
#include <vtkContourTriangulator.h>
#include <vtkMassProperties.h>
#include <vtkPointData.h>
#include <vtkClipPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkBYUWriter.h>

#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>
#include <itkImageRegionIterator.h>
#include <itkImageDuplicator.h>
using namespace std;

typedef itk::Image< unsigned char, 3 >	BinaryImageType;
typedef ImageType::IndexType IndexType;
typedef itk::Point< double, 3 > PointType;
typedef itk::ImageFileWriter< BinaryImageType > WriterType;
typedef itk::ImageDuplicator< ImageType > DuplicatorType;

typedef itk::SpatialOrientation::ValidCoordinateOrientationFlags OrientationType;


SpinalCord::SpinalCord()
{
	radialResolution_ = 0;
	centerline_ = new vector<CVector3>(0);
	crossSectionalArea_ = new vector<double>(0);
	length_ = 0.0;
}

SpinalCord::~SpinalCord()
{
	delete centerline_, crossSectionalArea_;
}

SpinalCord::SpinalCord(const SpinalCord& sp): Mesh(sp)
{
	radialResolution_ = sp.radialResolution_;
	centerline_ = new vector<CVector3>(*sp.centerline_);
	crossSectionalArea_ = new vector<double>(*sp.crossSectionalArea_);
	length_ = sp.length_;
}

SpinalCord::SpinalCord(const Mesh& m): Mesh(m)
{
	radialResolution_ = 0;
	centerline_ = new vector<CVector3>(0);
	crossSectionalArea_ = new vector<double>(0);
	length_ = 0.0;
}

void SpinalCord::Initialize(int radialResolution)
{
	radialResolution_ = radialResolution;
	centerline_ = new vector<CVector3>(0);
	crossSectionalArea_ = new vector<double>(0);
	length_ = 0.0;
}

vector<CVector3> SpinalCord::computeCenterline(bool saveFile, string filename)
{
	unsigned int numberDisks = points_.size()/radialResolution_;
	centerline_ = new vector<CVector3>(numberDisks);
	for (unsigned int i=0; i<numberDisks; i++)
	{
		CVector3 result;
		for (int k=0; k<radialResolution_; k++)
			result += points_[i*radialResolution_+k]->getPosition();
		result /= (double)radialResolution_;
		(*centerline_)[i] = result;
	}

	// compute length of spinal cord
	double length = 0.0;
	for (unsigned int i=0; i<centerline_->size()-1; i++)
		length += ((*centerline_)[i+1]-(*centerline_)[i]).Norm();
	length_ = length;

	if (saveFile) saveCenterline(filename);

	return *centerline_;
}

// need centerline computed
vector<double> SpinalCord::computeApproximateCircleRadius()
{
	// To determine approximate circle radius of disks, we have two solutions :
	// 1. take minimal distance of points to centerline
	// 2. take mean distance of points to centerline
	unsigned int numberDisks = points_.size()/radialResolution_;
	vector<double> *result = new vector<double>(numberDisks);
	for (unsigned int i=0; i<numberDisks; i++)
	{
		double minDistance = 10000.0, currentDistance;
		for (int k=0; k<radialResolution_; k++)
		{
			currentDistance = points_[i*radialResolution_+k]->distance((*centerline_)[i][0],(*centerline_)[i][1],(*centerline_)[i][2]);
			if (currentDistance < minDistance)
				minDistance = currentDistance;
		}
		(*result)[i] = minDistance;
	}
	return *result;
}


void SpinalCord::subdivision()
{
	subdivisionAxiale();
	subdivisionRadiale();
}

void SpinalCord::subdivisionAxiale()
{
	// On ajoute tout d'abord le premier disque. Ensuite, pour chaque bande, on ajoute un nouveau disque au milieu
	int numberOfDisks = getNbrOfPoints()/radialResolution_;
	vector<Vertex*> copyPoints = getListPoints(), points(copyPoints.size());
	for (unsigned int k=0; k<copyPoints.size(); k++)
		points[k] = new Vertex(*copyPoints[k]);
	this->clear();
	// Ajout du premier disque
	for (int i=0; i<radialResolution_; i++)
		addPoint(points[i]);
	// Ajout des disques suivant en subdivisiant
	for (int i=1; i<numberOfDisks; i++)
	{
		int offsetPoint = getNbrOfPoints(), offsetTriangles = getNbrOfTriangles();
		// Ajout des points intermédiaires
		for (int k=0; k<radialResolution_; k++)
			addPoint(new Vertex((points[i*radialResolution_+k]->getPosition()+points[(i-1)*radialResolution_+k]->getPosition())/2));
		// Ajout de la nouvelle bande de triangles
		for (int k=0; k<radialResolution_-1; k++)
		{
			addTriangle(offsetPoint-radialResolution_+k,offsetPoint-radialResolution_+k+1,offsetPoint+k);
			addTriangle(offsetPoint-radialResolution_+k+1,offsetPoint+k+1,offsetPoint+k);
		}
		// Adding two last triangles to close tube
		addTriangle(offsetPoint-radialResolution_+radialResolution_-1,offsetPoint-radialResolution_,offsetPoint+radialResolution_-1);
		addTriangle(offsetPoint-radialResolution_,offsetPoint,offsetPoint+radialResolution_-1);
		
		// Adding existing points
		offsetPoint = getNbrOfPoints();
		for (int k=0; k<radialResolution_; k++)
			addPoint(points[i*radialResolution_+k]);
		// Ajout de la nouvelle bande de triangles
		for (int k=0; k<radialResolution_-1; k++)
		{
			addTriangle(offsetPoint-radialResolution_+k,offsetPoint-radialResolution_+k+1,offsetPoint+k);
			addTriangle(offsetPoint-radialResolution_+k+1,offsetPoint+k+1,offsetPoint+k);
		}
		// Ajout des deux derniers triangles pour fermer le tube
		addTriangle(offsetPoint-radialResolution_+radialResolution_-1,offsetPoint-radialResolution_,offsetPoint+radialResolution_-1);
		addTriangle(offsetPoint-radialResolution_,offsetPoint,offsetPoint+radialResolution_-1);
	}
}

void SpinalCord::subdivisionRadiale()
{
	// Pour chaque disque, on doit ajouter un point entre deux points. La résolution radiale de sortie est donc 2*resolutionRadiale
	int numberOfDisks = getNbrOfPoints()/radialResolution_, newRadialResolution = 2*radialResolution_;
	vector<Vertex*> copyPoints = getListPoints(), points(copyPoints.size());
	for (unsigned int k=0; k<copyPoints.size(); k++)
		points[k] = new Vertex(*copyPoints[k]);
	this->clear();
	// Ajout du premier disque - Attention au dernier point
	for (int i=0; i<radialResolution_-1; i++)
	{
		addPoint(points[i]);
		addPoint(new Vertex((points[i]->getPosition()+points[i+1]->getPosition())/2));
	}
	addPoint(points[radialResolution_-1]); // Ajout des deux derniers points
	addPoint(new Vertex((points[radialResolution_-1]->getPosition()+points[0]->getPosition())/2));
	// Ajout des disques suivant en subdivisiant
	for (int i=1; i<numberOfDisks; i++)
	{
		int offsetPoint = getNbrOfPoints(), offsetTriangles = getNbrOfTriangles();
		// Ajout des points intermédiaires
		for (int k=0; k<radialResolution_-1; k++)
		{
			addPoint(points[i*radialResolution_+k]);
			addPoint(new Vertex((points[i*radialResolution_+k]->getPosition()+points[i*radialResolution_+k+1]->getPosition())/2));
		}
		addPoint(points[i*radialResolution_+radialResolution_-1]); // Ajout des deux derniers points
		addPoint(new Vertex((points[i*radialResolution_+radialResolution_-1]->getPosition()+points[i*radialResolution_]->getPosition())/2));
		// Ajout de la nouvelle bande de triangles
		for (int k=0; k<newRadialResolution-1; k++)
		{
			addTriangle(offsetPoint-newRadialResolution+k,offsetPoint-newRadialResolution+k+1,offsetPoint+k);
			addTriangle(offsetPoint-newRadialResolution+k+1,offsetPoint+k+1,offsetPoint+k);
		}
		// Ajout des deux derniers triangles pour fermer le tube
		addTriangle(offsetPoint-newRadialResolution+newRadialResolution-1,offsetPoint-newRadialResolution,offsetPoint+newRadialResolution-1);
		addTriangle(offsetPoint-newRadialResolution,offsetPoint,offsetPoint+newRadialResolution-1);
	}
	radialResolution_ = newRadialResolution;
}


void SpinalCord::saveCenterlineAsBinaryImage(ImageType::Pointer im, string filename, OrientationType orient)
{
	BinaryImageType::Pointer binaryCenterline = BinaryImageType::New();
	ImageType::RegionType region;
	ImageType::IndexType start;
	start[0] = 0; start[1] = 0; start[2] = 0;
	ImageType::SizeType size, imSize = im->GetLargestPossibleRegion().GetSize();
	size[0] = imSize[0]; size[1] = imSize[1]; size[2] = imSize[2];
	region.SetSize(size);
	region.SetIndex(start);
	binaryCenterline->CopyInformation(im);
	binaryCenterline->SetRegions(region);
	binaryCenterline->Allocate();
	binaryCenterline->FillBuffer(false);

	for (unsigned int i=0; i<centerline_->size(); i++)
	{
		PointType pt; pt[0] = (*centerline_)[i][0]; pt[1] = (*centerline_)[i][1]; pt[2] = (*centerline_)[i][2];
		IndexType ind;
		if (im->TransformPhysicalPointToIndex(pt,ind))
			binaryCenterline->SetPixel(ind,true);
	}
    
    OrientImage<BinaryImageType> orientationFilter;
    orientationFilter.setInputImage(binaryCenterline);
    orientationFilter.orientation(orient);
    binaryCenterline = orientationFilter.getOutputImage();

	WriterType::Pointer writer = WriterType::New();
	itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
	writer->SetImageIO(io);
	writer->SetFileName(filename+"_CenterlineBinaryImage.nii");
	writer->SetInput(binaryCenterline);
	try {
		writer->Write();
	} catch( itk::ExceptionObject & e ) {
		std::cerr << "Exception caught while writing image " << std::endl;
		std::cerr << e << std::endl;
	}
}


void SpinalCord::saveCenterline(string filename)
{
	ofstream myfile;
	filename += "_centerline.txt";
	myfile.open(filename.c_str());
	for (unsigned int i=0; i<centerline_->size(); i++)
		myfile << (*centerline_)[i][0] << " " << (*centerline_)[i][1] << " " << (*centerline_)[i][2] << endl;
	myfile.close();
}


vector<double> SpinalCord::computeCrossSectionalArea(bool saveFile, string filename)
{
	if (centerline_->size() == 0) computeCenterline();
	crossSectionalArea_ = new vector<double>(centerline_->size());

	// create polyData from points and triangles from mesh
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> pointNormalsArray = vtkSmartPointer<vtkDoubleArray>::New();
	pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)
	pointNormalsArray->SetNumberOfTuples(points_.size());
	CVector3 p, n;
	for (unsigned int i=0; i<points_.size(); i++) {
		p = points_[i]->getPosition();
		n = points_[i]->getNormal();
		points->InsertNextPoint(p[0],p[1],p[2]);
		pointNormalsArray->SetTuple3(i, n[0], n[1], n[2]) ;
	}
	vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
	source->SetPoints(points);
	source->GetPointData()->SetNormals(pointNormalsArray);
	source->Allocate(triangles_.size());
	for (unsigned int i=0; i<triangles_.size(); i+=3) {
		vtkIdType pts[3] = {triangles_[i],triangles_[i+1],triangles_[i+2]};
		source->InsertNextCell(VTK_TRIANGLE,3,pts);
	}

	// Computing area of cross-section. First and last values ar certainly wrong. Firstly because of the orientatation of points at the end of the mesh and for the last one, because of approximation of normal
	CVector3 point, normal;
	vtkSmartPointer<vtkPlane> plane = vtkSmartPointer<vtkPlane>::New();
	vtkSmartPointer<vtkCutter> cutter = vtkSmartPointer<vtkCutter>::New();
	vtkSmartPointer<vtkContourTriangulator> triangleFilter = vtkSmartPointer<vtkContourTriangulator>::New();
	vtkSmartPointer<vtkMassProperties> areaEvaluator = vtkSmartPointer<vtkMassProperties>::New();
	cutter->SetInputData(source);
	for (unsigned int i=1; i<centerline_->size()-3; i++)
	{
		point = (*centerline_)[i];
		normal = ((*centerline_)[i+3]-point).Normalize();
		plane->SetOrigin(point[0],point[1],point[2]);
		plane->SetNormal(normal[0],normal[1],normal[2]);
		cutter->SetCutFunction(plane);
		cutter->Update();
		triangleFilter->SetInputConnection(cutter->GetOutputPort());
		triangleFilter->Update();
		vtkSmartPointer<vtkPolyData> res = triangleFilter->GetOutput();
		areaEvaluator->SetInputConnection(triangleFilter->GetOutputPort());
		areaEvaluator->Update();
		(*crossSectionalArea_)[i] = areaEvaluator->GetSurfaceArea();
	}

	if (saveFile) saveCrossSectionalArea(filename);

	return *crossSectionalArea_;
}

double SpinalCord::computeLastCrossSectionalArea()
{
	computeCenterline();

	// create polyData from points and triangles from mesh
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> pointNormalsArray = vtkSmartPointer<vtkDoubleArray>::New();
	pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)
	pointNormalsArray->SetNumberOfTuples(points_.size());
	CVector3 p, n;
	for (unsigned int i=0; i<points_.size(); i++) {
		p = points_[i]->getPosition();
		n = points_[i]->getNormal();
		points->InsertNextPoint(p[0],p[1],p[2]);
		pointNormalsArray->SetTuple3(i, n[0], n[1], n[2]) ;
	}
	vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
	source->SetPoints(points);
	source->GetPointData()->SetNormals(pointNormalsArray);
	source->Allocate(triangles_.size());
	for (unsigned int i=0; i<triangles_.size(); i+=3) {
		vtkIdType pts[3] = {triangles_[i],triangles_[i+1],triangles_[i+2]};
		source->InsertNextCell(VTK_TRIANGLE,3,pts);
	}

	// Computing area of cross-section. First and last values ar certainly wrong. Firstly because of the orientatation of points at the end of the mesh and for the last one, because of approximation of normal
	CVector3 point, normal;
	vtkSmartPointer<vtkPlane> plane = vtkSmartPointer<vtkPlane>::New();
	vtkSmartPointer<vtkCutter> cutter = vtkSmartPointer<vtkCutter>::New();
	vtkSmartPointer<vtkContourTriangulator> triangleFilter = vtkSmartPointer<vtkContourTriangulator>::New();
	vtkSmartPointer<vtkMassProperties> areaEvaluator = vtkSmartPointer<vtkMassProperties>::New();
	cutter->SetInputData(source);

	unsigned int position = centerline_->size()/2;
	point = (*centerline_)[position];
	normal = ((*centerline_)[centerline_->size()-1]-point).Normalize();
	plane->SetOrigin(point[0],point[1],point[2]);
	plane->SetNormal(normal[0],normal[1],normal[2]);
	cutter->SetCutFunction(plane);
	cutter->Update();
	triangleFilter->SetInputConnection(cutter->GetOutputPort());
	triangleFilter->Update();
	vtkSmartPointer<vtkPolyData> res = triangleFilter->GetOutput();
	areaEvaluator->SetInputConnection(triangleFilter->GetOutputPort());
	areaEvaluator->Update();
	
	return areaEvaluator->GetSurfaceArea();
}

void SpinalCord::saveCrossSectionalArea(string filename)
{
	ofstream myfile;
	filename += "_CrossSectionalArea.txt";
	myfile.open(filename.c_str());
	for (unsigned int i=0; i<crossSectionalArea_->size(); i++)
		myfile << (*crossSectionalArea_)[i] << endl;
	myfile.close();
}


void SpinalCord::reduceMeshUpAndDown(CVector3 upperSlicePoint, CVector3 upperSliceNormal, CVector3 downSlicePoint, CVector3 downSliceNormal, string filename)
{
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkDoubleArray> pointNormalsArray = vtkSmartPointer<vtkDoubleArray>::New();
	pointNormalsArray->SetNumberOfComponents(3); //3d normals (ie x,y,z)
	pointNormalsArray->SetNumberOfTuples(points_.size());
	CVector3 p, n;
	for (unsigned int i=0; i<points_.size(); i++) {
		p = points_[i]->getPosition();
		n = points_[i]->getNormal();
		points->InsertNextPoint(p[0],p[1],p[2]);
		pointNormalsArray->SetTuple3(i, n[0], n[1], n[2]) ;
	}
	vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
	source->SetPoints(points);
	source->GetPointData()->SetNormals(pointNormalsArray);
	source->Allocate(triangles_.size());
	for (unsigned int i=0; i<triangles_.size(); i+=3) {
		vtkIdType pts[3] = {triangles_[i],triangles_[i+1],triangles_[i+2]};
		source->InsertNextCell(VTK_TRIANGLE,3,pts);
	}

	vtkSmartPointer<vtkPlane> downPlane = vtkSmartPointer<vtkPlane>::New(), upperPlane = vtkSmartPointer<vtkPlane>::New();
	downPlane->SetOrigin(downSlicePoint[0],downSlicePoint[1],downSlicePoint[2]);
	downPlane->SetNormal(downSliceNormal[0],downSliceNormal[1],downSliceNormal[2]);
	upperPlane->SetOrigin(upperSlicePoint[0],upperSlicePoint[1],upperSlicePoint[2]);
	upperPlane->SetNormal(upperSliceNormal[0],upperSliceNormal[1],upperSliceNormal[2]);

	vtkSmartPointer<vtkClipPolyData> downClipper = vtkSmartPointer<vtkClipPolyData>::New();
	downClipper->SetInputData(source);
	downClipper->SetClipFunction(downPlane);
	downClipper->InsideOutOn();
	downClipper->Update();

	vtkSmartPointer<vtkClipPolyData> upperClipper = vtkSmartPointer<vtkClipPolyData>::New();
	upperClipper->SetInputData(downClipper->GetOutput());
	upperClipper->SetClipFunction(upperPlane);
	upperClipper->InsideOutOn();
	upperClipper->Update();

	vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
	writer->SetFileName(filename.c_str());
	writer->SetInputData(upperClipper->GetOutput());
	writer->Write();
}
