#include "SpinalCord.h"
#include "OrientImage.h"
#include "Image3D.h"
#include "BSplineApproximation.h"
#include <fstream>
#include <string>
#include <map>
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
#include <vtkParametricSpline.h>
#include <vtkParametricFunctionSource.h>

#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>
#include <itkImageRegionIterator.h>
#include <itkImageDuplicator.h>
#include <itkPointSet.h>
#include <itkBSplineScatteredDataPointSetToImageFilter.h>
#include <itkBSplineControlPointImageFunction.h>
using namespace std;

typedef itk::Image< unsigned char, 3 >	BinaryImageType;
typedef ImageType::IndexType IndexType;
typedef itk::Point< double, 3 > PointType;
typedef itk::ImageFileWriter< BinaryImageType > WriterType;
typedef itk::ImageDuplicator< ImageType > DuplicatorType;

typedef itk::SpatialOrientation::ValidCoordinateOrientationFlags OrientationType;


template<class T1, class T2, class Pred = std::less<T1> >
struct sort_pair_first {
    bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
        Pred p;
        return p(left.second, right.second);
    }
};


SpinalCord::SpinalCord()
{
	radialResolution_ = 0;
	centerline_ = new vector<CVector3>(0);
    centerline_derivative_ = new vector<CVector3>(0);
	crossSectionalArea_ = new vector<double>(0);
	length_ = 0.0;
    completeCenterline_ = false;
}

SpinalCord::~SpinalCord()
{
	delete centerline_, crossSectionalArea_, centerline_derivative_;
}

SpinalCord::SpinalCord(const SpinalCord& sp): Mesh(sp)
{
	radialResolution_ = sp.radialResolution_;
	centerline_ = new vector<CVector3>(*sp.centerline_);
    centerline_derivative_ = new vector<CVector3>(0);
	crossSectionalArea_ = new vector<double>(*sp.crossSectionalArea_);
	length_ = sp.length_;
}

SpinalCord::SpinalCord(const Mesh& m): Mesh(m)
{
	radialResolution_ = 0;
	centerline_ = new vector<CVector3>(0);
    centerline_derivative_ = new vector<CVector3>(0);
	crossSectionalArea_ = new vector<double>(0);
	length_ = 0.0;
}

void SpinalCord::Initialize(int radialResolution)
{
	radialResolution_ = radialResolution;
	centerline_ = new vector<CVector3>(0);
    centerline_derivative_ = new vector<CVector3>(0);
	crossSectionalArea_ = new vector<double>(0);
	length_ = 0.0;
}

vector<CVector3> SpinalCord::computeCenterline(bool saveFile, string filename, bool spline)
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
    
    vector<CVector3> newCenterline, centerline_derivative;
    if (saveFile)
    {
        //double end1 = (*centerline_)[0][1], end2 = (*centerline_)[centerline_->size()-1][1];
        double range = centerline_->size();
    
        BSplineApproximation centerline_approximator = BSplineApproximation(centerline_);
    
        double start = 0, end = 2.0*range;
        
        double point = 0.0;
        for (double i=start; i<=end; i++) {
            point = i/(2.0*range);
            newCenterline.push_back(centerline_approximator.EvaluateBSpline(point));
            centerline_derivative.push_back(centerline_approximator.EvaluateGradient(point));
        }
        
    }
    
    // compute length of spinal cord
	double length = 0.0;
	for (unsigned int i=0; i<centerline_->size()-1; i++)
		length += ((*centerline_)[i+1]-(*centerline_)[i]).Norm();
	length_ = length;

	if (saveFile && !spline) {
        ofstream myfile;
        myfile.open(filename.c_str());
        for (unsigned int i=0; i<newCenterline.size(); i++)
            myfile << newCenterline[i][0] << " " << newCenterline[i][1] << " " << newCenterline[i][2] << endl;
        myfile.close();
    }
    else if (spline) {
        centerline_->clear();
        centerline_derivative_->clear();
        for (unsigned int k=0; k<newCenterline.size(); k++) {
            centerline_->push_back(newCenterline[k]);
            centerline_derivative_->push_back(centerline_derivative[k]);
        }
        double length = 0.0;
        for (unsigned int k=0; k<centerline_->size()-1; k++)
            length += ((*centerline_)[k+1]-(*centerline_)[k]).Norm();
        length_ = length;
        if (saveFile) saveCenterline(filename);
    }

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
		int offsetPoint = getNbrOfPoints();
		// Ajout des points interm�diaires
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
	// Pour chaque disque, on doit ajouter un point entre deux points. La r�solution radiale de sortie est donc 2*resolutionRadiale
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
		int offsetPoint = getNbrOfPoints();
		// Ajout des points interm�diaires
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
    double range = centerline_->size();
    
    BSplineApproximation centerline_approximator = BSplineApproximation(centerline_);
    
    vector<CVector3> newCenterline, centerline_derivative;
    double point = 0.0;
    for (double i=0; i<=2.0*range; i++) {
        point = i/(2.0*range);
        newCenterline.push_back(centerline_approximator.EvaluateBSpline(point));
        centerline_derivative.push_back(centerline_approximator.EvaluateGradient(point));
    }
    
	BinaryImageType::Pointer binaryCenterline = BinaryImageType::New();
	BinaryImageType::RegionType regionImage;
	BinaryImageType::IndexType startImage;
	startImage[0] = 0; startImage[1] = 0; startImage[2] = 0;
	BinaryImageType::SizeType sizeImage, imSizeImage = im->GetLargestPossibleRegion().GetSize();
	sizeImage[0] = imSizeImage[0]; sizeImage[1] = imSizeImage[1]; sizeImage[2] = imSizeImage[2];
	regionImage.SetSize(sizeImage);
	regionImage.SetIndex(startImage);
	binaryCenterline->CopyInformation(im);
	binaryCenterline->SetRegions(regionImage);
	binaryCenterline->Allocate();
	binaryCenterline->FillBuffer(false);

	for (unsigned int i=0; i<newCenterline.size(); i++)
	{
		PointType pt; pt[0] = newCenterline[i][0]; pt[1] = newCenterline[i][1]; pt[2] = newCenterline[i][2];
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
	writer->SetFileName(filename);
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
	myfile.open(filename.c_str());
	for (unsigned int i=0; i<centerline_->size(); i++)
		myfile << (*centerline_)[i][0] << " " << (*centerline_)[i][1] << " " << (*centerline_)[i][2] << endl;
	myfile.close();
}


vector<double> SpinalCord::computeCrossSectionalArea(bool saveFile, string filename, bool spline, Image3D* im)
{
	computeCenterline(saveFile, filename, spline);
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
    int step = 3;
    if (spline) step = 6;
	for (unsigned int i=1; i<centerline_->size()-step; i++)
	{
		point = (*centerline_)[i];
		normal = (*centerline_derivative_)[i].Normalize();
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

	if (saveFile) saveCrossSectionalArea(filename, im);

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

void SpinalCord::saveCrossSectionalArea(string filename, Image3D* im)
{
	ofstream myfile;
	myfile.open(filename.c_str());
    CVector3 index, last_index;
    double value_mean = 0.0, nb_mean = 0.0;
    int size_image = im->getLargeur();
    vector<pair<int,double> > cross;
	for (unsigned int i=0; i<crossSectionalArea_->size(); i++)
    {
        if (im != 0)
        {
            CVector3 point = (*centerline_)[i];
            im->TransformPhysicalPointToIndex(point, index);
            if (index[1] >= 0 && index[1] <= size_image)
            {
                if (index[1] == last_index[1]) {
                    value_mean += (*crossSectionalArea_)[i];
                    nb_mean += 1.0;
                    if (i == crossSectionalArea_->size() && index[1] < size_image)
                        cross.push_back(make_pair(last_index[1]+1,value_mean/nb_mean));
                }
                else {
                    if (index[1] <= size_image)
                        cross.push_back(make_pair(last_index[1]+1,value_mean/nb_mean));
                    last_index = index;
                    value_mean = (*crossSectionalArea_)[i];
                    nb_mean = 1.0;
                }
            }
        } else {
            myfile << (*crossSectionalArea_)[i] << endl;
        }
    }
    if (im != 0)
    {
        sort(cross.begin(),cross.end());
        for (unsigned int i=0; i<cross.size(); i++)
            myfile << cross[i].first << " " << cross[i].second << endl;
    }
	myfile.close();
}


vtkSmartPointer<vtkPolyData> SpinalCord::reduceMeshUpAndDown(CVector3 upperSlicePoint, CVector3 upperSliceNormal, CVector3 downSlicePoint, CVector3 downSliceNormal, string filename)
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

	/*vtkSmartPointer<vtkBYUWriter> writer = vtkSmartPointer<vtkBYUWriter>::New();
	string fileN = filename + "_CuttedMesh.byu";
	writer->SetGeometryFileName(fileN.c_str());
	writer->SetInputData(upperClipper->GetOutput());
	writer->Write();*/

    vtkSmartPointer<vtkPolyData> polyData = upperClipper->GetOutput();
	return polyData;
}

// Extract last disks from meshOutput
vector< vector<CVector3> > SpinalCord::extractLastDisks(int numberOfDisks)
{
    vector< vector<CVector3> > lastDisks;
	unsigned long size = points_.size();
	
	for (int i=1; i<=numberOfDisks; i++)
	{
		vector<CVector3> temp;
		for (int k=0; k<radialResolution_; k++)
		{
			temp.push_back(points_[size-i*radialResolution_+k]->getPosition());
		}
		lastDisks.push_back(temp);
	}
    return lastDisks;
}

CVector3 SpinalCord::computeGravityCenterLastDisk(int numberOfDisks)
{
    vector< vector<CVector3> > disks = extractLastDisks(numberOfDisks);
    
	CVector3 result;
	for (int k=0; k<radialResolution_; k++)
		result += disks[0][k];
	result /= (double)radialResolution_;
	return result;
}

CVector3 SpinalCord::computeGravityCenterFirstDisk(int numberOfDisks)
{
    vector< vector<CVector3> > disks = extractLastDisks(numberOfDisks);
    
	CVector3 result;
	unsigned int first = disks.size()-1;
	for (int k=0; k<radialResolution_; k++)
		result += disks[first][k];
	result /= (double)radialResolution_;
	return result;
}

CVector3 SpinalCord::computeGravityCenterSecondDisk()
{
	CVector3 result;
	for (int k=0; k<radialResolution_; k++)
		result += points_[radialResolution_+k]->getPosition();
	result /= (double)radialResolution_;
	return result;
}

CVector3 SpinalCord::computeLastDiskNormal(int numberOfDisks)
{
    vector< vector<CVector3> > lastDisks = extractLastDisks(numberOfDisks);
    
	CVector3 result;
    
	// Calcul des centres gravites des disques et on prend la normale passant par ces points
	CVector3 centreGravite1, centreGravite2;
	for (int k=0; k<radialResolution_; k++)
	{
		centreGravite1 += lastDisks[0][k];
		centreGravite2 += lastDisks[numberOfDisks-1][k];
	}
	centreGravite1 /= (double)radialResolution_;
	centreGravite2 /= (double)radialResolution_;
    
	result = (centreGravite1-centreGravite2).Normalize();
    
	return result;
}


SpinalCord* SpinalCord::extractLastDiskOfMesh(bool moving)
{
	SpinalCord* result = new SpinalCord;
    result->radialResolution_ = this->radialResolution_;
	vector< vector<CVector3> > lastDisks = extractLastDisks(1);
	// Ajout du deuxieme disque
	for (int j=0; j<radialResolution_; j++)
		result->addPoint(new Vertex(lastDisks[0][j],moving));
	return result;
}


SpinalCord* SpinalCord::extractPartOfMesh(int numberOfDisk, bool moving1, bool moving2)
{
	SpinalCord* result = new SpinalCord;
    result->radialResolution_ = this->radialResolution_;
	vector< vector<CVector3> > lastDisks = extractLastDisks(numberOfDisk);
	// Ajout du premier disque
	for (int j=0; j<radialResolution_; j++)
	{
		result->addPoint(new Vertex(lastDisks[numberOfDisk-1][j],moving1));
	}
	// Ajout des suivants
	for (int i=1; i<numberOfDisk; i++)
	{
		for (int j=0; j<radialResolution_; j++)
		{
			result->addPoint(new Vertex(lastDisks[numberOfDisk-1-i][j],moving2));
		}
		// Ajout des triangles - attention � la structure en cercle
		for (int k=0; k<radialResolution_-1; k++)
		{
			result->addTriangle((i-1)*radialResolution_+k,(i-1)*radialResolution_+k+1,i*radialResolution_+k);
			result->addTriangle((i-1)*radialResolution_+k+1,i*radialResolution_+k+1,i*radialResolution_+k);
		}
		// Ajout des deux derniers triangles pour fermer le tube
		result->addTriangle((i-1)*radialResolution_+radialResolution_-1,(i-1)*radialResolution_,i*radialResolution_+radialResolution_-1);
		result->addTriangle((i-1)*radialResolution_,i*radialResolution_,i*radialResolution_+radialResolution_-1);
	}
    
	return result;
}

void SpinalCord::assembleMeshes(SpinalCord* partOfMesh, int numberOfDisk, int radial_resolution_part)
{
	vector<Vertex*> pointsPart = partOfMesh->getListPoints();
	vector<int> trianglesPart = partOfMesh->getListTriangles();
	int nbrPoints = points_.size();
	int nbrPointsPart = pointsPart.size();
	int offsetTriangles = nbrPoints-radial_resolution_part, offsetTrianglesPart = nbrPointsPart-(numberOfDisk)*radial_resolution_part;
	for (int i=1; i<numberOfDisk; i++)
	{
		for (int j=0; j<radial_resolution_part; j++)
		{
			addPoint(new Vertex(*pointsPart[i*radial_resolution_part+j]));
			addTriangle(trianglesPart[6*((i-1)*radial_resolution_part+j)]+offsetTriangles-offsetTrianglesPart,trianglesPart[6*((i-1)*radial_resolution_part+j)+1]+offsetTriangles-offsetTrianglesPart,trianglesPart[6*((i-1)*radial_resolution_part+j)+2]+offsetTriangles-offsetTrianglesPart);
			addTriangle(trianglesPart[6*((i-1)*radial_resolution_part+j)+3]+offsetTriangles-offsetTrianglesPart,trianglesPart[6*((i-1)*radial_resolution_part+j)+4]+offsetTriangles-offsetTrianglesPart,trianglesPart[6*((i-1)*radial_resolution_part+j)+5]+offsetTriangles-offsetTrianglesPart);
		}
	}
}

