#include "Mesh.h"
#include <algorithm>
#include <vector>
#include <vtkPlane.h>
#include <vtkCutter.h>
#include <vtkClipPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkCellArray.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataReader.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkIterativeClosestPointTransform.h>
#include <vtkMatrix4x4.h>
#include <vtkLandmarkTransform.h>
#include <vtkDoubleArray.h>
#include <vtkDecimatePro.h>
#include <vtkPolyDataNormals.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkLoopSubdivisionFilter.h>
#include <vtkBYUWriter.h>

#include <itkImage.h>
#include <itkTriangleMeshToBinaryImageFilter.h>
#include <itkTriangleCell.h>
#include <itkCastImageFilter.h>
#include <itkImageRegionConstIterator.h>

using namespace std;

typedef itk::Image< double, 3 > ImageType;
typedef itk::Image< unsigned char, 3 > BinaryImageType;
typedef itk::Mesh< double, 3 > MeshTypeB;
typedef MeshTypeB::CellType CellInterfaceB;
typedef itk::TriangleCell<CellInterfaceB> CellTypeB;
typedef itk::TriangleMeshToBinaryImageFilter<MeshTypeB, BinaryImageType> MeshFilterType;
typedef itk::CastImageFilter< ImageType, BinaryImageType > CastFilterType;
typedef itk::ImageRegionConstIterator<BinaryImageType> ImageIterator;
typedef itk::Point< double, 3 > PointType;


Mesh::Mesh()
{
    label_ = 1;
    is_selected = false;
    repereLocalbool = false;
    to_draw = true;
    verbose_ = false;
}


Mesh::Mesh(const Mesh& m)
{
    label_ = m.label_;
    is_selected = m.is_selected;
    repereLocalbool = m.repereLocalbool;
    to_draw = m.to_draw;

    for (unsigned int i=0; i<m.points_.size(); i++)
        points_.push_back(new Vertex(*m.points_[i]));
    for (unsigned int i=0; i<m.pointsLocal_.size(); i++)
        pointsLocal_.push_back(new Vertex(*m.pointsLocal_[i]));
    triangles_ = m.triangles_;
    for (unsigned int i=0; i<m.trianglesBarycentre_.size(); i++)
        trianglesBarycentre_.push_back(new Vertex(*m.trianglesBarycentre_[i]));
    connectiviteTriangles_ = m.connectiviteTriangles_;
    neighbors_ = m.neighbors_;
    for (unsigned int i=0; i<m.markers_.size(); i++)
        markers_.push_back(new Vertex(*m.markers_[i]));
    verbose_ = false;
}


Mesh::~Mesh()
{
}


void Mesh::clear()
{
    for (unsigned int i=0; i<points_.size(); i++)
        delete points_[i];
    points_.clear();
    triangles_.clear();
}


int Mesh::addPoint(Vertex *v)
{
    points_.push_back(v);
    return points_.size()-1;
}


int Mesh::addPointLocal(Vertex *v)
{
    pointsLocal_.push_back(v);
    return points_.size()-1;
}


void Mesh::removeLastPoints(int number)
{
    for (int i=0; i<number; i++)
    {
        delete points_[points_.size()-1];
        points_.pop_back();
    }
}


void Mesh::addTriangle(int p1, int p2, int p3)
{
    triangles_.push_back(p1);
    triangles_.push_back(p2);
    triangles_.push_back(p3);
}


void Mesh::removeLastTriangles(int number)
{
    for (int i=0; i<3*number; i++)
        triangles_.pop_back();
}


void Mesh::setReferential(const Referential& ref, bool local)
{
    //cout << "Mise a jour du referentiel. Calcul des points locaux...";
    repereLocal_ = ref;
    repereLocalbool = true;
    CMatrix4x4 transformation = repereLocal_.getTransformation();
    CMatrix3x3 rotation = transformation;
    CVector3 translation(transformation[12],transformation[13],transformation[14]);
    calculateLocalPoints(rotation,translation);
    //cout << " Done" << endl;
}


void Mesh::calculateLocalPoints(CMatrix3x3 rotation, CVector3 translation)
{
    if (pointsLocal_.size() > 0) {
        //for (unsigned int i=0; i<pointsLocal_.size(); i++)
        //pointsLocal_[i]->setPosition(
    }
    else {
        for (unsigned int i=0; i<points_.size(); i++)
        {
            Vertex* newPoint = new Vertex(*points_[i]);
            newPoint->setPosition(rotation*(newPoint->getPosition()+translation));
            pointsLocal_.push_back(newPoint);
        }
    }
}


void Mesh::computeTrianglesBarycentre()
{
    if (trianglesBarycentre_.size() > 0) {
        for (unsigned int i=0; i<trianglesBarycentre_.size(); i++)
            delete trianglesBarycentre_[i];
        trianglesBarycentre_.clear();
    }
    CVector3 point1, point2, point3;
    for (unsigned int i=0; i<triangles_.size(); i+=3) {
        point1 = points_[triangles_[i]]->getPosition();
        point2 = points_[triangles_[i+1]]->getPosition();
        point3 = points_[triangles_[i+2]]->getPosition();
        trianglesBarycentre_.push_back(new Vertex((point1+point2+point3)/3,((point1-point2)^(point1-point3)).Normalize()));
    }
}


void Mesh::transform(CMatrix4x4 transformation)
{
    for (unsigned int i=0; i<points_.size(); i++)
        points_[i]->setPosition(transformation*points_[i]->getPosition());
    computeTrianglesBarycentre();
}


void Mesh::transform(CMatrix4x4 transformation, CVector3 rotationPoint)
{
    for (unsigned int i=0; i<points_.size(); i++)
        points_[i]->setPosition(transformation*(points_[i]->getPosition()-rotationPoint)+rotationPoint);
    computeTrianglesBarycentre();
}

void Mesh::localTransform(CMatrix4x4 transformation)
{
    if (repereLocalbool) {
        CMatrix3x3 rotation = transformation;
        CVector3 translation(transformation[12],transformation[13],transformation[14]);
        for (unsigned int i=0; i<pointsLocal_.size(); i++)
            pointsLocal_[i]->setPosition(rotation*(points_[i]->getPosition()+translation));
        CMatrix4x4 transfRef = repereLocal_.getTransformation();
        CMatrix3x3 rotationInv = transfRef;
        CVector3 translationInv(transfRef[12],transfRef[13],transfRef[14]);
        for (unsigned int i=0; i<points_.size(); i++)
            points_[i]->setPosition(rotationInv*(points_[i]->getPosition()+translationInv));
    }
}

void Mesh::computeConnectivity()
{
	unsigned int pointSize = points_.size();
	connectiviteTriangles_.resize(pointSize);
	unsigned int num, size;
	for (unsigned int i=0; i<pointSize; i++) {
		vector<int> temp;
		size = triangles_.size();
		for (unsigned int j=0; j<size; j++) {
			if (j==size-1) num = j/3;
			else num = (j+1)/3;
			if (triangles_[j] == i) temp.push_back(num);
		}
		connectiviteTriangles_[i] = temp;
	}

	neighbors_.resize(pointSize);
	int pointCourant;
	for (unsigned int i=0; i<pointSize; i++) {
		vector<int> temp;
		for (unsigned int j=0; j<connectiviteTriangles_[i].size(); j++) {
			for (int k=0; k<3; k++) {
				pointCourant = triangles_[3*connectiviteTriangles_[i][j]+k];
				if (pointCourant != i && count(temp.begin(), temp.end(), pointCourant) == 0)
					temp.push_back(pointCourant);
			}
		}
		neighbors_[i] = temp;
	}
}


void Mesh::save(string filename, ImageType::Pointer image_ref)
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
    
    if (image_ref.IsNotNull())
    {
        ImageType::RegionType largestRegion = image_ref->GetLargestPossibleRegion();
        ImageType::IndexType downSliceIndex, downSliceMIndex, upperSliceIndex, upperSliceMIndex;
        downSliceIndex.Fill(0);
        downSliceMIndex.Fill(0);
        downSliceMIndex[1] = 1;
        upperSliceIndex = largestRegion.GetUpperIndex();
        upperSliceMIndex = upperSliceIndex; upperSliceMIndex[1] = upperSliceMIndex[1]-1;
        PointType downSlicePoint, downSliceMPoint, upperSlicePoint, upperSliceMPoint;
        image_ref->TransformIndexToPhysicalPoint(downSliceIndex, downSlicePoint);
        image_ref->TransformIndexToPhysicalPoint(downSliceMIndex, downSliceMPoint);
        image_ref->TransformIndexToPhysicalPoint(upperSliceIndex, upperSlicePoint);
        image_ref->TransformIndexToPhysicalPoint(upperSliceMIndex, upperSliceMPoint);
        
        CVector3 downSliceNormal = CVector3(downSlicePoint[0]-downSliceMPoint[0],downSlicePoint[1]-downSliceMPoint[1],downSlicePoint[2]-downSliceMPoint[2]).Normalize(), upperSliceNormal = CVector3(upperSlicePoint[0]-upperSliceMPoint[0],upperSlicePoint[1]-upperSliceMPoint[1],upperSlicePoint[2]-upperSliceMPoint[2]).Normalize();
        
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
        source = upperClipper->GetOutput();
    }
    
	vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
	writer->SetFileName(filename.c_str());
	writer->SetInputData(source);
	writer->Write();
}


void Mesh::saveBYU(string filename)
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
	vtkSmartPointer<vtkBYUWriter> writer = vtkSmartPointer<vtkBYUWriter>::New();
	string fileN = filename + "_mesh.byu";
	writer->SetGeometryFileName(fileN.c_str());
	writer->SetInputData(source);
	writer->Write();
}


void Mesh::read(string filename)
{
	if (verbose_) cout << "Debut de lecture du fichier VTK " << filename << endl;

	vtkSmartPointer<vtkPolyDataReader> readerVTK = vtkSmartPointer<vtkPolyDataReader>::New();
	readerVTK->SetFileName(filename.c_str());
	readerVTK->Update();
	vtkSmartPointer<vtkPolyData> data = vtkSmartPointer<vtkPolyData>::New();
	data = readerVTK->GetOutput();

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	points = data->GetPoints();

	double *pt, *norm;
	CVector3 point, normale;
	label_ = 1;

	vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
	if (data->GetNumberOfPolys() == 0) {
		polys = data->GetStrips();
		vtkSmartPointer<vtkCellArray> polytemp = vtkSmartPointer<vtkCellArray>::New();
		vtkIdType nbTriangle, *po;
		vtkIdType pts[3];
		for (vtkIdType i = 0; i<polys->GetNumberOfCells(); i++)
		{
			polys->GetNextCell(nbTriangle,po);
			for (vtkIdType j=0; j<nbTriangle-2; j++) {
				if (j%2 == 1) {
					pts[0] = po[j];
					pts[1] = po[j+2];
					pts[2] = po[j+1];
				} else {
					pts[0] = po[j];
					pts[1] = po[j+1];
					pts[2] = po[j+2];
				}
				polytemp->InsertNextCell(3,pts);
			}
		}
		polys = polytemp;
	}
	else
		polys = data->GetPolys();

	Vertex* newPoint;
	for (vtkIdType i = 0; i<points->GetNumberOfPoints(); i++)
	{
		pt = points->GetPoint(i);
		norm = data->GetPointData()->GetNormals()->GetTuple(i);
		point = CVector3(pt[0],pt[1],pt[2]);
		normale = CVector3(norm[0],norm[1],norm[2]);
		newPoint = new Vertex(point,normale,1);
		points_.push_back(newPoint);
	}

	if (verbose_) cout << "Nombre de points : " << points->GetNumberOfPoints() << endl;
	if (verbose_) cout << "Nombre de triangles : " << polys->GetNumberOfCells() << endl;

	vtkIdType nbTriangle, *po;
	int nVert;
	for (vtkIdType i = 0; i<polys->GetNumberOfCells(); i++)
	{
		polys->GetNextCell(nbTriangle,po);
		addTriangle(po[0],po[1],po[2]);
	}

	/*cout << "Calcul des connectivites...";
	computeConnectivity();
	cout << " Done" << endl;*/
}


CMatrix4x4 Mesh::ICP(Mesh *vTarget)
{
	vtkSmartPointer<vtkPoints> points1 = vtkSmartPointer<vtkPoints>::New();
	for (unsigned int i=0; i<points_.size(); i++)
		points1->InsertNextPoint(points_[i]->getPosition()[0],points_[i]->getPosition()[1],points_[i]->getPosition()[2]);
	vtkSmartPointer<vtkCellArray> polys1 = vtkSmartPointer<vtkCellArray>::New();
	for (unsigned int i=0; i<triangles_.size(); i+=3) {
		vtkIdType pts[3] = {triangles_[i],triangles_[i+1],triangles_[i+2]};
		polys1->InsertNextCell(3,pts);
	}
	vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
	source->SetPoints(points1);
	source->SetPolys(polys1);

	vtkSmartPointer<vtkPoints> points2 = vtkSmartPointer<vtkPoints>::New();
	for (unsigned int i=0; i<vTarget->points_.size(); i++)
		points2->InsertNextPoint(vTarget->points_[i]->getPosition()[0],vTarget->points_[i]->getPosition()[1],vTarget->points_[i]->getPosition()[2]);
	vtkSmartPointer<vtkCellArray> polys2 = vtkSmartPointer<vtkCellArray>::New();
	for (unsigned int i=0; i<vTarget->triangles_.size(); i+=3) {
		vtkIdType pts[3] = {vTarget->triangles_[i],vTarget->triangles_[i+1],vTarget->triangles_[i+2]};
		polys2->InsertNextCell(3,pts);
	}
	vtkSmartPointer<vtkPolyData> target = vtkSmartPointer<vtkPolyData>::New();
	target->SetPoints(points2);
	target->SetPolys(polys2);


	// Setup ICP transform
	vtkSmartPointer<vtkIterativeClosestPointTransform> icp = vtkSmartPointer<vtkIterativeClosestPointTransform>::New();
	icp->SetSource(source);
	icp->SetTarget(target);
	icp->GetLandmarkTransform()->SetModeToRigidBody();
	icp->SetMaximumNumberOfIterations(20);
	//icp->StartByMatchingCentroidsOn();
	icp->Modified();
	icp->Update();
 
	// Get the resulting transformation matrix (this matrix takes the source points to the target points)
	vtkSmartPointer<vtkMatrix4x4> m = icp->GetMatrix();
	CMatrix4x4 transformation;
	for (int i=0; i<4; i++) {
		for (int j=0; j<4; j++)
			transformation[4*j+i] = (*m)[i][j];
	}
	
	return transformation;
}


void Mesh::decimation(float nb)
{
	double ratio = 1.0;
	if (nb >= 1.0)
		ratio = (double)nb/triangles_.size();
	else
		ratio = nb;
	if (verbose_) cout << "Demarage de la decimation. Cible : " << nb/3 << " Ratio : " << ratio << endl;
	int label = getLabel();

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	for (unsigned int i=0; i<points_.size(); i++)
		points->InsertNextPoint(points_[i]->getPosition()[0],points_[i]->getPosition()[1],points_[i]->getPosition()[2]);
	vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
	for (unsigned int i=0; i<triangles_.size(); i+=3) {
		vtkIdType pts[3] = {triangles_[i],triangles_[i+1],triangles_[i+2]};
		polys->InsertNextCell(3,pts);
	}

	vtkSmartPointer<vtkPolyData> pPolyData = vtkSmartPointer<vtkPolyData>::New();
	pPolyData->SetPoints(points);
	pPolyData->SetPolys(polys);
 
	vtkSmartPointer<vtkPolyData> input = vtkSmartPointer<vtkPolyData>::New();
	input->ShallowCopy(pPolyData);
 
    if (verbose_) {
        std::cout << "Before decimation" << std::endl << "------------" << std::endl;
        std::cout << "There are " << input->GetNumberOfPoints() << " points." << std::endl;
        std::cout << "There are " << input->GetNumberOfPolys() << " polygons." << std::endl;
    }
    
	vtkSmartPointer<vtkDecimatePro> decimate = vtkSmartPointer<vtkDecimatePro>::New();
	#if VTK_MAJOR_VERSION <= 5
		decimate->SetInputConnection(input->GetProducerPort());
	#else
		decimate->SetInputData(input);
	#endif
	decimate->SetTargetReduction(1-ratio); // exemple 0.1 : 10% reduction -> if there was 100 triangles, now there will be 90
	decimate->Update();

	vtkSmartPointer<vtkPolyDataNormals> skinNormals = vtkSmartPointer<vtkPolyDataNormals>::New();
	skinNormals->SetInputConnection(decimate->GetOutputPort());
	skinNormals->SetFeatureAngle(60.0);
	skinNormals->ComputePointNormalsOn();
	skinNormals->ComputeCellNormalsOff();
	skinNormals->ConsistencyOff();
	skinNormals->SplittingOff();
	skinNormals->Update();
 
	vtkSmartPointer<vtkPolyData> decimated = vtkSmartPointer<vtkPolyData>::New();
	decimated->ShallowCopy(skinNormals->GetOutput());
	
	this->clear();
	points = decimated->GetPoints();
	double *pt, *norm;
	for (vtkIdType i = 0; i<points->GetNumberOfPoints(); i++)
	{
		pt = points->GetPoint(i);
		norm = decimated->GetPointData()->GetNormals()->GetTuple(i);
		CVector3 p = CVector3(pt[0],pt[1],pt[2]), n = CVector3(norm[0],norm[1],norm[2]);
		addPoint(new Vertex(p,n,label));
	}
	polys = decimated->GetPolys();
	vtkSmartPointer<vtkIdList> cells = vtkSmartPointer<vtkIdList>::New();
	vtkIdType nbTriangle, *p;
	for (vtkIdType i = 0; i<polys->GetNumberOfCells(); i++)
	{
		polys->GetCell(4*i,nbTriangle,p);
		addTriangle(p[0],p[1],p[2]);
	}
 
    if (verbose_) {
        std::cout << "After decimation" << std::endl << "------------" << std::endl;
 
        std::cout << "There are " << decimated->GetNumberOfPoints() << " points." << std::endl;
        std::cout << "There are " << decimated->GetNumberOfPolys() << " polygons." << std::endl;
    }

	/*cout << "Calcul des connectivites...";
	computeConnectivity();
	cout << " Done" << endl;*/
}


void Mesh::subdivision(int numberOfSubdivision, bool computeFinalMesh)
{
	int label = getLabel();
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	for (unsigned int i=0; i<points_.size(); i++)
		points->InsertNextPoint(points_[i]->getPosition()[0],points_[i]->getPosition()[1],points_[i]->getPosition()[2]);
	vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
	for (unsigned int i=0; i<triangles_.size(); i+=3) {
		vtkIdType pts[3] = {triangles_[i],triangles_[i+1],triangles_[i+2]};
		polys->InsertNextCell(3,pts);
	}

	vtkSmartPointer<vtkPolyData> pPolyData = vtkSmartPointer<vtkPolyData>::New();
	pPolyData->SetPoints(points);
	pPolyData->SetPolys(polys);
 
	vtkSmartPointer<vtkPolyData> input = vtkSmartPointer<vtkPolyData>::New();
	input->ShallowCopy(pPolyData);

	/*std::cout << "Before subdivision" << std::endl << "------------" << std::endl;
	std::cout << "There are " << input->GetNumberOfPoints() << " points." << std::endl;
	std::cout << "There are " << input->GetNumberOfPolys() << " polygons." << std::endl;*/


	// Subdivision
	vtkSmartPointer<vtkLoopSubdivisionFilter> subdivisionFilter = vtkSmartPointer<vtkLoopSubdivisionFilter>::New();
	subdivisionFilter->SetNumberOfSubdivisions(numberOfSubdivision);
	#if VTK_MAJOR_VERSION <= 5
		subdivisionFilter->SetInputConnection(input->GetProducerPort());
	#else
		subdivisionFilter->SetInputData(input);
	#endif
		subdivisionFilter->Update();

	if (computeFinalMesh)
	{
		vtkSmartPointer<vtkPolyDataNormals> skinNormals = vtkSmartPointer<vtkPolyDataNormals>::New();
		skinNormals->SetInputConnection(subdivisionFilter->GetOutputPort());
		skinNormals->SetFeatureAngle(60.0);
		skinNormals->ComputePointNormalsOn();
		skinNormals->ComputeCellNormalsOff();
		skinNormals->ConsistencyOff();
		skinNormals->SplittingOff();
		skinNormals->Update();

		vtkSmartPointer<vtkPolyData> subdivised = skinNormals->GetOutput();


		this->clear();
		points = subdivised->GetPoints();
		double *pt, *norm;
		for (vtkIdType i = 0; i<points->GetNumberOfPoints(); i++)
		{
			pt = points->GetPoint(i);
			norm = subdivised->GetPointData()->GetNormals()->GetTuple(i);
			CVector3 p = CVector3(pt[0],pt[1],pt[2]), n = CVector3(norm[0],norm[1],norm[2]);
			addPoint(new Vertex(p,n,label));
		}
		polys = subdivised->GetPolys();
		vtkSmartPointer<vtkIdList> cells = vtkSmartPointer<vtkIdList>::New();
		vtkIdType nbTriangle, *p;
		for (vtkIdType i = 0; i<polys->GetNumberOfCells(); i++)
		{
			polys->GetCell(4*i,nbTriangle,p);
			addTriangle(p[0],p[1],p[2]);
		}
 
		/*std::cout << "After subdivision" << std::endl << "------------" << std::endl;
		std::cout << "There are " << subdivised->GetNumberOfPoints() << " points." << std::endl;
		std::cout << "There are " << subdivised->GetNumberOfPolys() << " polygons." << std::endl;*/
	}
	else
	{
		vtkSmartPointer<vtkPolyData> subdivised = subdivisionFilter->GetOutput();
		this->clear();
		points = subdivised->GetPoints();
		double *pt, *norm;
		for (vtkIdType i = 0; i<points->GetNumberOfPoints(); i++)
		{
			pt = points->GetPoint(i);
			addPoint(new Vertex(pt[0],pt[1],pt[2]));
		}
		polys = subdivised->GetPolys();
		vtkSmartPointer<vtkIdList> cells = vtkSmartPointer<vtkIdList>::New();
		vtkIdType nbTriangle, *p;
		for (vtkIdType i = 0; i<polys->GetNumberOfCells(); i++)
		{
			polys->GetCell(4*i,nbTriangle,p);
			addTriangle(p[0],p[1],p[2]);
		}
	}
}


void Mesh::smoothing(int numberOfIterations)
{
	if (verbose_) cout << "Smoothing du mesh...";
	int label = getLabel();

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	for (unsigned int i=0; i<points_.size(); i++)
		points->InsertNextPoint(points_[i]->getPosition()[0],points_[i]->getPosition()[1],points_[i]->getPosition()[2]);
	vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
	for (unsigned int i=0; i<triangles_.size(); i+=3) {
		vtkIdType pts[3] = {triangles_[i],triangles_[i+1],triangles_[i+2]};
		polys->InsertNextCell(3,pts);
	}

	vtkSmartPointer<vtkPolyData> pPolyData = vtkSmartPointer<vtkPolyData>::New();
	pPolyData->SetPoints(points);
	pPolyData->SetPolys(polys);
 
	vtkSmartPointer<vtkPolyData> input = vtkSmartPointer<vtkPolyData>::New();
	input->ShallowCopy(pPolyData);
 
	vtkSmartPointer<vtkSmoothPolyDataFilter> smooth = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
	#if VTK_MAJOR_VERSION <= 5
		smooth->SetInputConnection(input->GetProducerPort());
	#else
		smooth->SetInputData(input);
	#endif
	smooth->SetNumberOfIterations(numberOfIterations);
	smooth->Update();

	vtkSmartPointer<vtkPolyDataNormals> skinNormals = vtkSmartPointer<vtkPolyDataNormals>::New();
	skinNormals->SetInputConnection(smooth->GetOutputPort());
	skinNormals->SetFeatureAngle(60.0);
	skinNormals->ComputePointNormalsOn();
	skinNormals->ComputeCellNormalsOff();
	skinNormals->ConsistencyOff();
	skinNormals->SplittingOff();
	skinNormals->Update();
 
	vtkSmartPointer<vtkPolyData> smoothed = vtkSmartPointer<vtkPolyData>::New();
	smoothed->ShallowCopy(skinNormals->GetOutput());
	
	this->clear();
	points = smoothed->GetPoints();
	double *pt, *norm;
	for (vtkIdType i = 0; i<points->GetNumberOfPoints(); i++)
	{
		pt = points->GetPoint(i);
		norm = smoothed->GetPointData()->GetNormals()->GetTuple(i);
		CVector3 p = CVector3(pt[0],pt[1],pt[2]), n = CVector3(norm[0],norm[1],norm[2]);
		addPoint(new Vertex(p,n,label));
	}
	polys = smoothed->GetPolys();
	vtkSmartPointer<vtkIdList> cells = vtkSmartPointer<vtkIdList>::New();
	vtkIdType nbTriangle, *p;
	for (vtkIdType i = 0; i<polys->GetNumberOfCells(); i++)
	{
		polys->GetCell(4*i,nbTriangle,p);
		addTriangle(p[0],p[1],p[2]);
	}
	if (verbose_) cout << " Done" << endl;

	/*cout << "Calcul des connectivites...";
	computeConnectivity();
	cout << " Done" << endl;*/
}


// Les points entre les deux maillages doivent \EAtre en nombres \E9gaux et correspondants
double Mesh::distanceMean(Mesh *sp)
{
	double result = 0.0;
	unsigned int sizePoints = points_.size();
	for (unsigned int i=0; i<sizePoints; i++)
		result += points_[i]->distance(*sp->points_[i]);
	result /= (double)sizePoints;
	return result;
}



void Mesh::computeMeshNormals()
{
	int label = getLabel();

	if (verbose_) cout << "Calcul des normales";
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	for (unsigned int i=0; i<points_.size(); i++)
		points->InsertNextPoint(points_[i]->getPosition()[0],points_[i]->getPosition()[1],points_[i]->getPosition()[2]);
	vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
	for (unsigned int i=0; i<triangles_.size(); i+=3) {
		vtkIdType pts[3] = {triangles_[i],triangles_[i+1],triangles_[i+2]};
		polys->InsertNextCell(3,pts);
	}

	vtkSmartPointer<vtkPolyData> pPolyData = vtkSmartPointer<vtkPolyData>::New();
	pPolyData->SetPoints(points);
	pPolyData->SetPolys(polys);


	vtkSmartPointer<vtkPolyDataNormals> skinNormals = vtkSmartPointer<vtkPolyDataNormals>::New();
	skinNormals->SetInputData(pPolyData);
	skinNormals->SetFeatureAngle(60.0);
	skinNormals->ComputePointNormalsOn();
	skinNormals->ComputeCellNormalsOff();
	skinNormals->ConsistencyOff();
	skinNormals->SplittingOff();
	skinNormals->Update();
 
	vtkSmartPointer<vtkPolyData> output = vtkSmartPointer<vtkPolyData>::New();
	output->ShallowCopy(skinNormals->GetOutput());
	
	this->clear();
	points = output->GetPoints();
	double *pt, *norm;
	for (vtkIdType i = 0; i<points->GetNumberOfPoints(); i++)
	{
		pt = points->GetPoint(i);
		norm = output->GetPointData()->GetNormals()->GetTuple(i);
		CVector3 p = CVector3(pt[0],pt[1],pt[2]), n = CVector3(norm[0],norm[1],norm[2]);
		addPoint(new Vertex(p,n,label));
	}
	polys = output->GetPolys();
	vtkSmartPointer<vtkIdList> cells = vtkSmartPointer<vtkIdList>::New();
	vtkIdType nbTriangle, *p;
	for (vtkIdType i = 0; i<polys->GetNumberOfCells(); i++)
	{
		polys->GetCell(4*i,nbTriangle,p);
		addTriangle(p[0],p[1],p[2]);
	}
	if (verbose_) cout << " Done" << endl;
}


double Mesh::computeStandardDeviationFromPixelsInside(ImageType::Pointer image)
{
    MeshTypeB::Pointer mesh = MeshTypeB::New();
    typedef itk::Point< double, 3 > PointType;
    PointType pnt;
    ImageType::IndexType index;
    CVector3 p, n, min(10000,10000,10000), max;
    for (unsigned int i=0; i<points_.size(); i++) {
        p = points_[i]->getPosition();
        pnt[0] = p[0]; pnt[1] = p[1]; pnt[2] = p[2];
        mesh->SetPoint(i,pnt);
        
        image->TransformPhysicalPointToIndex(pnt, index);
        if (index[0]<min[0]) min[0]=index[0];
        if (index[1]<min[1]) min[1]=index[1];
        if (index[2]<min[2]) min[2]=index[2];
        if (index[0]>max[0]) max[0]=index[0];
        if (index[1]>max[1]) max[1]=index[1];
        if (index[2]>max[2]) max[2]=index[2];
    }
    for (unsigned int i=0; i<triangles_.size(); i+=3)
    {
        CellTypeB::CellAutoPointer triangle;
        triangle.TakeOwnership(new CellTypeB);
        triangle->SetPointId(0,triangles_[i]);
        triangle->SetPointId(1,triangles_[i+1]);
        triangle->SetPointId(2,triangles_[i+2]);
        mesh->SetCell((int)(i+1)/3,triangle);
    }
    
    
    CastFilterType::Pointer castFilter = CastFilterType::New();
    castFilter->SetInput(image);
    BinaryImageType::Pointer im = castFilter->GetOutput();
    
    BinaryImageType::IndexType indexRegion;
    BinaryImageType::SizeType sizeRegion;
    sizeRegion[0] = max[0]-min[0]+5; // adding two pixels at each side to be sure
    sizeRegion[1] = max[1]-min[1]+5;
    sizeRegion[2] = max[2]-min[2]+5;
    indexRegion[0] = min[0]-2;
    indexRegion[1] = min[1]-2;
    indexRegion[2] = min[2]-2;
    
    BinaryImageType::RegionType region(indexRegion,sizeRegion);
    im->SetRegions(region);
    
    MeshFilterType::Pointer meshFilter = MeshFilterType::New();
    meshFilter->SetInput(mesh);
    meshFilter->SetInfoImage(im);
    /*meshFilter->SetDirection(image->GetDirection());
    meshFilter->SetSpacing(image->GetSpacing());
    meshFilter->SetOrigin(image->GetOrigin());
    meshFilter->SetSize(sizeRegion);
    meshFilter->SetIndex(indexRegion);*/
    meshFilter->SetTolerance(1.0);
    meshFilter->SetInsideValue(1.0);
    meshFilter->SetOutsideValue(0.0);
    try {
        meshFilter->Update();
    }
    catch( itk::ExceptionObject & e )
    {
        cout << "Exception thrown ! " << endl;
        cout << "An error ocurred during creating binary image" << endl;
        cout << "Location    = " << e.GetLocation()    << endl;
        cout << "Description = " << e.GetDescription() << endl;
    }
    
    MeshFilterType::OutputImageType::Pointer binary = meshFilter->GetOutput();
    
    vector<double> valueVoxels;
    
    typedef ImageType::IndexType IndexType;
    IndexType ind;
    ImageIterator it( binary, binary->GetRequestedRegion() );
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        if (it.Get()==true)
        {
            ind = it.GetIndex();
            valueVoxels.push_back(image->GetPixel(ind));
        }
        ++it;
    }
    
    double mean = 0.0, sum = 0.0, std = 0.0, nbVoxels = valueVoxels.size();
    for (int i=0; i<nbVoxels; i++)
        mean += valueVoxels[i];
    mean /= nbVoxels;
    for (int i=0; i<nbVoxels; i++) {
        sum += valueVoxels[i]*valueVoxels[i];
    }
    std = sqrt(sum/nbVoxels-mean*mean);
    
    /*ofstream myfile;
	myfile.open("StandardDeviation.txt", ios_base::app);
	myfile << ind[1] << " " << mean << " " << std << endl;
	myfile.close();*/
    
    return mean;
}

double Mesh::computeMeanRadius(int numberOfPointsPerDisk)
{
    double result = 0.0;
    int number = points_.size()%numberOfPointsPerDisk;
    for (int i=0; i<number; i++)
    {
        CVector3 meanVertex;
        for (int k=0; k<numberOfPointsPerDisk; k++)
        {
            meanVertex += points_[i*numberOfPointsPerDisk+k]->getPosition();
        }
    }
    return result;
}

