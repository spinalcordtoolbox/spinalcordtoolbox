#define _SCL_SECURE_NO_WARNINGS
#include "DeformableModelBasicAdaptator.h"

#include <vector>
#include <itkTriangleCell.h>
#include <itkPointSet.h>
#include <itkVector.h>
#include <itkDefaultDynamicMeshTraits.h>
#include <itkConjugateGradientOptimizer.h>
#include <itkLBFGSOptimizer.h>

#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>


typedef itk::DefaultDynamicMeshTraits<double,3,3,double,double,double> MeshTraits;
typedef itk::Mesh<double,3,MeshTraits>	MeshType;
typedef MeshType::PointType		PointsType;
typedef MeshType::CellType		CellInterface;
typedef itk::TriangleCell<CellInterface> CellType;
typedef itk::Vector<double,3> VectorType;


typedef itk::ConjugateGradientOptimizer  OptimizerType;
//typedef itk::LBFGSOptimizer OptimizerType;
typedef OptimizerType::InternalOptimizerType  vnlOptimizerType;

class CommandIterationUpdateConjugateGradient : public itk::Command
{
public:
    typedef  CommandIterationUpdateConjugateGradient   Self;
    typedef  itk::Command             Superclass;
    typedef itk::SmartPointer<Self>  Pointer;
    itkNewMacro( Self );
protected:
    CommandIterationUpdateConjugateGradient()
    {
        m_IterationNumber=0;
        m_IterationDerivativeNumber=0;
    }
public:
    typedef itk::ConjugateGradientOptimizer   OptimizerType;
    typedef   const OptimizerType   *    OptimizerPointer;
    
    void Execute(itk::Object *caller, const itk::EventObject & event)
    {
        Execute( (const itk::Object *)caller, event);
    }
    
    void Execute(const itk::Object * object, const itk::EventObject & event)
    {
        OptimizerPointer optimizer =
        dynamic_cast< OptimizerPointer >( object );
        if( m_FunctionEvent.CheckEvent( &event ) )
        {
            m_IterationNumber++;
            if (m_IterationNumber>50000) {
                throw itk::ExceptionObject("ERROR: Too many iterations");
                //std::cout << m_IterationNumber << "   ";
                //std::cout << optimizer->GetCachedValue() << "   ";
                //std::cout << optimizer->GetCachedCurrentPosition() << std::endl;
            }
        }
        else if( m_GradientEvent.CheckEvent( &event ) )
        {
            m_IterationDerivativeNumber++;
            if (m_IterationDerivativeNumber>10000) {
                throw itk::ExceptionObject("ERROR: Too many iterations");
                //std::cout << m_IterationDerivativeNumber << "   ";
                //std::cout << "Gradient " << optimizer->GetCachedDerivative() << "   ";
            }
        }
        
    }
private:
    unsigned long m_IterationNumber, m_IterationDerivativeNumber;
    
    
    itk::FunctionEvaluationIterationEvent m_FunctionEvent;
    itk::GradientEvaluationIterationEvent m_GradientEvent;
};


DeformableModelBasicAdaptator::DeformableModelBasicAdaptator(Image3D* image, Mesh* m) : image_(image), mesh_(m), meshBool_(true), numberOfIteration_(1)
{
	changedParameters_ = false;
	contrast = 500;
	stopCondition = 0.4;
    numberOptimizerIteration = 500;
    progressiveLineSearchLength = false;
    
    verbose_ = false;

    line_search = 15;
	alpha = 25.0;
	beta = 0.0;
}

DeformableModelBasicAdaptator::DeformableModelBasicAdaptator(Image3D* image, Mesh* m, int nbIteration, double contrast, bool computeFinalMesh) : image_(image), mesh_(m), meshBool_(computeFinalMesh), numberOfIteration_(nbIteration)
{
	changedParameters_ = false;
	this->contrast = contrast;
	stopCondition = 0.4;
	numberOptimizerIteration = 500;
    progressiveLineSearchLength = false;

	tradeOff = 0.0;
	tradeoff_bool = false;
    
    verbose_ = false;

    line_search = 15;
	alpha = 25.0;
	beta = 0.0;
}

DeformableModelBasicAdaptator::DeformableModelBasicAdaptator(Image3D* image, Mesh* m, int nbIteration, vector<pair<CVector3,double> > contrast, bool computeFinalMesh) : image_(image), mesh_(m), meshBool_(computeFinalMesh), numberOfIteration_(nbIteration)
{
	changedParameters_ = false;
    this->contrast = -1.0;
	contrastvector = contrast;
	stopCondition = 0.4;
	numberOptimizerIteration = 500;
    progressiveLineSearchLength = false;

	tradeOff = 0.0;
	tradeoff_bool = false;
    
    verbose_ = false;

    line_search = 15;
	alpha = 25.0;
	beta = 0.0;
}

DeformableModelBasicAdaptator::~DeformableModelBasicAdaptator()
{
}

double DeformableModelBasicAdaptator::adaptation()
{
	//if (verbose_) cout << "Creation des variables, de l'optimiseur et de la fonction de cout..." << endl;
	vector<Vertex*> points = mesh_->getListPoints();
	int nbPoints = points.size();
	OptimizerType::ParametersType initialValue(3*nbPoints);
	CVector3 p;
	for (unsigned int i=0; i<nbPoints; i++) {
		p = points[i]->getPosition();
		initialValue[3*i] = p[0];
		initialValue[3*i+1] = p[1];
		initialValue[3*i+2] = p[2];
	}

	OptimizerType::ParametersType currentValue(3*nbPoints);
	currentValue = initialValue;


	vector<int> triangles = mesh_->getListTriangles();

	FoncteurDeformableBasicLocalAdaptation* costFunction = new FoncteurDeformableBasicLocalAdaptation(image_,mesh_,initialValue,nbPoints);
    costFunction->setVerbose(verbose_);
    costFunction->addCorrectionPoints(points_mask_correction_);
	/*if (contrast != -1.0) costFunction->setTradeOff(0.0001*contrast*contrast+0.026*contrast-1.6242);
    else {
        unsigned int index_nearest = 0;
        double distanceMin = 10000.0;
        for (unsigned int i=0; i<contrastvector.size(); i++)
        {
            double distance = sqrt((initialValue[0]*contrastvector[i].first[0])*(initialValue[0]*contrastvector[i].first[0])+(initialValue[0]*contrastvector[i].first[0])*(initialValue[0]*contrastvector[i].first[0])+(initialValue[0]*contrastvector[i].first[0])*(initialValue[0]*contrastvector[i].first[0]));
            if (distance < distanceMin) {
                distanceMin = distance;
                index_nearest = i;
            }
        }
        contrast = contrastvector[index_nearest].second;
        costFunction->setTradeOff(0.0001*contrast*contrast+0.026*contrast-1.6242);
    }*/
    //double a = 0.0001, b = 0.026, c = -1.6242; // initial
    double a, b, c;
    if (image_->getTypeImageFactor() == 1.0) { // T2
        a = 0.0001, b = 0.02, c = -1.6242;
    } else { // T1
        a = 0.0001, b = 0.02, c = -1.6242; //b = 0.013
    }
	if (tradeoff_bool) costFunction->setTradeOff(tradeOff);
    else if (contrast != -1.0) costFunction->setTradeOff(a*contrast*contrast+b*contrast+c);
    else {
        double mean = 0.0;
        for (unsigned int i=0; i<contrastvector.size(); i++)
            mean += contrastvector[i].second;
        mean /= (double)contrastvector.size();
        costFunction->setTradeOff(a*mean*mean+b*mean+c);
    }
    double lineSearchLength = costFunction->getLineSearchLength();
	//cout << "Contrast = " << contrast << " & TradeOff = " << 0.0001*contrast*contrast+0.026*contrast-1.6242 << endl;
	if (this->changedParameters_) {
		costFunction->setAlpha(alpha);
		costFunction->setBeta(beta);
		costFunction->setLineSearchLength(line_search);
		costFunction->computeOptimalPoints(initialValue);
	}

	OptimizerType::Pointer itkOptimizer = OptimizerType::New();
	itkOptimizer->SetCostFunction( costFunction );
	itkOptimizer->SetInitialPosition( currentValue );

	const double F_Tolerance      = 1e-4;  // Function value tolerance
	const double G_Tolerance      = 1e-6;  // Gradient magnitude tolerance 
	const double X_Tolerance      = 1e-8;  // Search space tolerance
	const double Epsilon_Function = 1e-10; // Step
	const int    Max_Iterations   = numberOptimizerIteration; // Maximum number of iterations

	vnlOptimizerType * vnlOptimizer = itkOptimizer->GetOptimizer();
	vnlOptimizer->set_f_tolerance( F_Tolerance );
	vnlOptimizer->set_g_tolerance( G_Tolerance );
	vnlOptimizer->set_x_tolerance( X_Tolerance ); 
	vnlOptimizer->set_epsilon_function( Epsilon_Function );
	vnlOptimizer->set_max_function_evals( Max_Iterations );

	vnlOptimizer->set_check_derivatives( 1 );
    
    CommandIterationUpdateConjugateGradient::Pointer observer =
    CommandIterationUpdateConjugateGradient::New();
    itkOptimizer->AddObserver( itk::IterationEvent(), observer );
    itkOptimizer->AddObserver( itk::FunctionEvaluationIterationEvent(), observer );
	
	//cout << "Valeur initiale de la fonction de cout = " << costFunction->GetInitialValue() << endl;
	//cout << "Valeur initiale de la norme des derivees = " << costFunction->getInitialNormDerivatives() << endl;
	
	//cout << "Optimisation du maillage..." << endl;
	bool done = false, error_occured = false;
	double distance = 0.0;
	Mesh* lastMesh = new Mesh;
	double newDistanceMeshInitial = 0.0;
	for (int i=0; i<numberOfIteration_ && !done; i++)
	{
        if (progressiveLineSearchLength) {
            double factor = i;
            if (factor > 4) factor = 4.0;
            costFunction->setTradeOff(costFunction->getTradeOff()*0.8);
            //costFunction->setTradeOff(costFunction->getTradeOff()/(1.5-factor/8.0));
            //costFunction->setLineSearchLength((3.0-factor/2.0)*lineSearchLength);
        }
		try
		{
		    //if (verbose_) cout << "start optimization" << endl;
			itkOptimizer->StartOptimization();
		}
		catch( itk::ExceptionObject & e )
		{
			cout << "Exception thrown ! " << endl;
			cout << "An error ocurred during Optimization" << endl;
			cout << "Location    = " << e.GetLocation()    << endl;
			cout << "Description = " << e.GetDescription() << endl;
            error_occured = true;
		}
        if (verbose_) {
            cout << "Report from vnl optimizer for deformation : " << endl;
            vnlOptimizer->diagnose_outcome( cout );
            cout << endl;
        }

		currentValue = itkOptimizer->GetCurrentPosition();
        if (currentValue.size() == 0)
            currentValue = itkOptimizer->GetCachedCurrentPosition();
        

		Mesh* newMesh = new Mesh(*mesh_);
		vector<Vertex*> pointsNewMesh = newMesh->getListPoints();
		for (unsigned int i=0; i<nbPoints; i++)
			pointsNewMesh[i]->setPosition(CVector3(currentValue[3*i],currentValue[3*i+1],currentValue[3*i+2]));
		CMatrix4x4 transform = mesh_->ICP(newMesh);
		newDistanceMeshInitial = mesh_->distanceMean(newMesh);
		if (verbose_) cout << "Distance from initial mesh [mm] = " << newDistanceMeshInitial << endl;
		double relativeDistance = newDistanceMeshInitial;
		if (i != 0) {
			relativeDistance = lastMesh->distanceMean(newMesh);
			if (verbose_) cout << "Distance from last mesh [mm] = " << relativeDistance << endl;
		}
		
		
		if (relativeDistance <= stopCondition) done = true;
		else {
			lastMesh = newMesh;
			costFunction->setTransformation(transform);

			for (unsigned int i=0; i<nbPoints; i++) {
				initialValue[3*i] = currentValue[3*i];
				initialValue[3*i+1] = currentValue[3*i+1];
				initialValue[3*i+2] = currentValue[3*i+2];
			}
			costFunction->setInitialParameters( initialValue );
			itkOptimizer->SetInitialPosition( currentValue );

			//if (costFunction->getAbsoluteMeanDistance() < 0.1)
			//	done = true;
		}
	}

	/*ofstream myfile;
	myfile.open("DistanceMeshPropagation.txt", ios_base::app);
	myfile << newDistanceMeshInitial << endl;
	myfile.close();*/
	// mean 0.3600 std 0.3555 max 2.1927 min 0.0318

	OptimizerType::ParametersType finalPosition;
	finalPosition = itkOptimizer->GetCurrentPosition();
    
    if (finalPosition.size() == 0)
        finalPosition = itkOptimizer->GetCachedCurrentPosition();

	if (!meshBool_)
	{
		meshOutput_ = new Mesh;
		for (unsigned int i=0; i<nbPoints; i++)
			meshOutput_->addPoint(new Vertex(finalPosition[3*i],finalPosition[3*i+1],finalPosition[3*i+2]));
		for (unsigned int i=0; i<triangles.size(); i+=3)
			meshOutput_->addTriangle(triangles[i],triangles[i+1],triangles[i+2]);
	}
	else
	{
		vtkSmartPointer<vtkPoints> pointsOut = vtkSmartPointer<vtkPoints>::New();
		CVector3 pointT;
		for (unsigned int i=0; i<nbPoints; i++)
			pointsOut->InsertNextPoint(finalPosition[3*i],finalPosition[3*i+1],finalPosition[3*i+2]);
		vtkSmartPointer<vtkPolyData> m_PolyData = vtkSmartPointer<vtkPolyData>::New();
		m_PolyData->SetPoints(pointsOut);
		m_PolyData->Allocate(triangles.size());
		for (unsigned int i=0; i<triangles.size(); i+=3) {
			vtkIdType pts[3] = {triangles[i],triangles[i+1],triangles[i+2]};
			m_PolyData->InsertNextCell(VTK_TRIANGLE,3,pts);
		}
		// Calcul des normales
		vtkSmartPointer<vtkPolyDataNormals> skinNormals = vtkSmartPointer<vtkPolyDataNormals>::New();
		skinNormals->SetInputData(m_PolyData);
		skinNormals->SetFeatureAngle(60.0);
		skinNormals->ComputePointNormalsOn();
		skinNormals->ComputeCellNormalsOff();
		skinNormals->ConsistencyOff();
		skinNormals->SplittingOff();
		skinNormals->Update();
		vtkSmartPointer<vtkPolyData> normales = vtkSmartPointer<vtkPolyData>::New();
		normales->ShallowCopy(skinNormals->GetOutput());

		meshOutput_ = new Mesh;
		int label = 1;
		vtkSmartPointer<vtkPoints> pointsMesh = normales->GetPoints();
		double *pt, *norm;
		CVector3 pointt, normale;
		for (vtkIdType i = 0; i<pointsMesh->GetNumberOfPoints(); i++)
		{
			pt = pointsMesh->GetPoint(i);
			norm = normales->GetPointData()->GetNormals()->GetTuple(i);
			pointt = CVector3(pt[0],pt[1],pt[2]);
			normale = CVector3(norm[0],norm[1],norm[2]);
			meshOutput_->addPoint(new Vertex(pointt,normale,label));
		}
		vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
		polys = normales->GetPolys();
		vtkIdType nbTriangle, *po;
		for (vtkIdType i = 0; i<polys->GetNumberOfCells(); i++)
		{
			polys->GetCell(4*i,nbTriangle,po);
			meshOutput_->addTriangle(po[0],po[1],po[2]);
		}
		meshOutput_->setLabel(2);
	}

	return newDistanceMeshInitial;
}
