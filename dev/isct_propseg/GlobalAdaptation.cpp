#include "GlobalAdaptation.h"
#include "SCRegion.h"
#include <itkAmoebaOptimizer.h>
#include <itkLBFGSBOptimizer.h>
using namespace std;

typedef itk::AmoebaOptimizer::ParametersType ParametersType;
typedef itk::LBFGSBOptimizer         OptimizerType;

GlobalAdaptation::GlobalAdaptation(Image3D* image, Mesh* v, CVector3 pointRotation, string mode) : image_(image), mesh_(v), pointRotation_(pointRotation), mode_(mode), badOrientation_(false), verbose_(false) {}
GlobalAdaptation::GlobalAdaptation(Image3D* image, Mesh* v, string mode) : image_(image), mesh_(v), mode_(mode), badOrientation_(false), verbose_(false) {}


CMatrix4x4 GlobalAdaptation::adaptation(bool itkAmoeba)
{
	unsigned int numberOfParameters;
	if (mode_ == "rotation")
		numberOfParameters = 3;
	else if (mode_ == "rotation+scaling")
		numberOfParameters = 4;
	else if (mode_ == "rotation+translation")
		numberOfParameters = 6;
	vector<double> p;
	if (!itkAmoeba) { // can only be used with rotation
		double	mean1 = -0.00924, std1 = 0.057,
				mean2 = -0.0022, std2 = 0.0615,
				mean3 = 0.0140, std3 = 0.0427;
		OptimizerType::BoundSelectionType boundSelect(3);
		boundSelect.Fill( 2 );
		OptimizerType::BoundValueType upperBound(3);
		upperBound[0] = mean1+2*std1; upperBound[1] = mean2+2*std2; upperBound[2] = mean3+2*std3;
		OptimizerType::BoundValueType lowerBound(3);
		lowerBound[0] = mean1-2*std1; lowerBound[1] = mean2-2*std2; lowerBound[2] = mean3-2*std3;

		FoncteurGlobalAdaptation* f = new FoncteurGlobalAdaptation(image_,mesh_->getListTrianglesBarycentre(),numberOfParameters);
		f->setPointRotation(pointRotation_);

		OptimizerType::ParametersType pInit(3);
		pInit.Fill(0.0);

		OptimizerType::Pointer LBFGSBOptimizer = OptimizerType::New();
		LBFGSBOptimizer->SetCostFunction(f);
		LBFGSBOptimizer->SetBoundSelection( boundSelect );
		LBFGSBOptimizer->SetUpperBound( upperBound );
		LBFGSBOptimizer->SetLowerBound( lowerBound );
		LBFGSBOptimizer->SetCostFunctionConvergenceFactor( 1.e7 );
		LBFGSBOptimizer->SetProjectedGradientTolerance( 1e-35);
		LBFGSBOptimizer->SetMaximumNumberOfIterations( 250 );
		LBFGSBOptimizer->SetMaximumNumberOfEvaluations( 250 );
		LBFGSBOptimizer->SetMaximumNumberOfCorrections( 7 );
		LBFGSBOptimizer->SetInitialPosition(pInit);

		try {
			LBFGSBOptimizer->StartOptimization();
		}
		catch ( std::bad_alloc & err ) {
			cerr << "BadAlloc caught !" << (char*)err.what() << endl;
		}
		catch( itk::ExceptionObject & err ) {
			cerr << "ExceptionObject caught !" << err << endl;
		}

		OptimizerType::ParametersType pFinal = LBFGSBOptimizer->GetCurrentPosition();

		cout << LBFGSBOptimizer->GetStopConditionDescription() << endl;

		for (int i=0; i<pFinal.size(); i++)
			p.push_back(pFinal[i]);
	}
	else {
		// Utilisation de la m�thode du Simplex impl�ment�e dans la librairie ITK
		itk::AmoebaOptimizer::Pointer optimizer = itk::AmoebaOptimizer::New();
	
		SCRegion* region = new SCRegion();
		int *sizeDesired = new int[3];
		double *spacingDesired = new double[3];
		sizeDesired[0] = 61; sizeDesired[1] = 61; sizeDesired[2] = 81;
		spacingDesired[0] = 1; spacingDesired[1] = 1; spacingDesired[2] = 1;
		region->setSize(sizeDesired);
		region->setSpacing(spacingDesired);
		region->setOrigin(pointRotation_[0],pointRotation_[1],pointRotation_[2]);
		region->setNormal(normal_mesh_[0],normal_mesh_[1],normal_mesh_[2]);
		region->setFactor(image_->getTypeImageFactor());
		region->build2DGaussian(1.5);

		FoncteurGlobalAdaptation* f = new FoncteurGlobalAdaptation(image_,mesh_->getListTrianglesBarycentre(),numberOfParameters);
		f->setPointRotation(pointRotation_);
		f->setGaussianRegion(region);
		optimizer->SetCostFunction(f);
		optimizer->SetOptimizeWithRestarts(false);
		optimizer->SetMaximumNumberOfIterations(250);
		optimizer->AutomaticInitialSimplexOn();
		
		ParametersType pInit, pFinal;
		//pInit.SetSize(6);
		double* init = new double[numberOfParameters];
		for (int i=0; i<numberOfParameters; i++) init[i] = 0.0;
		if (mode_ == "rotation+scaling")
			init[3] = 1.0;
		pInit.SetData(init,numberOfParameters,false);
		optimizer->SetFunctionConvergenceTolerance(1.0e-4);
		optimizer->SetParametersConvergenceTolerance(1.0e-8);
		optimizer->SetInitialPosition(pInit);
		try {
			optimizer->StartOptimization();
		}
		catch ( std::bad_alloc & err ) {
			cerr << "BadAlloc caught !" << (char*)err.what() << endl;
		}
		catch( itk::ExceptionObject & err ) {
			cerr << "ExceptionObject caught !" << err << endl;
		}
		//cout << optimizer->GetStopConditionDescription() << endl << optimizer->GetValue() << endl;
		pFinal = optimizer->GetCurrentPosition();
		for (int i=0; i<pFinal.size(); i++)
			p.push_back(pFinal[i]);
	}

	/*cout << "Affichage des resultats..." << endl;
	for (int i=0; i<p.size(); i++)
		cout << i << " " << p[i] << endl << endl;*/

	CMatrix4x4 transformation;
	double	mean1 = 0.0044, std1 = 0.0868,
			mean2 = -0.0107, std2 = 0.1170,
			mean3 = 0.0110, std3 = 0.1495;
	double factor = 2.7;
	if (p[0]<=mean1+factor*std1 && p[0]>=mean1-factor*std1 && p[1]<=mean2+factor*std2 && p[1]>=mean2-factor*std2 && p[2]<=mean3+factor*std3 && p[2]>=mean3-factor*std3)
	{
		transformation[0] = cos(p[0])*cos(p[1]),	transformation[4] = -cos(p[2])*sin(p[1]) + sin(p[2])*sin(p[0])*cos(p[1]),	transformation[8] = sin(p[2])*sin(p[1]) + cos(p[2])*sin(p[0])*cos(p[1]),
		transformation[1] = cos(p[0])*sin(p[1]),	transformation[5] = cos(p[2])*cos(p[1]) + sin(p[2])*sin(p[0])*sin(p[1]),	transformation[9] = -sin(p[2])*cos(p[1]) + cos(p[2])*sin(p[0])*sin(p[1]),
		transformation[2] = -sin(p[0]),				transformation[6] = sin(p[2])*cos(p[0]),									transformation[10] = cos(p[2])*cos(p[0]);
		if (mode_ == "rotation+translation") {
			transformation[12] = p[3], transformation[13] = p[4], transformation[14] = p[5];
		}
		if (mode_ == "rotation+scaling") {
			transformation[0] *= p[3];
			transformation[5] *= p[3];
			transformation[10] *= p[3];
		}
		//cout << transformation << endl;

		mesh_->transform(transformation,pointRotation_);
		badOrientation_ = false;
	}
	else
	{
		/*if (verbose_) {
            cout << "Rotations : " << endl;
            cout << mean1-factor*std1 << " <= " << p[0] << " <= " << mean1+factor*std1 << endl;
            cout << mean2-factor*std2 << " <= " << p[1] << " <= " << mean2+factor*std2 << endl;
            cout << mean3-factor*std3 << " <= " << p[2] << " <= " << mean3+factor*std3 << endl;
            cout << "Rotation Index Error :";
            if (p[0]>=mean1+factor*std1 || p[0]<=mean1-factor*std1) cout << " 0";
            if (p[1]>=mean2+factor*std2 || p[1]<=mean2-factor*std2) cout << " 1";
            if (p[2]>=mean3+factor*std3 || p[2]<=mean3-factor*std3) cout << " 2";
            cout << endl;
        }//*/
		badOrientation_ = true;
	}

	return transformation;
}

double GlobalAdaptation::getInitialValue()
{
	unsigned int numberOfParameters;
	if (mode_ == "rotation")
		numberOfParameters = 3;
	else if (mode_ == "rotation+scaling")
		numberOfParameters = 4;
	else if (mode_ == "rotation+translation")
		numberOfParameters = 6;
    else numberOfParameters = 3;

	ParametersType pInit;
	double* init = new double[numberOfParameters];
	for (int i=0; i<numberOfParameters; i++) init[i] = 0.0;
	if (mode_ == "rotation+scaling")
		init[3] = 1.0;
	pInit.SetData(init,numberOfParameters,false);

	SCRegion* region = new SCRegion();
	int *sizeDesired = new int[3];
	double *spacingDesired = new double[3];
	sizeDesired[0] = 61; sizeDesired[1] = 61; sizeDesired[2] = 81;
	spacingDesired[0] = 1; spacingDesired[1] = 1; spacingDesired[2] = 1;
	region->setSize(sizeDesired);
	region->setSpacing(spacingDesired);
	region->setOrigin(pointRotation_[0],pointRotation_[1],pointRotation_[2]);
	region->setNormal(normal_mesh_[0],normal_mesh_[1],normal_mesh_[2]);
	region->setFactor(image_->getTypeImageFactor());
	region->build2DGaussian(5);

	FoncteurGlobalAdaptation* f = new FoncteurGlobalAdaptation(image_,mesh_->getListTrianglesBarycentre(),numberOfParameters);
	f->setGaussianRegion(region);
	return f->GetValue(pInit);
}
