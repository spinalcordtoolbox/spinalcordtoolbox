#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif // !_USE_MATH_DEFINES
#include "PropagatedDeformableModel.h"
#include "DeformableModelBasicAdaptator.h"
#include "GlobalAdaptation.h"
#include "Orientation.h"
#include "Vertex.h"
#include "foncteurPlan.h"
#include "BSplineApproximation.h"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>



PropagatedDeformableModel::PropagatedDeformableModel()
{
	// Initial parameters
	resolutionRadiale_ = 32;
	resolutionAxiale_ = 1;
	rayon_ = 3.0;
	numberOfDeformIteration_ = 1;
	numberOfPropagationIteration_ = 10;
	deplacementAxial_ = 1.0;
	propagationLength_ = 10.0;
    stretchingFactor_ = 1.0;

	changedParameters_ = false;
	hasInitialPointAndNormals_ = false;

    meanContrast = 445.0;
	downLimit = -1000;
	upLimit = 1000;
    
    propCenterline_ = false;
    
    verbose_ = false;
    
    init_position_ = 0.5;
    
    maxDeformation = 2.5;
    maxArea = 120.0;
    minContrast = 50.0;
    
    range = 500;

	tradeoff_d_bool = false;
	tradeoff_d_ = 0.0;

	line_search = 15;
	alpha = 25.0;
	beta = 0.0;
}


PropagatedDeformableModel::PropagatedDeformableModel(int resolutionRadiale, int resolutionAxiale, double rayon, int numberOfDeformIteration, int numberOfPropagationIteration, double deplacementAxial, double propagationLength):
	resolutionRadiale_(resolutionRadiale),
	resolutionAxiale_(resolutionAxiale),
	rayon_(rayon),
	numberOfDeformIteration_(numberOfDeformIteration),
	numberOfPropagationIteration_(numberOfPropagationIteration),
	deplacementAxial_(deplacementAxial),
	propagationLength_(propagationLength)
{
	changedParameters_ = false;
	hasInitialPointAndNormals_ = false;

    meanContrast = 445.0;
    stretchingFactor_ = 1.0;
	downLimit = -1000;
	upLimit = 1000;
    
    propCenterline_ = false;
    
    verbose_ = false;
    
    init_position_ = 0.5;
    
    maxDeformation = 2.5;
    maxArea = 120.0;
    minContrast = 50.0;
    
    range = 500;

	tradeoff_d_bool = false;
	tradeoff_d_ = 0.0;

	line_search = 15;
	alpha = 25.0;
	beta = 0.0;
}


PropagatedDeformableModel::~PropagatedDeformableModel()
{
	delete initialTube1, initialTube2, meshOutput, meshOutputFinal;
}


void PropagatedDeformableModel::setInitialPointAndNormals(CVector3 initialPoint, CVector3 normal1, CVector3 normal2)
{
	hasInitialPointAndNormals_ = true;
	initialPoint_ = initialPoint;
	initialNormal1_ = normal1;
	initialNormal2_ = normal2;
}


void PropagatedDeformableModel::setUpAndDownLimits(int downLimit, int upLimit)
{
	this->upLimit = upLimit;
	this->downLimit = downLimit;
}


void PropagatedDeformableModel::computeMeshInitial()
{
    // centerline can be added to be followed. Points of centerline have to be added from bottom to top
    if (propCenterline_) {
        if (centerline.size() < 1) {
            cerr << "Error: Not enought points in centerline" << endl;
            return;
        } else {
            //int init = centerline.size()*init_position_;
            
            /******************************************************************************************
             * If a centerline is used for the orientation computation, we compute its BSpline approximation. This approximation is used for centerline position and orientation extraction
             * The parameter range is the centerline approximation accuracy
             *****************************************************************************************/
            initial_centerline = centerline;
            centerline_approximator = BSplineApproximation(&initial_centerline);
            
            initialPoint_ = centerline_approximator.EvaluateBSpline(init_position_);
            initialNormal1_ = -centerline_approximator.EvaluateGradient(init_position_-0.01).Normalize();
            initialNormal2_ = centerline_approximator.EvaluateGradient(init_position_+0.01).Normalize();;
            //initialNormal1_ = (centerline[init-1]-centerline[init]).Normalize();
            //initialNormal2_ = (centerline[init+1]-centerline[init]).Normalize();
            hasInitialPointAndNormals_ = true;
        }
    }
	if (centerline.size() < 1 && !hasInitialPointAndNormals_) {
		cerr << "Error: Not enought points in centerline" << endl;
	}
	else if (centerline.size() == 2) {
		initialTube1 = new SpinalCord;
		CVector3 directionInitiale = (centerline[1]-centerline[0]).Normalize(), directionInitialePerpendiculaire;
		if (directionInitiale[2] == 0.0) directionInitialePerpendiculaire = CVector3(0.0,0.0,1.0);
		else directionInitialePerpendiculaire = CVector3(1.0,2.0,-(directionInitiale[0]+2*directionInitiale[1])/directionInitiale[2]).Normalize();
		Referential refInitial = Referential(directionInitiale^directionInitialePerpendiculaire, directionInitialePerpendiculaire, directionInitiale, centerline[0]);
		CMatrix4x4 transformationFromOrigin = refInitial.getTransformationInverse();
		double angle;
		CMatrix3x3 trZ;
		CVector3 point, normale;
		// Compute initial disk
		for (int k=0; k<resolutionRadiale_; k++)
		{
			angle = 2*M_PI*k/(double)resolutionRadiale_;
			trZ[0] = cos(angle), trZ[1] = sin(angle), trZ[3] = -sin(angle), trZ[4] = cos(angle);
			point = transformationFromOrigin*(trZ*CVector3(rayon_,0.0,0.0));
			initialTube1->addPoint(new Vertex(point,(point-centerline[0]).Normalize()));
		}
		CVector3 nextPoint = centerline[0] + deplacementAxial_*directionInitiale;
		computeNewBand(initialTube1,centerline[0],nextPoint,resolutionAxiale_+2);

		initialTube1->computeConnectivity();
		initialTube1->computeTrianglesBarycentre();

		isMeshInitialized = true;

		meanContrast = computeContrast(refInitial);
		//cout << "Contrast = " << contrast << endl;
	}
	else if (hasInitialPointAndNormals_) {
		initialTube1 = new SpinalCord;
		CVector3 directionInitiale = initialNormal1_, directionInitialePerpendiculaire;
		if (directionInitiale[2] == 0.0) directionInitialePerpendiculaire = CVector3(0.0,0.0,1.0);
		else directionInitialePerpendiculaire = CVector3(1.0,2.0,-(directionInitiale[0]+2*directionInitiale[1])/directionInitiale[2]).Normalize();
		Referential refInitial = Referential(directionInitiale^directionInitialePerpendiculaire, directionInitialePerpendiculaire, directionInitiale, initialPoint_);
		CMatrix4x4 transformationFromOrigin = refInitial.getTransformationInverse();
		double angle;
		CMatrix3x3 trZ;
		CVector3 point, normale;
        CVector3 stretchingFactorWorld = image3D_->TransformContinuousIndexToPhysicalPoint(CVector3((1.0-1.0/stretchingFactor_), 0.0, 0.0))-image3D_->getOrigine();
		// Compute initial disk
		for (int k=0; k<resolutionRadiale_; k++)
		{
			angle = 2*M_PI*k/(double)resolutionRadiale_;
			trZ[0] = cos(angle), trZ[1] = sin(angle), trZ[3] = -sin(angle), trZ[4] = cos(angle);
			point = transformationFromOrigin*(trZ*CVector3(rayon_,0.0,0.0));
            CVector3 vecPoint = initialPoint_ - point;
            point[0] += stretchingFactorWorld[0]*vecPoint[0];
            point[1] += stretchingFactorWorld[1]*vecPoint[1];
            point[2] += stretchingFactorWorld[2]*vecPoint[2];
			initialTube1->addPoint(new Vertex(point,(point-initialPoint_).Normalize()));
		}
		CVector3 nextPoint = initialPoint_ + deplacementAxial_*directionInitiale;
		computeNewBand(initialTube1,initialPoint_,nextPoint,resolutionAxiale_+2);
        

		initialTube1->computeConnectivity();
		initialTube1->computeTrianglesBarycentre();

		initialTube2 = new SpinalCord;
		directionInitiale = initialNormal2_;
		if (directionInitiale[2] == 0.0) directionInitialePerpendiculaire = CVector3(0.0,0.0,1.0);
		else directionInitialePerpendiculaire = CVector3(1.0,2.0,-(directionInitiale[0]+2*directionInitiale[1])/directionInitiale[2]).Normalize();
		refInitial = Referential(directionInitiale^directionInitialePerpendiculaire, directionInitialePerpendiculaire, directionInitiale, initialPoint_);
		transformationFromOrigin = refInitial.getTransformationInverse();
		// Compute initial disk
		for (int k=0; k<resolutionRadiale_; k++)
		{
			angle = 2*M_PI*k/(double)resolutionRadiale_;
			trZ[0] = cos(angle), trZ[1] = sin(angle), trZ[3] = -sin(angle), trZ[4] = cos(angle);
			point = transformationFromOrigin*(trZ*CVector3(rayon_,0.0,0.0));
            CVector3 vecPoint = initialPoint_ - point;
            point[0] += stretchingFactorWorld[0]*vecPoint[0];
            point[1] += stretchingFactorWorld[1]*vecPoint[1];
            point[2] += stretchingFactorWorld[2]*vecPoint[2];
			initialTube2->addPoint(new Vertex(point,(point-initialPoint_).Normalize()));
		}
		nextPoint = initialPoint_ + deplacementAxial_*directionInitiale;
		computeNewBand(initialTube2,initialPoint_,nextPoint,resolutionAxiale_+2);

		initialTube2->computeConnectivity();
		initialTube2->computeTrianglesBarycentre();

		isMeshInitialized = true;

		meanContrast = computeContrast(refInitial);
	}
	else {
		initialTube1 = new SpinalCord;
		CVector3 directionInitiale = (centerline[1]-centerline[0]).Normalize(), directionInitialePerpendiculaire;
		if (directionInitiale[2] == 0.0) directionInitialePerpendiculaire = CVector3(0.0,0.0,1.0);
		else directionInitialePerpendiculaire = CVector3(1.0,2.0,-(directionInitiale[0]+2*directionInitiale[1])/directionInitiale[2]).Normalize();
		Referential refInitial = Referential(directionInitiale^directionInitialePerpendiculaire, directionInitialePerpendiculaire, directionInitiale, centerline[0]);
		CMatrix4x4 transformationFromOrigin = refInitial.getTransformationInverse();
		double angle;
		CMatrix3x3 trZ;
		CVector3 point, normale;
		// Compute initial disk
		for (int k=0; k<resolutionRadiale_; k++)
		{
			angle = 2*M_PI*k/(double)resolutionRadiale_;
			trZ[0] = cos(angle), trZ[1] = sin(angle), trZ[3] = -sin(angle), trZ[4] = cos(angle);
			point = transformationFromOrigin*(trZ*CVector3(rayon_,0.0,0.0));
			initialTube1->addPoint(new Vertex(point,(point-centerline[0]).Normalize()));
		}
		for (unsigned int i=1; i<centerline.size(); i++)
		{
			computeNewBand(initialTube1,centerline[i-1],centerline[i],resolutionAxiale_);
		}

		initialTube1->computeConnectivity();
		initialTube1->computeTrianglesBarycentre();

		isMeshInitialized = true;
	}
}


float PropagatedDeformableModel::computeContrast(Referential& refInitial)
{
	// Calcul du profil de la moelle et du LCR perpendiculairement au tube
	CVector3 pointS, indexS;
	vector<float> contrast(resolutionRadiale_);
	float angle;
	CMatrix3x3 trZ;
	CMatrix4x4 transformationFromOrigin = refInitial.getTransformationInverse();
	float factor = image3D_->getTypeImageFactor();
	for (int k=0; k<resolutionRadiale_; k++)
	{
		vector<float> profilIntensite;
		angle = 2*M_PI*k/(double)resolutionRadiale_;
		trZ[0] = cos(angle), trZ[1] = sin(angle), trZ[3] = -sin(angle), trZ[4] = cos(angle);
		for (int l=0; l<2.5*rayon_; l++) {
			pointS = transformationFromOrigin*(trZ*CVector3(l,0.0,0.0));
			if (image3D_->TransformPhysicalPointToIndex(pointS,indexS))
				profilIntensite.push_back(factor*image3D_->GetPixelOriginal(indexS));
		}
		float min = 0.0, max = 0.0, maxVal = 0.0, valCourante;
		unsigned int m = 0;
		for (unsigned int i=1; i<profilIntensite.size(); i++) {
			valCourante = profilIntensite[i]-profilIntensite[i-1];
			if (maxVal <= valCourante) {
				maxVal = valCourante;
				m = i;
			}
		}
		if (profilIntensite.size() > 0)
		{
			min = profilIntensite[m];
			for (unsigned int j=0; j<m; j++) {
				valCourante = profilIntensite[j];
				if (min > valCourante) min = valCourante;
			}
			max = profilIntensite[m];
			for (unsigned int j=m+1; j<profilIntensite.size(); j++) {
				valCourante = profilIntensite[j];
				if (max < valCourante) max = valCourante;
			}
		}
		contrast[k] = abs(max-min);
	}
	float result = 0.0;
	for (unsigned int i=0; i<contrast.size(); i++)
		result += contrast[i];
	result /= contrast.size();

	return result;
}


void PropagatedDeformableModel::computeNewBand(SpinalCord* mesh, CVector3 initialPoint, CVector3 nextPoint, int resolution)
{
	CMatrix4x4 transformationFromOrigin;
	double angle;
	CMatrix3x3 trZ;
	CVector3 point, normale;
	CVector3 directionCourante = nextPoint-initialPoint, lastNormal = (nextPoint-initialPoint).Normalize(), directionCourantePerpendiculaire;
	if (lastNormal[2] == 0.0) directionCourantePerpendiculaire = CVector3(0.0,0.0,1.0);
	else directionCourantePerpendiculaire = CVector3(1.0,2.0,-(lastNormal[0]+2*lastNormal[1])/lastNormal[2]).Normalize();

    CVector3 stretchingFactorWorld = image3D_->TransformContinuousIndexToPhysicalPoint(CVector3((1-1.0/stretchingFactor_), 0.0, 0.0))-image3D_->getOrigine();
    
	int offsetTriangles = mesh->getNbrOfPoints()-resolutionRadiale_;
	for (int len=1; len<=resolution; len++)
	{
		CVector3 pointIntermediaire = initialPoint + len*directionCourante/(double)resolutionAxiale_;
		Referential refCourant = Referential(lastNormal^directionCourantePerpendiculaire, directionCourantePerpendiculaire, lastNormal, pointIntermediaire);
		transformationFromOrigin = refCourant.getTransformationInverse();
		for (int k=0; k<resolutionRadiale_; k++)
		{
			angle = 2*M_PI*k/(double)resolutionRadiale_;
			trZ[0] = cos(angle), trZ[1] = sin(angle), trZ[3] = -sin(angle), trZ[4] = cos(angle);
			point = transformationFromOrigin*(trZ*CVector3(rayon_,0.0,0.0));
            CVector3 vecPoint = initialPoint - point;
            point[0] += stretchingFactorWorld[0]*vecPoint[0];
            point[1] += stretchingFactorWorld[1]*vecPoint[1];
            point[2] += stretchingFactorWorld[2]*vecPoint[2];
			mesh->addPoint(new Vertex(point,(point-pointIntermediaire).Normalize()));
		}
		// Ajout des triangles - attention � la structure en cercle
		for (int k=0; k<resolutionRadiale_-1; k++)
		{
			mesh->addTriangle(offsetTriangles+(len-1)*resolutionRadiale_+k,offsetTriangles+(len-1)*resolutionRadiale_+k+1,offsetTriangles+len*resolutionRadiale_+k);
			mesh->addTriangle(offsetTriangles+(len-1)*resolutionRadiale_+k+1,offsetTriangles+len*resolutionRadiale_+k+1,offsetTriangles+len*resolutionRadiale_+k);
		}
		// Ajout des deux derniers triangles pour fermer le tube
		mesh->addTriangle(offsetTriangles+(len-1)*resolutionRadiale_+resolutionRadiale_-1,offsetTriangles+(len-1)*resolutionRadiale_,offsetTriangles+len*resolutionRadiale_+resolutionRadiale_-1);
		mesh->addTriangle(offsetTriangles+(len-1)*resolutionRadiale_,offsetTriangles+len*resolutionRadiale_,offsetTriangles+len*resolutionRadiale_+resolutionRadiale_-1);
	}
}

void PropagatedDeformableModel::adaptationGlobale()
{
	if (isMeshInitialized)
	{
		if (hasInitialPointAndNormals_) // bidirectional propagation
		{
            // add up and down limit to not propagate more far than centerline border
            if (initial_centerline.size() != 0) {
                CVector3 down, up;
                bool downBool = image3D_->TransformPhysicalPointToIndex(initial_centerline[0],down);
                bool upBool = image3D_->TransformPhysicalPointToIndex(initial_centerline[centerline.size()-1],up);
                downLimit = down[1]-2;
                upLimit = up[1]+2;
            }
        
			SpinalCord *mesh1 = propagationMesh(1);
            SpinalCord *mesh2 = propagationMesh(2);
			// meshes merging
			meshOutput = mergeBidirectionalSpinalCord(mesh1,mesh2);
		}
		else propagationMesh(); // unidirectional propagation
	}
	else {
		cout << "Error: The initial mesh is not initialized" << endl;
	}
}

SpinalCord* PropagatedDeformableModel::mergeBidirectionalSpinalCord(SpinalCord* spinalCord1, SpinalCord* spinalCord2)
{
	int radialResolution = spinalCord1->getRadialResolution();
	SpinalCord* mesh = new SpinalCord();
	mesh->setRadialResolution(radialResolution);

	// Adding first mesh
	unsigned int numberOfPoints = spinalCord1->getNbrOfPoints(), numberOfTriangles = spinalCord2->getNbrOfTriangles();
	int numberOfDisk = numberOfPoints/radialResolution;
	vector<Vertex*> listPoints1 = spinalCord1->getListPoints();
	// Ajout du premier disque
	for (int j=0; j<radialResolution; j++)
	{
		mesh->addPoint(new Vertex(*listPoints1[numberOfDisk*radialResolution-1-j]));//(numberOfDisk-1)*radialResolution+j]));
	}
	// Ajout des suivants
	for (int i=1; i<numberOfDisk; i++)
	{
		for (int j=0; j<radialResolution; j++)
		{
			mesh->addPoint(new Vertex(*listPoints1[(numberOfDisk-i)*radialResolution-1-j]));//(numberOfDisk-1-i)*radialResolution+j]));
		}
		// Ajout des triangles - attention � la structure en cercle
		for (int k=0; k<radialResolution-1; k++)
		{
			mesh->addTriangle((i-1)*radialResolution+k,(i-1)*radialResolution+k+1,i*radialResolution+k);
			mesh->addTriangle((i-1)*radialResolution+k+1,i*radialResolution+k+1,i*radialResolution+k);
		}
		// Ajout des deux derniers triangles pour fermer le tube
		mesh->addTriangle((i-1)*radialResolution+radialResolution-1,(i-1)*radialResolution,i*radialResolution+radialResolution-1);
		mesh->addTriangle((i-1)*radialResolution,i*radialResolution,i*radialResolution+radialResolution-1);
	}

	// Adding second mesh
	numberOfPoints = spinalCord2->getNbrOfPoints(), numberOfTriangles = spinalCord2->getNbrOfTriangles();
	numberOfDisk = numberOfPoints/radialResolution;
	vector<Vertex*> listPoints2 = spinalCord2->getListPoints();
	int offsetTriangles = mesh->getNbrOfPoints()-radialResolution;
	// Points between meshes need offset to adjust properly triangles
	double distanceMin = 10000.0, dist;
	int indexMin = 0;
	Vertex* point = mesh->getListPoints()[offsetTriangles];
	for (int m=0; m<radialResolution; m++)
	{
		dist = point->distance(*listPoints2[m]);
		if (dist < distanceMin) {
			distanceMin = dist;
			indexMin = m;
		}
	}
	// Ajout des suivants
	for (int i=1; i<numberOfDisk; i++)
	{
		for (int j=0; j<radialResolution; j++)
		{
			mesh->addPoint(new Vertex(*listPoints2[i*radialResolution+(j+indexMin)%radialResolution]));
		}
		// Ajout des triangles - attention � la structure en cercle
		for (int k=0; k<radialResolution-1; k++)
		{
			mesh->addTriangle(offsetTriangles+(i-1)*radialResolution+k,offsetTriangles+(i-1)*radialResolution+k+1,offsetTriangles+i*radialResolution+k);
			mesh->addTriangle(offsetTriangles+(i-1)*radialResolution+k+1,offsetTriangles+i*radialResolution+k+1,offsetTriangles+i*radialResolution+k);
		}
		// Ajout des deux derniers triangles pour fermer le tube
		mesh->addTriangle(offsetTriangles+(i-1)*radialResolution+radialResolution-1,offsetTriangles+(i-1)*radialResolution,offsetTriangles+i*radialResolution+radialResolution-1);
		mesh->addTriangle(offsetTriangles+(i-1)*radialResolution,offsetTriangles+i*radialResolution,offsetTriangles+i*radialResolution+radialResolution-1);
	}
	return mesh;
}

SpinalCord* PropagatedDeformableModel::propagationMesh(int numberOfMesh)
{
    double const_contrast = 200.0;
    if (image3D_->getTypeImageFactor() == 1.0) const_contrast = 445.0; // if T2
    
    /******************************************************************************************
     * Initialization of the spinal cord mesh
     * Depending of the choice of propagation, the contrast vector is reversed to take the correct values
     *****************************************************************************************/
    SpinalCord* initialMesh = new SpinalCord();
    if (numberOfMesh == 1) initialMesh = initialTube1;
    else if (numberOfMesh == 2) {
        reverse(contrast.begin(),contrast.end());
        initialMesh = initialTube2;
    }
    vector< vector<CVector3> > lastDisks;
        
    /******************************************************************************************
     * Deformation of the initial mesh.
     * This deformation must be accurate to have a correct initialization. If not, errors can be propagated.
     * The number of iteration and the stop condition of the deformation is 0.05 mm by default
     *****************************************************************************************/
    if (verbose_) cout << endl << endl << "Initial deformation : " << initialMesh->getNbrOfPoints() << " points and " << initialMesh->getNbrOfTriangles() << " triangles" << endl;
    DeformableModelBasicAdaptator *deformableAdaptator = new DeformableModelBasicAdaptator(image3D_,initialMesh,numberOfDeformIteration_,const_contrast,false);
    deformableAdaptator->setVerbose(verbose_);
    deformableAdaptator->setNumberOfIteration(8); //8
    deformableAdaptator->setStopCondition(0.05);
	if (tradeoff_d_bool) deformableAdaptator->setTradeOff(tradeoff_d_);
    //deformableAdaptator->setProgressiveLineSearchLength(true);// tested but not optimal
    deformableAdaptator->addCorrectionPoints(points_mask_correction_);

    deformableAdaptator->adaptation(); // launch the deformation
    meshOutput = deformableAdaptator->getSpinalCordOutput(); // get the spinal cord segmentation mesh
    delete deformableAdaptator; // release memory
        
        
    meshOutput->setRadialResolution(resolutionRadiale_); // the output of DeformableModelBasicAdaptator is a mesh and we need to provide the radial resolution for further computation
    // we remove the last disk to prevent edges issues in the deformation process. Indeed, edges have less neighbors and the last disk retract itself.
    meshOutput->removeLastPoints(resolutionRadiale_);
    meshOutput->removeLastTriangles(2*resolutionRadiale_);
        
        
    /******************************************************************************************
     * Initialization of variables
     * numberOfDisks is a constant of propagation - don't change it
     * uniqueMesh is the part of the mesh that will be deformed at each iteration
     * newStartPoint is the center of the last disk of the mesh. It is the new start point of the propagation. The mesh is duplicated and translated on this point.
     *****************************************************************************************/
    unsigned int numberOfDisks = resolutionAxiale_+1;
    CVector3 nextPoint, newStartPoint, lastPoint, firstPoint = meshOutput->computeGravityCenterFirstDisk(numberOfDisks); // computation of first point = center of mass of the fisrt disk
    centerline.push_back(firstPoint); // add point to centerline - first point necessary
    SpinalCord	*uniqueMesh = meshOutput->extractPartOfMesh(numberOfDisks,true,true); // extraction of a part of the mesh
    uniqueMesh->computeConnectivity();
    uniqueMesh->computeTrianglesBarycentre();
    newStartPoint = meshOutput->computeGravityCenterFirstDisk(numberOfDisks); // compute first point of the mesh
        
    /******************************************************************************************
     * Computation of the initial rotation value.
     * It is used as a mesh refreshing condition. If the difference between initial and updated rotation value if too high, a new part of mesh is used as the template to be duplicated. Rotation value is the sum of intensity at vertices positions.
     *****************************************************************************************/
    GlobalAdaptation* gAdaptI = new GlobalAdaptation(image3D_,uniqueMesh,newStartPoint,"rotation");
    CVector3 normal_mesh = CVector3(initialNormal2_[0],initialNormal2_[1],initialNormal2_[2]);
    gAdaptI->setNormalMesh(normal_mesh);
    gAdaptI->setVerbose(verbose_);
    double initialRotationValue = gAdaptI->getInitialValue(), rotationValue = 0.0;
    delete gAdaptI;
    CVector3 position;
        
    /******************************************************************************************
     * initialization of stop condition variables
     *****************************************************************************************/
    bool done = false;
    area[0] = 0.0; area[1] = 0.0; area[2] = 0.0;
    double meanVoxels[3]; meanVoxels[0] = 0.0; meanVoxels[1] = 0.0; meanVoxels[2] = 0.0;
    //double meanVoxelsInit = meshOutput->computeStandardDeviationFromPixelsInside(image3D_->getImageOriginale()); // homogeneity stop condition - not optimal
    int numberOfBadOrientation = 0, numberOfBadOrientationTotal = 0, maxBadOrientation = 150;
    meanContrast = 0.0;
    
    
    /******************************************************************************************
     * Iterative deformation by adding a portion of mesh at each iteration
     *****************************************************************************************/
    int i;
    for (i=1; i<=numberOfPropagationIteration_ && !done; i++)
    {
        if (verbose_) cout << endl << "Propagation step " << i << "/" << numberOfPropagationIteration_ << endl;
        centerline = meshOutput->computeCenterline();
        double segmentationLength = 0.0;
        for (unsigned int c=1; c<centerline.size(); c++)
            segmentationLength += (centerline[c]-centerline[c-1]).Norm();
        if (verbose_) cout << "Propagation length [mm] : " << segmentationLength << " / " << propagationLength_ <<  endl;
            
        /******************************************************************************************
         * Referential computation on the last disk of the mesh
         * Computation of the local spinal cord / CSF contrast along the mesh
         * Computation of the mean contrast from last disk positions. The mean contrast is used as a parameter in the local deformation and as a stop condition
         *****************************************************************************************/
        CVector3 sourcePoint1 = centerline[centerline.size()-1], sourcePoint2 = centerline[centerline.size()-5];
        CVector3 lastNormal = (sourcePoint1-sourcePoint2).Normalize(), directionCourantePerpendiculaire;
        if (lastNormal[2] == 0.0) directionCourantePerpendiculaire = CVector3(0.0,0.0,1.0);
        else directionCourantePerpendiculaire = CVector3(1.0,2.0,-(lastNormal[0]+2*lastNormal[1])/lastNormal[2]).Normalize();
        Referential refCourant = Referential(lastNormal^directionCourantePerpendiculaire, directionCourantePerpendiculaire, lastNormal, sourcePoint2);
        contrast.push_back(pair<CVector3,double>(refCourant.getOrigine(),computeContrast(refCourant)));
        int nbContrast = contrast.size()-1;
        
        if (verbose_) cout << "Contrast = " << contrast[nbContrast].second << endl;
        if (nbContrast == 0) meanContrast = (contrast[0].second+2*const_contrast)/3.0;
        else if (nbContrast == 1) meanContrast = (contrast[nbContrast].second+contrast[nbContrast-1].second+const_contrast)/3.0;
        else meanContrast = (contrast[nbContrast].second+contrast[nbContrast-1].second+contrast[nbContrast-2].second)/3.0;
        // mean 445.8476 std 113.5695 max 708.9200 min 148.6900
        if (verbose_) cout << "Iteration Position = " << sourcePoint1 << endl;
        
        /******************************************************************************************
         * newStartPoint is the last point of the mesh and will be the first point of the new section (for duplication and translation)
         *****************************************************************************************/
        newStartPoint = meshOutput->computeGravityCenterLastDisk(numberOfDisks);
            
        CVector3 indexPosition, temp;
        bool inOut = image3D_->TransformPhysicalPointToIndex(newStartPoint,indexPosition); // stop condition
        CVector3 lastPointReal = lastPoint;
        if (lastPointReal == CVector3::ZERO) lastPointReal = newStartPoint;
        
        /******************************************************************************************
         * stop conditions:
         * length of propagation
         * mean local contrast between CSF and spinal cord
         * abnormalities - not good if the new starting point is behind the last starting point
         * inferior and superior limits can be imposed by the user
         *****************************************************************************************/
        if (segmentationLength < propagationLength_ && meanContrast > minContrast && abs(newStartPoint[1]-lastPointReal[1]) <= 15.0 && indexPosition[1]<upLimit && indexPosition[1]>downLimit)
        {
            lastPoint = meshOutput->computeGravityCenterFirstDisk(numberOfDisks);
            if (position == CVector3()) position = lastPoint;
                
            SpinalCord *partMesh = meshOutput->extractLastDiskOfMesh(false);
                
            CMatrix4x4 translation, transformation; translation[12] = newStartPoint[0]-position[0]; translation[13] = newStartPoint[1]-position[1]; translation[14] = newStartPoint[2]-position[2];
            position = newStartPoint;
                
            uniqueMesh->transform(translation);
            
            bool orientationBool = true;
                
            /******************************************************************************************
             * GlobalAdaptation compute the rigid transformation of the mesh to the image using the gradient magnitude
             * It can provide a value of the gradient at the surface vertice positions for a quality control with the function getInitialValue()
             * This value is used to change the mesh when it doesn't correspond anymore to the spinal cord edges - this value is negative
             * The function adaptation() compute the orientation and transform the mesh
             *****************************************************************************************/
            GlobalAdaptation* gAdapt = new GlobalAdaptation(image3D_,uniqueMesh,newStartPoint,"rotation");
            normal_mesh = CVector3(translation[12],translation[13],translation[14]);
            gAdapt->setNormalMesh(normal_mesh);
            gAdapt->setVerbose(verbose_);
            rotationValue = gAdapt->getInitialValue();
            if (rotationValue >= 0.75*initialRotationValue || rotationValue <= 1.5*initialRotationValue) { // if the value of GlobalAdaptation isn't in range, we replace the mesh
                // release the memory and create a new mesh using the last disks
                delete uniqueMesh;
                uniqueMesh = meshOutput->extractPartOfMesh(numberOfDisks,true,true);
                uniqueMesh->computeConnectivity();
                uniqueMesh->computeTrianglesBarycentre();
                lastPoint = meshOutput->computeGravityCenterFirstDisk(numberOfDisks);
                CMatrix4x4 translation, transformation; translation[12] = newStartPoint[0]-lastPoint[0]; translation[13] = newStartPoint[1]-lastPoint[1]; translation[14] = newStartPoint[2]-lastPoint[2];
                position = newStartPoint;
                uniqueMesh->transform(translation); // translate the mesh to its new position
                gAdapt = new GlobalAdaptation(image3D_,uniqueMesh,newStartPoint,"rotation");
                normal_mesh = CVector3(translation[12],translation[13],translation[14]);
                gAdapt->setNormalMesh(normal_mesh);
                gAdapt->setVerbose(verbose_);
                initialRotationValue = gAdapt->getInitialValue();
                rotationValue = gAdapt->getInitialValue();
            }
            
            /******************************************************************************************
             * if the centerline of the spinal cord is provided by the user, it is what it's used to compute the rotation at each propagation iteration
             * TODO: spline interpolation
             *****************************************************************************************/
            if (propCenterline_)
            {
                // Computation of the rotation based on the centerline
                // Find the point in centerline that is the nearest point to our new starting point.
                double nearest_point_value = centerline_approximator.getNearestPoint(newStartPoint, range);
                
                // Evaluate the derivative of the centerline at this location
                CVector3 normal = centerline_approximator.EvaluateGradient(nearest_point_value).Normalize();
                
                // Compute normal of our mesh at the starting position
                if (numberOfMesh == 1) normal = -normal; // the normal is inverted for the first mesh
                CVector3 lastNormalMesh = (newStartPoint-lastPoint).Normalize();
                
                // Compute rotation between the two normals. We need to compute all the necessary axis for establishing referentials.
                CVector3 directionCourantePerpendiculaireMesh, directionCourantePerpendiculaireCenterline;
                if (lastNormalMesh[2] == 0.0) directionCourantePerpendiculaireMesh = CVector3(0.0,0.0,1.0);
                else directionCourantePerpendiculaireMesh = CVector3(1.0,2.0,-(lastNormalMesh[0]+2*lastNormalMesh[1])/lastNormalMesh[2]).Normalize();
                Referential refMesh = Referential(lastNormalMesh^directionCourantePerpendiculaireMesh, directionCourantePerpendiculaireMesh, lastNormalMesh, newStartPoint);
                if (normal[2] == 0.0) directionCourantePerpendiculaireCenterline = CVector3(0.0,0.0,1.0);
                else directionCourantePerpendiculaireCenterline = CVector3(1.0,2.0,-(normal[0]+2*normal[1])/normal[2]).Normalize();
                Referential refCenterline = Referential(normal^directionCourantePerpendiculaireCenterline, directionCourantePerpendiculaireCenterline, normal, newStartPoint);
                CMatrix4x4 transformationRotation = refMesh.getTransformation(refCenterline);
                // As the referential origin is the starting point of our mesh, the transformation that is computed is only a rotation and does not contain any translation.
                
                uniqueMesh->transform(transformationRotation,newStartPoint);
            }
            /******************************************************************************************
             * if the centerline is not provided, the rotation computation used GlobalAdaptation
             *****************************************************************************************/
            else
            {
                gAdapt->adaptation(true);
                // bad orientation (out of known orientation range) can happened
                if (gAdapt->getBadOrientation()) {
                    numberOfBadOrientation++;
                    numberOfBadOrientationTotal++;
                }
                else numberOfBadOrientation = 0;
            }
            
            if (orientationBool) // The program can stop if too much bad orientation - not used for now
            {
                /******************************************************************************************
                 * Creation of a new mesh with the last disk of meshOutput and the new mesh (uniqueMesh) correctly oriented
                 *****************************************************************************************/
                partMesh->assembleMeshes(uniqueMesh,numberOfDisks,resolutionRadiale_);
                partMesh->computeConnectivity();
                partMesh->computeTrianglesBarycentre();
                if (verbose_) cout << "Mesh deformation : " << partMesh->getNbrOfPoints() << " points and " << partMesh->getNbrOfTriangles() << " triangles" << endl;
                
                /******************************************************************************************
                 * Creation of the deformation object, including the image, the mesh and the deformation parameters
                 *****************************************************************************************/
                deformableAdaptator = new DeformableModelBasicAdaptator(image3D_,partMesh,numberOfDeformIteration_,meanContrast,false);
                if (tradeoff_d_bool) deformableAdaptator->setTradeOff(tradeoff_d_);
				deformableAdaptator->setVerbose(verbose_);
                if (this->changedParameters_) {
                    deformableAdaptator->changedParameters();
                    deformableAdaptator->setAlpha(alpha);
                    deformableAdaptator->setBeta(beta);
                    deformableAdaptator->setLineSearch(line_search);
                }
                deformableAdaptator->addCorrectionPoints(points_mask_correction_);
                
                /******************************************************************************************
                 * Deformation of the mesh
                 *****************************************************************************************/
                double deformation = deformableAdaptator->adaptation();
                
                /******************************************************************************************
                 * Extraction of the deformation result and verification of stop conditions:
                 * mesh consistency
                 * maximum deformation
                 * maximum cross-sectional area
                 * number of wrong orientation (sign of a wrong orientation)
                 *****************************************************************************************/
                SpinalCord* deformed_spinalcord = deformableAdaptator->getSpinalCordOutput();
                deformed_spinalcord->computeConnectivity();
                deformed_spinalcord->computeTrianglesBarycentre();
                deformed_spinalcord->Initialize(resolutionRadiale_);
                CVector3 secondPoint = deformed_spinalcord->computeGravityCenterSecondDisk();
                double lastCrossSectionalArea = deformed_spinalcord->computeLastCrossSectionalArea();
                area[0] = area[1]; area[1] = area[2]; area[2] = lastCrossSectionalArea; meanArea = (area[0]+area[1]+area[2])/3.0;
                
                if ((secondPoint-lastPoint)*(newStartPoint-lastPoint)<0.0) {
                    if (verbose_) cout << "Stop by deformation error : overlap during propagation" << endl;
                    done = true;
                }
                if (deformation >= maxDeformation) {
                    if (verbose_) cout << "Stop by too large deformation : " << deformation << "/" << maxDeformation << endl;
                    done = true;
                }
                if ((meanArea >= maxArea && abs(area[2]-area[1]) >= maxArea/7.0) || meanArea >= 1.2*maxArea) {// || (area[1]>=maxArea && area[2]<maxArea)) {
                    if (verbose_) cout << "Stop by too large cross sectionnal area : " << meanArea << "/" << maxArea << endl;
                    done = true;
                }
                if (numberOfBadOrientation >= maxBadOrientation)
                {
                    if (verbose_) cout << "Stop by too much bad orientation during propagation : " << numberOfBadOrientation << "/" << maxBadOrientation << endl;
                    done = true;
                }
                
                /******************************************************************************************
                 * Assembling meshOutput (whole spinal cord segmentation) and the new part of the deformed mesh
                 *****************************************************************************************/
                if (!done)
                {
                    meshOutput->assembleMeshes(deformed_spinalcord,numberOfDisks,resolutionRadiale_);
                    meshOutput->computeConnectivity();
                    meshOutput->computeTrianglesBarycentre();
                    
                    // if the mesh get out the image, the propagation has to stop
                    if(!inOut || indexPosition[1]>upLimit || indexPosition[1]<downLimit) {
                        done = true;
                        if (!inOut && verbose_) cout << "Stop because out of image" << endl;
                        if (indexPosition[1]>upLimit && verbose_) cout << "Stop because out of range: up" << endl;
                        if (indexPosition[1]<downLimit && verbose_) cout << "Stop because out of range: down" << endl;
                    }
                }
                delete deformed_spinalcord;
            }
            else
            {
                if (verbose_) cout << "Stop by bad orientation" << endl;
                done = true;
            }
                
            delete partMesh, gAdapt, deformableAdaptator;//, orientationFilter;
        }
        else
        {
            done = true;
            if (!inOut && verbose_) cout << "Stop because out of image" << endl;
            if (indexPosition[1]>=upLimit && verbose_) cout << "Stop because out of range: up" << endl;
            if (indexPosition[1]<=downLimit && verbose_) cout << "Stop because out of range: down" << endl;
            if (abs(newStartPoint[1]-lastPointReal[1]) > 15.0 && verbose_) cout << "Stop because bad direction" << endl;
        }
        
        if (verbose_) cout << "Number of bad orientation = " << numberOfBadOrientationTotal << " / " << i << endl;
        if (verbose_) cout << "Contrast : " << meanContrast << " / " << minContrast << endl;
		
        /******************************************************************************************
         * Computation of the new spinal cord centerline
         *****************************************************************************************/
        centerline = meshOutput->computeCenterline();
        if (verbose_) {
            double distanceSegmentation = 0.0;
            int interval = 1;
            for (unsigned int c=interval; c<centerline.size(); c+=interval)
                distanceSegmentation += (centerline[c]-centerline[c-interval]).Norm();
            cout << "Distance of segmentation [mm] = " << distanceSegmentation << endl;
        }
    }
    
    /******************************************************************************************
     * Smoothing of the low-resolution mesh
     *****************************************************************************************/
	meshOutput->smoothing(70);

	return meshOutput;
}


void PropagatedDeformableModel::rafinementGlobal()
{
	if (verbose_) cout << endl << "Global deformation after subdivision... ";
	meshOutputFinal = new SpinalCord(*meshOutput);
	meshOutputFinal->setRadialResolution(resolutionRadiale_);
	//meshOutputFinal = subdivisionRadiale(meshOutput,resolutionRadiale_);
	meshOutputFinal->subdivision();
	meshOutputFinal->computeConnectivity();
	if (verbose_) cout << meshOutputFinal->getNbrOfPoints() << " points and " << meshOutputFinal->getNbrOfTriangles() << " triangles" << endl;
	
	//DeformableModelBasicAdaptator *deformableAdaptator = new DeformableModelBasicAdaptator(image3D_,meshOutputFinal,numberOfDeformIteration_,445.00);
    DeformableModelBasicAdaptator *deformableAdaptator = new DeformableModelBasicAdaptator(image3D_,meshOutputFinal,numberOfDeformIteration_,contrast);
    if (tradeoff_d_bool) deformableAdaptator->setTradeOff(tradeoff_d_);
	deformableAdaptator->setVerbose(verbose_);
	deformableAdaptator->changedParameters();
	deformableAdaptator->setLineSearch(15);
	deformableAdaptator->setAlpha(25);
	deformableAdaptator->setBeta(50);
	deformableAdaptator->setNumberOptimizerIteration(250);
	deformableAdaptator->setNumberOfIteration(3);
	deformableAdaptator->addCorrectionPoints(points_mask_correction_);
	deformableAdaptator->adaptation();
	delete meshOutputFinal;
	meshOutputFinal = deformableAdaptator->getSpinalCordOutput();
	meshOutputFinal->setRadialResolution(2*resolutionRadiale_);
	delete deformableAdaptator;

    meshOutputFinal->smoothing(20);

	centerline = meshOutputFinal->computeCenterline();
	if (verbose_) {
        double distanceSegmentation = 0.0;
        int interval = 1;
        for (unsigned int c=interval; c<centerline.size(); c+=interval)
            distanceSegmentation += (centerline[c]-centerline[c-interval]).Norm();
        cout << "Distance of segmentation [mm] = " << distanceSegmentation << endl;
    }
}

void PropagatedDeformableModel::blockBothExtremesOfMesh(SpinalCord* m, int resolutionRadiale)
{
	vector<Vertex*> points = m->getListPoints();
	//unsigned int endPoints = points.size()-1;
	for (int i=0; i<resolutionRadiale; i++) {
		points[i]->setDeform(false);
		//points[endPoints-i]->setDeform(false);
	}
}


void PropagatedDeformableModel::readCenterline(string filename)
{
	ifstream myfile;
	string l;
	double x, y, z;
	CVector3 point, pointPrecedent;
	int i = 0;
	myfile.open(filename.c_str());
	if (myfile.is_open())
	{
		while ( myfile.good() )
		{
			getline(myfile,l);
			stringstream ss(l);
			ss >> x >> z >> y;
			point = image3D_->TransformIndexToPhysicalPoint(CVector3(x,y,z));
			if ((point-pointPrecedent).Norm() > 0) {
				pointPrecedent = point;
				//point[1] = -point[1];
				centerline.push_back(point);
			}
			i++;
		}
	}
	myfile.close();
}