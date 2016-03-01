#ifndef __PROPOGATED_DEFORMABLE_MODEL__
#define __PROPOGATED_DEFORMABLE_MODEL__

/*!
 * \file PropagatedDeformableModel.h
 * \brief Perform the propagated segmentation of the spinal cord on MR image
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include <vector>
#include "Image3D.h"
#include "util/Vector3.h"
#include "SpinalCord.h"
#include "BSplineApproximation.h"
using namespace std;

/*!
 * \class PropagatedDeformableModel
 * \brief Perform the propagated segmentation of the spinal cord on MR image
 * 
 * This class performs the segmentation of the spinal cord by propagating an elliptical tubular mesh in MR volume as T1- or T2-weighted image.
 */
class PropagatedDeformableModel
{
public:
	PropagatedDeformableModel();
	PropagatedDeformableModel(int resolutionRadiale, int resolutionAxiale, double rayon, int numberOfDeformIteration, int numberOfPropagationIteration, double deplacementAxial, double propagationLength);
	~PropagatedDeformableModel();

	void addPointToCenterline(CVector3 point) { centerline.push_back(point); };
    void propagationWithCenterline() { propCenterline_ = !propCenterline_; };
	void setInitialPointAndNormals(CVector3 initialPoint, CVector3 normal1, CVector3 normal2);
    void setStretchingFactor(double stretchingFactor) { stretchingFactor_ = stretchingFactor; };
	void setUpAndDownLimits(int downLimit, int upLimit);
	void computeMeshInitial();
	void adaptationGlobale();
	void rafinementGlobal();
	

	vector<CVector3> getCenterline() { return centerline; };
	SpinalCord* getInitialMesh() { return initialTube1; };
	SpinalCord* getInverseInitialMesh() { return initialTube2; };
	SpinalCord* getOutput() { return meshOutput; };
	SpinalCord* getOutputFinal() { return meshOutputFinal; };

	void readCenterline(string filename);
	void setImage3D(Image3D* image) { image3D_ = image; };

	void changedParameters() { this->changedParameters_ = true; };
	void setAlpha(double alpha) { this->alpha = alpha; };
	void setBeta(double beta) { this->beta = beta; };
	void setLineSearch(double line_search) { this->line_search = line_search; };
	void setContrast(double contrast) { this->meanContrast = contrast; };
	double getContrast() { return meanContrast; };
    
    void setInitPosition(double init) { init_position_ = init; };
    double getInitPosition() { return init_position_; };
    
    void setMaxDeformation(double max) { maxDeformation = max; };
    void setMaxArea(double max) { maxArea =  max; };
    void setMinContrast(double min) { minContrast = min; };

	void setTradeOffDistanceFeature(double tradeoff_d) { tradeoff_d_ = tradeoff_d; tradeoff_d_bool = true; };
    
    void setVerbose(bool verbose) { verbose_ = verbose; };
    bool getVerbose() { return verbose_; };

    void addCorrectionPoints(vector<CVector3> points_mask_correction) { points_mask_correction_ = points_mask_correction; };

private:
	SpinalCord* mergeBidirectionalSpinalCord(SpinalCord* spinalCord1, SpinalCord* spinalCord2);
	SpinalCord* propagationMesh(int numberOfMesh=1);
	float computeContrast(Referential& refInitial);
	void computeNewBand(SpinalCord* mesh, CVector3 initialPoint, CVector3 nextPoint, int resolution);
	void blockBothExtremesOfMesh(SpinalCord* m, int resolutionRadiale);

	vector<CVector3> centerline, initial_centerline;
	CVector3 initialPoint_, initialNormal1_, initialNormal2_;
    double stretchingFactor_;
	bool hasInitialPointAndNormals_, propCenterline_;
	SpinalCord *initialTube1, *initialTube2, *meshOutput, *meshOutputFinal;
	bool isMeshInitialized;

	int resolutionRadiale_, resolutionAxiale_;
	double rayon_, deplacementAxial_, propagationLength_;
	int numberOfDeformIteration_, numberOfPropagationIteration_;

	vector< vector<CVector3> > lastDisks;
	CVector3 lastNormal;

	Image3D* image3D_;

	int downLimit, upLimit;
    double init_position_;

	// Deformable models adaptator parameters
	bool changedParameters_;
	double line_search, alpha, beta, meanContrast, area[3], meanArea;
    vector< pair<CVector3,double> > contrast;
    
    double maxDeformation, maxArea, minContrast;

	double tradeoff_d_;
	bool tradeoff_d_bool;
    
    BSplineApproximation centerline_approximator;
    double range;
    
    bool verbose_;

    vector<CVector3> points_mask_correction_;
};

#endif
