#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif // !_USE_MATH_DEFINES

#include <cmath>
#include <map>
#include <list>

#include <itkImage.h>
#include <itkNiftiImageIO.h>
#include <itkPNGImageIO.h>
#include <itkIndex.h>
#include <itkContinuousIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkGradientMagnitudeImageFilter.h>
#include <itkGradientImageFilter.h>
#include <itkImageAlgorithm.h>
#include <itkInvertIntensityImageFilter.h>
#include <itkGradientMagnitudeRecursiveGaussianImageFilter.h>
#include <itkGradientRecursiveGaussianImageFilter.h>
#include "itkHoughTransform2DCirclesImageFilter.h"
#include <itkExtractImageFilter.h>
#include <itkEllipseSpatialObject.h>
#include <itkImageDuplicator.h>
#include <itkImageFileWriter.h>
#include <itkScaleTransform.h>
#include <itkResampleImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkRGBPixel.h>

#include "Initialisation.h"
#include "OrientImage.h"

using namespace std;

typedef itk::Image< double, 3 >	ImageType;
typedef itk::Image< double, 2 >	ImageType2D;
typedef itk::Image< unsigned char, 3 > BinaryImageType;
typedef itk::ExtractImageFilter< ImageType, ImageType2D > FilterType;
typedef itk::MinimumMaximumImageCalculator<ImageType2D> MinMaxCalculatorType;
typedef itk::ImageDuplicator< ImageType2D > DuplicatorType;
typedef itk::HoughTransform2DCirclesImageFilter< double, double > HoughCirclesFilter;
typedef itk::DiscreteGaussianImageFilter<ImageType2D,ImageType2D> GaussianFilterType;
typedef itk::ImageFileWriter< BinaryImageType >  WriterBinaryType;
typedef ImageType::IndexType Index;
typedef itk::ContinuousIndex<double,3> ContinuousIndex;
typedef itk::Point< double, 3 > PointType;
typedef itk::GradientMagnitudeImageFilter< ImageType2D, ImageType2D > GradientMFilterType;
typedef itk::IdentityTransform<double, 2> TransformType;
typedef itk::ResampleImageFilter<ImageType2D, ImageType2D> ResampleImageFilterType;
typedef itk::DiscreteGaussianImageFilter<ImageType2D, ImageType2D> SmoothFilterType;
typedef itk::ImageRegionConstIterator<BinaryImageType> ImageIterator;
typedef itk::RescaleIntensityImageFilter< ImageType, ImageType > RescaleFilterType;

double round(double number)
{
    return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}

class Node
{
public:
	Node(CVector3 point, double radius, double accumulator, CVector3 center, double radiusStretch, double stretchingFactor=0.0):point_(point),radius_(radius),accumulator_(accumulator),stretchingFactor_(stretchingFactor),hasNext_(false),hasPrevious_(false),centerStretch_(center),radiusStretch_(radiusStretch) {};
	~Node(){};
    
	CVector3 getPosition() { return point_; };
	double getRadius() { return radius_; };
    CVector3 getCenterStretch() { return centerStretch_; };
	double getRadiusStretch() { return radiusStretch_; };
	double getAccumulator() { return accumulator_; };
    double getStretchingFactor() {return stretchingFactor_; };
	bool hasNext() { return hasNext_; };
	bool hasPrevious() { return hasPrevious_; };
	void addNext(Node* n)
	{
		nextPoint_ = n;
		hasNext_ = true;
	};
	void addPrevious(Node* p)
	{
		previousPoint_ = p;
		hasPrevious_ = true;
	};
	Node* getNext() { return nextPoint_; };
	Node* getPrevious() { return previousPoint_; };
    
private:
	CVector3 point_, centerStretch_;
	double radius_, radiusStretch_, accumulator_;
    double stretchingFactor_;
    
	Node *nextPoint_, *previousPoint_;
	bool hasNext_, hasPrevious_;
};

Initialisation::Initialisation()
{
	typeImageFactor_ = 1;
    gap_ = 1.0;
    startSlice_ = -1.0;
    numberOfSlices_ = 5;
    radius_ = 4.0;
	verbose_ = false;
}


Initialisation::Initialisation(ImageType::Pointer image, double imageFactor, double gap)
{
	OrientImage<ImageType> orientationFilter;
    orientationFilter.setInputImage(image);
    orientationFilter.orientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL);
    inputImage_ = orientationFilter.getOutputImage();
    orientation_ = orientationFilter.getInitialImageOrientation();
    
	typeImageFactor_ = imageFactor;
    gap_ = gap;
    startSlice_ = -1.0;
    numberOfSlices_ = 5;
    radius_ = 4.0;

	verbose_ = false;
}


void Initialisation::setInputImage(ImageType::Pointer image)
{
    OrientImage<ImageType> orientationFilter;
    orientationFilter.setInputImage(image);
    orientationFilter.orientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL);
    inputImage_ = orientationFilter.getOutputImage();
    orientation_ = orientationFilter.getInitialImageOrientation();
}


bool Initialisation::computeInitialParameters(float startFactor)
{  
	ImageType::SizeType desiredSize = inputImage_->GetLargestPossibleRegion().GetSize();
	
    // The spinal cord detection is performed on a bunch of axial slices. The choice of which slices will be analyzed depends on the startFactor. Default is the middle axial slice. The parameter startFactor must be the number of the slice, or a number between 0 and 1 representing the pourcentage of the image.
    // For exemple, startFactor=0.5 means the detection will start in the middle axial slice.
    float startZ;
    if (startFactor != -1.0) startSlice_ = startFactor;
    if (startSlice_ == -1.0) {
        startZ = desiredSize[1]/2;
        startSlice_ = startZ;
    }
    else if (startSlice_ < 1.0) {
        startZ = desiredSize[1]*startSlice_;
        startSlice_ = startZ;
    }
	else startZ = startSlice_;

	// Adapt radius to the image spacing to provide a radius in pixels - use average spacing of axial slice
    ImageType::SpacingType spacing = inputImage_->GetSpacing();
    mean_resolution_ = (spacing[0]+spacing[2])/2;
    
    
    // Adapt the gap between detection axial slices to the spacing
	if (round(spacing[1]) != 0 && (int)gap_ % (int)round(spacing[1]) != 0)
	{
		gap_ = spacing[1];
	}
    
    // Adapt the number of axial slices used for the detection to the spacing and the image dimensions.
	if (startZ-((numberOfSlices_-1.0)/2.0)*(gap_/spacing[1]) < 0 || startZ+((numberOfSlices_-1.0)/2.0)*(gap_/spacing[1]) >= desiredSize[1])
	{
		numberOfSlices_ = numberOfSlices_-2;
		//gap_ = 1;
		if (verbose_) {
			cout << "WARNING: number of slices and gap between slices are not adapted to the image dimensions for the initilization. Default parameters will be used." << endl;
			cout << "New parameters:" << endl << "Gap inter slices = " << gap_ << endl << "Number of slices = " << numberOfSlices_ << endl;
		}
	}
    
    // Initalisation of the paremeters for the spinal cord detection
	ImageType::IndexType desiredStart;
	desiredStart[0] = 0;
	desiredStart[1] = startZ;
	desiredStart[2] = 0;
	desiredSize[1] = 0;

    // First extraction of the axial slice to check if the image contains information (not null)
	ImageType::RegionType desiredRegionImage(desiredStart, desiredSize);
    typedef itk::ExtractImageFilter< ImageType, ImageType2D > Crop2DFilterType;
    Crop2DFilterType::Pointer cropFilter = Crop2DFilterType::New();
    cropFilter->SetExtractionRegion(desiredRegionImage);
    cropFilter->SetInput(inputImage_);
	#if ITK_VERSION_MAJOR >= 4
    cropFilter->SetDirectionCollapseToIdentity(); // This is required.
	#endif
    try {
        cropFilter->Update();
    } catch( itk::ExceptionObject & e ) {
        std::cerr << "Exception caught while updating cropFilter " << std::endl;
        std::cerr << e << std::endl;
    }
    ImageType2D::Pointer image_test_minmax = cropFilter->GetOutput();
	MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
	minMaxCalculator->SetImage(image_test_minmax);
	minMaxCalculator->ComputeMaximum();
	minMaxCalculator->ComputeMinimum();
	ImageType2D::PixelType maxIm = minMaxCalculator->GetMaximum(), minIm = minMaxCalculator->GetMinimum();
	if (maxIm == minIm) {
		cerr << "WARNING: The principal axial slice where the spinal cord detection will be performed (slice " << startZ << ") is full of constant value (" << maxIm << "). You can change it using -init parameter." << endl;
	}
    
    // Starting the spinal cord detection process
    if (verbose_) cout << "Initialization" << endl;
    
    // Creation of a matrix of potential spinal cord centers
	vector<vector <vector <Node*> > > centers;
    
    // Start of the detection of circles and ellipses. For each axial slices, a Hough transform is performed to detect circles. Each axial image is stretched in the antero-posterior direction in order to detect the spinal cord as a ellipse as well as a circle.
	for (int i=round(-((numberOfSlices_-1.0)/2.0)*(gap_/spacing[1])); i<=round(((numberOfSlices_-1.0)/2.0)*(gap_/spacing[1])); i+=round(gap_/spacing[1]))
	{
        // Cropping of the image
		if (verbose_) cout << "Slice num " << i << endl;
		desiredStart[1] = startZ+i;
		ImageType::RegionType desiredRegion(desiredStart, desiredSize);
		FilterType::Pointer filter = FilterType::New();
		filter->SetExtractionRegion(desiredRegion);
		filter->SetInput(inputImage_);
        #if ITK_VERSION_MAJOR >= 4
		filter->SetDirectionCollapseToIdentity(); // This is required.
        #endif
		try {
			filter->Update();
		} catch( itk::ExceptionObject & e ) {
			std::cerr << "Exception caught while updating cropFilter " << std::endl;
			std::cerr << e << std::endl;
			cout << inputImage_->GetLargestPossibleRegion().GetSize() << endl;
			cout << desiredRegion << endl;
		}
        
        // The image is duplicated to allow multiple processing on the image.
		ImageType2D::Pointer im = filter->GetOutput();
		DuplicatorType::Pointer duplicator = DuplicatorType::New();
		duplicator->SetInputImage(im);
		duplicator->Update();
		ImageType2D::Pointer clonedImage = duplicator->GetOutput();
		ImageType::DirectionType imageDirection = inputImage_->GetDirection();
		ImageType2D::DirectionType clonedImageDirection;
		clonedImageDirection[0][0] = imageDirection[0][0];
		clonedImageDirection[0][1] = imageDirection[0][2];
		clonedImageDirection[1][0] = imageDirection[1][0];
		clonedImageDirection[1][1] = imageDirection[1][2];
		clonedImage->SetDirection(clonedImageDirection);
        
		// Initialization of resulting spinal cord center list.
		vector<vector <Node*> > vecNode;
        
        // Initialization of stretching parameters
        // A stretchingFactor equals to 1 doesn't change the image
		double stretchingFactor = 1.0, step = 0.25;
		while (stretchingFactor <= 2.0)
		{
			if (verbose_) cout << "Stretching factor " << stretchingFactor << endl;
            // Stretching the image in the antero-posterior direction. This direction is chosen because potential elliptical spinal cord will be transformed to circles and wil be detected by the Hough transform. The resulting circles will then be stretch in the other direction.
			if (stretchingFactor != 1.0)
			{
				ImageType2D::SizeType inputSize = clonedImage->GetLargestPossibleRegion().GetSize(), outputSize;
				outputSize[0] = inputSize[0]*stretchingFactor;
				outputSize[1] = inputSize[1];
				ImageType2D::SpacingType outputSpacing;
				outputSpacing[0] = static_cast<double>(clonedImage->GetSpacing()[0] * inputSize[0] / outputSize[0]);
				outputSpacing[1] = static_cast<double>(clonedImage->GetSpacing()[1] * inputSize[1] / outputSize[1]);
                
				ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
				resample->SetInput(clonedImage);
				resample->SetSize(outputSize);
				resample->SetOutputDirection(clonedImage->GetDirection());
				resample->SetOutputOrigin(clonedImage->GetOrigin());
				resample->SetOutputSpacing(outputSpacing);
				resample->SetTransform(TransformType::New());
				resample->Update();
                
				im = resample->GetOutput();
			}
            
            // Searching the circles in the image using circular Hough transform, adapted from ITK
            // The list of radii and accumulator values are then extracted for analyses
			vector<CVector3> vecCenter;
			vector<double> vecRadii, vecAccumulator;
			searchCenters(im,vecCenter,vecRadii,vecAccumulator,startZ+i);
			
            // Reformating of the detected circles in the image. Each detected circle is push in a Node with all its information.
            // The radii are transformed in mm using mean axial resolution
			vector<Node*> vecNodeTemp;
			for (unsigned int k=0; k<vecCenter.size(); k++) {
				if (vecRadii[k] != 0.0) {
					CVector3 center = vecCenter[k]; center[0] /= stretchingFactor;
					vecNodeTemp.push_back(new Node(center,mean_resolution_*vecRadii[k]/stretchingFactor,vecAccumulator[k],vecCenter[k],mean_resolution_*vecRadii[k],stretchingFactor));
				}
			}
			vecNode.push_back(vecNodeTemp);
			
            // Preparing next iteration of the spinal cord detection
			stretchingFactor += step;
		}
        // Saving the detected centers
		centers.push_back(vecNode);
	}
    
	// All centers are ordoned by slice
	// First step -> delete points without neighbour
	double limitDistance = sqrt(2.0*gap_*gap_); // in mm
	list<Node*> listPoints;
	for (unsigned int k=0; k<numberOfSlices_; k++)
	{
		// For every point in a slice, we search on next and previous slice for neighbors
        // Potential neighbours are circles that have a similar radius (less than 20% of difference)
		for (unsigned int i=0; i<centers[k].size(); i++)
		{
			for (unsigned int m=0; m<centers[k][i].size(); m++)
			{
				bool hasNeighbor = false;
				double radius = centers[k][i][m]->getRadius();
				if (k != 0) // search down
				{
                    // All the point are sorted by the distance
					map<double,Node*> listNeighbors;
					for (unsigned int j=0; j<centers[k-1][i].size(); j++)
					{
                        // Compute the distance between two adjacent centers (in mm)
                        // If this distance is less or equal to the limit distance, the two centers are attached to each others
						double currentDistance = mean_resolution_*sqrt(pow(centers[k][i][m]->getPosition()[0]-centers[k-1][i][j]->getPosition()[0],2)+pow(centers[k][i][m]->getPosition()[1]-centers[k-1][i][j]->getPosition()[1],2)+pow(centers[k][i][m]->getPosition()[2]-centers[k-1][i][j]->getPosition()[2],2));
						if (currentDistance <= limitDistance)
							listNeighbors[currentDistance] = centers[k-1][i][j];
					}
					while (!listNeighbors.empty())
					{
						double radiusCurrent = listNeighbors.begin()->second->getRadius();
						if (radiusCurrent >= radius*0.8 && radiusCurrent <= radius*1.2)
						{
							hasNeighbor = true;
							centers[k][i][m]->addPrevious(listNeighbors.begin()->second);
							break;
						}
						listNeighbors.erase(listNeighbors.begin());
					}
				}
				if (k != numberOfSlices_-1) // search up
				{
					map<double,Node*> listNeighbors;
					for (unsigned int j=0; j<centers[k+1][i].size(); j++)
					{
						double currentDistance = mean_resolution_*sqrt(pow(centers[k][i][m]->getPosition()[0]-centers[k+1][i][j]->getPosition()[0],2)+pow(centers[k][i][m]->getPosition()[1]-centers[k+1][i][j]->getPosition()[1],2)+pow(centers[k][i][m]->getPosition()[2]-centers[k+1][i][j]->getPosition()[2],2));
						if (currentDistance <= limitDistance)
							listNeighbors[currentDistance] = centers[k+1][i][j];
					}
					while (!listNeighbors.empty())
					{
						double radiusCurrent = listNeighbors.begin()->second->getRadius();
						if (radiusCurrent >= radius*0.8 && radiusCurrent <= radius*1.2)
						{
							hasNeighbor = true;
							centers[k][i][m]->addNext(listNeighbors.begin()->second);
							break;
						}
						listNeighbors.erase(listNeighbors.begin());
					}
				}
				if (hasNeighbor) // if point has at least one neighbor, we keep it
					listPoints.push_back(centers[k][i][m]);
			}
		}
	}
    
	// Second step -> assembling points
	vector<vector <Node*> > chains;
	while (listPoints.size() != 0)
	{
		vector<Node*> temp;
		Node* current = listPoints.front();
		temp.push_back(current);
		listPoints.pop_front();
		while(current->hasNext())
		{
			current = current->getNext();
			temp.push_back(current);
		}
		chains.push_back(temp);
	}
	// And search for the longest and with larger accumulation value and small angle between normals 
	unsigned int maxLenght = 0, max = 0;
	double maxAccumulator = 0.0, angleMax = 15.0;
	for (unsigned int j=0; j<chains.size(); j++)
	{
		unsigned int length = chains[j].size();
        double angle = 0.0;
        if (length >= 3)
        {
            CVector3 vector1 = chains[j][0]->getPosition()-chains[j][length/2]->getPosition(), vector2 = (chains[j][length/2]->getPosition()-chains[j][length-1]->getPosition());
            angle = 360.0*acos((vector1*vector2)/(vector1.Norm()*vector2.Norm()))/(2.0*M_PI);
        }
		if (length > maxLenght && angle <= angleMax)
		{
			maxLenght = chains[j].size();
			max = j;
			maxAccumulator = 0.0;
			for (unsigned int k=0; k<length; k++)
				maxAccumulator += chains[j][k]->getAccumulator();
		}
		else if (length == maxLenght && angle <= angleMax)
		{
			double accumulator = 0.0;
			for (unsigned int k=0; k<length; k++)
				accumulator += chains[j][k]->getAccumulator();
			if (accumulator > maxAccumulator) {
				maxLenght = chains[j].size();
				max = j;
				maxAccumulator = accumulator;
			}
		}
	}
    
	if (chains.size() > 1)
	{
		unsigned int sizeMaxChain = chains[max].size();
		//cout << "Results : " << endl;
        points_.clear();
		for (unsigned int j=0; j<sizeMaxChain; j++) {
            points_.push_back(chains[max][j]->getPosition());
			//cout << chains[max][j]->getPosition() << " " << chains[max][j]->getRadius() << endl;
        }
        if (verbose_) cout << "Stretching factor of circle found = " << chains[max][0]->getStretchingFactor() << endl;
		if (sizeMaxChain < numberOfSlices_) {
			if (verbose_) cout << "Warning: Number of center found on slices (" << sizeMaxChain << ") doesn't correspond to number of analyzed slices. An error may occur. To improve results, you can increase the number of analyzed slices (option -n must be impair)" << endl;
            
			// we have to transform pixel points to physical points
			CVector3 finalPoint, initPointT = chains[max][0]->getPosition(), finalPointT = chains[max][sizeMaxChain-1]->getPosition();
			ContinuousIndex initPointIndex, finalPointIndex;
			initPointIndex[0] = initPointT[0]; initPointIndex[1] = initPointT[1]; initPointIndex[2] = initPointT[2];
			finalPointIndex[0] = finalPointT[0]; finalPointIndex[1] = finalPointT[1]; finalPointIndex[2] = finalPointT[2];
			PointType initPoint, finPoint;
			inputImage_->TransformContinuousIndexToPhysicalPoint(initPointIndex,initPoint);
			inputImage_->TransformContinuousIndexToPhysicalPoint(finalPointIndex,finPoint);
			initialPoint_ = CVector3(initPoint[0],initPoint[1],initPoint[2]);
			finalPoint = CVector3(finPoint[0],finPoint[1],finPoint[2]);
			initialNormal1_ = (finalPoint-initialPoint_).Normalize();
			initialRadius_ = 0.0;
			for (unsigned int j=0; j<sizeMaxChain; j++)
				initialRadius_ += chains[max][j]->getRadiusStretch();
			initialRadius_ /= sizeMaxChain;
            stretchingFactor_ = chains[max][0]->getStretchingFactor();
		}
		else
		{
			// we have to transform pixel points to physical points
			CVector3 finalPoint1, finalPoint2, initPointT = chains[max][(int)(sizeMaxChain/2)]->getPosition(), finalPointT1 = chains[max][0]->getPosition(), finalPointT2 = chains[max][sizeMaxChain-1]->getPosition();
			ContinuousIndex initPointIndex, finalPoint1Index, finalPoint2Index;
			initPointIndex[0] = initPointT[0]; initPointIndex[1] = initPointT[1]; initPointIndex[2] = initPointT[2];
			finalPoint1Index[0] = finalPointT1[0]; finalPoint1Index[1] = finalPointT1[1]; finalPoint1Index[2] = finalPointT1[2];
			finalPoint2Index[0] = finalPointT2[0]; finalPoint2Index[1] = finalPointT2[1]; finalPoint2Index[2] = finalPointT2[2];
			PointType initPoint, finPoint1, finPoint2;
			inputImage_->TransformContinuousIndexToPhysicalPoint(initPointIndex,initPoint);
			inputImage_->TransformContinuousIndexToPhysicalPoint(finalPoint1Index,finPoint1);
			inputImage_->TransformContinuousIndexToPhysicalPoint(finalPoint2Index,finPoint2);
			initialPoint_ = CVector3(initPoint[0],initPoint[1],initPoint[2]);
			finalPoint1 = CVector3(finPoint1[0],finPoint1[1],finPoint1[2]);
			finalPoint2 = CVector3(finPoint2[0],finPoint2[1],finPoint2[2]);
			initialNormal1_ = (finalPoint1-initialPoint_).Normalize();
			initialNormal2_ = (finalPoint2-initialPoint_).Normalize();
			initialRadius_ = 0.0;
			for (unsigned int j=0; j<sizeMaxChain; j++)
				initialRadius_ += chains[max][j]->getRadiusStretch();
			initialRadius_ /= sizeMaxChain;
            stretchingFactor_ = chains[max][0]->getStretchingFactor();
		}
		return true;
	}
	else {
		cout << "Error: No point detected..." << endl;
		return false;
	}
}


void Initialisation::searchCenters(ImageType2D::Pointer im, vector<CVector3> &vecCenter, vector<double> &vecRadii, vector<double> &vecAccumulator, float startZ)
{
	DuplicatorType::Pointer duplicator = DuplicatorType::New();
	duplicator->SetInputImage(im);
	duplicator->Update();
	ImageType2D::Pointer clonedOutput = duplicator->GetOutput();
    
	MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
    
	unsigned int numberOfCircles = 20;
	double **center_result_small = new double*[numberOfCircles], **center_result_large = new double*[numberOfCircles];
	for (unsigned int k=0; k<numberOfCircles; k++) {
		center_result_small[k] = new double[2];
		center_result_large[k] = new double[2];
	}
	double *radius_result_small = new double[numberOfCircles], *radius_result_large = new double[numberOfCircles];
	double *accumulator_result_small = new double[numberOfCircles], *accumulator_result_large = new double[numberOfCircles];
	unsigned int numSmall = houghTransformCircles(im,numberOfCircles,center_result_small,radius_result_small,accumulator_result_small,radius_/mean_resolution_,-1.0);
	unsigned int numLarge = houghTransformCircles(clonedOutput,numberOfCircles,center_result_large,radius_result_large,accumulator_result_large,radius_/mean_resolution_+6.0,1.0);
    
	// search along results for nested circles
	vector<unsigned int> listMostPromisingCenters;
	vector<unsigned int> listMostPromisingCentersLarge;
	double distance = 0.0;
	for (unsigned int i=0; i<numSmall; i++)
	{
		for (unsigned int j=0; j<numLarge; j++)
		{
			// distance between center + small_radius must be smaller than large_radius
			distance = sqrt(pow(center_result_small[i][0]-center_result_large[j][0],2)+pow(center_result_small[i][1]-center_result_large[j][1],2));
			if ((distance+radius_result_small[i])*0.8 <= radius_result_large[j]) {
				listMostPromisingCenters.push_back(i);
				listMostPromisingCentersLarge.push_back(j);
			}
		}
	}
	for (unsigned int i=0; i<listMostPromisingCenters.size(); i++)
	{
		vecCenter.push_back(CVector3(center_result_small[listMostPromisingCenters[i]][0],startZ,center_result_small[listMostPromisingCenters[i]][1]));
		vecRadii.push_back(radius_result_small[listMostPromisingCenters[i]]);
		vecAccumulator.push_back(accumulator_result_small[listMostPromisingCenters[i]]);
	}
}


unsigned int Initialisation::houghTransformCircles(ImageType2D* im, unsigned int numberOfCircles, double** center_result, double* radius_result, double* accumulator_result, double meanRadius, double valPrint)
{
	MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
	minMaxCalculator->SetImage(im);
	minMaxCalculator->ComputeMaximum();
	minMaxCalculator->ComputeMinimum();
	ImageType2D::PixelType maxIm = minMaxCalculator->GetMaximum(), minIm = minMaxCalculator->GetMinimum();
	double val_Print = maxIm;
    
    double min_radius = meanRadius-3.0;
    if (min_radius < 0) min_radius = 0;
	
	HoughCirclesFilter::Pointer houghfilter = HoughCirclesFilter::New();
	houghfilter->SetInput(im);
	houghfilter->SetMinimumRadius(min_radius);
	houghfilter->SetMaximumRadius(meanRadius+3.0);
	houghfilter->SetSigmaGradient(2);
	houghfilter->SetGradientFactor(valPrint*typeImageFactor_);
	houghfilter->SetSweepAngle(M_PI/180.0*5.0);
	houghfilter->SetThreshold((maxIm-minIm)/20.0);
	houghfilter->Update();
	
    
	const double nPI = 4.0 * vcl_atan( 1.0 );
	ImageType2D::IndexType index;
    
	ImageType2D::Pointer m_Accumulator= houghfilter->GetOutput();
    
	ImageType2D::Pointer m_RadiusImage= houghfilter->GetRadiusImage();
    
	/** Blur the accumulator in order to find the maximum */
	ImageType2D::Pointer m_PostProcessImage = ImageType2D::New();
	GaussianFilterType::Pointer gaussianFilter = GaussianFilterType::New();
	gaussianFilter->SetInput(m_Accumulator);
	double variance[2];
	variance[0]=10;
	variance[1]=10;
	gaussianFilter->SetVariance(variance);
	gaussianFilter->SetMaximumError(.01f);
	gaussianFilter->Update();
	m_PostProcessImage = gaussianFilter->GetOutput();
    
	ImageType2D::SizeType bound = m_PostProcessImage->GetLargestPossibleRegion().GetSize();
    
	itk::ImageRegionIterator<ImageType2D> it_output(im,im->GetLargestPossibleRegion());
	itk::ImageRegionIterator<ImageType2D> it_input(m_PostProcessImage,m_PostProcessImage->GetLargestPossibleRegion());
    
    
	/** Set the disc ratio */
	double discRatio = 1.1;
    
	/** Search for maxima */
	unsigned int circles=0, maxIteration=100, it=0;
	do{
		it++;
		minMaxCalculator->SetImage(m_PostProcessImage);
		minMaxCalculator->ComputeMaximum();
		ImageType2D::PixelType max = minMaxCalculator->GetMaximum();
        
		it_output.GoToBegin();
		for(it_input.GoToBegin();!it_input.IsAtEnd();++it_input)
		{
			if(it_input.Get() == max)
			{
				it_output.Set(val_Print);
				index = it_output.GetIndex();
				double radius2 = m_RadiusImage->GetPixel(index);
				if (index[0]!=0 && index[0]!=bound[0]-1 && index[1]!=0 && index[1]!=bound[1]-1)
				{
					center_result[circles][0] = it_output.GetIndex()[0];
					center_result[circles][1] = it_output.GetIndex()[1];
					radius_result[circles] = radius2;
					accumulator_result[circles] = m_PostProcessImage->GetPixel(index);
                    
					// Draw the circle
					for(double angle = 0; angle <= 2 * nPI; angle += nPI / 1000)
					{
						index[0] = (long int)(it_output.GetIndex()[0] + radius2 * cos(angle));
						index[1] = (long int)(it_output.GetIndex()[1] + radius2 * sin(angle));
						if (index[0]>=0 && index[0]<bound[0] && index[1]>=0 && index[1]<bound[1])
							im->SetPixel(index,val_Print);
                        
						// Remove the maximum from the accumulator
						for(double length = 0; length < discRatio*radius2;length+=1)
						{
							index[0] = (long int)(it_output.GetIndex()[0] + length * cos(angle));
							index[1] = (long int)(it_output.GetIndex()[1] + length* sin(angle));
							if (index[0]>=0 && index[0]<bound[0] && index[1]>=0 && index[1]<bound[1])
								m_PostProcessImage->SetPixel(index,0);
						}
					}
					circles++;
					if(circles == numberOfCircles) break;
				}
				else
				{
					// Draw the circle
					for(double angle = 0; angle <= 2 * nPI; angle += nPI / 1000)
					{
						// Remove the maximum from the accumulator
						for(double length = 0; length < discRatio*radius2;length+=1)
						{
							index[0] = (long int)(it_output.GetIndex()[0] + length * cos(angle));
							index[1] = (long int)(it_output.GetIndex()[1] + length* sin(angle));
							if (index[0]>=0 && index[0]<bound[0] && index[1]>=0 && index[1]<bound[1])
								m_PostProcessImage->SetPixel(index,0);
						}
					}
				}
				minMaxCalculator->SetImage(m_PostProcessImage);
				minMaxCalculator->ComputeMaximum();
				max = minMaxCalculator->GetMaximum();
			}
			++it_output;
		}
	}
	while(circles<numberOfCircles && it<=maxIteration);
    
	return circles;
}

void Initialisation::getPoints(CVector3 &point, CVector3 &normal1, CVector3 &normal2, double &radius, double &stretchingFactor)
{
	point = initialPoint_;
	normal1 = initialNormal1_;
	normal2 = initialNormal2_;
    if (initialRadius_/stretchingFactor_ > radius_) radius = radius_*stretchingFactor_; // initialRadius_ majored by radius_
    else radius = initialRadius_;
    stretchingFactor = stretchingFactor_;
}

void Initialisation::savePointAsBinaryImage(ImageType::Pointer initialImage, string filename, OrientationType orientation)
{
    if (points_.size() > 0)
    {
        typedef itk::Image< unsigned char, 3 > BinaryImageType;
        BinaryImageType::Pointer binary = BinaryImageType::New();
        ImageType::RegionType region;
        ImageType::IndexType start;
        start[0] = 0; start[1] = 0; start[2] = 0;
        ImageType::SizeType size, imSize = initialImage->GetLargestPossibleRegion().GetSize();
        size[0] = imSize[0]; size[1] = imSize[1]; size[2] = imSize[2];
        region.SetSize(size);
        region.SetIndex(start);
        binary->CopyInformation(initialImage);
        binary->SetRegions(region);
        binary->Allocate();
        binary->FillBuffer(false);
        
        typedef ImageType::IndexType IndexType;
        ContinuousIndex ind;
        IndexType ind2;
        unsigned int pSize = points_.size();
        unsigned int indexMiddle = 0;
        for (unsigned int i=0; i<pSize; i++) {
            if (points_[i][1] == startSlice_)
                indexMiddle = i;
        }
        ind[0] = points_[indexMiddle][0]; ind[1] = points_[indexMiddle][1]; ind[2] = points_[indexMiddle][2];
        PointType pt;
        inputImage_->TransformContinuousIndexToPhysicalPoint(ind, pt);
        initialImage->TransformPhysicalPointToIndex(pt, ind2);
        binary->SetPixel(ind2,true);
        
        OrientImage<BinaryImageType> orientationFilter;
        orientationFilter.setInputImage(binary);
        orientationFilter.orientation(orientation);
        binary = orientationFilter.getOutputImage();
        
        ImageIterator it( binary, binary->GetRequestedRegion() );
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            if (it.Get()==true)
            {
                ind2 = it.GetIndex();
                break;
            }
            ++it;
        }
        if (verbose_) cout << "Center of spinal cord saved on pixel : " << ind2 << endl;
        
        WriterBinaryType::Pointer writer = WriterBinaryType::New();
        itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
        writer->SetImageIO(io);
        writer->SetFileName(filename);
        writer->SetInput(binary);
        try {
            writer->Write();
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while writing image " << std::endl;
            std::cerr << e << std::endl;
        }
    }
    else cout << "Error: Spinal cord center not detected" << endl;
}

void Initialisation::savePointAsAxialImage(ImageType::Pointer initialImage, string filename)
{
    if (points_.size() > 0)
    {
        double radius = 2.0;
        //if (initialRadius_/stretchingFactor_ > radius_) radius = radius_*stretchingFactor_; // initialRadius_ majored by radius_
        //else radius = initialRadius_;
        typedef itk::ImageDuplicator< ImageType > DuplicatorType3D;
        DuplicatorType3D::Pointer duplicator = DuplicatorType3D::New();
        duplicator->SetInputImage(initialImage);
        duplicator->Update();
        ImageType::Pointer clonedImage = duplicator->GetOutput();
        
        // Intensity normalization
        RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
        rescaleFilter->SetInput(clonedImage);
        rescaleFilter->SetOutputMinimum(0);
        rescaleFilter->SetOutputMaximum(255);
        try {
            rescaleFilter->Update();
        } catch( itk::ExceptionObject & e ) {
            cerr << "Exception caught while normalizing input image " << endl;
            cerr << e << endl;
        }
        clonedImage = rescaleFilter->GetOutput();
        
        typedef itk::RGBPixel<unsigned char> RGBPixelType;
        typedef itk::Image<RGBPixelType, 2> RGBImageType;
		typedef itk::ExtractImageFilter< ImageType, RGBImageType > ExtractorTypeRGB;

		PointType pt; pt[0] = initialPoint_[0]; pt[1] = initialPoint_[1]; pt[2] = initialPoint_[2];
        ImageType::IndexType ind;
        clonedImage->TransformPhysicalPointToIndex(pt,ind);

		ImageType::SizeType desiredSize = clonedImage->GetLargestPossibleRegion().GetSize();
		ImageType::IndexType desiredStart;
		desiredStart[0] = 0; desiredStart[1] = ind[1]; desiredStart[2] = 0;
		desiredSize[1] = 0;
		ImageType::RegionType desiredRegion(desiredStart, desiredSize);
		ExtractorTypeRGB::Pointer filter = ExtractorTypeRGB::New();
		filter->SetExtractionRegion(desiredRegion);
		filter->SetInput(clonedImage);
		#if ITK_VERSION_MAJOR >= 4
		filter->SetDirectionCollapseToIdentity(); // This is required.
		#endif
		try {
			filter->Update();
		} catch( itk::ExceptionObject & e ) {
			std::cerr << "Exception caught while updating ExtractorTypeRGB " << std::endl;
			std::cerr << e << std::endl;
		}
		RGBImageType::Pointer image = filter->GetOutput();
        
		// draw cross
        RGBPixelType pixel; pixel[0] = 255; pixel[1] = 255; pixel[2] = 255;
		for (int x=-radius; x<=radius; x++) {
			RGBImageType::IndexType ind_x, ind_y;
			ind_x[0] = ind[0]+x; ind_x[1] = ind[2]; ind_y[0] = ind[0]; ind_y[1] = ind[2]+x;
			image->SetPixel(ind_x, pixel);
			image->SetPixel(ind_y, pixel);
		}

		typedef itk::ImageFileWriter< RGBImageType > WriterRGBType;
		itk::PNGImageIO::Pointer ioPNG = itk::PNGImageIO::New();
		WriterRGBType::Pointer writerPNG = WriterRGBType::New();
		writerPNG->SetInput(image);
		writerPNG->SetImageIO(ioPNG);
		writerPNG->SetFileName(filename);
		try {
		    writerPNG->Update();
		}
		catch( itk::ExceptionObject & e )
		{
			cout << "Exception thrown ! " << endl;
			cout << "An error ocurred during Writing PNG" << endl;
			cout << "Location    = " << e.GetLocation()    << endl;
			cout << "Description = " << e.GetDescription() << endl;
		}
    }
    else cout << "Error: Spinal cord center not detected" << endl;
}
