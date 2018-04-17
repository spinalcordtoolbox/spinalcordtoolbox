#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif // !_USE_MATH_DEFINES

#include "Orientation.h"
#include "referential.h"
#include "itkHoughTransform2DCirclesImageFilter.h"
#include <itkExtractImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkImageDuplicator.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>
#include <vector>
#include <map>
#include <math.h>
using namespace std;

typedef itk::Image< double, 3 >	ImageType;
typedef itk::Image< double, 2 >	ImageType2D;
typedef itk::ImageFileWriter<ImageType2D> WriterType;

Orientation::Orientation(Image3D* image, SpinalCord* s)
    : image_(image), mesh_(s), badOrientation_(false)
{
    typeImageFactor_ = image_->getTypeImageFactor();
}

bool Orientation::computeOrientation(double &distance)
{
    vector<CVector3> centerline = mesh_->computeCenterline();
    CVector3 targetPoint = centerline[centerline.size()-1], sourcePoint = centerline[0], indexTargetPoint;
    image_->TransformPhysicalPointToContinuousIndex(targetPoint, indexTargetPoint);
    CVector3 lastNormal = (targetPoint-sourcePoint).Normalize();
    
    typedef itk::ExtractImageFilter< ImageType, ImageType2D > ExtractFilterType;
    ImageType::Pointer inputImage = image_->getCroppedImageOriginale();
    ImageType::SizeType desiredSizeExtract = inputImage->GetLargestPossibleRegion().GetSize();
    ImageType::IndexType desiredStartExtract;
	desiredStartExtract[0] = 0;
	desiredStartExtract[1] = indexTargetPoint[1];
	desiredStartExtract[2] = 0;
	desiredSizeExtract[1] = 0;
    ImageType::RegionType desiredRegion(desiredStartExtract, desiredSizeExtract);
    ExtractFilterType::Pointer filter = ExtractFilterType::New();
    filter->SetExtractionRegion(desiredRegion);
    filter->SetInput(inputImage);
    #if ITK_VERSION_MAJOR >= 4
    filter->SetDirectionCollapseToIdentity();
    #endif
    try {
        filter->Update();
    } catch( itk::ExceptionObject & e ) {
        std::cerr << "Exception caught while updating cropFilter " << std::endl;
        std::cerr << e << std::endl;
    }
    ImageType2D::Pointer image2D = filter->GetOutput();
    
    ImageType2D::SizeType size2D = image2D->GetLargestPossibleRegion().GetSize();
    ImageType2D::SpacingType spacingInput = image2D->GetSpacing();
    double width = 25.0; // width of square region in millimeters
    ImageType2D::SizeType desiredSize2D;
    desiredSize2D[0] = width/spacingInput[0];
    desiredSize2D[1] = width/spacingInput[1];
    ImageType2D::IndexType desiredStart2D;
    desiredStart2D[0] = indexTargetPoint[0]-width/(2*spacingInput[0]);
    desiredStart2D[1] = indexTargetPoint[2]-width/(2*spacingInput[1]);
    if (desiredStart2D[0] < 0) desiredStart2D[0] = 0;
    if (desiredStart2D[1] < 0) desiredStart2D[1] = 0;
    if (desiredStart2D[0]+desiredSize2D[0] > size2D[0]) desiredSize2D[0] = size2D[0]-desiredStart2D[0];
    if (desiredStart2D[1]+desiredSize2D[1] > size2D[1]) desiredSize2D[1] = size2D[1]-desiredStart2D[1];
    ImageType2D::RegionType desiredRegion2D(desiredStart2D,desiredSize2D);
    
    double stretchingFactor = 1.0, step = 0.5;
    
    typedef itk::ExtractImageFilter< ImageType2D, ImageType2D > ExtractFilterType2D;
    ExtractFilterType2D::Pointer extractFilter2D = ExtractFilterType2D::New();
    extractFilter2D->SetExtractionRegion(desiredRegion2D);
    extractFilter2D->SetInput(image2D);
#if ITK_VERSION_MAJOR >= 4
    extractFilter2D->SetDirectionCollapseToIdentity();
#endif
    try {
        extractFilter2D->Update();
    } catch( itk::ExceptionObject & e ) {
        std::cerr << "Exception caught while updating cropFilter " << std::endl;
        std::cerr << e << std::endl;
    }
    ImageType2D::Pointer image2DE = extractFilter2D->GetOutput();
    
    typedef itk::ImageDuplicator< ImageType2D > DuplicatorType;
	DuplicatorType::Pointer duplicator = DuplicatorType::New();
	duplicator->SetInputImage(image2DE);
	duplicator->Update();
	ImageType2D::Pointer im = duplicator->GetOutput();
    
    /*WriterType::Pointer writer = WriterType::New();
    itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
    writer->SetImageIO(io);
    writer->SetFileName("/home/django/benjamindeleener/data/segmentation_spinalCord/test.nii");
    writer->SetInput(im);
    writer->Update();*/
    
    typedef itk::ResampleImageFilter<ImageType2D, ImageType2D> ResampleImageFilterType;
    typedef itk::IdentityTransform<double, 2> TransformType;
    map<double,CVector3,greater<double> > centers;
    while (stretchingFactor <= 1.0)
    {
        if (stretchingFactor != 1.0)
        {
            ImageType2D::SizeType inputSize = image2D->GetLargestPossibleRegion().GetSize(), outputSize;
            outputSize[0] = inputSize[0]*stretchingFactor;
            outputSize[1] = inputSize[1];
            ImageType2D::SpacingType outputSpacing;
            outputSpacing[0] = static_cast<double>(image2D->GetSpacing()[0] * inputSize[0] / outputSize[0]);
            outputSpacing[1] = static_cast<double>(image2D->GetSpacing()[1] * inputSize[1] / outputSize[1]);
            
            ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
            resample->SetInput(image2D);
            resample->SetSize(outputSize);
            resample->SetOutputDirection(image2D->GetDirection());
            resample->SetOutputOrigin(image2D->GetOrigin());
            resample->SetOutputSpacing(outputSpacing);
            resample->SetTransform(TransformType::New());
            resample->Update();
            
            im = resample->GetOutput();
            
            ExtractFilterType2D::Pointer extractFilter2D = ExtractFilterType2D::New();
            ImageType2D::SizeType desiredSize2DE;
            desiredSize2DE[0] = stretchingFactor*width/spacingInput[0];
            desiredSize2DE[1] = width/spacingInput[1];
            ImageType2D::RegionType desiredRegion2DE(desiredStart2D,desiredSize2DE);
            extractFilter2D->SetExtractionRegion(desiredRegion2DE);
            extractFilter2D->SetInput(resample->GetOutput());
#if ITK_VERSION_MAJOR >= 4
            extractFilter2D->SetDirectionCollapseToIdentity();
#endif
            try {
                extractFilter2D->Update();
            } catch( itk::ExceptionObject & e ) {
                std::cerr << "Exception caught while updating cropFilter " << std::endl;
                std::cerr << e << std::endl;
            }
            im = extractFilter2D->GetOutput();
        }
        
        vector<CVector3> center;
        vector<double> radius, accumulator;
        searchCenters(im,center,radius,accumulator,indexTargetPoint[1],indexTargetPoint);
        double maxAccumulator = 0.0;
        unsigned int indexMax = -1;
        for (unsigned int i=0; i<center.size(); i++) {
            if (accumulator[i] > maxAccumulator) {
                maxAccumulator = accumulator[i];
                indexMax = i;
            }
        }
        if (indexMax != -1) {
            CVector3 cent(center[indexMax][0],center[indexMax][1],center[indexMax][2]);
            centers[accumulator[indexMax]] = cent;
        }
        
        stretchingFactor += step;
    }
    
    indexTargetPoint = centers.begin()->second;
    CVector3 lastTP = targetPoint;
    targetPoint = image_->TransformContinuousIndexToPhysicalPoint(indexTargetPoint);
    if (verbose_) cout << "Target Point : " << targetPoint << endl;
    double distanceTarget = sqrt((targetPoint[0]-lastTP[0])*(targetPoint[0]-lastTP[0])+(targetPoint[1]-lastTP[1])*(targetPoint[1]-lastTP[1])+(targetPoint[2]-lastTP[2])*(targetPoint[2]-lastTP[2]));
    double angle = (acos((targetPoint*lastTP)/(targetPoint.Norm()*lastTP.Norm()))/(2.0*M_PI))*360.0;
    if (verbose_) cout << "Angle = " << angle << endl;
    distance = distanceTarget;
    if (distanceTarget >= 3.0) {
        if (verbose_) cout << "Distance error : " << distanceTarget << endl;
        return false;
    }
    else if (angle > 1.0 && angle < 3.0) {
        CVector3 normalNew = (targetPoint-sourcePoint).Normalize();
        CVector3 axe = (lastNormal^normalNew).Normalize();
        Referential refCourant = Referential(lastNormal^axe, axe, lastNormal, sourcePoint);
        Referential newReferential = Referential(normalNew^axe, axe, normalNew, sourcePoint);
        CMatrix3x3 transformation = refCourant.getTransformation(newReferential);
        mesh_->transform(transformation,sourcePoint);
    }
    
    return true;
}


void Orientation::searchCenters(ImageType2D::Pointer im, vector<CVector3> &center, vector<double> &radius, vector<double> &accumulator, float z, CVector3 c)
{
    typedef itk::ImageDuplicator< ImageType2D > DuplicatorType;
	DuplicatorType::Pointer duplicator = DuplicatorType::New();
	duplicator->SetInputImage(im);
	duplicator->Update();
	ImageType2D::Pointer clonedOutput = duplicator->GetOutput();
    
    unsigned int numberOfCircles = 15;
	double **center_result_small = new double*[numberOfCircles], **center_result_large = new double*[numberOfCircles];
	for (unsigned int k=0; k<numberOfCircles; k++) {
		center_result_small[k] = new double[2];
		center_result_large[k] = new double[2];
	}
	double *radius_result_small = new double[numberOfCircles], *radius_result_large = new double[numberOfCircles];
	double *accumulator_result_small = new double[numberOfCircles], *accumulator_result_large = new double[numberOfCircles];
	unsigned int numSmall = houghTransformCircles(im,numberOfCircles,center_result_small,radius_result_small,accumulator_result_small,4.0,-1.0);
	unsigned int numLarge = houghTransformCircles(clonedOutput,numberOfCircles,center_result_large,radius_result_large,accumulator_result_large,4.0+4.0,1.0);
    
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
    map< double,CVector3,greater<double> > centers;
	for (unsigned int i=0; i<listMostPromisingCenters.size(); i++)
	{
		centers[accumulator_result_small[listMostPromisingCenters[i]]] = CVector3(center_result_small[listMostPromisingCenters[i]][0],z,center_result_small[listMostPromisingCenters[i]][1]);
	}
    
    for (int l=0; l<3; l++ ) {
        if (centers.size() != 0) {
            center.push_back(centers.begin()->second);
            radius.push_back(0.0);
            accumulator.push_back(centers.begin()->first);
            centers.erase(centers.begin());
        }
    }
}


unsigned int Orientation::houghTransformCircles(ImageType2D* im, unsigned int numberOfCircles, double** center_result, double* radius_result, double* accumulator_result, double meanRadius, double valPrint)
{
    typedef itk::MinimumMaximumImageCalculator<ImageType2D> MinMaxCalculatorType;
	MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
	minMaxCalculator->SetImage(im);
	minMaxCalculator->ComputeMaximum();
	minMaxCalculator->ComputeMinimum();
	ImageType2D::PixelType maxIm = minMaxCalculator->GetMaximum(), minIm = minMaxCalculator->GetMinimum();
	double val_Print = maxIm;
    
    double min_radius = meanRadius-3.0;
    if (min_radius < 0) min_radius = 0;
	
    typedef itk::HoughTransform2DCirclesImageFilter< double, double > HoughCirclesFilter;
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
    typedef itk::DiscreteGaussianImageFilter<ImageType2D,ImageType2D> GaussianFilterType;
	GaussianFilterType::Pointer gaussianFilter = GaussianFilterType::New();
	gaussianFilter->SetInput(m_Accumulator);
	double variance[2];
	variance[0]=10;
	variance[1]=10;
	gaussianFilter->SetVariance(variance);
	gaussianFilter->SetMaximumError(.01f);
	gaussianFilter->Update();
	m_PostProcessImage = gaussianFilter->GetOutput();
    
    ImageType2D::IndexType start = m_PostProcessImage->GetLargestPossibleRegion().GetIndex();
	ImageType2D::SizeType bound = m_PostProcessImage->GetLargestPossibleRegion().GetSize();
    
	itk::ImageRegionIterator<ImageType2D> it_output(im,im->GetLargestPossibleRegion());
	itk::ImageRegionIterator<ImageType2D> it_input(m_PostProcessImage,m_PostProcessImage->GetLargestPossibleRegion());
    
    
	/** Set the disc ratio */
	double discRatio = 1.1;
    
	/** Search for maxima */
	unsigned int circles=0, maxIteration=100, it=0;
	do {
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
				if (index[0]!=start[0] && index[0]!=start[0]+bound[0]-1 && index[1]!=start[1] && index[1]!=start[1]+bound[1]-1)
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
						if (index[0]>=start[0] && index[0]<start[0]+bound[0] && index[1]>=start[1] && index[1]<start[1]+bound[1])
							im->SetPixel(index,val_Print);
                        
						// Remove the maximum from the accumulator
						for(double length = 0; length < discRatio*radius2;length+=1)
						{
							index[0] = (long int)(it_output.GetIndex()[0] + length * cos(angle));
							index[1] = (long int)(it_output.GetIndex()[1] + length* sin(angle));
							if (index[0]>=start[0] && index[0]<start[0]+bound[0] && index[1]>=start[1] && index[1]<start[1]+bound[1])
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
							if (index[0]>=start[0] && index[0]<start[0]+bound[0] && index[1]>=start[1] && index[1]<start[1]+bound[1])
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
