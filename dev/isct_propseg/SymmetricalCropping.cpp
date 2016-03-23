//
//  SymmetricalCropping.cpp
//  Test
//
//  Created by benji_admin on 2013-09-23.
//  Copyright (c) 2013 benji_admin. All rights reserved.
//

#include "SymmetricalCropping.h"
#include <itkExtractImageFilter.h>
#include <itkFixedArray.h>
#include <itkFlipImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkMattesMutualInformationImageToImageMetric.h>
#include <itkIdentityTransform.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <map>
using namespace std;

typedef itk::Image< double, 3 > ImageType;


SymmetricalCropping::SymmetricalCropping()
{
    cropWidth_ = 50.0;
    bandWidth_ = 40.0;
    middleSlice_ = -1;
	initSlice_ = -1.0;
}

int SymmetricalCropping::symmetryDetection(int dimension)
{
    ImageType::SpacingType spacingIm = inputImage_->GetSpacing();
    int cropSize = cropWidth_/spacingIm[2];
    ImageType::SizeType desiredSize = inputImage_->GetLargestPossibleRegion().GetSize();
    
    ImageType::SizeType desiredSizeInitial = inputImage_->GetLargestPossibleRegion().GetSize();
    
    map<double,int> mutualInformation;
    int startSlice = desiredSizeInitial[2]/4, endSlice = desiredSizeInitial[2]/4*3;
    if (desiredSizeInitial[2] < cropSize*2) {
        startSlice = cropSize/2;
        endSlice = desiredSizeInitial[2]-cropSize/2;
    }

	float init_slice;
	if (initSlice_ != -1.0) init_slice = initSlice_;
    if (initSlice_ == -1.0) {
        init_slice = 3*desiredSizeInitial[1]/4;
    }
    else if (initSlice_ < 1.0) {
        init_slice = desiredSize[1]*init_slice;
    }
	else init_slice = initSlice_;

	// Check for non-null intensity in the image. If null, mutual information cannot be computed...
	ImageType::IndexType desiredStart_i;
    ImageType::SizeType desiredSize_i = desiredSizeInitial;
    desiredStart_i[0] = 0;
    desiredStart_i[1] = (int)init_slice;
    desiredStart_i[2] = 0;
    desiredSize_i[1] = 0;
    desiredSize_i[2] = desiredSizeInitial[2];
	ImageType::RegionType desiredRegionImage(desiredStart_i, desiredSize_i);
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
    ImageType2D::Pointer image = cropFilter->GetOutput();
	typedef itk::MinimumMaximumImageCalculator<ImageType2D> MinMaxCalculatorType;
	MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
	minMaxCalculator->SetImage(image);
	minMaxCalculator->ComputeMaximum();
	minMaxCalculator->ComputeMinimum();
	ImageType2D::PixelType maxIm = minMaxCalculator->GetMaximum(), minIm = minMaxCalculator->GetMinimum();
	if (maxIm == minIm) {
		cerr << "ERROR: The axial slice where the symmetry will be detected (slice " << init_slice << ") is full of constant value (" << maxIm << "). You can change it using -init parameter." << endl;
		return -1;
	}

    for (int i=startSlice; i<endSlice; i++)
    {
        float startCrop = i, size;
        if (startCrop < desiredSizeInitial[2]/2 && startCrop <= bandWidth_+1) size = startCrop-1;
        else if (startCrop >= desiredSizeInitial[2]/2 && startCrop >= desiredSizeInitial[2]-bandWidth_-1) size = desiredSizeInitial[2]-startCrop-1;
        else size = bandWidth_;
        ImageType::IndexType desiredStart;
        ImageType::SizeType desiredSize = desiredSizeInitial;
        desiredStart[0] = 0;
        desiredStart[1] = (int)init_slice;
        desiredStart[2] = startCrop;
        desiredSize[1] = 0;
        desiredSize[2] = size;
        
        // Right Image
        ImageType::RegionType desiredRegionImageRight(desiredStart, desiredSize);
        typedef itk::ExtractImageFilter< ImageType, ImageType2D > Crop2DFilterType;
        Crop2DFilterType::Pointer cropFilterRight = Crop2DFilterType::New();
        cropFilterRight->SetExtractionRegion(desiredRegionImageRight);
        cropFilterRight->SetInput(inputImage_);
#if ITK_VERSION_MAJOR >= 4
        cropFilterRight->SetDirectionCollapseToIdentity(); // This is required.
#endif
        try {
            cropFilterRight->Update();
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while updating cropFilter " << std::endl;
            std::cerr << e << std::endl;
        }
        ImageType2D::Pointer imageRight = cropFilterRight->GetOutput();
        
        // Left Image
        desiredStart[2] = startCrop-size;
        if (desiredStart[2] < 0) desiredStart[2] = 0;
        ImageType::RegionType desiredRegionImageLeft(desiredStart, desiredSize);
        Crop2DFilterType::Pointer cropFilterLeft = Crop2DFilterType::New();
        cropFilterLeft->SetExtractionRegion(desiredRegionImageLeft);
        cropFilterLeft->SetInput(inputImage_);
#if ITK_VERSION_MAJOR >= 4
        
        cropFilterLeft->SetDirectionCollapseToIdentity(); // This is required.
#endif
        try {
            cropFilterLeft->Update();
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while updating cropFilter " << std::endl;
            std::cerr << e << std::endl;
        }
        ImageType2D::Pointer imageLeft = cropFilterLeft->GetOutput();
        
        ImageType2D::IndexType desIndex; desIndex.Fill(0);
        ImageType2D::SizeType desSize; desSize[0] = desiredSize[0]; desSize[1] = desiredSize[2];
        ImageType2D::RegionType desired2DRegionImageRight(desIndex, desSize);
        ImageType2D::RegionType desired2DRegionImageLeft(desIndex, desSize);
        imageRight->SetLargestPossibleRegion(desired2DRegionImageRight);
        imageRight->SetRequestedRegion(desired2DRegionImageRight);
        imageRight->SetRegions(desired2DRegionImageRight);
        imageLeft->SetLargestPossibleRegion(desired2DRegionImageLeft);
        imageLeft->SetRequestedRegion(desired2DRegionImageLeft);
        imageLeft->SetRegions(desired2DRegionImageLeft);
        
        itk::FixedArray<bool, 2> flipAxes;
        flipAxes[0] = false;
        flipAxes[1] = true;
        typedef itk::FlipImageFilter <ImageType2D> FlipImageFilterType;
        FlipImageFilterType::Pointer flipFilter = FlipImageFilterType::New ();
        flipFilter->SetInput(imageRight);
        flipFilter->SetFlipAxes(flipAxes);
        flipFilter->Update();
        imageRight = flipFilter->GetOutput();
        
        ImageType2D::PointType origin = imageLeft->GetOrigin();
        imageRight->SetOrigin(origin);
        
        // Better value is minimum
        typedef itk::MattesMutualInformationImageToImageMetric< ImageType2D, ImageType2D > MattesMutualInformationFilter;
        MattesMutualInformationFilter::Pointer correlationFilter = MattesMutualInformationFilter::New();
        typedef itk::IdentityTransform< double,2 > IdentityTransform;
        IdentityTransform::Pointer transform = IdentityTransform::New();
        correlationFilter->SetTransform(transform);
        typedef itk::NearestNeighborInterpolateImageFunction< ImageType2D, double > InterpolatorType;
        InterpolatorType::Pointer interpolator = InterpolatorType::New();
        interpolator->SetInputImage(imageRight);
        correlationFilter->SetInterpolator(interpolator);
        correlationFilter->SetFixedImage(imageLeft);
        correlationFilter->SetMovingImage(imageRight);
        correlationFilter->SetFixedImageRegion(imageLeft->GetLargestPossibleRegion());
        correlationFilter->UseAllPixelsOn();
        correlationFilter->Initialize();
        MattesMutualInformationFilter::TransformParametersType id(3);
        
        id[0] = 0; id[1] = 0; id[2] = 0;
        double value = 0.0;
        try {
            value = correlationFilter->GetValue( id );
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while getting value " << std::endl;
            std::cerr << e << std::endl;
        }
        mutualInformation[value] = startCrop;
    }
    //cout << "Cropping around slice = " << mutualInformation.begin()->second << endl;
    middleSlice_ = mutualInformation.begin()->second;
    return middleSlice_;
}

ImageType::Pointer SymmetricalCropping::cropping()
{
    ImageType::SpacingType spacingIm = inputImage_->GetSpacing();
    int cropSize = cropWidth_/spacingIm[2];
    float start = middleSlice_-cropSize/2;
    ImageType::SizeType desiredSize = inputImage_->GetLargestPossibleRegion().GetSize();
    
    ImageType::IndexType desiredStart;
    desiredStart[0] = 0;
    desiredStart[1] = 0;
    desiredStart[2] = start;
    desiredSize[2] = cropSize;
    
    ImageType::RegionType desiredRegion(desiredStart, desiredSize);
    typedef itk::ExtractImageFilter< ImageType, ImageType > CropFilterType;
    CropFilterType::Pointer cropFilter = CropFilterType::New();
    cropFilter->SetExtractionRegion(desiredRegion);
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
    
    outputImage_ = cropFilter->GetOutput();
    outputImage_->DisconnectPipeline();
    desiredStart[2] = 0;
    desiredRegion.SetIndex(desiredStart);
    ImageType::PointType origin;
    ImageType::IndexType index = outputImage_->GetLargestPossibleRegion().GetIndex();
    outputImage_->TransformIndexToPhysicalPoint(index, origin);
    outputImage_->SetOrigin(origin);
    outputImage_->SetLargestPossibleRegion(desiredRegion);
    outputImage_->SetRequestedRegion(desiredRegion);
    outputImage_->SetRegions(desiredRegion);
    
    /*Writer3DType::Pointer writer = Writer3DType::New();
    itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
    writer->SetImageIO(io);
    writer->SetFileName(path+"cropped_"+filename);
    writer->SetInput(image);
    
    try {
        writer->Update();
    } catch( itk::ExceptionObject & e ) {
        std::cerr << "Exception caught while writting cropped image " << std::endl;
        std::cerr << e << std::endl;
    }*/
    
    return outputImage_;
}