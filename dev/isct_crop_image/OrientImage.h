//
//  OrientImage.h
//  Test
//
//  Created by benji_admin on 2013-09-22.
//  Copyright (c) 2013 benji_admin. All rights reserved.
//

#ifndef __Test__OrientImage__
#define __Test__OrientImage__

#include <itkImage.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientationAdapter.h>

typedef itk::SpatialOrientationAdapter SpatialOrientationAdapterType;
typedef itk::SpatialOrientation::ValidCoordinateOrientationFlags OrientationType;

template <typename InputImageType>
class OrientImage
{
public:
    typedef typename InputImageType::Pointer    InputImagePointer;
    
    OrientImage() {};
    ~OrientImage() {};
    
    void setInputImage(InputImagePointer image)
    {
        image_ = image;
        SpatialOrientationAdapterType adapt;
        initialImageOrientation_ = adapt.FromDirectionCosines(image_->GetDirection());
    };
    InputImagePointer getOutputImage() { return outputImage_; };
    OrientationType getInitialImageOrientation() { return initialImageOrientation_; };
    void orientation(OrientationType desiredOrientation)
    {
        typename itk::OrientImageFilter<InputImageType,InputImageType>::Pointer orienter = itk::OrientImageFilter<InputImageType,InputImageType>::New();
        orienter->UseImageDirectionOn();
        orienter->SetDesiredCoordinateOrientation(desiredOrientation);
        orienter->SetInput(image_);
        orienter->Update();
        outputImage_ = orienter->GetOutput();
    };
    
private:
    InputImagePointer image_, outputImage_;
    OrientationType initialImageOrientation_;
};

#endif /* defined(__Test__OrientImage__) */
