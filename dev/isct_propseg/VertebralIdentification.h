//
//  VertebralIdentification.h
//  sct_segmentation_propagation
//
//  Created by Benjamin De Leener on 2014-03-10.
//  Copyright (c) 2014 Benjamin De Leener. All rights reserved.
//

#ifndef __sct_segmentation_propagation__VertebralIdentification__
#define __sct_segmentation_propagation__VertebralIdentification__

#include <vector>
#include <string>

#include <itkImage.h>

#include "util/Vector3.h"


typedef itk::Image< double, 3 >	ImageType;

class VertebralIdentification
{
public:
    VertebralIdentification() {};
    VertebralIdentification(ImageType::Pointer image, std::vector<CVector3> centerline):image_(image),centerline_(centerline){};
    ~VertebralIdentification() {};
    
    void getIntensityProfile();
    
private:
    std::vector<double> detrend(std::vector<double> data, int degree, std::string func_type="");
    
    ImageType::Pointer image_;
    std::vector<CVector3> centerline_; // in world coordinates
};

#endif /* defined(__sct_segmentation_propagation__VertebralIdentification__) */
