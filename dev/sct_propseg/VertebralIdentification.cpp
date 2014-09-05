//
//  VertebralIdentification.cpp
//  sct_segmentation_propagation
//
//  Created by Benjamin De Leener on 2014-03-10.
//  Copyright (c) 2014 Benjamin De Leener. All rights reserved.
//

#define _USE_MATH_DEFINES

#include "VertebralIdentification.h"
#include <string>
#include <fstream>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMedianImageFilter.h>

#include "util/MatrixNxM.h"
using namespace std;

typedef itk::MedianImageFilter<ImageType, ImageType > MedianFilterType;
typedef itk::LinearInterpolateImageFunction< ImageType, double > InterpolateIntensityFilter;
typedef itk::Point< double, 3 > PointType;
typedef itk::ContinuousIndex<double, 3> ContinuousIndexType;

void VertebralIdentification::getIntensityProfile()
{
    MedianFilterType::Pointer medianFilter = MedianFilterType::New();
    MedianFilterType::InputSizeType radius;
    radius.Fill(1.0);
    medianFilter->SetRadius(radius);
    medianFilter->SetInput(image_);
    medianFilter->Update();
    ImageType::Pointer image_median = medianFilter->GetOutput();
    
    InterpolateIntensityFilter::Pointer imageInterpolator = InterpolateIntensityFilter::New();
    imageInterpolator->SetInputImage(image_median);
    
    // displacement of the centerline in the antero-posterior direction
    PointType point;
    ContinuousIndexType continuous_index, continuous_index_temp;
    vector<double> intensity_profile(centerline_.size());
    for (unsigned int i=0; i<centerline_.size(); i++)
    {
        centerline_[i][1] -= 15; //TO BE UPDATED WITH THE DIRECTION OF THE IMAGE
        point[0] = centerline_[i][0];
        point[1] = centerline_[i][1];
        point[2] = centerline_[i][2];
        if (image_->TransformPhysicalPointToContinuousIndex(point, continuous_index))
        {
            continuous_index_temp = continuous_index;
            intensity_profile[i] = 0.0;
            for(int x=-1; x<=1; x++) {
                for(int y=-1; y<=1; y++) {
                    continuous_index_temp[0] = continuous_index[0];
                    continuous_index_temp[1] = continuous_index[1];
                    intensity_profile[i] += imageInterpolator->EvaluateAtContinuousIndex(continuous_index_temp);
                }
            }
            intensity_profile[i] /= 9.0;
        }
    }
    
    // detrend data
    intensity_profile = detrend(intensity_profile,6);
    //for (int i=0; i<intensity_profile.size(); i++) cout << intensity_profile[i] << endl;
    
    // normalize with maximum
    double maximum = 0.0;
    int index_max = 0;
    for (int i=0; i<intensity_profile.size(); i++) {
        if (intensity_profile[i] > maximum) {
            maximum = intensity_profile[i];
            index_max = i;
        }
    }
    for (int i=0; i<intensity_profile.size(); i++) intensity_profile[i] /= maximum;
    
    
    // writing intensity profile in file.txt
    ofstream myfile;
	myfile.open("/home/django/benjamindeleener/data/segmentation_spinalCord/errsm_11/T1_test_cropped/intensity_profile.txt");
    for (unsigned int i=0; i<intensity_profile.size(); i++)
        myfile << intensity_profile[i] << endl;
	myfile.close();
}

vector<double> VertebralIdentification::detrend(vector<double> data, int degree, string func_type)
{
    // get data size
    int nb_samples = data.size();
    
    Matrice D;
    if (func_type == "linear")
    {
        // Linear trend
        D = Matrice(nb_samples,1);
        for (double i=-1; i<=1; i+=2.0/((double)nb_samples-1.0))
            D(i,0) = i;
    }
    else
    {
        // create DCT basis of regressors,
        D = Matrice(nb_samples,degree);
        for (int i=0; i<nb_samples; i++) D(i,0) = 0.0;
        for (int i=0; i<nb_samples; i++) D(i,1) = sqrt(nb_samples)/(double)nb_samples;
        for (int k=0; k<degree; k++) {
            for (int i=0; i<nb_samples; i++) D(i,k) = sqrt((2.0*sqrt(nb_samples))/(double)nb_samples)*cos(M_PI*(2.0*(double)i+1.0)*(k-1.0)/(2.0*(double)nb_samples));
        }
    }
    
    Matrice data_m = Matrice(nb_samples,1);
    for (int i=0; i<data.size(); i++) data_m(i,0) = data[i];
    // detrend data
    Matrice l = (D.transpose()*D).pinv()*(D.transpose()*data_m);
    // reconstruct drift signal
    Matrice Dl = D*l;
    // reconstruct data2d without low frequency drifts
    data_m = data_m-Dl;
    
    vector<double> result(nb_samples);
    for (int i=0; i<nb_samples; i++) result[i] = data_m(i,0);
                 
    return result;
}
