//
//  SCRegion.h
//  SpinalCordDetectionValidation
//
//  Created by Benjamin De Leener on 2013-12-10.
//  Copyright (c) 2013 Benjamin De Leener. All rights reserved.
//

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif // !_USE_MATH_DEFINES

#ifndef SpinalCordDetectionValidation_SCRegion_h
#define SpinalCordDetectionValidation_SCRegion_h

#include <exception>
#include <string>
#include <vector>
#include <cmath>

#include <itkImage.h>
#include <itkImageAlgorithm.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMattesMutualInformationImageToImageMetric.h>
#include <itkNormalizedCorrelationImageToImageMetric.h>
#include <itkMeanReciprocalSquareDifferenceImageToImageMetric.h>
#include <itkIdentityTransform.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>

#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>
#include <itkContinuousIndex.h>
#include <itkMedianImageFilter.h>

#include "SCTemplate.h"
#include "referential.h"

typedef itk::Image< double, 3 >	ImageType;
typedef itk::Point< double, 3 > PointType;
typedef itk::MattesMutualInformationImageToImageMetric< ImageType, ImageType > MattesMutualInformationFilter;
typedef itk::NormalizedCorrelationImageToImageMetric< ImageType, ImageType > NormalizedCorrelationFilter;
typedef itk::MeanReciprocalSquareDifferenceImageToImageMetric< ImageType, ImageType > MeanSquaredDifferenceFilter;
typedef itk::LinearInterpolateImageFunction< ImageType, double > LinearInterpolatorType;
typedef itk::NearestNeighborInterpolateImageFunction< ImageType, double > NNInterpolatorType;
typedef itk::IdentityTransform< double, 3 > IdentityTransform;
typedef itk::ContinuousIndex<double, 3> ContinuousIndexType;
typedef itk::LinearInterpolateImageFunction< ImageType, double > InterpolateIntensityFilter;
typedef itk::MedianImageFilter<ImageType, ImageType > MedianFilterType;

class SCRegion: public SCTemplate
{
public:
    SCRegion(): SCTemplate()
    {
        origin_[0] = 0.0; origin_[1] = 0.0; origin_[2] = 0.0;
        spacing_[0] = 0.0; spacing_[1] = 0.0; spacing_[2] = 0.0;
    }
    virtual ~SCRegion() {};
    
    virtual void setOrigin(double* origin) { origin_[0] = origin[0]; origin_[1] = origin[1]; origin_[2] = origin[2]; };
    virtual void setOrigin(double x, double y, double z) { origin_[0] = x; origin_[1] = y; origin_[2] = z; };
    virtual double* getOrigin() { return origin_; };
    virtual void setNormal(double* normal) { normal_[0] = normal[0]; normal_[1] = normal[1]; normal_[2] = normal[2]; };
    virtual void setNormal(double x, double y, double z) { normal_[0] = x; normal_[1] = y; normal_[2] = z; };
    virtual double* getNormal() { return normal_; };
    virtual void setSpacing(double* spacing) { spacing_[0] = spacing[0]; spacing_[1] = spacing[1]; spacing_[2] = spacing[2]; };
    virtual double* getSpacing() { return spacing_; };
    virtual void setFactor(float factor) { factor_ = factor; };
    virtual float getFactor() { return factor_; };
    
    void readImage(ImageType::Pointer image)
    {
        if (origin_[0] == 0.0 && origin_[1] == 0.0 && origin_[2] == 0.0) throw std::string("Error: reading image requires a defined origin");
        else if (normal_[0] == 0.0 && normal_[1] == 0.0 && normal_[2] == 0.0) throw std::string("Error: reading image requires a defined origin");
        else if (size_[0] == 0.0 && size_[1] == 0.0 && size_[2] == 0.0) throw std::string("Error: reading image requires a defined size");
        else if (spacing_[0] == 0.0 && spacing_[1] == 0.0 && spacing_[2] == 0.0) throw std::string("Error: reading image requires a defined spacing");
        else
        {
            if (content_ != 0) {
                for (unsigned int x=0; x<size_[0]; x++) {
                    for (unsigned int y=0; y<size_[1]; y++)
                        delete [] content_[x][y];
                    delete [] content_[x];
                }
                delete [] content_;
            }
            content_ = new double**[size_[0]];
            for (unsigned int x=0; x<size_[0]; x++)
            {
                content_[x] = new double*[size_[1]];
                for (unsigned int y=0; y<size_[1]; y++)
                    content_[x][y] = new double[size_[2]];
            }
            
            LinearInterpolatorType::Pointer imageInterpolator = LinearInterpolatorType::New();
            imageInterpolator->SetInputImage(image);
            ContinuousIndexType ind;
            
            ImageType::DirectionType directionImage = image->GetDirection();
            CVector3 axe_AP_image = CVector3(*directionImage[1],*directionImage[4],*directionImage[7]).Normalize();
            //cout << *directionImage[1] << " " << *directionImage[4] << " " << *directionImage[7] << endl;
            
            CVector3 normal(normal_[0],normal_[1],normal_[2]), directionCourantePerpendiculaire, origin(origin_[0],origin_[1],origin_[2]);
            directionCourantePerpendiculaire = -(normal^axe_AP_image).Normalize();
            
            //if (normal[2] == 0.0) directionCourantePerpendiculaire = CVector3(0.0,0.0,1.0);
            //else directionCourantePerpendiculaire = CVector3(1.0,2.0,-(normal[0]+2*normal[1])/normal[2]).Normalize();
            refCourant_ = Referential(normal^directionCourantePerpendiculaire, directionCourantePerpendiculaire, normal, origin);
            
            CMatrix4x4 transformationFromOrigin = refCourant_.getTransformationInverse();
            transformationFromOrigin[0] = spacing_[0]*transformationFromOrigin[0];
            transformationFromOrigin[5] = spacing_[1]*transformationFromOrigin[5];
            transformationFromOrigin[10] = spacing_[2]*transformationFromOrigin[10];
            for (int x=0; x<size_[0]; x++)
            {
                for (int y=0; y<size_[1]; y++)
                {
                    for (int z=0; z<size_[2]; z++)
                    {
                        PointType pt;
                        CVector3 ptInOrigin(x-(size_[0]-1)/2,y-(size_[1]-1)/2,z-(size_[2]-1)/2);
                        CVector3 ptInWorld = transformationFromOrigin*ptInOrigin;
                        pt[0] = ptInWorld[0]; pt[1] = ptInWorld[1]; pt[2] = ptInWorld[2];
                        //pt[0] = origin_[0]+normal_[0]*spacing_[0]*(x-(size_[0]-1)/2);
                        //pt[1] = origin_[1]+normal_[1]*spacing_[1]*(y-(size_[1]-1)/2);
                        //pt[2] = origin_[2]+normal_[2]*spacing_[2]*(z-(size_[2]-1)/2);
                        bool result = image->TransformPhysicalPointToContinuousIndex(pt,ind);
                        if (!result) throw std::string("Error: region of interest exceeds image dimension");
                        content_[x][y][z] = imageInterpolator->EvaluateAtContinuousIndex(ind);
                    }
                }
            }
        }
    };
    void readImage(ImageType::Pointer image, double* spacing)
    {
        spacing_[0] = spacing[0]; spacing_[1] = spacing[1]; spacing_[2] = spacing[2];
        readImage(image);
    };
    void writeImage(std::string filename)
    {
        typedef itk::ImageFileWriter< ImageType > WriterType;
        WriterType::Pointer writer = WriterType::New();
        itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
        writer->SetImageIO(io);
        writer->SetFileName(filename);
        writer->SetInput(image_);
        try {
            writer->Write();
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while writing median image " << std::endl;
            std::cerr << e << std::endl;
        }
    };
    void fromImage(ImageType::Pointer image)
    {
        ImageType::SizeType sizeImage = image->GetLargestPossibleRegion().GetSize();
        size_[0] = sizeImage[0]; size_[1] = sizeImage[1]; size_[2] = sizeImage[2];
        ImageType::SpacingType spacingImage = image->GetSpacing();
        spacing_[0] = spacingImage[0]; spacing_[1] = spacingImage[1]; spacing_[2] = spacingImage[2];
        origin_[0] = 0.0; origin_[1] = 0.0; origin_[2] = 0.0;
        
        if (content_ != 0) {
            for (unsigned int x=0; x<size_[0]; x++) {
                for (unsigned int y=0; y<size_[1]; y++)
                    delete [] content_[x][y];
                delete [] content_[x];
            }
            delete [] content_;
        }
        content_ = new double**[size_[0]];
        for (unsigned int x=0; x<size_[0]; x++)
        {
            content_[x] = new double*[size_[1]];
            for (unsigned int y=0; y<size_[1]; y++)
                content_[x][y] = new double[size_[2]];
        }
        ImageType::IndexType ind;
        for (unsigned int x=0; x<size_[0]; x++) {
            for (unsigned int y=0; y<size_[1]; y++) {
                for (unsigned int z=0; z<size_[2]; z++) {
                    ind[0] = x; ind[1] = y; ind[2] = z;
                    content_[x][y][z] = image->GetPixel(ind);
                }
            }
        }
    };
    
    void createImage()
    {
        image_ = ImageType::New();
        ImageType::IndexType start; start.Fill(0);
        ImageType::SizeType size; size[0] = size_[0]; size[1] = size_[1]; size[2] = size_[2];
        ImageType::RegionType region(start, size);
        ImageType::PointType origin; origin.Fill(0.0);
        
        image_->SetRegions(region);
        image_->SetOrigin(origin);
        image_->Allocate();
        image_->FillBuffer(0);
        
        itk::ImageRegionConstIterator<ImageType> imageIterator(image_,region);
        ImageType::IndexType ind;
        for (unsigned int x=0; x<size_[0]; x++)
        {
            for (unsigned int y=0; y<size_[1]; y++)
            {
                for (unsigned int z=0; z<size_[2]; z++)
                {
                    ind[0] = x; ind[1] = y; ind[2] = z;
                    image_->SetPixel(ind,content_[x][y][z]);
                }
            }
        }
        
        medianFilter();
        
        /*typedef itk::ImageFileWriter< ImageType > WriterType;
        WriterType::Pointer writer = WriterType::New();
        itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
        writer->SetImageIO(io);
        writer->SetFileName("/home/django/benjamindeleener/data/data_montreal/errsm_11/t2/test/im.nii");
        writer->SetInput(image_);
        try {
            writer->Write();
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while writing median image " << std::endl;
            std::cerr << e << std::endl;
        }*/
    };
    ImageType::Pointer getImageCreated() { return image_; };
    
    void medianFilter()
    {
        MedianFilterType::Pointer medianFilter = MedianFilterType::New();
        MedianFilterType::InputSizeType radius;
        radius.Fill(3.0);
        medianFilter->SetRadius(radius);
        medianFilter->SetInput(image_);
        medianFilter->Update();
        image_ = medianFilter->GetOutput();
    };

    double comparisonMutualInformation(SCRegion* source)
    {
        ImageType::Pointer imageSource = source->getImageCreated();
        MattesMutualInformationFilter::Pointer correlationFilter = MattesMutualInformationFilter::New();
        IdentityTransform::Pointer transform = IdentityTransform::New();
        correlationFilter->SetTransform(transform);
        
        LinearInterpolatorType::Pointer interpolator = LinearInterpolatorType::New();
        interpolator->SetInputImage(image_);
        correlationFilter->SetInterpolator(interpolator);
        correlationFilter->SetFixedImage(imageSource);
        correlationFilter->SetMovingImage(image_);
        correlationFilter->SetFixedImageRegion(imageSource->GetLargestPossibleRegion());
        correlationFilter->UseAllPixelsOn();
        correlationFilter->Initialize();
        MattesMutualInformationFilter::TransformParametersType id(3);
        
        id[0] = 0; id[1] = 0; id[2] = 0;
        double result = 0.0;
        try {
            result = correlationFilter->GetValue( id );
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while getting value " << std::endl;
            std::cerr << e << std::endl;
        }
        return result;
    };
    double comparisonCorrelation(SCRegion* source)
    {
        ImageType::Pointer imageSource = source->getImageCreated();
        NormalizedCorrelationFilter::Pointer correlationFilter = NormalizedCorrelationFilter::New();
        IdentityTransform::Pointer transform = IdentityTransform::New();
        correlationFilter->SetTransform(transform);
        LinearInterpolatorType::Pointer interpolator = LinearInterpolatorType::New();
        interpolator->SetInputImage(image_);
        correlationFilter->SetInterpolator(interpolator);
        correlationFilter->SetFixedImage(imageSource);
        correlationFilter->SetMovingImage(image_);
        correlationFilter->SetFixedImageRegion(imageSource->GetLargestPossibleRegion());
        correlationFilter->UseAllPixelsOn();
        correlationFilter->Initialize();
        NormalizedCorrelationFilter::TransformParametersType id(3);
        
        id[0] = 0; id[1] = 0; id[2] = 0;
        double result = 0.0;
        try {
            result = correlationFilter->GetValue( id );
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while getting value " << std::endl;
            std::cerr << e << std::endl;
        }
        return result;
    };
    double comparisonMeanSquareDifference(SCRegion* source)
    {
        ImageType::Pointer imageSource = source->getImageCreated();
        MeanSquaredDifferenceFilter::Pointer correlationFilter = MeanSquaredDifferenceFilter::New();
        IdentityTransform::Pointer transform = IdentityTransform::New();
        correlationFilter->SetTransform(transform);
        LinearInterpolatorType::Pointer interpolator = LinearInterpolatorType::New();
        interpolator->SetInputImage(image_);
        correlationFilter->SetInterpolator(interpolator);
        correlationFilter->SetFixedImage(imageSource);
        correlationFilter->SetMovingImage(image_);
        correlationFilter->SetFixedImageRegion(imageSource->GetLargestPossibleRegion());
        correlationFilter->UseAllPixelsOn();
        try {
            correlationFilter->Initialize();
        } catch (std::string const& e) {
            cerr << e << endl;
            throw e;
        }
        MeanSquaredDifferenceFilter::TransformParametersType id(3);
        
        id[0] = 0; id[1] = 0; id[2] = 0;
        double result = 0.0;
        try {
            result = correlationFilter->GetValue( id );
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while getting value " << std::endl;
            std::cerr << e << std::endl;
        }
        return result;
    };
    
    float computeContrast(double& mean_distance, double& std_distance, int nbPoint=20)
    {
        InterpolateIntensityFilter::Pointer imageInterpolator = InterpolateIntensityFilter::New();
        imageInterpolator->SetInputImage(image_);
        
        CVector3 centerImage = CVector3(((size_[0]-1)/2),((size_[1]-1)/2),((size_[2]-1)/2));
        
        mean_distance = 0.0;
        std_distance = 0.0;
        double rayon_ = 6.0;
        std::vector<double> distance(nbPoint);
        // Calcul du profil de la moelle et du LCR perpendiculairement au tube
        itk::Point<double,3> pointS;
        ContinuousIndexType indexS;
        std::vector<float> contrast(nbPoint);
        float angle;
        CMatrix3x3 trZ;
        for (int k=0; k<nbPoint; k++)
        {
            std::vector<float> profilIntensite;
            angle = 2*M_PI*k/(double)nbPoint;
            trZ[0] = cos(angle), trZ[1] = sin(angle), trZ[3] = -sin(angle), trZ[4] = cos(angle);
            for (double l=0.0; l<=2.5*rayon_; l+=1) {
                CVector3 pointS_temp = trZ*CVector3(l,0.0,0.0)+centerImage;
                pointS[0] = pointS_temp[0]; pointS[1] = pointS_temp[1]; pointS[2] = pointS_temp[2];
                if (image_->TransformPhysicalPointToContinuousIndex(pointS,indexS))
                    profilIntensite.push_back(factor_*imageInterpolator->EvaluateAtContinuousIndex(indexS));
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
            distance[k] = m;
        }
        float result = 0.0;
        for (unsigned int i=0; i<contrast.size(); i++)
            result += contrast[i];
        result /= contrast.size();
        
        for (int n=0; n<nbPoint; n++) {
            mean_distance += distance[n];
            std_distance += distance[n]*distance[n];
        }
        mean_distance /= (double)nbPoint;
        std_distance /= (double)nbPoint;
        std_distance = sqrt(std_distance - mean_distance*mean_distance);
        return result;
    }

	void build2DGaussian(double sigma)
	{
		image_ = ImageType::New();
        ImageType::IndexType start; start.Fill(0);
        ImageType::SizeType size; size[0] = size_[0]; size[1] = size_[1]; size[2] = size_[2];
        ImageType::RegionType region(start, size);
        ImageType::PointType origin; origin.Fill(0.0);
        
        image_->SetRegions(region);
        image_->SetOrigin(origin);
        image_->Allocate();
        image_->FillBuffer(0);
        
        itk::ImageRegionConstIterator<ImageType> imageIterator(image_,region);
        ImageType::IndexType ind;
		double value;
		CVector3 centerImage = CVector3(((size_[0]-1)/2),((size_[1]-1)/2),((size_[2]-1)/2));
        for (unsigned int x=0; x<size_[0]; x++)
        {
            for (unsigned int y=0; y<size_[1]; y++)
            {
                for (unsigned int z=0; z<size_[2]; z++)
                {
                    ind[0] = x; ind[1] = y; ind[2] = z;
					value = exp(-(x-centerImage[0])*(x-centerImage[0])/(2*sigma*sigma)-(y-centerImage[1])*(y-centerImage[1])/(2*sigma*sigma))/(2*M_PI*sigma*sigma);
                    image_->SetPixel(ind,value);
                }
            }
        }
		imageInterpolator = InterpolateIntensityFilter::New();
		imageInterpolator->SetInputImage(image_);
	}

	double GetPixelMagnitudeGradient(const CVector3& index)
	{
		IndexType ind = {static_cast<itk::IndexValueType>(index[0]),static_cast<itk::IndexValueType>(index[1]),static_cast<itk::IndexValueType>(index[2])};
		return image_->GetPixel(ind);
	}

	double GetContinuousPixelMagnitudeGradient(const CVector3& index)
	{
		ContinuousIndexType ind; ind[0] = index[0]; ind[1] = index[1]; ind[2] = index[2];
		return imageInterpolator->EvaluateAtContinuousIndex(ind);
	}


private:
    double origin_[3], normal_[3], spacing_[3];
    float factor_;
    
    ImageType::Pointer image_;
	InterpolateIntensityFilter::Pointer imageInterpolator;
};
    
#endif
