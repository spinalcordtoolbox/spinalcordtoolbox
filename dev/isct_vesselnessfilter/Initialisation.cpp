#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif // !_USE_MATH_DEFINES
#include "Initialisation.h"
#include "OrientImage.h"
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
#include <itkMedianImageFilter.h>
#include <itkFlipImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkMattesMutualInformationImageToImageMetric.h>
#include <itkMeanReciprocalSquareDifferenceImageToImageMetric.h>
#include <itkNormalizedCorrelationImageToImageMetric.h>
#include <itkIdentityTransform.h>
#include <itkHessianRecursiveGaussianImageFilter.h>
#include <itkMultiScaleHessianBasedMeasureImageFilter.h>
#include <itkHessianToObjectnessMeasureImageFilter.h>
//#include <itkGradientImageFilter.h>
#include "itkGradientVectorFlowImageFilter.h" // local version
#include <itkGradientVectorFlowImageFilter.h> // ITK version
#include "itkRecursiveGaussianImageFilter.h"
#include "itkImageRegionIterator.h"
#include <itkSymmetricSecondRankTensor.h>
#include "itkHessian3DToVesselnessMeasureImageFilter.h"
#include <itkStatisticsImageFilter.h>
#include "itkTileImageFilter.h"
#include "itkPermuteAxesImageFilter.h"
#include <itkDiscreteGaussianImageFilter.h>


#include "BSplineApproximation.h"
#include "MatrixNxM.h"
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
typedef itk::ImageFileWriter< ImageType >     WriterType;

double roundIn(double number)
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

vector<CVector3> Initialisation::getCenterlineUsingMinimalPath(vector<int> middle_slices, double alpha, double beta, double gamma, double sigmaMinimum, double sigmaMaximum, unsigned int numberOfSigmaSteps, double sigmaDistance)
{
    if (verbose_) cout << "Starting vesselness filtering and minimal path identification...";
    // Apply vesselness filter on the image
    ImageType::Pointer vesselnessImage = vesselnessFilter(middle_slices, inputImage_, alpha, beta, gamma, sigmaMinimum, sigmaMaximum, numberOfSigmaSteps, sigmaDistance);
    
    // Compute the minimal path on the vesselnessfilter result to find the centerline
    // The vesselness image is multiplied by a 3D gaussian mask, around the median plane, in the left-right direction.
    vector<CVector3> centerline;
    ImageType::Pointer minimalImage = minimalPath3d(vesselnessImage, centerline, true, true);
    
    // Transform centerline coordinates in world coordinates as requested by PropSeg (previously in image coordinates).
    vector<CVector3> centerline_worldcoordinate;
    ImageType::IndexType ind;
    itk::Point<double,3> point;
    for (int i=0; i<centerline.size(); i++)
    {
        ind[0] = centerline[i][0]; ind[1] = centerline[i][1]; ind[2] = centerline[i][2];
        inputImage_->TransformIndexToPhysicalPoint(ind, point);
        centerline_worldcoordinate.push_back(CVector3(point[0],point[1],point[2]));
    }
    
    if (verbose_) cout << "Done" << endl;
    
    return centerline_worldcoordinate;
}

ImageType::Pointer Initialisation::minimalPath3d(ImageType::Pointer image, vector<CVector3> &centerline, bool homoInt, bool invert, double factx)
{

/*% MINIMALPATH Recherche du chemin minimum de Haut vers le bas et de
% bas vers le haut tel que dÈcrit par Luc Vincent 1998
% [sR,sC,S] = MinimalPath(I,factx)
%
%   I     : Image d'entrÔøΩe dans laquelle on doit trouver le
%           chemin minimal
%   factx : Poids de linearite [1 10]
%
% Programme par : Ramnada Chav
% Date : 22 fÈvrier 2007
% ModifiÈ le 16 novembre 2007*/
    
    typedef itk::ImageDuplicator< ImageType > DuplicatorType3D;
    typedef itk::InvertIntensityImageFilter <ImageType> InvertIntensityImageFilterType;
    typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
    
    ImageType::Pointer inverted_image = image;
    
    if (invert)
    {
        StatisticsImageFilterType::Pointer statisticsImageFilterInput = StatisticsImageFilterType::New();
        statisticsImageFilterInput->SetInput(image);
        statisticsImageFilterInput->Update();
        double maxIm = statisticsImageFilterInput->GetMaximum();
        InvertIntensityImageFilterType::Pointer invertIntensityFilter = InvertIntensityImageFilterType::New();
        invertIntensityFilter->SetInput(image);
        invertIntensityFilter->SetMaximum(maxIm);
        invertIntensityFilter->Update();
        inverted_image = invertIntensityFilter->GetOutput();
    }
    
    
    ImageType::SizeType sizeImage = image->GetLargestPossibleRegion().GetSize();
    int m = sizeImage[0]; // x to change because we are in AIL
    int n = sizeImage[2]; // y
    int p = sizeImage[1]; // z
    
    // create image with high values J1
    DuplicatorType3D::Pointer duplicator = DuplicatorType3D::New();
    duplicator->SetInputImage(inverted_image);
    duplicator->Update();
    ImageType::Pointer J1 = duplicator->GetOutput();
    typedef itk::ImageRegionIterator<ImageType> ImageIterator3D;
    ImageIterator3D vItJ1( J1, J1->GetBufferedRegion() );
    vItJ1.GoToBegin();
    while ( !vItJ1.IsAtEnd() )
    {
        vItJ1.Set(100000000);
        ++vItJ1;
    }
    
    // create image with high values J2
    DuplicatorType3D::Pointer duplicatorJ2 = DuplicatorType3D::New();
    duplicatorJ2->SetInputImage(inverted_image);
    duplicatorJ2->Update();
    ImageType::Pointer J2 = duplicatorJ2->GetOutput();
    ImageIterator3D vItJ2( J2, J2->GetBufferedRegion() );
    vItJ2.GoToBegin();
    while ( !vItJ2.IsAtEnd() )
    {
        vItJ2.Set(100000000);
        ++vItJ2;
    }
    
    DuplicatorType3D::Pointer duplicatorCPixel = DuplicatorType3D::New();
    duplicatorCPixel->SetInputImage(inverted_image);
    duplicatorCPixel->Update();
    ImageType::Pointer cPixel = duplicatorCPixel->GetOutput();
    
    ImageType::IndexType index;
    
    // iterate on slice from slice 1 (start=0) to slice p-2. Basically, we avoid first and last slices.
    // IMPORTANT: first slice of J1 and last slice of J2 must be set to 0...
    for (int x=0; x<m; x++)
    {
        for (int y=0; y<n; y++)
        {
            index[0] = x; index[1] = 0; index[2] = y;
            J1->SetPixel(index, 0.0);
        }
    }
    for (int slice=1; slice<p; slice++)
    {
        // 1. extract pJ = the (slice-1)th slice of the image J1
        Matrice pJ = Matrice(m,n);
        for (int x=0; x<m; x++)
        {
            for (int y=0; y<n; y++)
            {
                index[0] = x; index[1] = slice-1; index[2] = y;
                pJ(x,y) = J1->GetPixel(index);
            }
        }
        
        // 2. extract cP = the (slice)th slice of the image cPixel
        Matrice cP = Matrice(m,n);
        for (int x=0; x<m; x++)
        {
            for (int y=0; y<n; y++)
            {
                index[0] = x; index[1] = slice; index[2] = y;
                cP(x,y) = cPixel->GetPixel(index);
            }
        }
        
        // 2'
        Matrice cPm = Matrice(m,n);
        if (homoInt)
        {
            for (int x=0; x<m; x++)
            {
                for (int y=0; y<n; y++)
                {
                    index[0] = x; index[1] = slice-1; index[2] = y;
                    cP(x,y) = cPixel->GetPixel(index);
                }
            }
        }
        
        // 3. Create a matrix VI with 5 slices, that are exactly a repetition of cP without borders
        // multiply all elements of all slices of VI except the middle one by factx
        Matrice VI[5];
        for (int i=0; i<5; i++)
        {
            // Create VI
            Matrice cP_in = Matrice(m-1, n-1);
            for (int x=0; x<m-2; x++)
            {
                for (int y=0; y<n-2; y++)
                {
                    cP_in(x,y) = cP(x+1,y+1);
                    if (i!=2)
                        cP_in(x,y) *= factx;
                }
            }
            VI[i] = cP_in;
        }
        
        // 3'.
        Matrice VIm[5];
        if (homoInt)
        {
            for (int i=0; i<5; i++)
            {
                // Create VIm
                Matrice cPm_in = Matrice(m-1, n-1);
                for (int x=0; x<m-2; x++)
                {
                    for (int y=0; y<n-2; y++)
                    {
                        cPm_in(x,y) = cPm(x+1,y+1);
                        if (i!=2)
                            cPm_in(x,y) *= factx;
                    }
                }
                VIm[i] = cPm_in;
            }
        }
        
        // 4. create a matrix of 5 slices, containing pJ(vectx-1,vecty),pJ(vectx,vecty-1),pJ(vectx,vecty),pJ(vectx,vecty+1),pJ(vectx+1,vecty) where vectx=2:m-1; and vecty=2:n-1;
        Matrice Jq[5];
        int s = 0;
        Matrice pJ_temp = Matrice(m-1, n-1);
        for (int x=0; x<m-2; x++)
        {
            for (int y=0; y<n-2; y++)
            {
                pJ_temp(x,y) = pJ(x+1,y+1);
            }
        }
        Jq[2] = pJ_temp;
        for (int k=-1; k<=1; k+=2)
        {
            for (int l=-1; l<=1; l+=2)
            {
                Matrice pJ_temp = Matrice(m-1, n-1);
                for (int x=0; x<m-2; x++)
                {
                    for (int y=0; y<n-2; y++)
                    {
                        pJ_temp(x,y) = pJ(x+k+1,y+l+1);
                    }
                }
                Jq[s] = pJ_temp;
                s++;
                if (s==2) s++; // we deal with middle slice before
            }
        }
        
        // 4'. An alternative is to minimize the difference in intensity between slices.
        if (homoInt)
        {
            Matrice VI_temp[5];
            // compute the difference between VI and VIm
            for (int i=0; i<5; i++)
                VI_temp[i] = VI[i] - VIm[i];
            
            // compute the minimum value for each element of the matrices
            for (int i=0; i<5; i++)
            {
                for (int x=0; x<m-2; x++)
                {
                    for (int y=0; y<n-2; y++)
                    {
                        if (VI_temp[i](x,y) > 0)
                            VI[i](x,y) = abs(VI_temp[i](x,y));///VIm[i](x,y);
                        else
                            VI[i](x,y) = abs(VI_temp[i](x,y));///VI[i](x,y);
                    }
                }
            }
        }
        
        // 5. sum Jq and Vi voxel by voxel to produce JV
        Matrice JV[5];
        for (int i=0; i<5; i++)
            JV[i] = VI[i] + Jq[i];
        
        // 6. replace each pixel of the (slice)th slice of J1 with the minimum value of the corresponding column in JV
        for (int x=0; x<m-2; x++)
        {
            for (int y=0; y<n-2; y++)
            {
                double min_value = 1000000;
                for (int i=0; i<5; i++)
                {
                    if (JV[i](x,y) < min_value)
                        min_value = JV[i](x,y);
                }
                index[0] = x+1; index[1] = slice; index[2] = y+1;
                J1->SetPixel(index, min_value);
            }
        }
    }
    
    // iterate on slice from slice n-1 to slice 1. Basically, we avoid first and last slices.
    // IMPORTANT: first slice of J1 and last slice of J2 must be set to 0...
    for (int x=0; x<m; x++)
    {
        for (int y=0; y<n; y++)
        {
            index[0] = x; index[1] = p-1; index[2] = y;
            J2->SetPixel(index, 0.0);
        }
    }
    for (int slice=p-2; slice>=0; slice--)
    {
        // 1. extract pJ = the (slice-1)th slice of the image J1
        Matrice pJ = Matrice(m,n);
        for (int x=0; x<m; x++)
        {
            for (int y=0; y<n; y++)
            {
                index[0] = x; index[1] = slice+1; index[2] = y;
                pJ(x,y) = J2->GetPixel(index);
            }
        }
        
        // 2. extract cP = the (slice)th slice of the image cPixel
        Matrice cP = Matrice(m,n);
        for (int x=0; x<m; x++)
        {
            for (int y=0; y<n; y++)
            {
                index[0] = x; index[1] = slice; index[2] = y;
                cP(x,y) = cPixel->GetPixel(index);
            }
        }
        
        // 2'
        Matrice cPm = Matrice(m,n);
        if (homoInt)
        {
            for (int x=0; x<m; x++)
            {
                for (int y=0; y<n; y++)
                {
                    index[0] = x; index[1] = slice+1; index[2] = y;
                    cPm(x,y) = cPixel->GetPixel(index);
                }
            }
        }
        
        // 3. Create a matrix VI with 5 slices, that are exactly a repetition of cP without borders
        // multiply all elements of all slices of VI except the middle one by factx
        Matrice VI[5];
        for (int i=0; i<5; i++)
        {
            // Create VI
            Matrice cP_in = Matrice(m-1, n-1);
            for (int x=0; x<m-2; x++)
            {
                for (int y=0; y<n-2; y++)
                {
                    cP_in(x,y) = cP(x+1,y+1);
                    if (i!=2)
                        cP_in(x,y) *= factx;
                }
            }
            VI[i] = cP_in;
        }
        
        // 3'.
        Matrice VIm[5];
        if (homoInt)
        {
            for (int i=0; i<5; i++)
            {
                // Create VI
                Matrice cPm_in = Matrice(m-1, n-1);
                for (int x=0; x<m-2; x++)
                {
                    for (int y=0; y<n-2; y++)
                    {
                        cPm_in(x,y) = cPm(x+1,y+1);
                        if (i!=2)
                            cPm_in(x,y) *= factx;
                    }
                }
                VIm[i] = cPm_in;
            }
        }
        
        // 4. create a matrix of 5 slices, containing pJ(vectx-1,vecty),pJ(vectx,vecty-1),pJ(vectx,vecty),pJ(vectx,vecty+1),pJ(vectx+1,vecty) where vectx=2:m-1; and vecty=2:n-1;
        Matrice Jq[5];
        int s = 0;
        Matrice pJ_temp = Matrice(m-1, n-1);
        for (int x=0; x<m-2; x++)
        {
            for (int y=0; y<n-2; y++)
            {
                pJ_temp(x,y) = pJ(x+1,y+1);
            }
        }
        Jq[2] = pJ_temp;
        for (int k=-1; k<=1; k+=2)
        {
            for (int l=-1; l<=1; l+=2)
            {
                Matrice pJ_temp = Matrice(m-1, n-1);
                for (int x=0; x<m-2; x++)
                {
                    for (int y=0; y<n-2; y++)
                    {
                        pJ_temp(x,y) = pJ(x+k+1,y+l+1);
                    }
                }
                Jq[s] = pJ_temp;
                s++;
                if (s==2) s++; // we deal with middle slice before
            }
        }
        
        // 4'. An alternative is to minimize the difference in intensity between slices.
        if (homoInt)
        {
            Matrice VI_temp[5];
            // compute the difference between VI and VIm
            for (int i=0; i<5; i++)
                VI_temp[i] = VI[i] - VIm[i];
            
            // compute the minimum value for each element of the matrices
            for (int i=0; i<5; i++)
            {
                for (int x=0; x<m-2; x++)
                {
                    for (int y=0; y<n-2; y++)
                    {
                        if (VI_temp[i](x,y) > 0)
                            VI[i](x,y) = abs(VI_temp[i](x,y));///VIm[i](x,y);
                        else
                            VI[i](x,y) = abs(VI_temp[i](x,y));///VI[i](x,y);
                    }
                }
            }
        }
        
        // 5. sum Jq and Vi voxel by voxel to produce JV
        Matrice JV[5];
        for (int i=0; i<5; i++)
            JV[i] = VI[i] + Jq[i];
        
        // 6. replace each pixel of the (slice)th slice of J1 with the minimum value of the corresponding column in JV
        for (int x=0; x<m-2; x++)
        {
            for (int y=0; y<n-2; y++)
            {
                double min_value = 10000000;
                for (int i=0; i<5; i++)
                {
                    if (JV[i](x,y) < min_value)
                        min_value = JV[i](x,y);
                }
                index[0] = x+1; index[1] = slice; index[2] = y+1;
                J2->SetPixel(index, min_value);
            }
        }
    }
    
    // add J1 and J2 to produce "S" which is actually J1 here.
    ImageIterator3D vItS( J1, J1->GetBufferedRegion() );
    ImageIterator3D vItJ2b( J2, J2->GetBufferedRegion() );
    vItS.GoToBegin();
    vItJ2b.GoToBegin();
    while ( !vItS.IsAtEnd() )
    {
        vItS.Set(vItS.Get()+vItJ2b.Get());
        ++vItS;
        ++vItJ2b;
    }

    // Find the minimal value of S for each slice and create a binary image with all the coordinates
    // TO DO: the minimal path shouldn't be a pixelar path. It should be a continuous spline that is minimum.
    double val_temp;
    vector<CVector3> list_index;
    for (int slice=1; slice<p-1; slice++)
    {
        double min_value_S = 10000000;
        ImageType::IndexType index_min;
        for (int x=1; x<m-1; x++)
        {
            for (int y=1; y<n-1; y++)
            {
                index[0] = x; index[1] = slice; index[2] = y;
                val_temp = J1->GetPixel(index);
                if (val_temp < min_value_S)
                {
                    min_value_S = val_temp;
                    index_min = index;
                }
            }
        }
        list_index.push_back(CVector3(index_min[0], index_min[1], index_min[2]));
    }
    
    //BSplineApproximation centerline_approximator = BSplineApproximation(&list_index);
    //list_index = centerline_approximator.EvaluateBSplinePoints(list_index.size());
    
    /*// create image with high values J1
    ImageType::Pointer result_bin = J2;
    ImageIterator3D vItresult( result_bin, result_bin->GetBufferedRegion() );
    vItresult.GoToBegin();
    while ( !vItresult.IsAtEnd() )
    {
        vItresult.Set(0.0);
        ++vItresult;
    }
    for (int i=0; i<list_index.size(); i++)
    {
        index[0] = list_index[i][0]; index[1] = list_index[i][1]; index[2] = list_index[i][2];
        result_bin->SetPixel(index,1.0);
    }
    
    typedef itk::ImageFileWriter< ImageType > WriterTypeM;
    WriterTypeM::Pointer writerMin = WriterTypeM::New();
    itk::NiftiImageIO::Pointer ioV = itk::NiftiImageIO::New();
    writerMin->SetImageIO(ioV);
    writerMin->SetInput( J1 ); // result_bin
    writerMin->SetFileName("minimalPath.nii.gz");
    try {
        writerMin->Update();
    }
    catch( itk::ExceptionObject & e )
    {
        cout << "Exception thrown ! " << endl;
        cout << "An error ocurred during Writing Min" << endl;
        cout << "Location    = " << e.GetLocation()    << endl;
        cout << "Description = " << e.GetDescription() << endl;
    }*/

    centerline = list_index;
    return J1; // return image with minimal path
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
	if (roundIn(spacing[1]) != 0 && (int)gap_ % (int)roundIn(spacing[1]) != 0)
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
    
    // Apply vesselness filter on the image
    ImageType::Pointer vesselnessImage = vesselnessFilter(vector<int>(), inputImage_);
    
    // Start of the detection of circles and ellipses. For each axial slices, a Hough transform is performed to detect circles. Each axial image is stretched in the antero-posterior direction in order to detect the spinal cord as a ellipse as well as a circle.
    for (int i=roundIn(-((numberOfSlices_-1.0)/2.0)*(gap_/spacing[1])); i<=roundIn(((numberOfSlices_-1.0)/2.0)*(gap_/spacing[1])); i+=roundIn(gap_/spacing[1]))
	{
        // Cropping of the image
		if (verbose_) cout << "Slice num " << i << endl;
		desiredStart[1] = startZ+i;
		ImageType::RegionType desiredRegion(desiredStart, desiredSize);
		FilterType::Pointer filter = FilterType::New();
		filter->SetExtractionRegion(desiredRegion);
		//filter->SetInput(vesselnessImage);
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
        ImageType2D::DirectionType directionIm = im->GetDirection();
		ImageType::DirectionType imageDirection = inputImage_->GetDirection();
		directionIm[0][0] = imageDirection[0][0];
		directionIm[0][1] = imageDirection[0][2];
		directionIm[1][0] = imageDirection[1][0];
		directionIm[1][1] = imageDirection[1][2];
		im->SetDirection(directionIm);
        
		DuplicatorType::Pointer duplicator = DuplicatorType::New();
		duplicator->SetInputImage(im);
		duplicator->Update();
		ImageType2D::Pointer clonedImage = duplicator->GetOutput();
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
            
            // Search for symmetry in image
            //cout << "Symmetry = " << symmetryDetection3D(inputImage_, 40, 40) << endl;
            
            // Searching the circles in the image using circular Hough transform, adapted from ITK
            // The list of radii and accumulator values are then extracted for analyses
			vector<CVector3> vecCenter;
			vector<double> vecRadii, vecAccumulator;
			searchCenters(im,vecCenter,vecRadii,vecAccumulator,startZ+i, vesselnessImage);
			
            // Reformating of the detected circles in the image. Each detected circle is push in a Node with all its information.
            // The radii are transformed in mm using mean axial resolution
			vector<Node*> vecNodeTemp;
			for (unsigned int k=0; k<vecCenter.size(); k++) {
				if (vecRadii[k] != 0.0) {
					CVector3 center = vecCenter[k];// center[0] /= stretchingFactor;
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
						//double currentDistance = sqrt(pow(centers[k][i][m]->getPosition()[0]-centers[k-1][i][j]->getPosition()[0],2)+pow(centers[k][i][m]->getPosition()[1]-centers[k-1][i][j]->getPosition()[1],2)+pow(centers[k][i][m]->getPosition()[2]-centers[k-1][i][j]->getPosition()[2],2));
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
						//double currentDistance = sqrt(pow(centers[k][i][m]->getPosition()[0]-centers[k+1][i][j]->getPosition()[0],2)+pow(centers[k][i][m]->getPosition()[1]-centers[k+1][i][j]->getPosition()[1],2)+pow(centers[k][i][m]->getPosition()[2]-centers[k+1][i][j]->getPosition()[2],2));
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
    
    
    // Search for the longest chain that is not far from the the center of the image (detected as the symmetry in the image) and with the largest accumulation value
    // metric used to compute the best chain : accumulation_value * (exp(1/(average_distance_to_center_of_image+(1/log(2))))-1)
    // metric accumulation_value * tanh(2.5-0.5*average_distance_to_center_of_image)/2+0.5
	unsigned int maxLenght = 0, max = 0;
	double maxMetric = 0.0, angleMax = 30.0;
    
    map<double, int, greater<double> > map_metric;
    
	for (unsigned int j=0; j<chains.size(); j++)
	{
		unsigned int length = chains[j].size();
        double angle = 0.0;
        int middle_slice = inputImage_->GetLargestPossibleRegion().GetSize()[2]/2;
        double average_distance_to_center_of_image = 0.0;
        for (int k=0; k<length; k++)
            average_distance_to_center_of_image += abs(chains[j][k]->getPosition()[2]-middle_slice);
        average_distance_to_center_of_image /= (double)length;
        //cout << j << " " << average_distance_to_center_of_image << endl;
        
        //average_distance_to_center_of_image = abs(chains[j][length/2]->getPosition()[2]-middle_slice);
        //double weighted_distance = (exp(1/(average_distance_to_center_of_image+(1/log(2))))-1);
        double weighted_distance = tanh(2.5-0.5*average_distance_to_center_of_image)/2+0.5;
        //double weighted_distance = 1;
        
        if (length >= 3)
        {
            CVector3 vector1 = chains[j][0]->getPosition()-chains[j][length/2]->getPosition(), vector2 = (chains[j][length/2]->getPosition()-chains[j][length-1]->getPosition());
            angle = 360.0*acos((vector1*vector2)/(vector1.Norm()*vector2.Norm()))/(2.0*M_PI);
        }
		if (length > maxLenght && angle <= angleMax)
		{
			maxLenght = chains[j].size();
			max = j;
			maxMetric = 0.0;
			for (unsigned int k=0; k<length; k++)
				maxMetric += chains[j][k]->getAccumulator() * weighted_distance * length;
            map_metric[maxMetric] = max;
		}
		else if (length == maxLenght && angle <= angleMax)
		{
			double metric = 0.0;
			for (unsigned int k=0; k<length; k++)
				metric += chains[j][k]->getAccumulator() * weighted_distance * length;
			if (metric > maxMetric) {
				maxLenght = chains[j].size();
				max = j;
				maxMetric = metric;
                map_metric[maxMetric] = max;
			}
		}
	}
    
	if (chains.size() > 1)
	{
		unsigned int sizeMaxChain = chains[map_metric.begin()->second].size();
		//cout << "Results : " << endl;
        points_.clear();
		for (unsigned int j=0; j<sizeMaxChain; j++) {
            points_.push_back(chains[map_metric.begin()->second][j]->getPosition());
			//cout << chains[max][j]->getPosition() << " " << chains[max][j]->getRadius() << endl;
        }
        if (verbose_) cout << "Stretching factor of circle found = " << chains[map_metric.begin()->second][0]->getStretchingFactor() << endl;
		if (sizeMaxChain < numberOfSlices_) {
			if (verbose_) cout << "Warning: Number of center found on slices (" << sizeMaxChain << ") doesn't correspond to number of analyzed slices. An error may occur. To improve results, you can increase the number of analyzed slices (option -n must be impair)" << endl;
            
			// we have to transform pixel points to physical points
			CVector3 finalPoint, initPointT = chains[map_metric.begin()->second][0]->getPosition(), finalPointT = chains[map_metric.begin()->second][sizeMaxChain-1]->getPosition();
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
				initialRadius_ += chains[map_metric.begin()->second][j]->getRadiusStretch();
			initialRadius_ /= sizeMaxChain;
            stretchingFactor_ = chains[map_metric.begin()->second][0]->getStretchingFactor();
		}
		else
		{
			// we have to transform pixel points to physical points
			CVector3 finalPoint1, finalPoint2, initPointT = chains[map_metric.begin()->second][(int)(sizeMaxChain/2)]->getPosition(), finalPointT1 = chains[map_metric.begin()->second][0]->getPosition(), finalPointT2 = chains[map_metric.begin()->second][sizeMaxChain-1]->getPosition();
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
				initialRadius_ += chains[map_metric.begin()->second][j]->getRadiusStretch();
			initialRadius_ /= sizeMaxChain;
            stretchingFactor_ = chains[map_metric.begin()->second][0]->getStretchingFactor();
		}
		return true;
	}
	else {
		cout << "Error: No point detected..." << endl;
		return false;
	}
}

void Initialisation::searchCenters(ImageType2D::Pointer im, vector<CVector3> &vecCenter, vector<double> &vecRadii, vector<double> &vecAccumulator, float startZ, ImageType::Pointer imageVesselness)
{
	DuplicatorType::Pointer duplicator = DuplicatorType::New();
	duplicator->SetInputImage(im);
	duplicator->Update();
	ImageType2D::Pointer clonedOutput = duplicator->GetOutput();
    
	MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
    
	unsigned int numberOfCircles = 20;
	double **center_result_small = new double*[numberOfCircles], **center_result_large = new double*[numberOfCircles];
	for (unsigned int k=0; k<numberOfCircles; k++) {
		center_result_small[k] = new double[3]; // Detected centers are in mm
		center_result_large[k] = new double[3];
	}
	double *radius_result_small = new double[numberOfCircles], *radius_result_large = new double[numberOfCircles]; // radius is in pixel
	double *accumulator_result_small = new double[numberOfCircles], *accumulator_result_large = new double[numberOfCircles];
	unsigned int numSmall = houghTransformCircles(im,numberOfCircles,center_result_small,radius_result_small,accumulator_result_small,radius_/mean_resolution_,imageVesselness,startZ,-1.0);
	unsigned int numLarge = houghTransformCircles(clonedOutput,numberOfCircles,center_result_large,radius_result_large,accumulator_result_large,radius_/mean_resolution_+4.0,imageVesselness,startZ,1.0);
    
	// search along results for nested circles
	vector<unsigned int> listMostPromisingCenters;
	vector<unsigned int> listMostPromisingCentersLarge;
	double distance = 0.0;
	for (unsigned int i=0; i<numSmall; i++)
	{
		for (unsigned int j=0; j<numLarge; j++)
		{
			// distance between center + small_radius must be smaller than large_radius
			distance = sqrt(pow(center_result_small[i][0]-center_result_large[j][0],2)+pow(center_result_small[i][2]-center_result_large[j][2],2));
			if ((distance+radius_result_small[i])*0.7 <= radius_result_large[j]) {
				listMostPromisingCenters.push_back(i);
				listMostPromisingCentersLarge.push_back(j);
			}
		}
	}
    
    // If circular structure surroundIned by other circular shapes were detected, we add them in the list of promising circles
    if (listMostPromisingCenters.size() > 0)
    {
        for (unsigned int i=0; i<listMostPromisingCenters.size(); i++)
        {
            vecCenter.push_back(CVector3(center_result_small[listMostPromisingCenters[i]][0],center_result_small[listMostPromisingCenters[i]][1],center_result_small[listMostPromisingCenters[i]][2]));
            vecRadii.push_back(radius_result_small[listMostPromisingCenters[i]]);
            vecAccumulator.push_back(accumulator_result_small[listMostPromisingCenters[i]]);
        }
    }
    // If no double circular shapes were detected, spinal cord may not be surroundIned by another circular shape. Therefore, we add the 5 (chosen arbitrarely) most promising points (in terms of accumulator values) to the vector.
    else
    {
        // center_result_small are already sorted by accumulator values
        for (unsigned int i=0; i<5; i++)
        {
            vecCenter.push_back(CVector3(center_result_small[i][0],center_result_small[i][2],center_result_small[i][2]));
            vecRadii.push_back(radius_result_small[i]);
            vecAccumulator.push_back(accumulator_result_small[i]);
        }
    }
}


unsigned int Initialisation::houghTransformCircles(ImageType2D* im, unsigned int numberOfCircles, double** center_result, double* radius_result, double* accumulator_result, double meanRadius, ImageType::Pointer VesselnessImage, float slice, double valPrint)
{
	MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
	minMaxCalculator->SetImage(im);
	minMaxCalculator->ComputeMaximum();
	minMaxCalculator->ComputeMinimum();
	ImageType2D::PixelType maxIm = minMaxCalculator->GetMaximum(), minIm = minMaxCalculator->GetMinimum();
	double val_Print = maxIm;
    
    double min_radius = meanRadius-3.0/mean_resolution_;
    if (min_radius < 0) min_radius = 0;
    
    /*// Application of a median filter to reduce noise and improve circle detection
    typedef itk::MedianImageFilter<ImageType2D, ImageType2D > MedianFilterType;
    MedianFilterType::Pointer medianFilter = MedianFilterType::New();
    MedianFilterType::InputSizeType radius;
    radius.Fill(1);
    medianFilter->SetRadius(radius);
    medianFilter->SetInput( im );
    medianFilter->Update();
    im = medianFilter->GetOutput();*/
	
	HoughCirclesFilter::Pointer houghfilter = HoughCirclesFilter::New();
	houghfilter->SetInput(im);
	houghfilter->SetMinimumRadius(min_radius);
	houghfilter->SetMaximumRadius(meanRadius+3.0/mean_resolution_);
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
    
    typedef itk::Point< double, 2 > PointType2D;
    PointType2D point2d;
    PointType point3d;
    ImageType::IndexType index3d;
    for(it_input.GoToBegin();!it_input.IsAtEnd();++it_input)
    {
        im->TransformIndexToPhysicalPoint(it_input.GetIndex(),point2d);
        point3d[0] = point2d[0]; point3d[1] = point2d[1]; point3d[2] = 0;
        VesselnessImage->TransformPhysicalPointToIndex(point3d, index3d);
        index3d[1] = slice;
        it_input.Set(it_input.Get()*VesselnessImage->GetPixel(index3d));
    }
    VesselnessImage->TransformIndexToPhysicalPoint(index3d, point3d);
    
    typedef itk::TileImageFilter< ImageType2D, ImageType > TilerType;
    TilerType::Pointer resample = TilerType::New();
    
    TilerType::Pointer tiler = TilerType::New();
    
    itk::FixedArray< unsigned int, 3 > layout;
    layout[0] = 1; layout[1] = 1; layout[2] = 0;
    tiler->SetLayout( layout );
    tiler->SetInput( 0, m_PostProcessImage );
    ImageType::PixelType filler = 0;
    tiler->SetDefaultPixelValue( filler );
    tiler->Update();
    ImageType::Pointer m_PostProcessImage3D = tiler->GetOutput();
    
    typedef itk::PermuteAxesImageFilter <ImageType> PermuteAxesImageFilterType;
    itk::FixedArray<unsigned int, 3> order;
    order[0] = 0; order[1] = 2; order[2] = 1;
    PermuteAxesImageFilterType::Pointer permuteAxesFilter = PermuteAxesImageFilterType::New();
    permuteAxesFilter->SetInput(m_PostProcessImage3D);
    permuteAxesFilter->SetOrder(order);
    permuteAxesFilter->Update();
    m_PostProcessImage3D = permuteAxesFilter->GetOutput();
    
    
    PointType origin_im = inputImage_->GetOrigin();
    origin_im[0] = m_PostProcessImage3D->GetOrigin()[0];
    origin_im[1] = m_PostProcessImage3D->GetOrigin()[1];
    origin_im[2] = point3d[2];
    m_PostProcessImage3D->SetOrigin(origin_im);
    m_PostProcessImage3D->SetDirection(inputImage_->GetDirection());
    m_PostProcessImage3D->SetSpacing(inputImage_->GetSpacing());
    
    typedef itk::ImageFileWriter< ImageType > WriterTypeP;
    WriterTypeP::Pointer writerVesselNess = WriterTypeP::New();
    itk::NiftiImageIO::Pointer ioV = itk::NiftiImageIO::New();
    writerVesselNess->SetImageIO(ioV);
    writerVesselNess->SetInput( m_PostProcessImage3D );
    writerVesselNess->SetFileName("im_proc.nii.gz");
    try {
        writerVesselNess->Update();
    }
    catch( itk::ExceptionObject & e )
    {
        cout << "Exception thrown ! " << endl;
        cout << "An error ocurred during Writing 1" << endl;
        cout << "Location    = " << e.GetLocation()    << endl;
        cout << "Description = " << e.GetDescription() << endl;
    }
    
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
                    typedef itk::Point< double, 2 > PointType2D;
                    PointType2D point2d;
                    im->TransformIndexToPhysicalPoint(it_output.GetIndex(),point2d);
                    PointType point3d;
                    point3d[0] = point2d[0]; point3d[1] = point2d[1]; point3d[2] = 0;
                    ImageType::IndexType index3d;
                    VesselnessImage->TransformPhysicalPointToIndex(point3d, index3d);
                    index3d[1] = slice;
                    center_result[circles][0] = index3d[0];
                    center_result[circles][1] = index3d[1];
                    center_result[circles][2] = index3d[2];
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

ImageType::Pointer Initialisation::vesselnessFilter(vector<int> middle_slices, ImageType::Pointer im, double alpha, double beta, double gamma, double sigmaMinimum, double sigmaMaximum, unsigned int numberOfSigmaSteps, double sigmaDistance)
{
    typedef itk::ImageDuplicator< ImageType > DuplicatorTypeIm;
    DuplicatorTypeIm::Pointer duplicator = DuplicatorTypeIm::New();
    duplicator->SetInputImage(im);
    duplicator->Update();
    ImageType::Pointer clonedImage = duplicator->GetOutput();
    
    typedef itk::SymmetricSecondRankTensor< double, 3 > HessianPixelType;
    typedef itk::Image< HessianPixelType, 3 >           HessianImageType;
    typedef itk::HessianToObjectnessMeasureImageFilter< HessianImageType, ImageType > ObjectnessFilterType;
    ObjectnessFilterType::Pointer objectnessFilter = ObjectnessFilterType::New();
    objectnessFilter->SetBrightObject( 1-typeImageFactor_ );
    objectnessFilter->SetScaleObjectnessMeasure( false );
    objectnessFilter->SetAlpha( alpha );
    objectnessFilter->SetBeta( beta );
    objectnessFilter->SetGamma( gamma );
    objectnessFilter->SetObjectDimension(1);
    
    typedef itk::MultiScaleHessianBasedMeasureImageFilter< ImageType, HessianImageType, ImageType > MultiScaleEnhancementFilterType;
    MultiScaleEnhancementFilterType::Pointer multiScaleEnhancementFilter =
    MultiScaleEnhancementFilterType::New();
    multiScaleEnhancementFilter->SetInput( clonedImage );
    multiScaleEnhancementFilter->SetHessianToMeasureFilter( objectnessFilter );
    multiScaleEnhancementFilter->SetSigmaStepMethodToLogarithmic();
    multiScaleEnhancementFilter->SetSigmaMinimum( sigmaMinimum );
    multiScaleEnhancementFilter->SetSigmaMaximum( sigmaMaximum );
    multiScaleEnhancementFilter->SetNumberOfSigmaSteps( numberOfSigmaSteps );
    multiScaleEnhancementFilter->Update();
    
    ImageType::Pointer vesselnessImage = multiScaleEnhancementFilter->GetOutput();
    
    /*// Multiplying the vesselness filter results by a sigmoid mask to concentrate spinal cord detection aroung the median sagittal plane.
    // sigmoid: tanh(5.0-0.5*left_right_distance)/2+0.5
    // gaussian: e^(-left_right_distance^2/(2*10^2))
    // left-right is the first dimension in world coordinates
    vector<double> pointCenters;
    ImageType::IndexType index;
    PointType point, pointCenter;
    if (middle_slices.empty()) {
        for (int i=0; i<vesselnessImage->GetLargestPossibleRegion().GetSize()[2]; i++)
        {
            int middle = vesselnessImage->GetLargestPossibleRegion().GetSize()[2]/2;
            index[0] = 0; index[1] = 0; index[2] = middle;
            inputImage_->TransformIndexToPhysicalPoint(index, pointCenter);
            pointCenters.push_back(pointCenter[0]);
        }
    }
    else {
        for (int i=0; i<middle_slices.size(); i++)
        {
            index[0] = 0; index[1] = 0; index[2] = middle_slices[i];
            inputImage_->TransformIndexToPhysicalPoint(index, pointCenter);
            pointCenters.push_back(pointCenter[0]);
        }
    }
    
    double factor = 1.0;
    typedef itk::ImageRegionIterator< ImageType > ImageIterator;
    ImageIterator vIt( vesselnessImage, vesselnessImage->GetLargestPossibleRegion() );
    vIt.GoToBegin();
    while ( !vIt.IsAtEnd() )
    {
        index = vIt.GetIndex();
        vesselnessImage->TransformIndexToPhysicalPoint(index, point);
        //factor = tanh(5.0-0.5*abs(point[0]-pointCenter[0]))/2.0+0.5;
        factor = exp(-pow(point[0]-pointCenters[index[1]],2)/(2*sigmaDistance*sigmaDistance));
        vIt.Set(vIt.Get()*factor);
        ++vIt;
    }*/
    
    WriterType::Pointer writerVesselNess = WriterType::New();
    itk::NiftiImageIO::Pointer ioV = itk::NiftiImageIO::New();
    writerVesselNess->SetImageIO(ioV);
    writerVesselNess->SetInput( vesselnessImage );
    writerVesselNess->SetFileName("imageVesselNessFilter.nii.gz");
    try {
        writerVesselNess->Update();
    }
    catch( itk::ExceptionObject & e )
    {
        cout << "Exception thrown ! " << endl;
        cout << "An error ocurred during Writing 1" << endl;
        cout << "Location    = " << e.GetLocation()    << endl;
        cout << "Description = " << e.GetDescription() << endl;
    }
    
    return vesselnessImage;
}


ImageType::Pointer Initialisation::vesselnessFilter2(ImageType::Pointer im)
{
    
    typedef itk::SymmetricSecondRankTensor< double, 3 > MatrixType;
    typedef itk::Image< MatrixType, 3> HessianImageType;
    typedef itk::ImageRegionIterator< HessianImageType > HessianImageIterator;
    
    typedef itk::InvertIntensityImageFilter <ImageType> InvertIntensityImageFilterType;
    typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
    
    typedef itk::ImageDuplicator< ImageType > DuplicatorTypeIm;
    DuplicatorTypeIm::Pointer duplicator = DuplicatorTypeIm::New();
    duplicator->SetInputImage(im);
    duplicator->Update();
    ImageType::Pointer clonedImage = duplicator->GetOutput();
    
    if (typeImageFactor_ == 1.0)
    {
        StatisticsImageFilterType::Pointer statisticsImageFilterInput = StatisticsImageFilterType::New();
        statisticsImageFilterInput->SetInput(clonedImage);
        statisticsImageFilterInput->Update();
        double maxIm = statisticsImageFilterInput->GetMaximum();
        InvertIntensityImageFilterType::Pointer invertIntensityFilter = InvertIntensityImageFilterType::New();
        invertIntensityFilter->SetInput(clonedImage);
        invertIntensityFilter->SetMaximum(maxIm);
        invertIntensityFilter->Update();
        clonedImage = invertIntensityFilter->GetOutput();
    }
    
    typedef itk::HessianRecursiveGaussianImageFilter< ImageType >     HessianFilterType;
    typedef itk::Hessian3DToVesselnessMeasureImageFilter< double > VesselnessMeasureFilterType;
    HessianFilterType::Pointer hessianFilter = HessianFilterType::New();
    VesselnessMeasureFilterType::Pointer vesselnessFilter = VesselnessMeasureFilterType::New();
    hessianFilter->SetInput( clonedImage );
    hessianFilter->SetSigma( 3.0 );
    hessianFilter->Update();
    
    vesselnessFilter->SetInput( hessianFilter->GetOutput() );
    vesselnessFilter->SetAlpha1( 0.5 );
    vesselnessFilter->SetAlpha2( 2.0 );
    vesselnessFilter->Update();
    
    ImageType::Pointer vesselnessImage = vesselnessFilter->GetOutput();
    
    // Normalization of the vesselness image
    StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New();
    statisticsImageFilter->SetInput(vesselnessImage);
    statisticsImageFilter->Update();
    double meanIm = statisticsImageFilter->GetMean();
    double sigmaIm = statisticsImageFilter->GetSigma();
    double minIm = statisticsImageFilter->GetMinimum();
    double maxIm = statisticsImageFilter->GetMaximum();
    
    typedef itk::ImageRegionIterator< ImageType > ImageIterator;
    ImageIterator vIt( vesselnessImage, vesselnessImage->GetBufferedRegion() );
    vIt.GoToBegin();
    double newMin = 0, newMax = 1;
    minIm = meanIm-sigmaIm;
    maxIm = meanIm+sigmaIm;
    // normalization that remove extrema
    while ( !vIt.IsAtEnd() )
    {
        vIt.Set((vIt.Get()-minIm)*(newMax-newMin)/(maxIm-minIm)+newMin);
        ++vIt;
    }
    //double newMin = 0, newMax = 1; // normalization
    /*while ( !vIt.IsAtEnd() )
     {
     vIt.Set((vIt.Get()-minIm)*(newMax-newMin)/(maxIm-minIm)+newMin);
     ++vIt;
     }*/
    
    WriterType::Pointer writerVesselNess = WriterType::New();
    itk::NiftiImageIO::Pointer ioV = itk::NiftiImageIO::New();
    writerVesselNess->SetImageIO(ioV);
    writerVesselNess->SetInput( vesselnessImage );
    writerVesselNess->SetFileName("imageVesselNessFilter.nii.gz");
    try {
        writerVesselNess->Update();
    }
    catch( itk::ExceptionObject & e )
    {
        cout << "Exception thrown ! " << endl;
        cout << "An error ocurred during Writing 1" << endl;
        cout << "Location    = " << e.GetLocation()    << endl;
        cout << "Description = " << e.GetDescription() << endl;
    }
    
    
    return vesselnessImage;
}


/**************************************************/
// Non used methods
/**************************************************/

int Initialisation::symmetryDetection3D(ImageType::Pointer im, double cropWidth_, double bandWidth_)
{
    
    
    ImageType::SpacingType spacingIm = im->GetSpacing();
    int cropSize = cropWidth_/spacingIm[0];
    ImageType::SizeType desiredSize = im->GetLargestPossibleRegion().GetSize();
    
    ImageType::SizeType desiredSizeInitial = im->GetLargestPossibleRegion().GetSize();
    
    map<double,int> mutualInformation;
    //int startSlice = desiredSizeInitial[0]/4, endSlice = desiredSizeInitial[0]/4*3;
    //int startSlice = 160, endSlice = 161;
    int startSlice = 127, endSlice = 128;
    if (desiredSizeInitial[0] < cropSize*2) {
        startSlice = cropSize/2;
        endSlice = desiredSizeInitial[0]-cropSize/2;
    }
    
	typedef itk::MinimumMaximumImageCalculator<ImageType> MinMaxCalculatorType;
	MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
	minMaxCalculator->SetImage(im);
	minMaxCalculator->ComputeMaximum();
	minMaxCalculator->ComputeMinimum();
	ImageType::PixelType maxIm = minMaxCalculator->GetMaximum(), minIm = minMaxCalculator->GetMinimum();
	if (maxIm == minIm) {
		cerr << "ERROR: The image where the symmetry will be detected is full of constant value (" << maxIm << "). You can change it using -init parameter." << endl;
		return -1;
	}
    
    
    for (int i=startSlice; i<endSlice; i++)
    {
        float startCrop = i, size;
        if (startCrop < desiredSizeInitial[0]/2 && startCrop <= bandWidth_+1) size = startCrop-1;
        else if (startCrop >= desiredSizeInitial[0]/2 && startCrop >= desiredSizeInitial[0]-bandWidth_-1) size = desiredSizeInitial[0]-startCrop-1;
        else size = bandWidth_;
        ImageType::IndexType desiredStart;
        ImageType::SizeType desiredSize = desiredSizeInitial;
        desiredStart[0] = startCrop;
        desiredStart[1] = 0;
        desiredStart[2] = desiredSizeInitial[2]/2-(cropSize+5)/2;
        desiredSize[0] = size;
        desiredSize[2] = cropSize+5;
        
        typedef itk::ExtractImageFilter< ImageType, ImageType > CropFilterType;
        
        // middle image
        desiredStart[0] = startCrop-size;
        if (desiredStart[0] < 0) desiredStart[0] = 0;
        desiredSize[0] = size*2;
        ImageType::RegionType desiredRegionImage(desiredStart, desiredSize);
        CropFilterType::Pointer cropFilter = CropFilterType::New();
        cropFilter->SetExtractionRegion(desiredRegionImage);
        cropFilter->SetInput(im);
#if ITK_VERSION_MAJOR >= 4
        cropFilter->SetDirectionCollapseToIdentity(); // This is required.
#endif
        try {
            cropFilter->Update();
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while updating cropFilter " << std::endl;
            std::cerr << e << std::endl;
        }
        ImageType::Pointer imageMiddle = cropFilter->GetOutput();
        
        WriterType::Pointer writer = WriterType::New();
        itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
        writer->SetImageIO(io);
        writer->SetFileName("imageMiddle.nii.gz");
        writer->SetInput(imageMiddle);
        try {
            writer->Update();
        }
        catch( itk::ExceptionObject & e )
        {
            cout << "Exception thrown ! " << endl;
            cout << "An error ocurred during Writing" << endl;
            cout << "Location    = " << e.GetLocation()    << endl;
            cout << "Description = " << e.GetDescription() << endl;
        }
        
        /*// Computation of Hessian matrix based on the GGVF of the image
         typedef itk::CovariantVector< double, 3 > GradientPixelType;
         typedef itk::Image< GradientPixelType, 3 > GradientImageType;
         typedef itk::GradientImageFilter< ImageType, float, double, GradientImageType > VectorGradientFilterType;
         typedef itk::GradientRecursiveGaussianImageFilter< ImageType, GradientImageType > RVectorGradientFilterType;
         
         RVectorGradientFilterType::Pointer gradientMapFilter = RVectorGradientFilterType::New();
         gradientMapFilter->SetInput( imageMiddle );
         gradientMapFilter->SetSigma(2.0);
         try {
         gradientMapFilter->Update();
         } catch( itk::ExceptionObject & e ) {
         cerr << "Exception caught while updating gradientMapFilter " << endl;
         cerr << e << endl;
         return EXIT_FAILURE;
         }
         GradientImageType::Pointer imageVectorGradient_temp = gradientMapFilter->GetOutput();
         
         typedef itk::ImageRegionIterator< GradientImageType > GradientImageIterator;
         typedef itk::ImageRegionIterator< ImageType > ImageIterator;
         GradientImageIterator GradientIt( imageVectorGradient_temp, imageVectorGradient_temp->GetBufferedRegion() );
         ImageType::Pointer magnitudeGradientInit = ImageType::New();
         magnitudeGradientInit->SetLargestPossibleRegion( imageMiddle->GetLargestPossibleRegion() );
         magnitudeGradientInit->SetRequestedRegionToLargestPossibleRegion();
         magnitudeGradientInit->SetBufferedRegion( magnitudeGradientInit->GetRequestedRegion() );
         magnitudeGradientInit->SetOrigin(imageMiddle->GetOrigin());
         magnitudeGradientInit->SetDirection(imageMiddle->GetDirection());
         magnitudeGradientInit->SetSpacing(imageMiddle->GetSpacing());
         magnitudeGradientInit->Allocate();
         ImageIterator gradientMagnInitIt( magnitudeGradientInit, magnitudeGradientInit->GetBufferedRegion() );
         gradientMagnInitIt.GoToBegin();
         GradientIt.GoToBegin();
         while ( !GradientIt.IsAtEnd() )
         {
         //cout << GradientIt.GetIndex() << " " << GradientIt.Get() << endl;
         //cout << sqrt(GradientIt.Get()[0]*GradientIt.Get()[0]+GradientIt.Get()[1]*GradientIt.Get()[1]+GradientIt.Get()[2]*GradientIt.Get()[2]) << endl;
         gradientMagnInitIt.Set(sqrt(GradientIt.Get()[0]*GradientIt.Get()[0]+GradientIt.Get()[1]*GradientIt.Get()[1]+GradientIt.Get()[2]*GradientIt.Get()[2]));
         ++GradientIt;
         ++gradientMagnInitIt;
         }
         WriterType::Pointer writerImM = WriterType::New();
         itk::NiftiImageIO::Pointer ioM = itk::NiftiImageIO::New();
         writerImM->SetImageIO(ioM);
         writerImM->SetFileName("/home/django/benjamindeleener/data/PropSeg_data/t1/errsm_11/imageMagnInit.nii.gz");
         writerImM->SetInput(magnitudeGradientInit);
         try {
         writerImM->Update();
         }
         catch( itk::ExceptionObject & e )
         {
         cout << "Exception thrown ! " << endl;
         cout << "An error ocurred during Writing M" << endl;
         cout << "Location    = " << e.GetLocation()    << endl;
         cout << "Description = " << e.GetDescription() << endl;
         }
         
         typedef itk::GradientVectorFlowImageFilter< GradientImageType, GradientImageType >  GradientVectorFlowFilterType;
         GradientVectorFlowFilterType::Pointer gradientVectorFlowFilter = GradientVectorFlowFilterType::New();
         gradientVectorFlowFilter->SetInput(imageVectorGradient_temp);
         gradientVectorFlowFilter->SetIterationNum( 500 );
         gradientVectorFlowFilter->SetNoiseLevel( 200 );
         gradientVectorFlowFilter->SetNormalize(false);
         try {
         gradientVectorFlowFilter->Update();
         } catch( itk::ExceptionObject & e ) {
         cerr << "Exception caught while updating gradientMapFilter " << endl;
         cerr << e << endl;
         return EXIT_FAILURE;
         }
         GradientImageType::Pointer imageVectorGradient = gradientVectorFlowFilter->GetOutput();
         GradientImageIterator GradientGVFIt( imageVectorGradient, imageVectorGradient->GetBufferedRegion() );
         ImageType::Pointer magnitudeGVFInit = ImageType::New();
         magnitudeGVFInit->SetLargestPossibleRegion( imageMiddle->GetLargestPossibleRegion() );
         magnitudeGVFInit->SetRequestedRegionToLargestPossibleRegion();
         magnitudeGVFInit->SetBufferedRegion( magnitudeGVFInit->GetRequestedRegion() );
         magnitudeGVFInit->SetOrigin(imageMiddle->GetOrigin());
         magnitudeGVFInit->SetDirection(imageMiddle->GetDirection());
         magnitudeGVFInit->SetSpacing(imageMiddle->GetSpacing());
         magnitudeGVFInit->Allocate();
         ImageIterator gradientMagnGVFIt( magnitudeGVFInit, magnitudeGVFInit->GetBufferedRegion() );
         gradientMagnInitIt.GoToBegin();
         GradientGVFIt.GoToBegin();
         while ( !GradientGVFIt.IsAtEnd() )
         {
         //cout << sqrt(outputIt.Get()[0]*outputIt.Get()[0]+outputIt.Get()[1]*outputIt.Get()[1]+outputIt.Get()[2]*outputIt.Get()[2]) << endl;
         gradientMagnGVFIt.Set(sqrt(GradientGVFIt.Get()[0]*GradientGVFIt.Get()[0]+GradientGVFIt.Get()[1]*GradientGVFIt.Get()[1]+GradientGVFIt.Get()[2]*GradientGVFIt.Get()[2]));
         //gradientMagnGVFIt.Set(GradientGVFIt.Get()[2]);
         ++GradientGVFIt;
         ++gradientMagnGVFIt;
         }
         WriterType::Pointer writerImMGVF = WriterType::New();
         writerImMGVF->SetImageIO(ioM);
         writerImMGVF->SetFileName("/home/django/benjamindeleener/data/PropSeg_data/t1/errsm_11/imageMagnGVFInit.nii.gz");
         writerImMGVF->SetInput(magnitudeGVFInit);
         try {
         writerImMGVF->Update();
         }
         catch( itk::ExceptionObject & e )
         {
         cout << "Exception thrown ! " << endl;
         cout << "An error ocurred during Writing M" << endl;
         cout << "Location    = " << e.GetLocation()    << endl;
         cout << "Description = " << e.GetDescription() << endl;
         }
         
         // Construction of three image from gradient image
         
         ImageType::Pointer gradientX = ImageType::New();
         gradientX->SetLargestPossibleRegion( imageMiddle->GetLargestPossibleRegion() );
         gradientX->SetRequestedRegionToLargestPossibleRegion();
         gradientX->SetBufferedRegion( gradientX->GetRequestedRegion() );
         gradientX->SetOrigin(imageMiddle->GetOrigin());
         gradientX->SetDirection(imageMiddle->GetDirection());
         gradientX->SetSpacing(imageMiddle->GetSpacing());
         gradientX->Allocate();
         ImageType::Pointer gradientY = ImageType::New();
         gradientY->SetLargestPossibleRegion( imageMiddle->GetLargestPossibleRegion() );
         gradientY->SetRequestedRegionToLargestPossibleRegion();
         gradientY->SetBufferedRegion( gradientY->GetRequestedRegion() );
         gradientY->SetOrigin(imageMiddle->GetOrigin());
         gradientY->SetDirection(imageMiddle->GetDirection());
         gradientY->SetSpacing(imageMiddle->GetSpacing());
         gradientY->Allocate();
         ImageType::Pointer gradientZ = ImageType::New();
         gradientZ->SetLargestPossibleRegion( imageMiddle->GetLargestPossibleRegion() );
         gradientZ->SetRequestedRegionToLargestPossibleRegion();
         gradientZ->SetBufferedRegion( gradientZ->GetRequestedRegion() );
         gradientZ->SetOrigin(imageMiddle->GetOrigin());
         gradientZ->SetDirection(imageMiddle->GetDirection());
         gradientZ->SetSpacing(imageMiddle->GetSpacing());
         gradientZ->Allocate();
         GradientImageIterator outputIt( imageVectorGradient, imageVectorGradient->GetBufferedRegion() );
         ImageIterator gradientXIt( gradientX, gradientX->GetBufferedRegion() );
         ImageIterator gradientYIt( gradientY, gradientY->GetBufferedRegion() );
         ImageIterator gradientZIt( gradientZ, gradientZ->GetBufferedRegion() );
         outputIt.GoToBegin();
         gradientXIt.GoToBegin();
         gradientYIt.GoToBegin();
         gradientZIt.GoToBegin();
         
         ImageType::Pointer magnitudeGradientIm = ImageType::New();
         magnitudeGradientIm->SetLargestPossibleRegion( imageMiddle->GetLargestPossibleRegion() );
         magnitudeGradientIm->SetRequestedRegionToLargestPossibleRegion();
         magnitudeGradientIm->SetBufferedRegion( magnitudeGradientIm->GetRequestedRegion() );
         magnitudeGradientIm->SetOrigin(imageMiddle->GetOrigin());
         magnitudeGradientIm->SetDirection(imageMiddle->GetDirection());
         magnitudeGradientIm->SetSpacing(imageMiddle->GetSpacing());
         magnitudeGradientIm->Allocate();
         ImageIterator gradientMagnIt( magnitudeGradientIm, magnitudeGradientIm->GetBufferedRegion() );
         gradientMagnIt.GoToBegin();
         
         bool normalize = true;
         
         while ( !outputIt.IsAtEnd() )
         {
         if (normalize)
         {
         double norm = sqrt(outputIt.Get()[0]*outputIt.Get()[0]+outputIt.Get()[1]*outputIt.Get()[1]+outputIt.Get()[2]*outputIt.Get()[2]);
         gradientXIt.Set(outputIt.Get()[0]/norm);
         gradientYIt.Set(outputIt.Get()[1]/norm);
         gradientZIt.Set(outputIt.Get()[2]/norm);
         }
         else
         {
         gradientXIt.Set(outputIt.Get()[0]);
         gradientYIt.Set(outputIt.Get()[1]);
         gradientZIt.Set(outputIt.Get()[2]);
         }
         
         //cout << sqrt(outputIt.Get()[0]*outputIt.Get()[0]+outputIt.Get()[1]*outputIt.Get()[1]+outputIt.Get()[2]*outputIt.Get()[2]) << endl;
         gradientMagnIt.Set(sqrt(outputIt.Get()[0]*outputIt.Get()[0]+outputIt.Get()[1]*outputIt.Get()[1]+outputIt.Get()[2]*outputIt.Get()[2]));
         
         ++outputIt;
         ++gradientXIt;
         ++gradientYIt;
         ++gradientZIt;
         ++gradientMagnIt;
         }
         
         WriterType::Pointer writerImMM = WriterType::New();
         writerImMM->SetImageIO(ioM);
         writerImMM->SetFileName("/home/django/benjamindeleener/data/PropSeg_data/t1/errsm_11/imageMagn.nii.gz");
         writerImMM->SetInput(magnitudeGradientIm);
         try {
         writerImMM->Update();
         }
         catch( itk::ExceptionObject & e )
         {
         cout << "Exception thrown ! " << endl;
         cout << "An error ocurred during Writing M" << endl;
         cout << "Location    = " << e.GetLocation()    << endl;
         cout << "Description = " << e.GetDescription() << endl;
         }
         
         
         // Computation of the Hessian matrix
         //typedef itk::GradientImageFilter< ImageType, float, double, GradientImageType > VectorGradientFilterType;
         typedef itk::GradientRecursiveGaussianImageFilter< ImageType, GradientImageType > VectorRGradientFilterType;
         
         VectorGradientFilterType::Pointer gradientMapFilterX = VectorGradientFilterType::New();
         gradientMapFilterX->SetInput( gradientX );
         try {
         gradientMapFilterX->Update();
         } catch( itk::ExceptionObject & e ) {
         cerr << "Exception caught while updating gradientMapFilter " << endl;
         cerr << e << endl;
         return EXIT_FAILURE;
         }
         GradientImageType::Pointer gradientX_gradient = gradientMapFilterX->GetOutput();
         VectorGradientFilterType::Pointer gradientMapFilterY = VectorGradientFilterType::New();
         gradientMapFilterY->SetInput( gradientY );
         try {
         gradientMapFilterY->Update();
         } catch( itk::ExceptionObject & e ) {
         cerr << "Exception caught while updating gradientMapFilter " << endl;
         cerr << e << endl;
         return EXIT_FAILURE;
         }
         GradientImageType::Pointer gradientY_gradient = gradientMapFilterY->GetOutput();
         VectorGradientFilterType::Pointer gradientMapFilterZ = VectorGradientFilterType::New();
         gradientMapFilterZ->SetInput( gradientZ );
         try {
         gradientMapFilterZ->Update();
         } catch( itk::ExceptionObject & e ) {
         cerr << "Exception caught while updating gradientMapFilter " << endl;
         cerr << e << endl;
         return EXIT_FAILURE;
         }
         GradientImageType::Pointer gradientZ_gradient = gradientMapFilterZ->GetOutput();
         
         
         ImageType::Pointer imageT = ImageType::New();
         imageT->SetLargestPossibleRegion( imageMiddle->GetLargestPossibleRegion() );
         imageT->SetRequestedRegionToLargestPossibleRegion();
         imageT->SetBufferedRegion( imageT->GetRequestedRegion() );
         imageT->SetOrigin(imageMiddle->GetOrigin());
         imageT->SetDirection(imageMiddle->GetDirection());
         imageT->SetSpacing(imageMiddle->GetSpacing());
         imageT->Allocate();
         ImageIterator imageTIt( imageT, imageT->GetBufferedRegion() );
         GradientImageIterator gradientX_gradientIt( gradientX_gradient, gradientX_gradient->GetBufferedRegion() );
         GradientImageIterator gradientY_gradientIt( gradientY_gradient, gradientY_gradient->GetBufferedRegion() );
         GradientImageIterator gradientZ_gradientIt( gradientZ_gradient, gradientZ_gradient->GetBufferedRegion() );
         imageTIt.GoToBegin();
         gradientX_gradientIt.GoToBegin();
         gradientY_gradientIt.GoToBegin();
         gradientZ_gradientIt.GoToBegin();
         
         double alpha = 0.5, beta = 0.5, gamma = 100;*/
        
        typedef itk::SymmetricSecondRankTensor< double, 3 > MatrixType;
        typedef itk::Image< MatrixType, 3> HessianImageType;
        typedef itk::ImageRegionIterator< HessianImageType > HessianImageIterator;
        /*HessianImageType::Pointer imageHessian = HessianImageType::New();
         imageHessian->SetLargestPossibleRegion( imageMiddle->GetLargestPossibleRegion() );
         imageHessian->SetRequestedRegionToLargestPossibleRegion();
         imageHessian->SetBufferedRegion( imageHessian->GetRequestedRegion() );
         imageHessian->SetOrigin(imageMiddle->GetOrigin());
         imageHessian->SetDirection(imageMiddle->GetDirection());
         imageHessian->SetSpacing(imageMiddle->GetSpacing());
         imageHessian->Allocate();
         HessianImageIterator hessianImageIt( imageHessian, imageHessian->GetBufferedRegion() );
         hessianImageIt.GoToBegin();
         
         double max_T = 0.0;
         while ( !hessianImageIt.IsAtEnd() )
         {
         MatrixType hessianMatrix;
         hessianMatrix(0,0) = gradientX_gradientIt.Get()[0];
         hessianMatrix(1,0) = gradientX_gradientIt.Get()[1];
         hessianMatrix(2,0) = gradientX_gradientIt.Get()[2];
         hessianMatrix(1,1) = gradientY_gradientIt.Get()[1];
         hessianMatrix(2,1) = gradientY_gradientIt.Get()[2];
         hessianMatrix(2,2) = gradientZ_gradientIt.Get()[2];
         
         
         hessianImageIt.Set(hessianMatrix);
         
         ++gradientX_gradientIt;
         ++gradientY_gradientIt;
         ++gradientZ_gradientIt;
         ++hessianImageIt;
         }*/
        
        
        
        typedef itk::HessianRecursiveGaussianImageFilter< ImageType >     HessianFilterType;
        typedef itk::Hessian3DToVesselnessMeasureImageFilter< double > VesselnessMeasureFilterType;
        HessianFilterType::Pointer hessianFilter = HessianFilterType::New();
        VesselnessMeasureFilterType::Pointer vesselnessFilter = VesselnessMeasureFilterType::New();
        hessianFilter->SetInput( imageMiddle );
        hessianFilter->SetSigma( 3.0 );
        hessianFilter->Update();
        HessianImageType::Pointer imageHessianITK = hessianFilter->GetOutput();
        
        
        //HessianImageIterator hessianImageUseIt( imageHessianITK, imageHessianITK->GetBufferedRegion() ); // Use ITK hessian image
        /*HessianImageIterator hessianImageUseIt( imageHessian, imageHessian->GetBufferedRegion() ); // Use my Hessian image
         hessianImageUseIt.GoToBegin();
         while ( !imageTIt.IsAtEnd() )
         {
         MatrixType hessianMatrix = hessianImageUseIt.Get();
         MatrixType::EigenValuesArrayType eigenValues;
         hessianMatrix.ComputeEigenValues(eigenValues);
         vector<double> temp = vector<double>(3); temp[0] = eigenValues[0]; temp[1] = eigenValues[1]; temp[2] = eigenValues[2];
         sort(temp.begin(), temp.end(),[](double a, double b){ return abs(a)<abs(b); });
         eigenValues[0] = temp[0]; eigenValues[1] = temp[1]; eigenValues[2] = temp[2];
         
         if (eigenValues[1] > 0 || eigenValues[2] > 0)
         {
         imageTIt.Set(0.0);
         }
         else
         {
         double Ra = abs(eigenValues[0])/(sqrt(abs(eigenValues[1])*abs(eigenValues[2])));
         double Rb = abs(eigenValues[1])/abs(eigenValues[2]);
         double S = sqrt(eigenValues[0]*eigenValues[0]+eigenValues[1]*eigenValues[1]+eigenValues[2]*eigenValues[2]);
         double T = (1-exp(-(Ra*Ra)/(2*alpha*alpha)))*exp(-(Rb*Rb)/(2*beta*beta))*(1-exp(-(S*S)/(2*gamma*gamma)));
         imageTIt.Set(T);
         cout << imageTIt.GetIndex() << " " << eigenValues << " " << Ra << " " << Rb << " " << S << " " << T << endl;
         if (T > max_T) max_T = T;
         }
         
         ++imageTIt;
         ++hessianImageUseIt;
         }
         cout << "Maximum T = " << max_T << endl;
         imageTIt.GoToBegin();
         while ( !imageTIt.IsAtEnd() )
         {
         imageTIt.Set(imageTIt.Get()/max_T);
         ++imageTIt;
         }
         
         WriterType::Pointer writerIm1 = WriterType::New();
         writerIm1->SetImageIO(io);
         writerIm1->SetFileName("/home/django/benjamindeleener/data/PropSeg_data/t1/errsm_11/imageT.nii.gz");
         writerIm1->SetInput(imageT);
         try {
         writerIm1->Update();
         }
         catch( itk::ExceptionObject & e )
         {
         cout << "Exception thrown ! " << endl;
         cout << "An error ocurred during Writing T" << endl;
         cout << "Location    = " << e.GetLocation()    << endl;
         cout << "Description = " << e.GetDescription() << endl;
         }*/
        
        
        
        
        
        vesselnessFilter->SetInput( imageHessianITK ); // output of the itk hessian image*/
        //vesselnessFilter->SetInput( imageHessian ); // output of the my hessian image
        vesselnessFilter->SetAlpha1( 0.5 );
        vesselnessFilter->SetAlpha2( 2.0 );
        vesselnessFilter->Update();
        WriterType::Pointer writerVesselNess = WriterType::New();
        itk::NiftiImageIO::Pointer ioV = itk::NiftiImageIO::New();
        writerVesselNess->SetImageIO(ioV);
        writerVesselNess->SetInput( vesselnessFilter->GetOutput() );
        writerVesselNess->SetFileName("imageVesselNessFilter.nii.gz");
        try {
            writerVesselNess->Update();
        }
        catch( itk::ExceptionObject & e )
        {
            cout << "Exception thrown ! " << endl;
            cout << "An error ocurred during Writing 1" << endl;
            cout << "Location    = " << e.GetLocation()    << endl;
            cout << "Description = " << e.GetDescription() << endl;
        }
        
        
        //mutualInformation[value] = startCrop;
    }
    return 0;
}





int Initialisation::symmetryDetection(ImageType2D::Pointer im, double cropWidth_, double bandWidth_)
{
    
    
    ImageType2D::SpacingType spacingIm = im->GetSpacing();
    int cropSize = cropWidth_/spacingIm[0];
    ImageType2D::SizeType desiredSize = im->GetLargestPossibleRegion().GetSize();
    
    ImageType2D::SizeType desiredSizeInitial = im->GetLargestPossibleRegion().GetSize();
    
    map<double,int> mutualInformation;
    int startSlice = desiredSizeInitial[0]/4, endSlice = desiredSizeInitial[0]/4*3;
    if (desiredSizeInitial[0] < cropSize*2) {
        startSlice = cropSize/2;
        endSlice = desiredSizeInitial[0]-cropSize/2;
    }
    
	typedef itk::MinimumMaximumImageCalculator<ImageType2D> MinMaxCalculatorType;
	MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
	minMaxCalculator->SetImage(im);
	minMaxCalculator->ComputeMaximum();
	minMaxCalculator->ComputeMinimum();
	ImageType2D::PixelType maxIm = minMaxCalculator->GetMaximum(), minIm = minMaxCalculator->GetMinimum();
	if (maxIm == minIm) {
		cerr << "ERROR: The image where the symmetry will be detected is full of constant value (" << maxIm << "). You can change it using -init parameter." << endl;
		return -1;
	}
    
    
    for (int i=startSlice; i<endSlice; i++)
    {
        float startCrop = i, size;
        if (startCrop < desiredSizeInitial[0]/2 && startCrop <= bandWidth_+1) size = startCrop-1;
        else if (startCrop >= desiredSizeInitial[0]/2 && startCrop >= desiredSizeInitial[0]-bandWidth_-1) size = desiredSizeInitial[0]-startCrop-1;
        else size = bandWidth_;
        ImageType2D::IndexType desiredStart;
        ImageType2D::SizeType desiredSize = desiredSizeInitial;
        desiredStart[0] = startCrop;
        desiredStart[1] = desiredSizeInitial[1]/2-(cropSize+5)/2;
        desiredSize[0] = size;
        desiredSize[1] = cropSize+5;
        
        // Right Image
        ImageType2D::RegionType desiredRegionImageRight(desiredStart, desiredSize);
        typedef itk::ExtractImageFilter< ImageType2D, ImageType2D > Crop2DFilterType;
        Crop2DFilterType::Pointer cropFilterRight = Crop2DFilterType::New();
        cropFilterRight->SetInput(im);
        cropFilterRight->SetExtractionRegion(desiredRegionImageRight);
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
        
        // middle image
        desiredStart[0] = startCrop-size;
        if (desiredStart[0] < 0) desiredStart[0] = 0;
        desiredSize[0] = size*2;
        ImageType2D::RegionType desiredRegionImage(desiredStart, desiredSize);
        Crop2DFilterType::Pointer cropFilter = Crop2DFilterType::New();
        cropFilter->SetExtractionRegion(desiredRegionImage);
        cropFilter->SetInput(im);
#if ITK_VERSION_MAJOR >= 4
        cropFilter->SetDirectionCollapseToIdentity(); // This is required.
#endif
        try {
            cropFilter->Update();
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while updating cropFilter " << std::endl;
            std::cerr << e << std::endl;
        }
        ImageType2D::Pointer imageMiddle = cropFilter->GetOutput();
        
        // Computation of Hessian matrix based on the GGVF of the image
        typedef itk::CovariantVector< double, 2 > Gradient2DPixelType;
        typedef itk::Image< Gradient2DPixelType, 2 > Gradient2DImageType;
        typedef itk::GradientImageFilter< ImageType2D, float, double, Gradient2DImageType > VectorGradient2DFilterType;
        VectorGradient2DFilterType::Pointer gradientMapFilter = VectorGradient2DFilterType::New();
        gradientMapFilter->SetInput( imageMiddle );
        try {
            gradientMapFilter->Update();
        } catch( itk::ExceptionObject & e ) {
            cerr << "Exception caught while updating gradientMapFilter " << endl;
            cerr << e << endl;
            return EXIT_FAILURE;
        }
        Gradient2DImageType::Pointer imageVectorGradient_temp = gradientMapFilter->GetOutput();
        
        typedef itk::GradientVectorFlowImageFilter< Gradient2DImageType, Gradient2DImageType >  GradientVectorFlowFilterType;
        GradientVectorFlowFilterType::Pointer gradientVectorFlowFilter = GradientVectorFlowFilterType::New();
        gradientVectorFlowFilter->SetInput(imageVectorGradient_temp);
        gradientVectorFlowFilter->SetIterationNum( 100 );
        gradientVectorFlowFilter->SetNoiseLevel( 1 );
        //gradientVectorFlowFilter->SetNormalize(true);
        try {
            gradientVectorFlowFilter->Update();
        } catch( itk::ExceptionObject & e ) {
            cerr << "Exception caught while updating gradientMapFilter " << endl;
            cerr << e << endl;
            return EXIT_FAILURE;
        }
        Gradient2DImageType::Pointer imageVectorGradient = gradientVectorFlowFilter->GetOutput();
        
        
        // Computation of the Hessian matrix
        typedef itk::RecursiveGaussianImageFilter<Gradient2DImageType,Gradient2DImageType> DerivativeFilterType;
        DerivativeFilterType::Pointer m_DerivativeFilter = DerivativeFilterType::New();
        m_DerivativeFilter->SetOrder(DerivativeFilterType::FirstOrder);
        m_DerivativeFilter->SetInput( imageVectorGradient );
        
        
        // Left Image
        desiredStart[0] = startCrop-size;
        if (desiredStart[0] < 0) desiredStart[0] = 0;
        desiredSize[0] = size;
        ImageType2D::RegionType desiredRegionImageLeft(desiredStart, desiredSize);
        Crop2DFilterType::Pointer cropFilterLeft = Crop2DFilterType::New();
        cropFilterLeft->SetExtractionRegion(desiredRegionImageLeft);
        cropFilterLeft->SetInput(im);
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
        ImageType2D::SizeType desSize; desSize[0] = desiredSize[0]; desSize[1] = desiredSize[1];
        ImageType2D::RegionType desired2DRegionImageRight(desIndex, desSize);
        ImageType2D::RegionType desired2DRegionImageLeft(desIndex, desSize);
        imageRight->SetLargestPossibleRegion(desired2DRegionImageRight);
        imageRight->SetRequestedRegion(desired2DRegionImageRight);
        imageRight->SetRegions(desired2DRegionImageRight);
        imageLeft->SetLargestPossibleRegion(desired2DRegionImageLeft);
        imageLeft->SetRequestedRegion(desired2DRegionImageLeft);
        imageLeft->SetRegions(desired2DRegionImageLeft);
        
        
        itk::FixedArray<bool, 2> flipAxes;
        flipAxes[0] = true;
        flipAxes[1] = false;
        typedef itk::FlipImageFilter <ImageType2D> FlipImageFilterType;
        FlipImageFilterType::Pointer flipFilter = FlipImageFilterType::New ();
        flipFilter->SetInput(imageRight);
        flipFilter->SetFlipAxes(flipAxes);
        flipFilter->Update();
        imageRight = flipFilter->GetOutput();
        
        ImageType2D::PointType origin = imageLeft->GetOrigin();
        imageRight->SetOrigin(origin);
        
        typedef itk::Image< unsigned short, 2 >	ImageType2DUI;
        typedef itk::ResampleImageFilter<ImageType2D, ImageType2DUI> ResampleImageFilterTypeunsignedint;
        ResampleImageFilterTypeunsignedint::Pointer resfilter = ResampleImageFilterTypeunsignedint::New();
        resfilter->SetInput(imageRight);
        resfilter->SetSize(imageRight->GetLargestPossibleRegion().GetSize());
        resfilter->SetOutputDirection(imageRight->GetDirection());
        resfilter->SetOutputOrigin(imageRight->GetOrigin());
        resfilter->SetOutputSpacing(imageRight->GetSpacing());
        resfilter->SetTransform(TransformType::New());
        resfilter->Update();
        typedef itk::ImageFileWriter< ImageType2DUI > WriterRGBType;
		itk::PNGImageIO::Pointer ioPNG = itk::PNGImageIO::New();
		WriterRGBType::Pointer writerPNG = WriterRGBType::New();
		writerPNG->SetInput(resfilter->GetOutput());
		writerPNG->SetImageIO(ioPNG);
        stringstream ss;
        ss << i;
        string str = ss.str();
		writerPNG->SetFileName("/home/django/benjamindeleener/data/PropSeg_data/t2star/errsm_01/imageRight"+ss.str()+".png");
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
        ResampleImageFilterTypeunsignedint::Pointer resfilter2 = ResampleImageFilterTypeunsignedint::New();
        resfilter2->SetInput(imageLeft);
        resfilter2->SetSize(imageLeft->GetLargestPossibleRegion().GetSize());
        resfilter2->SetOutputDirection(imageLeft->GetDirection());
        resfilter2->SetOutputOrigin(imageLeft->GetOrigin());
        resfilter2->SetOutputSpacing(imageLeft->GetSpacing());
        resfilter2->SetTransform(TransformType::New());
        resfilter2->Update();
        WriterRGBType::Pointer writerPNG2 = WriterRGBType::New();
		writerPNG2->SetInput(resfilter2->GetOutput());
		writerPNG2->SetImageIO(ioPNG);
		writerPNG2->SetFileName("/home/django/benjamindeleener/data/PropSeg_data/t2star/errsm_01/imageLeft"+ss.str()+".png");
		try {
		    writerPNG2->Update();
		}
		catch( itk::ExceptionObject & e )
		{
			cout << "Exception thrown ! " << endl;
			cout << "An error ocurred during Writing PNG" << endl;
			cout << "Location    = " << e.GetLocation()    << endl;
			cout << "Description = " << e.GetDescription() << endl;
		}
        
        MinMaxCalculatorType::Pointer minMaxCalculatorLeft = MinMaxCalculatorType::New();
        minMaxCalculatorLeft->SetImage(imageLeft);
        minMaxCalculatorLeft->ComputeMaximum();
        minMaxCalculatorLeft->ComputeMinimum();
        ImageType2D::PixelType maxImLeft = minMaxCalculatorLeft->GetMaximum(), minImLeft = minMaxCalculatorLeft->GetMinimum();
        //if (maxImLeft-minImLeft <200) cout << "ATTENTION!! " << i << " " << maxImLeft-minImLeft << endl;
        MinMaxCalculatorType::Pointer minMaxCalculatorRight = MinMaxCalculatorType::New();
        minMaxCalculatorRight->SetImage(imageRight);
        minMaxCalculatorRight->ComputeMaximum();
        minMaxCalculatorRight->ComputeMinimum();
        ImageType2D::PixelType maxImRight = minMaxCalculatorRight->GetMaximum(), minImRight = minMaxCalculatorRight->GetMinimum();
        //if (maxImRight-minImRight <200) cout << "ATTENTION!! " << i << " " << maxImRight-minImRight << endl;
        
        if (maxImLeft-minImLeft > 250 && maxImRight-minImRight > 250)
        {
            // Better value is minimum
            //typedef itk::MattesMutualInformationImageToImageMetric< ImageType2D, ImageType2D > SimilarityFilter;
            typedef itk::MeanReciprocalSquareDifferenceImageToImageMetric< ImageType2D, ImageType2D > SimilarityFilter;
            //typedef itk::NormalizedCorrelationImageToImageMetric< ImageType2D, ImageType2D > SimilarityFilter;
            SimilarityFilter::Pointer correlationFilter = SimilarityFilter::New();
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
            SimilarityFilter::TransformParametersType id(2);
            
            id[0] = 0; id[1] = 0;
            double value = 0.0;
            try {
                value = correlationFilter->GetValue( id );
            } catch( itk::ExceptionObject & e ) {
                std::cerr << "Exception caught while getting value " << std::endl;
                std::cerr << e << std::endl;
            }
            mutualInformation[value] = startCrop;
        }
    }
    //cout << "Cropping aroundIn slice = " << mutualInformation.begin()->second << endl;
    int middleSlice_ = mutualInformation.begin()->second;
    for (map<double,int>::iterator it=mutualInformation.begin(); it!=mutualInformation.end(); it++)
        cout << it->first << " " << it->second << endl;
    int k;
    cin >> k;
    return middleSlice_;
}