#include <itkMultiScaleHessianBasedMeasureImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImage.h>
#include <string>
#include <itkNiftiImageIO.h>
#include <itkPNGImageIO.h>
#include <itkIndex.h>
#include <itkContinuousIndex.h>
#include <itkImageRegionConstIterator.h>
#include

/*
 * This program is meant as an example to test results of a Vesselness filter
 */

int main(int argc, char *argv[])
{
	string input = "input.nii.gz", string output = "vesselnessOutput.nii.gz";
	double sigmaMin = 2.5;
	double sigmaMax = 10;
	double alpha;
	double beta;
	double gamma;
	unsigned int numberOfSigmaSteps;
	double sigmaDistance;

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
