#ifndef __SymmetricalCropping__
#define __SymmetricalCropping__

/*!
 * \file SymmetricalCropping.h
 * \brief Compute a detection of the symmetry inside the body.
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include <itkImage.h>
#include <vector>
using namespace std;

/*!
 * \class SymmetricalCropping
 * \brief Compute a detection of the symmetry inside the body.
 * 
 * This class detects the symmetry in an 3D image in the left-right direction by performing a mutual information correlation between two parts of a 2D axial slice of the image and return the index of the sagittal symmetric slice.
 */
class SymmetricalCropping {
public:
    typedef itk::Image< double, 3 > ImageType;
    typedef itk::Image< double, 2 >	ImageType2D;
    
    SymmetricalCropping();
    ~SymmetricalCropping() {};
    
    void setInputImage(ImageType::Pointer inputImage) { inputImage_=inputImage; };
    ImageType::Pointer getOutput() { return outputImage_; };

	void setInitSlice(float initSlice) { initSlice_=initSlice; };
    
    //vector<int> symmetryDetectionFull(int dimension=1);
    int symmetryDetection(int dimension=1);
    ImageType::Pointer cropping();
    
private:
    ImageType::Pointer inputImage_, outputImage_;
    double cropWidth_; // in millimeters
    double bandWidth_; // width of band to correlate
    int middleSlice_; // -1 if not yet computed
	float initSlice_;
};
#endif /* defined(__Test__SymmetricalCropping__) */
