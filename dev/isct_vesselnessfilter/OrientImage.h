#ifndef __OrientImage__
#define __OrientImage__

/*!
 * \file OrientImage.h
 * \brief Compute and change orientation of an image
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include <itkImage.h>
#include <itkOrientImageFilter.h>
#include <itkSpatialOrientationAdapter.h>

/*!
 * \class OrientImage
 * \brief Compute and change orientation of an image.
 *
 * This class compute the orientation of an image and can change its orientation to any one available on ITK (itk::SpatialOrientation::ValidCoordinateOrientationFlags).
 */
template <typename InputImageType>
class OrientImage
{
public:
	typedef itk::SpatialOrientationAdapter SpatialOrientationAdapterType;
	typedef itk::SpatialOrientation::ValidCoordinateOrientationFlags OrientationType;
    typedef typename InputImageType::Pointer InputImagePointer;
    
	/*!
	* \brief Constructor
	*
	* Constructor of OrientImage class
	*/
    OrientImage() {};

	/*!
	* \brief Destructor
	*
	* Destructor of OrientImage class
	*/
    ~OrientImage() {};
    
	/*!
	* \brief Set the image to compute
	*
	* Set the image to compute
	*/
    void setInputImage(InputImagePointer image)
    {
        image_ = image;
        SpatialOrientationAdapterType adapt;
        initialImageOrientation_ = adapt.FromDirectionCosines(image_->GetDirection());
    };

	/*!
	* \brief Get the computed image
	*
	* Get the computed image
	*/
    InputImagePointer getOutputImage() { return outputImage_; };

	/*!
	* \brief Get the initial orientation of the image
	*
	* Get the initial orientation of the image. The type provided comes from itk::SpatialOrientation.
	*/
    OrientationType getInitialImageOrientation() { return initialImageOrientation_; };

	/*!
	* \brief Change orientation of the input image
	*
	* Change the orientation of the input image.
	*
	* \param Desired orientation. Available orientation are listed in itk::SpatialOrientation documentation.
	*/
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

#endif /* defined(__OrientImage__) */
