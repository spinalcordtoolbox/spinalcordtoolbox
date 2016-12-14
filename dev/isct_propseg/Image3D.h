#ifndef __IMAGE3D__
#define __IMAGE3D__

/*!
 * \file Image3D.h
 * \brief Container for image, gradient magnitude image and gradient vector image.
 * \author Benjamin De Leener - NeuroPoly (http://www.neuropoly.info)
 */

#include "util/Vector3.h"
#include "util/Matrix3x3.h"
#include "Mesh.h"
#include <itkImageAlgorithm.h>
#include <itkImageFileReader.h>
#include <itkImage.h>
#include <itkTriangleMeshToBinaryImageFilter.h>
#include <itkTriangleCell.h>
#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkCastImageFilter.h>
using namespace std;

typedef itk::Image< double, 3 > ImageType;
typedef itk::Image< unsigned char, 3 > BinaryImageType;
typedef itk::CovariantVector<double,3> PixelType;
typedef itk::Image< PixelType, 3 > ImageVectorType;
typedef ImageVectorType::IndexType IndexType;

typedef itk::LinearInterpolateImageFunction< ImageType, double > InterpolateIntensityFilter;
typedef itk::VectorLinearInterpolateImageFunction< ImageVectorType, double > InterpolateVectorFilter;

typedef itk::SpatialOrientation::ValidCoordinateOrientationFlags OrientationType;

/*!
 * \class Image3D
 * \brief Container for images
 * 
 * This class contains an image, its gradient magnitude and gradient vector, compute discrete and continuous pixel values, and providing transformation method.
 * This class also provide methods for transforming a mesh to a binary image.
 */
class Image3D
{
public:
	Image3D();
	Image3D(ImageVectorType::Pointer im, int hauteur, int largeur, int profondeur, CVector3 origine, CVector3 directionX, CVector3 directionY, CVector3 directionZ, CVector3 spacing, double typeImageFactor);
	~Image3D() {};

	ImageVectorType::Pointer getImage() { return image_; };
	
	float GetPixelOriginal(const CVector3& index);
	float GetPixelMagnitudeGradient(const CVector3& index);
	float GetContinuousPixelMagnitudeGradient(const CVector3& index);
	PixelType GetPixel(const CVector3& index);
	CVector3 GetPixelVector(const CVector3& index);
	CVector3 GetContinuousPixelVector(const CVector3& index);
	CVector3 GetPixelVectorLaplacian(const CVector3& index);
	PixelType GetPixel(const IndexType& index) { return image_->GetPixel(index); };
	int getHauteur() { return hauteur_; };
	int getLargeur() { return largeur_; };
	int getProfondeur() { return profondeur_; };
	CVector3 getOrigine() { return origine_; };
	CVector3 getDirectionX() { return directionX_; };
	CVector3 getDirectionY() { return directionY_; };
	CVector3 getDirectionZ() { return directionZ_; };
	CVector3 getSpacing() { return spacing_; };

	bool TransformPhysicalPointToIndex(const CVector3& point, CVector3& index);
	bool TransformPhysicalPointToContinuousIndex(const CVector3& point, CVector3& index);
	CVector3 TransformIndexToPhysicalPoint(const CVector3& index);
    CVector3 TransformContinuousIndexToPhysicalPoint(const CVector3& index);

	double GetMaximumNorm();
	void NormalizeByMaximum();
	void DeleteHighVector();

	void TransformMeshToBinaryImage(Mesh* m, string filename, OrientationType orient, bool sub_segmentation=false, bool cropUpDown=false, CVector3* upperSlicePoint=0, CVector3* upperSliceNormal=0, CVector3* downSlicePoint=0, CVector3* downSliceNormal=0);

	void setImageOriginale(ImageType::Pointer i);
	ImageType::Pointer getImageOriginale() { return imageOriginale_; };

	void setCroppedImageOriginale(ImageType::Pointer i);
	ImageType::Pointer getCroppedImageOriginale() { return croppedOriginalImage_; };

	void setImageMagnitudeGradient(ImageType::Pointer i);
	ImageType::Pointer getImageMagnitudeGradient() { return imageMagnitudeGradient_; };

	void setLaplacianImage(ImageVectorType::Pointer i);
	ImageVectorType::Pointer getLaplacianImage() { return laplacianImage_; };

	void setTypeImageFactor(double f) { type_image_factor_ = f; };
	double getTypeImageFactor() { return type_image_factor_; };

	void releaseMemory();

private:
	ImageType::Pointer imageOriginale_, croppedOriginalImage_, imageMagnitudeGradient_;
    BinaryImageType::Pointer imageSegmentation_;
	ImageVectorType::Pointer image_, laplacianImage_;
	bool boolImageOriginale_, boolCroppedOriginalImage_, boolImageMagnitudeGradient_, boolImage_, boolLaplacianImage_;
	int hauteur_, largeur_, profondeur_;
	CVector3 origine_, directionX_, directionY_, directionZ_, spacing_, extremePoint_;
	CMatrix3x3 direction, directionInverse;
	double type_image_factor_;

	InterpolateIntensityFilter::Pointer imageInterpolator;
	InterpolateVectorFilter::Pointer vectorImageInterpolator;
};

#endif
