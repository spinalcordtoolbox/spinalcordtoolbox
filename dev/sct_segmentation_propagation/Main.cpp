/*! \file Main.cpp
 * \mainpage sct_segmentation_propagation
 *
 * \section description Description
 * Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>
 *
 * This program segments automatically the spinal cord on T1- and T2-weighted images, for any field of view. You must provide the type of contrast and the image.
 * 
 * Primary output is the binary mask of the spinal cord segmentation (a voxel is inside the spinal cord when its center is inside the segmentation surface). This method must provide VTK triangular mesh of the segmentation (option -mesh). Spinal cord centerline is available as a binary image (-centerline-binary) or a text file with coordinates in world referential (-centerline-coord). It also provide the cross-sectional areas of the spinal cord, for each "z" slice.
 * 
 * Several tips on segmentation correction can be found on the \ref correction_tips "Correction Tips" page of the documentation while advices on parameters adjustments can be found on the \ref parameters_adjustment "Parameters" page.
 * 
 * If the segmentation fails at some location (e.g. due to poor contrast between spinal cord and CSF), edit your anatomical image (e.g. with fslview) and manually enhance the contrast by adding bright values around the spinal cord for T2-weighted images (dark values for T1-weighted). Then, launch the segmentation again.
 *
 * \section usage Usage
 * \code sct_segmentation_propagation -i <inputfilename> -o <outputfolderpath> -t <imagetype> [options] \endcode
 * 
 * \section input Input parameters
 *
 * General options: 
 *		* -i <inputfilename>            (no default)
 *      * -i-dicom <inputfolderpath>    (replace -i, read DICOM series, output still in NIFTI)
 *		* -o <outputfolderpath>         (default is current folder)
 *		* -t <imagetype> {t1,t2}        (string, type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark, no default)
 *		* -down <down_slice>            (int, down limit of the propagation, default is 0)
 *		* -up <up_slice>                (int, up limit of the propagation, default is higher slice of the image)
 *		* -verbose                      (display on)
 *		* -help
 *	
 * Output options:
 *		* -detect-display               (output: spinal cord detection as a PNG image)
 *		* -mesh                         (output: mesh of the spinal cord segmentation)
 *		* -centerline-binary            (output: centerline as a binary image)
 *		* -centerline-coord             (output: centerline as world coordinates)
 *		* -cross                        (output: cross-sectional areas)
 *      * -init-tube                    (output: initial tubular meshes)
 *      * -low-resolution-mesh          (output: low-resolution mesh)
 *      * -CSF                          (output: CSF segmentation)
 *	
 * Initialization - Spinal cord detection module options:
 *		* -init <init_position>         (axial slice where the propagation starts, default is middle axial slice)
 *		* -detect-n <numberslice>       (int, number of axial slices computed in the detection process, default is 4)
 *		* -detect-gap <gap>             (int, gap between two axial slices in the detection process, default is 4)
 *		* -detect-radius <radius>       (double, approximate radius of the spinal cord, default is 4 mm)
 *      * -init-mask <filename>         (string, mask containing three center of the spinal cord, used to initiate the propagation)
 *      * -init-validation              (enable validation on spinal cord detection)
 * 
 * Propagation module options:
 *      * -init-centerline <filename>   (filename of centerline to use for the propagation, format .txt or .nii, see file structure in documentation)
 *      * -nbiter <number>              (int, stop condition: number of iteration for the propagation for both direction, default is 200)
 *      * -max-area <number>            (double, in mm^2, stop condition: maximum cross-sectional area, default is 120 mm^2)
 *      * -max-deformation <number>     (double, in mm, stop condition: maximum deformation per iteration, default is 2.5 mm)

 *
 * \section dep Dependencies
 * - ITK (http://www.itk.org/)
 * - VTK (http://www.vtk.org/)
 *
 * \section com Comments
 * Copyright (c) 2014 NeuroPoly, Polytechnique Montreal \<http://www.neuro.polymtl.ca\>
 *
 * Author: Benjamin De Leener
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 * 
 */
#define _SCL_SECURE_NO_WARNINGS
//#define _CRT_SECURE_NO_WARNINGS

// std libraries
#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>

// local references
#include "referential.h"
#include "util/Matrix4x4.h"
#include "SpinalCord.h"
#include "Initialisation.h"
#include "OrientImage.h"
#include "GlobalAdaptation.h"
#include "PropagatedDeformableModel.h"
#include "SymmetricalCropping.h"
#include "Image3D.h"
#include "SCRegion.h"
#include "VertebralIdentification.h"

// ITK libraries
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkNiftiImageIO.h>
#include <itkGradientMagnitudeImageFilter.h>
#include <itkGradientImageFilter.h>
#include <itkImageAlgorithm.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkFileTools.h>
#include <itkPointSet.h>
#include <itkBSplineScatteredDataPointSetToImageFilter.h>
#include <itkBSplineControlPointImageFunction.h>
#include <itkImageSeriesReader.h>
//#include <itkGDCMImageIO.h>
//#include <itkGDCMSeriesFileNames.h>

using namespace std;

// Type definitions
typedef itk::Image< double, 3 >	ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageRegionConstIterator<ImageType> ImageIterator;
typedef itk::RescaleIntensityImageFilter< ImageType, ImageType > RescaleFilterType;
typedef itk::CovariantVector< double, 3 > GradientPixelType;
typedef itk::Image< GradientPixelType, 3 > GradientImageType;
typedef itk::GradientImageFilter< ImageType, float, double, GradientImageType > VectorGradientFilterType;
typedef itk::GradientMagnitudeImageFilter< ImageType, ImageType > GradientMFilterType;

typedef itk::Image< unsigned char, 3 >	BinaryImageType;
typedef itk::ImageFileReader<BinaryImageType> BinaryReaderType;
typedef itk::ImageRegionConstIterator<BinaryImageType> BinaryImageIterator;

typedef itk::ImageSeriesReader< ImageType > DICOMReaderType;
//typedef itk::GDCMImageIO ImageIOType;
//typedef itk::GDCMSeriesFileNames InputNamesGeneratorType;

bool extractPointAndNormalFromMask(string filename, CVector3 &point, CVector3 &normal1, CVector3 &normal2);
vector<CVector3> extractCenterline(string filename);

void help()
{
    cout << "sct_segmentation_propagation - Version 0.2.7 (2014-06-12)" << endl;
    cout << "Author : Benjamin De Leener - NeuroPoly lab - Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>" << endl << endl;
    
    cout << "Description:" << endl;
	cout << "This program segments automatically the spinal cord on T1- and T2-weighted images, for any field of view. You must provide the type of contrast, the image as well as the output folder path." << endl;
	cout << "Initialization is provided by a spinal cord detection module based on the elliptical Hough transform on multiple axial slices. The result of the detection is available as a PNG image using option -detection-display." << endl;
	cout << "Parameters of the spinal cord detection are :" << endl << " - the position (in inferior-superior direction) of the initialization" << endl << " - the number of axial slices" << endl << " - the gap (in pixel) between two axial slices" << endl << " - the approximate radius of the spinal cord" << endl << endl;
	
	cout << "Primary output is the binary mask of the spinal cord segmentation. This method must provide VTK triangular mesh of the segmentation (option -mesh). Spinal cord centerline is available as a binary image (-centerline-binary) or a text file with coordinates in world referential (-centerline-coord)." << endl;
	cout << "Cross-sectional areas along the spinal cord can be available (-cross)." << endl;
    
    cout << "Several tips on segmentation correction can be found on the \"Correction Tips\" page of the documentation while advices on parameters adjustments can be found on the \"Parameters\" page." << endl;
    cout << "If the segmentation fails at some location (e.g. due to poor contrast between spinal cord and CSF), edit your anatomical image (e.g. with fslview) and manually enhance the contrast by adding bright values around the spinal cord for T2-weighted images (dark values for T1-weighted). Then, launch the segmentation again." << endl;
	
    cout << "Usage: \t sct_segmentation_propagation -i <inputfilename> -o <outputfolderpath> -t <imagetype> [options]" << endl << endl;
    cout << endl << "WARNING: be sure your image has the correct type !!" << endl << endl;
    
    cout << "General options: " << endl;
    cout << "\t-i <inputfilename> \t (no default)" << endl;
    //cout << "\t-i-dicom <inputfolderpath> \t (replace -i, read DICOM series, output still in NIFTI)" << endl;
    cout << "\t-o <outputfolderpath> \t (default is current folder)" << endl;
    cout << "\t-t <imagetype> {t1,t2} \t (string, type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark, no default)" << endl;
	cout << "\t-down <down_slice> \t (int, down limit of the propagation, default is 0)" << endl;
    cout << "\t-up <up_slice> \t\t (int, up limit of the propagation, default is higher slice of the image)" << endl;
	cout << "\t-verbose \t\t (display on)" << endl;
    cout << "\t-help" << endl;
    cout << endl;
	
	// Output files
	cout << "Output options:" << endl;
	cout << "\t-detect-display \t (output of spinal cord detection as a PNG image)" << endl;
	cout << "\t-mesh \t (output: mesh of the spinal cord segmentation)" << endl;
	cout << "\t-centerline-binary \t (output: centerline as a binary image)" << endl;
	cout << "\t-centerline-coord \t (output: centerline as world coordinates)" << endl;
	cout << "\t-cross \t (output: cross-sectional areas)" << endl;
    cout << "\t-init-tube \t (output: initial tubular meshes)" << endl;
    cout << "\t-low-resolution-mesh \t (output: low-resolution mesh)" << endl;
    cout << "\t-CSF \t (output: CSF segmentation)" << endl;
	cout << endl;
	
    // Initialization
	cout << "Initialization - Spinal cord detection module options:" << endl;
    cout << "\t-init <init_position> \t (axial slice where the propagation starts, default is middle axial slice)" << endl;
	cout << "\t-detect-n <numberslice> \t (int, number of axial slices computed in the detection process, default is 4)" << endl;
	cout << "\t-detect-gap <gap> \t (int, gap between two axial slices in the detection process, default is 4)" << endl;
	cout << "\t-detect-radius <radius> \t (double, approximate radius of the spinal cord, default is 4 mm)" << endl;
    cout << "\t-init-mask <filename> \t (string, mask containing three center of the spinal cord, used to initiate the propagation)" << endl;
	cout << "\t-init-validation \t (enable validation on spinal cord detection)" << endl;
    cout << endl;
    
    // Propagation
	cout << "Propagation module options:" << endl;
	cout << "\t-init-centerline <filename> \t (filename of centerline to use for the propagation, format .txt or .nii, see file structure in documentation)" << endl;
    cout << "\t-nbiter <number> \t (int, stop condition: number of iteration for the propagation for both direction, default is 200)" << endl;
    cout << "\t-max-area <number> \t (double, in mm^2, stop condition: maximum cross-sectional area, default is 120 mm^2)" << endl;
    cout << "\t-max-deformation <number> \t (double, in mm, stop condition: maximum deformation per iteration, default is 2.5 mm)" << endl;
    cout << "\t-min-contrast <number> \t (double, in intensity value, stop condition: minimum local SC/CSF contrast, default is 50)" << endl;
	
}

int main(int argc, char *argv[])
{
    srand (time(NULL));
    
	if (argc == 1)
    {
        help();
        return EXIT_FAILURE;
    }
    string inputFilename = "", outputPath = "", outputFilenameBinary = "", outputFilenameMesh = "", outputFilenameBinaryCSF = "", outputFilenameMeshCSF = "", outputFilenameAreas = "", outputFilenameAreasCSF = "", outputFilenameCenterline = "", outputFilenameCenterlineBinary = "", inputCenterlineFilename = "", initMaskFilename = "";
    double typeImageFactor = 0.0, initialisation = 0.5;
    int downSlice = -10000, upSlice = 10000;
    string suffix;
	bool input_dicom = false, output_detection = false, output_mesh = false, output_centerline_binary = false, output_centerline_coord = false, output_cross = false, init_with_centerline = false, init_with_mask = false, verbose = false, output_init_tube = false, completeCenterline = false, init_validation = false, low_res_mesh = false, CSF_segmentation = false;
	int gapInterSlices = 4, nbSlicesInitialisation = 5;
	double radius = 4.0;
    int numberOfPropagationIteration = 200;
    double maxDeformation = 0.0, maxArea = 0.0, minContrast = 50.0;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i],"-i")==0) {
            i++;
            inputFilename = argv[i];
        }
        else if (strcmp(argv[i],"-i-dicom")==0) {
            i++;
            //inputFilename = argv[i];
            //input_dicom = true;
        }
        else if (strcmp(argv[i],"-o")==0) {
            i++;
            outputPath = argv[i];
        }
        else if (strcmp(argv[i],"-t")==0) {
            i++;
            if (strcmp(argv[i],"t1")==0) {
				typeImageFactor = -1.0;
				if (verbose) cout << endl << "WARNING: be sure your image is a T1-weighted image." << endl << endl;
			}
            else if (strcmp(argv[i],"t2")==0) {
				typeImageFactor = 1.0;
				if (verbose) cout << endl << "WARNING: be sure your image is a T2-weighted image." << endl << endl;
			}
            else {
				cout << "Error: Invalid type or image (need to be \"t1\" or \"t2\")" << endl << endl;
				help();
                return EXIT_FAILURE;
			}
        }
        else if (strcmp(argv[i],"-init")==0) {
            i++;
            initialisation = atof(argv[i]);
        }
        else if (strcmp(argv[i],"-down")==0) {
            i++;
            downSlice = atoi(argv[i]);
        }
        else if (strcmp(argv[i],"-up")==0) {
            i++;
            upSlice = atoi(argv[i]);
        }
		else if (strcmp(argv[i],"-detect-display")==0) {
            output_detection = true;
        }
		else if (strcmp(argv[i],"-mesh")==0) {
            output_mesh = true;
        }
		else if (strcmp(argv[i],"-centerline-binary")==0) {
            output_centerline_binary = true;
        }
		else if (strcmp(argv[i],"-centerline-coord")==0) {
            output_centerline_coord = true;
        }
		else if (strcmp(argv[i],"-cross")==0) {
            output_cross = true;
        }
		else if (strcmp(argv[i],"-detect-n")==0) {
            i++;
            nbSlicesInitialisation = atoi(argv[i]);
        }
		else if (strcmp(argv[i],"-detect-gap")==0) {
            i++;
            gapInterSlices = atoi(argv[i]);
        }
		else if (strcmp(argv[i],"-detect-radius")==0) {
            i++;
            radius = atof(argv[i]);
        }
		else if (strcmp(argv[i],"-init-centerline")==0) {
            i++;
            inputCenterlineFilename = argv[i];
            init_with_centerline = true;
        }
        else if (strcmp(argv[i],"-nbiter")==0) {
            i++;
            numberOfPropagationIteration = atoi(argv[i]);
        }
        else if (strcmp(argv[i],"-init-mask")==0) {
            i++;
            initMaskFilename = argv[i];
            init_with_mask = true;
        }
        else if (strcmp(argv[i],"-init-tube")==0) {
            output_init_tube = true;
        }
		else if (strcmp(argv[i],"-init-validation")==0) {
            init_validation = true;
        }
        else if (strcmp(argv[i],"-low-resolution-mesh")==0) {
            low_res_mesh = true;
        }
        else if (strcmp(argv[i],"-max-deformation")==0) {
            i++;
            maxDeformation = atof(argv[i]);
        }
        else if (strcmp(argv[i],"-max-area")==0) {
            i++;
            maxArea = atof(argv[i]);
        }
        else if (strcmp(argv[i],"-min-contrast")==0) {
            i++;
            minContrast = atof(argv[i]);
        }
        else if (strcmp(argv[i],"-CSF")==0) {
            CSF_segmentation = true;
            if (maxArea == 0.0) maxArea = 120;
            if (maxDeformation == 0.0) maxDeformation = 2.5;
        }
		else if (strcmp(argv[i],"-verbose")==0) {
            verbose = true;
        }
        else if (strcmp(argv[i],"-help")==0) {
            help();
            return EXIT_FAILURE;
        }
    }
    if (inputFilename == "")
    {
        cerr << "Input filename or folder (if DICOM) not provided" << endl;
		help();
        return EXIT_FAILURE;
    }
    if (typeImageFactor == 0)
    {
        cerr << "Error: The type of contrast not provided (option -t)" << endl;
		help();
        return EXIT_FAILURE;
    }
    
    // output files must have the same extension as input file
    string nii=".nii", niigz=".nii.gz"; suffix=niigz;
    size_t pos = inputFilename.find(niigz);
    if (pos == string::npos) {
        pos = inputFilename.find(nii);
        suffix = nii;
    }
    if (outputPath!="" && outputPath.compare(outputPath.length()-1,1,"/")) outputPath += "/"; // add "/" if missing
    outputFilenameBinary = outputPath+"segmentation_binary"+suffix;
    outputFilenameMesh = outputPath+"segmentation_mesh.vtk";
    outputFilenameBinaryCSF = outputPath+"segmentation_CSF_binary"+suffix;
    outputFilenameMeshCSF = outputPath+"segmentation_CSF_mesh.vtk";
    outputFilenameAreas = outputPath+"cross_sectional_areas.txt";
    outputFilenameAreasCSF = outputPath+"cross_sectional_areas_CSF.txt";
    outputFilenameCenterline = outputPath+"segmentation_centerline.txt";
    outputFilenameCenterlineBinary = outputPath+"segmentation_centerline_binary"+suffix;
    // if output path doesn't exist, we create it
    if (outputPath!="") itk::FileTools::CreateDirectory(outputPath.c_str());
    
    // Image reading - image can be T1 or T2 (or Tx-like) depending on contrast between spinal cord and CSF
	// typeImageFactor depend of contrast type and is equal to +1 when CSF is brighter than spinal cord and equal to -1 inversely
    ImageType::Pointer initialImage, image = ImageType::New();
    
    if (input_dicom)
    {
        /*ImageIOType::Pointer gdcmIO = ImageIOType::New();
        InputNamesGeneratorType::Pointer inputNames = InputNamesGeneratorType::New();
        inputNames->SetInputDirectory( inputFilename );
        
        const DICOMReaderType::FileNamesContainer & filenames = inputNames->GetInputFileNames();
        
        DICOMReaderType::Pointer reader = DICOMReaderType::New();
        reader->SetImageIO( gdcmIO );
        reader->SetFileNames( filenames );
        try
        {
            reader->Update();
        } catch (itk::ExceptionObject &excp) {
            std::cerr << "Exception thrown while reading the DICOM series" << std::endl;
            std::cerr << excp << std::endl;
            return EXIT_FAILURE;
        }
        initialImage = reader->GetOutput();*/
    }
    else
    {
        ReaderType::Pointer reader = ReaderType::New();
        itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
        reader->SetImageIO(io);
        reader->SetFileName(inputFilename);
        try {
            reader->Update();
        } catch( itk::ExceptionObject & e ) {
            cerr << "Exception caught while reading input image" << endl;
            cerr << e << endl;
            return EXIT_FAILURE;
        }
        initialImage = reader->GetOutput();
    }
    
    // Change orientation of input image to AIL. Output images will have the same orientation as input image
    OrientImage<ImageType> orientationFilter;
    orientationFilter.setInputImage(initialImage);
    orientationFilter.orientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL);
    initialImage = orientationFilter.getOutputImage();
	
	// Crop image if it is too large in left-right direction. No need to compute the initialization on the whole image. We assume the spinal cord is included in a 5cm large region.
	ImageType::SizeType desiredSize = initialImage->GetLargestPossibleRegion().GetSize();
    ImageType::SpacingType spacingI = initialImage->GetSpacing();
	if (desiredSize[2]*spacingI[2] > 60 && !init_with_mask && !init_with_centerline)
	{
		SymmetricalCropping symCroppingFilter;
        symCroppingFilter.setInputImage(initialImage);
		symCroppingFilter.setInitSlice(initialisation);
		int crop_slice = -1;
		try {
			crop_slice = symCroppingFilter.symmetryDetection();
		} catch(exception & e) {
		    cerr << "Exception caught while computing symmetry" << endl;
            cerr << e.what() << endl;
            return EXIT_FAILURE;
		}
		if (crop_slice != -1) {
			if (verbose) cout << "Cropping input image in left-right direction around slice = " << crop_slice << endl;
			image = symCroppingFilter.cropping();
		} else {
			if (verbose) cout << "Image non cropped for symmetry" << endl;
			image = initialImage;
		}
	}
	else image = initialImage;
    
	// Intensity normalization
	RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
	rescaleFilter->SetInput(image);
	rescaleFilter->SetOutputMinimum(0);
	rescaleFilter->SetOutputMaximum(1000);
    try {
        rescaleFilter->Update();
    } catch( itk::ExceptionObject & e ) {
        cerr << "Exception caught while normalizing input image " << endl;
        cerr << e << endl;
        return EXIT_FAILURE;
    }
	image = rescaleFilter->GetOutput();
	
	// computing magnitude and direction of gradient image
	GradientMFilterType::Pointer gradientMagnitudeFilter = GradientMFilterType::New();
	gradientMagnitudeFilter->SetInput(image);
	try {
		gradientMagnitudeFilter->Update();
	} catch( itk::ExceptionObject & e ) {
		cerr << "Exception caught while updating gradientMagnitudeFilter " << endl;
		cerr << e << endl;
        return EXIT_FAILURE;
	}
	ImageType::Pointer imageGradient = gradientMagnitudeFilter->GetOutput();
    
	VectorGradientFilterType::Pointer gradientMapFilter = VectorGradientFilterType::New();
	gradientMapFilter->SetInput( image );
	try {
		gradientMapFilter->Update();
	} catch( itk::ExceptionObject & e ) {
		cerr << "Exception caught while updating gradientMapFilter " << endl;
		cerr << e << endl;
        return EXIT_FAILURE;
	}
	GradientImageType::Pointer imageVectorGradient = gradientMapFilter->GetOutput();
    
	// Creation of 3D image with origin, orientation and scaling, containing original image, gradient image, vector gradient image
	ImageType::SizeType regionSize = image->GetLargestPossibleRegion().GetSize();
	ImageType::PointType origineI = image->GetOrigin();
	CVector3 origine = CVector3(origineI[0],origineI[1],origineI[2]);
	ImageType::DirectionType directionI = image->GetInverseDirection();
	CVector3	directionX = CVector3(directionI[0][0],directionI[0][1],directionI[0][2]),
    directionY = CVector3(directionI[1][0],directionI[1][1],directionI[1][2]),
    directionZ = CVector3(directionI[2][0],directionI[2][1],directionI[2][2]);
	CVector3 spacing = CVector3(spacingI[0],spacingI[1],spacingI[2]);
	Image3D* image3DGrad = new Image3D(imageVectorGradient,regionSize[0],regionSize[1],regionSize[2],origine,directionX,directionY,directionZ,spacing,typeImageFactor);
	image3DGrad->setImageOriginale(initialImage);
	image3DGrad->setCroppedImageOriginale(image);
	image3DGrad->setImageMagnitudeGradient(imageGradient);
    
    
	/******************************************
     // Initialization of Propagated Deformable Model of Spinal Cord
     ******************************************/
	int radialResolution, axialResolution, numberOfDeformIteration = 3;
	double axialStep, propagationLength = 800.0;
	// Definition of parameters for T1 and T2 images. T1 need better resolution to provide accurate segmentation.
    if (typeImageFactor == 1.0) { // T2
        radialResolution = 15;
        axialResolution = 3;
        axialStep = 6.0;
    }
    else { // T1
        radialResolution = 20; //30
        axialResolution = 3; //5
        axialStep = 6.0; //8
    }
    
    CVector3 point, normal1, normal2; // normal1 and normal2 and normals in both direction from initial point
    double stretchingFactor = 1.0;
    vector<CVector3> centerline;
    
    if (init_with_centerline)
    {
        if (verbose) cout << "Initialization - using given centerline" << endl;
        centerline = extractCenterline(inputCenterlineFilename);
        if (centerline.size() == 0) return EXIT_FAILURE;
    }
    else if (init_with_mask)
    {
        if (verbose) cout << "Initialization - using given mask" << endl;
        bool result_init = extractPointAndNormalFromMask(initMaskFilename, point, normal1, normal2);
        if (!result_init) return EXIT_FAILURE;
        if (verbose) {
            cout << "Point = " << point << endl;
            cout << "Normal 1 = " << normal1 << endl;
            cout << "Normal 2 = " << normal2 << endl;
        }
    }
    else
    {
        bool isSpinalCordDetected = false;
        int countFailure = 0;
        double step = 0.025; // step of displacement in pourcentage of the image
        do
        {
            if (verbose) cout << "Initialization - spinal cord detection on axial slices" << endl;
            Initialisation init(image,typeImageFactor);
            init.setVerbose(verbose);
            init.setGap(gapInterSlices); // gap between slices is necessary to provide good normals
            init.setRadius(radius); // approximate radius of spinal cord. This parameter is used to initiate Hough transform
            init.setNumberOfSlices(nbSlicesInitialisation);
            
            // if the initialization fails at the first position in the image, an other spinal cord detection process is launch higher.
            int d = rand() % 2; if (d==0) d = -1;
            isSpinalCordDetected = init.computeInitialParameters(initialisation+(double)countFailure*step*(double)d);
            //if (!isSpinalCordDetected) isSpinalCordDetected = init.computeInitialParameters(0.7);
            if(isSpinalCordDetected)
            {
                if (output_detection) init.savePointAsAxialImage(image,outputPath+"result_detection.png");
                
                init.getPoints(point,normal1,normal2,radius,stretchingFactor);
                if (normal2 == CVector3::ZERO) normal2 = -normal1;
                if (verbose) {
                    cout << "Initialization - Spinal Cord Detection:" << endl;
                    cout << "Point = " << point << endl;
                    cout << "Normal 1 = " << normal1 << endl;
                    cout << "Normal 2 = " << normal2 << endl;
                    cout << "Radius = " << radius << endl;
                }
                
				if(init_validation)
				{
					// Definition of discrimination surface for the validation of the spinal cord detection module
					double K_T1 = 15.7528, K_T2 = 3.2854;
					CVector3 L_T1 = CVector3(-0.0762,-2.5921,0.3472), L_T2 = CVector3(-0.0022,-1.2995,0.4909);
					CMatrix3x3 Q_T1, Q_T2;
					Q_T1[0] = 0.0; Q_T1[1] = 0.0; Q_T1[2] = 0.0; Q_T1[3] = 0.0; Q_T1[4] = 0.1476; Q_T1[5] = 0.0; Q_T1[6] = 0.0; Q_T1[7] = 0.0; Q_T1[8] = 0.6082;
					Q_T2[0] = 0.0; Q_T2[1] = 0.0; Q_T2[2] = 0.0; Q_T2[3] = 0.0; Q_T2[4] = 0.0687; Q_T2[5] = 0.0; Q_T2[6] = 0.0; Q_T2[7] = 0.0; Q_T2[8] = 0.3388;
					double contrast = 0.0, mean_distance = 0.0, std_distance = 0.0;
                
					// validation of the spinal cord detetion
					int *sizeDesired = new int[3];
					double *spacingDesired = new double[3];
					sizeDesired[0] = 61; sizeDesired[1] = 61; sizeDesired[2] = 11;
					spacingDesired[0] = 0.5; spacingDesired[1] = 0.5; spacingDesired[2] = 0.5;
					SCRegion* spinal_cord_verif = new SCRegion();
					spinal_cord_verif->setSize(sizeDesired);
					spinal_cord_verif->setSpacing(spacingDesired);
					spinal_cord_verif->setOrigin(point[0],point[1],point[2]);
					spinal_cord_verif->setNormal(normal1[0],normal1[1],normal1[2]);
					spinal_cord_verif->setFactor(typeImageFactor);
					try {
						spinal_cord_verif->readImage(initialImage);
						spinal_cord_verif->createImage();
						contrast = spinal_cord_verif->computeContrast(mean_distance,std_distance,15);
					} catch (string const& e) {
						cerr << e << endl;
						contrast = -1.0;
					}
                
					CVector3 vec = CVector3(contrast,mean_distance,std_distance);
					double discrim = 0.0;
					if (typeImageFactor == -1) // if T1
					{
						CVector3 temp = vec*Q_T1;
						double quad = 0.0;
						for(int r=0; r<3; r++) {
							quad += temp[r]*vec[r];
						}
						discrim = K_T1 + vec*L_T1 + quad;
					}
					else{
						CVector3 temp = vec*Q_T2;
						double quad = 0.0;
						for(int r=0; r<3; r++) {
							quad += temp[r]*vec[r];
						}
						discrim = K_T2 + vec*L_T2 + quad;
					}
                
					if (discrim > 0.0)
					{
						countFailure++;
						isSpinalCordDetected = false;
						if (verbose) cout << "WARNING: Bad initialization. Attempt to locate spinal cord at an other level." << endl << endl;
					}
					else
					{
						isSpinalCordDetected = true;
					}
					delete sizeDesired, spacingDesired, spinal_cord_verif;
				}
				else {
					isSpinalCordDetected = true;
				}
            } else {
                countFailure++;
            }
        }
        while (!isSpinalCordDetected && countFailure<10);
        if (!isSpinalCordDetected)
        {
            cerr << "Error: Enable to detect the spinal cord. Please provide the initial position and orientation of the spinal cord (-init, -init-mask)" << endl;
            return EXIT_FAILURE;
        }
    }
    
	/******************************************
     // Launch of Propagated Deformable Model of Spinal Cord. Propagation have to be done in both direction
     ******************************************/
	PropagatedDeformableModel* prop = new PropagatedDeformableModel(radialResolution,axialResolution,radius,numberOfDeformIteration,numberOfPropagationIteration,axialStep,propagationLength);
    if (maxDeformation != 0.0) prop->setMaxDeformation(maxDeformation);
    if (maxArea != 0.0) prop->setMaxArea(maxArea);
    prop->setMinContrast(minContrast);
	prop->setInitialPointAndNormals(point,normal1,normal2);
    prop->setStretchingFactor(stretchingFactor);
	prop->setUpAndDownLimits(downSlice,upSlice);
	prop->setImage3D(image3DGrad);
    if (init_with_centerline) {
        prop->propagationWithCenterline();
        for (unsigned int k=0; k<centerline.size(); k++) prop->addPointToCenterline(centerline[k]);
        if (initialisation <= 1) prop->setInitPosition(initialisation);
    }
    prop->setVerbose(verbose);
	prop->computeMeshInitial();
    if (output_init_tube) {
        SpinalCord *tube1 = prop->getInitialMesh(), *tube2 = prop->getInverseInitialMesh();
        tube1->save(outputPath+"InitialTube1.vtk");
        tube2->save(outputPath+"InitialTube2.vtk");
    }
	
	prop->adaptationGlobale(); // Propagation
	// Saving low resolution mesh
	if (low_res_mesh)
    {
        SpinalCord *meshOutputLowResolution = prop->getOutput();
        meshOutputLowResolution->save(outputPath+"segmentation_mesh_low_resolution.vtk",initialImage);
        //meshOutput->computeCenterline(true,path+"LowResolution");
        //meshOutput->computeCrossSectionalArea(true,path+"LowResolution");
        //image3DGrad->TransformMeshToBinaryImage(meshOutput,path+"LowResolution",orientationFilter.getInitialImageOrientation());
        //meshOutput->saveCenterlineAsBinaryImage(initialImage,path+"LowResolution",orientationFilter.getInitialImageOrientation());
    }
    
	/******************************************
     // High Resolution Deformation
     ******************************************/
    prop->rafinementGlobal();
	SpinalCord* meshOutputFinal = prop->getOutputFinal();
	if (output_mesh) meshOutputFinal->save(outputFilenameMesh,initialImage);
	if (output_centerline_coord) meshOutputFinal->computeCenterline(true,outputFilenameCenterline,true);
	if (output_cross) meshOutputFinal->computeCrossSectionalArea(true,outputFilenameAreas,true,image3DGrad);
	image3DGrad->TransformMeshToBinaryImage(meshOutputFinal,outputFilenameBinary,orientationFilter.getInitialImageOrientation());
	if (output_centerline_binary) meshOutputFinal->saveCenterlineAsBinaryImage(initialImage,outputFilenameCenterlineBinary,orientationFilter.getInitialImageOrientation());
    
	if (verbose) {
		double lengthPropagation = meshOutputFinal->getLength();
		cout << "Total propagation length = " << lengthPropagation << " mm" << endl;
	}
    
    
    
    if (CSF_segmentation)
    {
        /******************************************
         // Launch of Propagated Deformable Model on the CSF. Propagation have to be done in both direction
         ******************************************/
        double factor_CSF = 2;
        PropagatedDeformableModel* prop_CSF = new PropagatedDeformableModel(radialResolution,axialResolution,radius*factor_CSF,numberOfDeformIteration,numberOfPropagationIteration,axialStep,propagationLength);
        if (maxDeformation != 0.0) prop_CSF->setMaxDeformation(maxDeformation*factor_CSF);
        if (maxArea != 0.0) prop_CSF->setMaxArea(maxArea*factor_CSF*2);
        prop->setMinContrast(minContrast);
        prop_CSF->setInitialPointAndNormals(point,normal1,normal2);
        prop_CSF->setStretchingFactor(stretchingFactor);
        prop_CSF->setUpAndDownLimits(downSlice,upSlice);
        image3DGrad->setTypeImageFactor(-image3DGrad->getTypeImageFactor());
        prop_CSF->setImage3D(image3DGrad);
        if (init_with_centerline) {
            prop_CSF->propagationWithCenterline();
            for (unsigned int k=0; k<centerline.size(); k++) prop->addPointToCenterline(centerline[k]);
            if (initialisation <= 1) prop->setInitPosition(initialisation);
        }
        prop_CSF->setVerbose(verbose);
        prop_CSF->computeMeshInitial();
        if (output_init_tube) {
            SpinalCord *tube1 = prop_CSF->getInitialMesh(), *tube2 = prop_CSF->getInverseInitialMesh();
            tube1->save(outputPath+"InitialTubeCSF1.vtk");
            tube2->save(outputPath+"InitialTubeCSF2.vtk");
        }
        
        prop_CSF->adaptationGlobale(); // Propagation
        // Saving low resolution mesh
        if (low_res_mesh)
        {
            SpinalCord *meshOutputLowResolution = prop_CSF->getOutput();
            meshOutputLowResolution->save(outputPath+"segmentation_CSF_mesh_low_resolution.vtk",initialImage);
            //meshOutput->computeCenterline(true,path+"LowResolution");
            //meshOutput->computeCrossSectionalArea(true,path+"LowResolution");
            //image3DGrad->TransformMeshToBinaryImage(meshOutput,path+"LowResolution",orientationFilter.getInitialImageOrientation());
            //meshOutput->saveCenterlineAsBinaryImage(initialImage,path+"LowResolution",orientationFilter.getInitialImageOrientation());
        }
        
        /******************************************
         // High Resolution Deformation
         ******************************************/
        prop_CSF->rafinementGlobal();
        SpinalCord* meshOutputFinal = prop_CSF->getOutputFinal();
        if (output_mesh) meshOutputFinal->save(outputFilenameMeshCSF,initialImage);
        if (output_cross) meshOutputFinal->computeCrossSectionalArea(true,outputFilenameAreasCSF,true,image3DGrad);
        image3DGrad->TransformMeshToBinaryImage(meshOutputFinal,outputFilenameBinaryCSF,orientationFilter.getInitialImageOrientation());
        
        if (verbose) {
            double lengthPropagation = meshOutputFinal->getLength();
            cout << "Total propagation length = " << lengthPropagation << " mm" << endl;
        }
        delete prop_CSF;
    }
    
    
	
	delete image3DGrad, prop;
    return EXIT_SUCCESS;
}

bool extractPointAndNormalFromMask(string filename, CVector3 &point, CVector3 &normal1, CVector3 &normal2)
{
    ReaderType::Pointer reader = ReaderType::New();
	itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
	reader->SetImageIO(io);
	reader->SetFileName(filename);
    try {
        reader->Update();
    } catch( itk::ExceptionObject & e ) {
        cerr << "Exception caught while reading input image " << endl;
        cerr << e << endl;
        return false;
    }
    ImageType::Pointer image = reader->GetOutput();
    
    vector<CVector3> result;
    ImageType::IndexType ind;
    itk::Point<double,3> pnt;
    ImageIterator it( image, image->GetRequestedRegion() );
    it.GoToBegin();
    while(!it.IsAtEnd())
    {
        if (it.Get()!=0)
        {
            ind = it.GetIndex();
            image->TransformIndexToPhysicalPoint(ind, pnt);
            bool added = false;
            if (result.size() == 0) {
                result.push_back(CVector3(pnt[0],pnt[1],pnt[2]));
                added = true;
            }
            else {
                for (vector<CVector3>::iterator it=result.begin(); it!=result.end(); it++) {
                    if (pnt[2] < (*it)[2]) {
                        result.insert(it, CVector3(pnt[0],pnt[1],pnt[2]));
                        added = true;
                        break;
                    }
                }
            }
            if (!added) result.push_back(CVector3(pnt[0],pnt[1],pnt[2]));
        }
        ++it;
    }
    
    if (result.size() != 3) {
        cerr << "Error: Not enough of too many points in the binary mask. Number of point needed = 3. Detected points = " << result.size() << endl;
        return false;
    }
    point = result[1];
    normal1 = (result[0]-result[1]).Normalize();
    normal2 = (result[2]-result[1]).Normalize();
    
    return true;
}

vector<CVector3> extractCenterline(string filename)
{
    vector<CVector3> result;
    
    string nii=".nii", niigz=".nii.gz", txt=".txt", suffix="";
    size_t pos_niigz = filename.find(niigz), pos_nii = filename.find(nii), pos_txt = filename.find(txt);
    if (pos_niigz != string::npos || pos_nii != string::npos)
    {
        ReaderType::Pointer reader = ReaderType::New();
        itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
        reader->SetImageIO(io);
        reader->SetFileName(filename);
        try {
            reader->Update();
        } catch( itk::ExceptionObject & e ) {
            cerr << "Exception caught while reading centerline input image " << endl;
            cerr << e << endl;
        }
        ImageType::Pointer image_centerline = reader->GetOutput();
        
        OrientImage<ImageType> orientationFilter;
        orientationFilter.setInputImage(image_centerline);
        orientationFilter.orientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL);
        image_centerline = orientationFilter.getOutputImage();
        
        ImageType::IndexType ind;
        itk::Point<double,3> point;
        ImageIterator it( image_centerline, image_centerline->GetRequestedRegion() );
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            if (it.Get()!=0)
            {
                ind = it.GetIndex();
                image_centerline->TransformIndexToPhysicalPoint(ind, point);
                bool added = false;
                if (result.size() == 0) {
                    result.push_back(CVector3(point[0],point[1],point[2]));
                    added = true;
                }
                else {
                    for (vector<CVector3>::iterator it=result.begin(); it!=result.end(); it++) {
                        if (point[2] < (*it)[2]) {
                            result.insert(it, CVector3(point[0],point[1],point[2]));
                            added = true;
                            break;
                        }
                    }
                }
                if (!added) result.push_back(CVector3(point[0],point[1],point[2]));
            }
            ++it;
        }
        
        /*// spline approximation to produce correct centerline
        
        double range = result.size()/4.0 ;
        const unsigned int ParametricDimension = 1; const unsigned int DataDimension = 3;
        typedef double RealType;
        typedef itk::Vector<RealType, DataDimension> VectorType; typedef itk::Image<VectorType, ParametricDimension> ImageType;
        typedef itk::PointSet <VectorType , ParametricDimension > PointSetType; PointSetType::Pointer pointSet = PointSetType::New();
        // Sample the helix.
        int nb = result.size();
        for (unsigned long i=0; i<nb; i++) {
            PointSetType::PointType point; point[0] = (double)i/(double)(nb-1);
            pointSet ->SetPoint( i, point );
            VectorType V;
            V[0] = result[i][0]; V[1] = result[i][1]; V[2] = result[i][2];
            pointSet ->SetPointData( i, V );
        }
        
        typedef itk::BSplineScatteredDataPointSetToImageFilter <PointSetType , ImageType > FilterType;
        FilterType::Pointer filter = FilterType::New();
        ImageType::SpacingType spacing; spacing.Fill( 1.0 ); ImageType::SizeType size; size.Fill( 2.0); ImageType::PointType origin; origin.Fill( 0.0 );
        ImageType::RegionType region(size); FilterType::ArrayType closedim; closedim.Fill(0);
        filter->SetSize( size ); filter->SetOrigin( origin ); filter->SetSpacing( spacing ); filter->SetInput( pointSet );
        int splineOrder = 3; filter->SetSplineOrder( splineOrder ); FilterType::ArrayType ncps;
        ncps.Fill( splineOrder + 1 ); filter->SetNumberOfControlPoints( ncps ); filter->SetNumberOfLevels( 5 ); filter->SetGenerateOutputImage( false );
        try
        { filter->Update();
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while reading input image " << std::endl;
            std::cerr << e << std::endl;
        }
        
        typedef itk::BSplineControlPointImageFunction < ImageType, double > BSplineType;
        BSplineType::Pointer bspline = BSplineType::New();
        bspline->SetSplineOrder(filter->GetSplineOrder());
        bspline->SetOrigin(origin);
        bspline->SetSpacing(spacing);
        bspline->SetSize(size);
        bspline->SetInputImage(filter->GetPhiLattice());
        
        result.clear();
        for (double i=0; i<=2.0*range; i++) {
            PointSetType::PointType point; point[0] = i/(2.0*range);
            VectorType V = bspline->Evaluate( point );
            result.push_back(CVector3(V[0],V[1],V[2]));
        }*/
        
    }
    else if (pos_txt != string::npos)
    {
        ifstream myfile;
        string l;
        double x, y, z;
        CVector3 point, pointPrecedent;
        int i = 0;
        myfile.open(filename.c_str());
        if (myfile.is_open())
        {
            while (myfile.good())
            {
                getline(myfile,l);
                stringstream ss(l);
                ss >> x >> z >> y;
                point = CVector3(x,y,z);
                if ((point-pointPrecedent).Norm() > 0) {
                    pointPrecedent = point;
                    //point[1] = -point[1];
                    result.push_back(point);
                }
                i++;
            }
        }
        myfile.close();
    }
    else cerr << "Error: Centerline input file not supported" << endl;
    
    return result;
}