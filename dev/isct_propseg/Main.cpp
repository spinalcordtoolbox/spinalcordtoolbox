/*! \file Main.cpp
 * \mainpage sct_propseg
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
 * \code sct_propseg -i <inputfilename> -o <outputfolderpath> -t <imagetype> [options] \endcode
 * 
 * \section input Input parameters
 *
 * MANDATORY ARGUMENTS:
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
#include <itkIntensityWindowingImageFilter.h>
#include <itkFileTools.h>
#include <itkPointSet.h>
#include <itkBSplineScatteredDataPointSetToImageFilter.h>
#include <itkBSplineControlPointImageFunction.h>
#include <itkImageSeriesReader.h>
#include <itkMedianImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
//#include <itkGDCMImageIO.h>
//#include <itkGDCMSeriesFileNames.h>

using namespace std;

// Type definitions
typedef itk::Image< double, 3 >	ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileWriter<ImageType> WriterType;
typedef itk::ImageRegionConstIterator<ImageType> ImageIterator;
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
vector<CVector3> extractPointsFromMask(string filename);

// Small procedure to manage length of string
string StrPad(string original, size_t charCount, string prefix="")
{
	if (original.size() < charCount)
    	original.resize(charCount,' ');
	else {
		string tempString = "";
		int nbString = (original.size()/charCount)+1;
		for (int i=0; i<nbString; i++) {
			string subString = original.substr(i*charCount,charCount);
			if (i != nbString-1)
				subString += "\n";
			if (i != 0) subString = prefix+subString;
			tempString += subString;
		}
		original = tempString;
	}
    return original;
}

void help()
{
    cout << "sct_propseg - Version 1.1 (2015-03-24)" << endl;
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \nPart of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox> \nAuthor : Benjamin De Leener" << endl << endl;
    
    cout << "DESCRIPTION" << endl;
	cout << "This program segments automatically the spinal cord on T1- and T2-weighted images, for any field of view. You must provide the type of contrast, the image as well as the output folder path." << endl;
	cout << "Initialization is provided by a spinal cord detection module based on the elliptical Hough transform on multiple axial slices. The result of the detection is available as a PNG image using option -detection-display." << endl;
	cout << "Parameters of the spinal cord detection are :" << endl << " - the position (in inferior-superior direction) of the initialization" << endl << " - the number of axial slices" << endl << " - the gap (in pixel) between two axial slices" << endl << " - the approximate radius of the spinal cord" << endl << endl;
	
	cout << "Primary output is the binary mask of the spinal cord segmentation. This method must provide VTK triangular mesh of the segmentation (option -mesh). Spinal cord centerline is available as a binary image (-centerline-binary) or a text file with coordinates in world referential (-centerline-coord)." << endl;
	cout << "Cross-sectional areas along the spinal cord can be available (-cross)." << endl;
    
    cout << "Several tips on segmentation correction can be found on the \"Correction Tips\" page of the documentation while advices on parameters adjustments can be found on the \"Parameters\" page." << endl;
    cout << "If the segmentation fails at some location (e.g. due to poor contrast between spinal cord and CSF), edit your anatomical image (e.g. with fslview) and manually enhance the contrast by adding bright values around the spinal cord for T2-weighted images (dark values for T1-weighted). Then, launch the segmentation again." << endl;
	
    cout << "USAGE" << endl;
    cout << "  sct_propseg -i <inputfilename> -o <outputfolderpath> -t <imagetype> [options]" << endl << endl;
    cout << endl << "WARNING: be sure your image has the correct type !!" << endl << endl;
    
    cout << "MANDATORY ARGUMENTS" << endl;
    cout << StrPad("  -i <inputfilename>",30) << StrPad("no default",70,StrPad("",30)) << endl;
    //cout << "\t-i-dicom <inputfolderpath> \t (replace -i, read DICOM series, output still in NIFTI)" << endl;
    cout << StrPad("  -o <outputfolderpath>",30) << StrPad("default is current folder",70,StrPad("",30)) << endl;
    cout << StrPad("  -t {t1,t2}",30) << StrPad("string, type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark, no default",70,StrPad("",30)) << endl;
    cout << endl;
	
	// Output files
	cout << "OPTIONAL ARGUMENTS" << endl;
    cout << "General options" << endl;
    cout << StrPad("  -down <down_slice>",30) << StrPad("int, down limit of the propagation, default is 0",70,StrPad("",30)) << endl;
    cout << StrPad("  -up <up_slice>",30) << StrPad("int, up limit of the propagation, default is the higher slice of the image",70,StrPad("",30)) << endl;
    cout << StrPad("  -verbose",30) << StrPad("display on",70,StrPad("",30)) << endl;
    cout << StrPad("  -help",30) << endl;
	cout << endl;
    
    cout << "Output options" << endl;
    cout << StrPad("  -mesh",30) << StrPad("output: mesh of the spinal cord segmentation",70,StrPad("",30)) << endl;
    cout << StrPad("  -centerline-binary",30) << StrPad("output: centerline as a binary image",70,StrPad("",30)) << endl;
    cout << StrPad("  -CSF",30) << StrPad("output: CSF segmentation",70,StrPad("",30)) << endl;
	cout << StrPad("  -centerline-coord",30) << StrPad("output: centerline as world coordinates",70,StrPad("",30)) << endl;
	cout << StrPad("  -cross",30) << StrPad("output: cross-sectional areas",70,StrPad("",30)) << endl;
    cout << StrPad("  -init-tube",30) << StrPad("output: initial tubular meshes",70,StrPad("",30)) << endl;
    cout << StrPad("  -low-resolution-mesh",30) << StrPad("output: low-resolution mesh",70,StrPad("",30)) << endl;
    cout << StrPad("  -detect-nii",30) << StrPad("output of spinal cord detection as a nifti image",70,StrPad("",30)) << endl;
    cout << StrPad("  -detect-png",30) << StrPad("output of spinal cord detection as a PNG image",70,StrPad("",30)) << endl;
	cout << endl;
	
	cout << "Options helping the segmentation" << endl;
	cout << StrPad("  -init-centerline <filename>",30) << StrPad("filename of centerline to use for the propagation, format .txt or .nii, see file structure in documentation",70,StrPad("",30)) << endl;
    cout << StrPad("  -init <init_position>",30) << StrPad("int, axial slice where the propagation starts, default is middle axial slice",70,StrPad("",30)) << endl;
    cout << StrPad("  -init-mask <filename>",30) << StrPad("string, mask containing three center of the spinal cord, used to initiate the propagation",70,StrPad("",30)) << endl;
	cout << StrPad("  -radius <radius>",30) << StrPad("double, approximate radius of the spinal cord, default is 4 mm",70,StrPad("",30)) << endl;
	cout << endl;
	cout << StrPad("  -detect-n <numberslice>",30) << StrPad("int, number of axial slices computed in the detection process, default is 4",70,StrPad("",30)) << endl;
	cout << StrPad("  -detect-gap <gap>",30) << StrPad("int, gap in Z direction for the detection process, default is 4",70,StrPad("",30)) << endl;
	cout << StrPad("  -init-validation",30) << StrPad("enable validation on spinal cord detection based on discriminant analysis",70,StrPad("",30)) << endl;
    cout << StrPad("  -nbiter <number>",30) << StrPad("int, stop condition: number of iteration for the propagation for both direction, default is 200",70,StrPad("",30)) << endl;
    cout << StrPad("  -max-area <number>",30) << StrPad("double, in mm^2, stop condition: maximum cross-sectional area, default is 120 mm^2",70,StrPad("",30)) << endl;
    cout << StrPad("  -max-deformation <number>",30) << StrPad("double, in mm, stop condition: maximum deformation per iteration, default is 2.5 mm",70,StrPad("",30)) << endl;
    cout << StrPad("  -min-contrast <number>",30) << StrPad("double, in intensity value, stop condition: minimum local SC/CSF contrast, default is 50",70,StrPad("",30)) << endl;
	cout << StrPad("  -d <number>",30) << StrPad("double, trade-off between distance of most promising point and feature strength, default depend on the contrast. Range of values from 0 to 50. 15-25 values show good results.",70,StrPad("",30)) << endl;
	cout << endl;
}

int main(int argc, char *argv[])
{
    srand (time(NULL));
    
	if (argc == 1)
    {
        help();
        return EXIT_FAILURE;
    }
    string inputFilename = "", outputPath = "", outputFilenameBinary = "", outputFilenameMesh = "", outputFilenameBinaryCSF = "", outputFilenameMeshCSF = "", outputFilenameAreas = "", outputFilenameAreasCSF = "", outputFilenameCenterline = "", outputFilenameCenterlineBinary = "", inputCenterlineFilename = "", initMaskFilename = "", maskCorrectionFilename = "";
    double typeImageFactor = 0.0, initialisation = 0.5;
    int downSlice = -10000, upSlice = 10000;
    string suffix;
	bool input_dicom = false, output_detection = false, output_detection_nii = false, output_mesh = false, output_centerline_binary = false, output_centerline_coord = false, output_cross = false, init_with_centerline = false, init_with_mask = false, verbose = false, output_init_tube = false, completeCenterline = false, init_validation = false, low_res_mesh = false, CSF_segmentation = false;
	int gapInterSlices = 4, nbSlicesInitialisation = 5;
	double radius = 4.0;
    int numberOfPropagationIteration = 200;
    double maxDeformation = 0.0, maxArea = 0.0, minContrast = 50.0, tradeoff_d;
	bool tradeoff_d_bool = false;
	bool bool_maskCorrectionFilename = false;
	double distance_search = -1.0;
	double alpha_param = -1.0;
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
        else if (strcmp(argv[i],"-detect-nii")==0) {
            output_detection_nii = true;
        }
		else if (strcmp(argv[i],"-detect-png")==0) {
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
		else if (strcmp(argv[i],"-radius")==0) {
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
        else if (strcmp(argv[i],"-mask-correction")==0) {
            i++;
            maskCorrectionFilename = argv[i];
            bool_maskCorrectionFilename = true;
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
		else if (strcmp(argv[i],"-d")==0) {
            i++;
            tradeoff_d = atof(argv[i]);
			tradeoff_d_bool = true;
        }
        else if (strcmp(argv[i],"-CSF")==0) {
            CSF_segmentation = true;
            if (maxArea == 0.0) maxArea = 120;
            if (maxDeformation == 0.0) maxDeformation = 2.5;
        }
        else if (strcmp(argv[i],"-dsearch")==0) {
            i++;
            distance_search = atof(argv[i]);
        }
        else if (strcmp(argv[i],"-alpha")==0) {
            i++;
            alpha_param = atof(argv[i]);
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
    

    
    // Extract the input file name
    unsigned found_slash = inputFilename.find_last_of("/\\");
    string inputFilename_nameonly = inputFilename.substr(found_slash+1);

    // output files must have the same extension as input file
    string nii=".nii", niigz=".nii.gz"; suffix=niigz;
    size_t pos = inputFilename_nameonly.find(niigz);
    if (pos == string::npos) {
        pos = inputFilename_nameonly.find(nii);
        suffix = nii;
    }
    inputFilename_nameonly = inputFilename_nameonly.substr(0,pos);

    // Check if output folder ends with /
    if (outputPath!="" && outputPath.compare(outputPath.length()-1,1,"/")) outputPath += "/"; // add "/" if missing
    
    // Set output filenames
    outputFilenameBinary = outputPath+inputFilename_nameonly+"_seg"+suffix;
    outputFilenameMesh = outputPath+inputFilename_nameonly+"_mesh.vtk";
    outputFilenameBinaryCSF = outputPath+inputFilename_nameonly+"_CSF_seg"+suffix;
    outputFilenameMeshCSF = outputPath+inputFilename_nameonly+"_CSF_mesh.vtk";
    outputFilenameAreas = outputPath+inputFilename_nameonly+"_cross_sectional_areas.txt";
    outputFilenameAreasCSF = outputPath+inputFilename_nameonly+"_cross_sectional_areas_CSF.txt";
    outputFilenameCenterline = outputPath+inputFilename_nameonly+"_centerline.txt";
    outputFilenameCenterlineBinary = outputPath+inputFilename_nameonly+"_centerline"+suffix;
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
    
	// Robust intensity normalization
    // Min and max values are detected after median filter.
    typedef itk::MedianImageFilter< ImageType, ImageType > MedianFilterType;
    MedianFilterType::Pointer medianFilter = MedianFilterType::New();
    medianFilter->SetInput(image);
    MedianFilterType::InputSizeType radiusMedianFilter;
    radiusMedianFilter.Fill(2);
    medianFilter->SetRadius(radiusMedianFilter);
    medianFilter->Update();
    
    typedef itk::MinimumMaximumImageCalculator< ImageType > MinMaxCalculatorType;
    MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
    minMaxCalculator->SetImage(medianFilter->GetOutput());
    minMaxCalculator->Compute();
    
    typedef itk::IntensityWindowingImageFilter< ImageType, ImageType > RescaleFilterType;
	RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
	rescaleFilter->SetInput(image);
    rescaleFilter->SetWindowMinimum(minMaxCalculator->GetMinimum());
    rescaleFilter->SetWindowMaximum(minMaxCalculator->GetMaximum());
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
            init.setVerbose(false);
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
                if (output_detection_nii) init.savePointAsBinaryImage(image,outputPath+inputFilename_nameonly+"_detection"+suffix, orientationFilter.getInitialImageOrientation());
                
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
            cerr << "Error: Unable to detect the spinal cord. Please provide the initial position and orientation of the spinal cord (-init, -init-mask)" << endl;
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
	prop->setUpAndDownLimits(downSlice-5,upSlice+5);
	prop->setImage3D(image3DGrad);
	if (distance_search != -1.0)
	{
	    prop->changedParameters();
	    prop->setLineSearch(distance_search);
	}
	if (alpha_param != -1.0)
	{
	    prop->changedParameters();
	    prop->setAlpha(alpha_param);
	}
    if (init_with_centerline) {
        prop->propagationWithCenterline();
        for (unsigned int k=0; k<centerline.size(); k++) prop->addPointToCenterline(centerline[k]);
        if (initialisation <= 1) prop->setInitPosition(initialisation);
    }
	if (tradeoff_d_bool) {
		prop->setTradeOffDistanceFeature(tradeoff_d);
	}
    prop->setVerbose(false);
	prop->computeMeshInitial();
    if (output_init_tube) {
        SpinalCord *tube1 = prop->getInitialMesh(), *tube2 = prop->getInverseInitialMesh();
        tube1->save(outputPath+"InitialTube1.vtk");
        tube2->save(outputPath+"InitialTube2.vtk");
    }

    /******************************************
    // Extraction of points from correction mask, if provided
    ******************************************/
    vector<CVector3> points_mask_correction;
    if (bool_maskCorrectionFilename)
    {
        points_mask_correction = extractPointsFromMask(maskCorrectionFilename);
        prop->addCorrectionPoints(points_mask_correction);
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
	if (upSlice != 10000 || downSlice != -10000)
	{

	    vector<CVector3> centerline_spinalcord = meshOutputFinal->computeCenterline();
	    CVector3* upSlicePoint = new CVector3(image3DGrad->TransformIndexToPhysicalPoint(CVector3(0,upSlice+1,0)));
	    (*upSlicePoint)[2] = (*upSlicePoint)[2] - 0.45*spacingI[2];
	    CVector3* upSliceNormal = new CVector3(0,0,1);
	    CVector3* downSlicePoint = new CVector3(image3DGrad->TransformIndexToPhysicalPoint(CVector3(0,downSlice-1,0)));
	    (*downSlicePoint)[2] = (*downSlicePoint)[2] + 0.55*spacingI[2];
	    CVector3* downSliceNormal = new CVector3(0,0,-1);
	    image3DGrad->TransformMeshToBinaryImage(meshOutputFinal,outputFilenameBinary,orientationFilter.getInitialImageOrientation(), false, true, upSlicePoint, upSliceNormal, downSlicePoint, downSliceNormal);
	}
	else
	{
	    image3DGrad->TransformMeshToBinaryImage(meshOutputFinal,outputFilenameBinary,orientationFilter.getInitialImageOrientation());
	}
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
        prop_CSF->setMinContrast(minContrast);
        prop_CSF->setInitialPointAndNormals(point,normal1,normal2);
        prop_CSF->setStretchingFactor(stretchingFactor);
        prop_CSF->setUpAndDownLimits(downSlice,upSlice);
        image3DGrad->setTypeImageFactor(-image3DGrad->getTypeImageFactor());
        prop_CSF->setImage3D(image3DGrad);
        if (init_with_centerline)
        {
            prop_CSF->propagationWithCenterline();
            for (unsigned int k=0; k<centerline.size(); k++)
            {
                //cout << centerline[k] << endl;
                prop_CSF->addPointToCenterline(centerline[k]);
            }
            //cout << endl;
            if (initialisation <= 1) prop_CSF->setInitPosition(initialisation);
        }
        else
        {
            prop_CSF->propagationWithCenterline();
            centerline = extractCenterline(outputFilenameBinary);
            for (unsigned int k=0; k<centerline.size(); k++)
            {
                //cout << centerline[k] << endl;
                prop_CSF->addPointToCenterline(centerline[k]);
            }
            //cout << endl;
            if (initialisation <= 1) prop_CSF->setInitPosition(initialisation);
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
        image3DGrad->TransformMeshToBinaryImage(meshOutputFinal,outputFilenameBinaryCSF,orientationFilter.getInitialImageOrientation(),true);
        
        if (verbose) {
            double lengthPropagation = meshOutputFinal->getLength();
            cout << "Total propagation length = " << lengthPropagation << " mm" << endl;
        }
        delete prop_CSF;
    }
    
    if (verbose) {
        cout << endl << "Segmentation finished. To view results, type:" << endl;
        cout << "fslview " << inputFilename << " " << outputFilenameBinary << " &" << endl;
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
        cerr << "Error: Not enough or too many points in the binary mask. Number of point needed = 3. Detected points = " << result.size() << endl;
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
        vector<int> result_no_double;
        vector<int> result_no_double_number;
        while(!it.IsAtEnd())
        {
            double pixel_value = it.Get();
            if (pixel_value!=0)
            {
                ind = it.GetIndex();
                image_centerline->TransformIndexToPhysicalPoint(ind, point);
                CVector3 point_cvec = CVector3(point[0],point[1],point[2]);

                bool hasDouble = false;
                int double_point = 0;
                for (int j=0; j<result_no_double.size(); j++)
                {
                    if (ind[1] == result_no_double[j])
                    {
                        hasDouble = true;
                        double_point = j;
                        break;
                    }
                }

                if (hasDouble)
                {
                    result[double_point] += point_cvec;
                    result_no_double_number[double_point] += pixel_value;
                }
                else
                {
                    bool added = false;
                    if (result.size() == 0) {
                        result.push_back(point_cvec);
                        result_no_double.push_back(ind[1]);
                        result_no_double_number.push_back(pixel_value);
                        added = true;
                    }
                    else
                    {
                        vector<CVector3> result_average = result;
                        for (int j=0; j<result_average.size(); j++)
                        {
                            result_average[j] /= result_no_double_number[j];
                        }
                        for (int k=0; k<result_average.size(); k++)
                        {
                            if (point[2] < result_average[k][2])
                            {
                                result.insert(result.begin()+k, point_cvec);
                                result_no_double.insert(result_no_double.begin()+k, ind[1]);
                                result_no_double_number.insert(result_no_double_number.begin()+k, pixel_value);
                                added = true;
                                break;
                            }
                        }
                    }
                    if (!added)
                    {
                        result.push_back(point_cvec);
                        result_no_double.push_back(ind[1]);
                        result_no_double_number.push_back(pixel_value);
                    }
                }
            }
            ++it;
        }

        for (int j=0; j<result.size(); j++)
        {
            result[j] /= result_no_double_number[j];
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

vector<CVector3> extractPointsFromMask(string filename)
{
    vector<CVector3> result;

    ReaderType::Pointer reader = ReaderType::New();
	itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
	reader->SetImageIO(io);
	reader->SetFileName(filename);
    try {
        reader->Update();
    } catch( itk::ExceptionObject & e ) {
        cerr << "Exception caught while reading input image " << endl;
        cerr << e << endl;
    }
    ImageType::Pointer image = reader->GetOutput();

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
            result.push_back(CVector3(pnt[0],pnt[1],pnt[2]));
        }
        ++it;
    }

    return result;
}
