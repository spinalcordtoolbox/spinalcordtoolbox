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
#include <sstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <vector>

// ITK libraries
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>
#include <itkGradientMagnitudeImageFilter.h>
#include <itkGradientImageFilter.h>
#include "itkGradientVectorFlowImageFilter.h"
#include <itkImageAlgorithm.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkFileTools.h>
#include <itkPointSet.h>
#include <itkBSplineScatteredDataPointSetToImageFilter.h>
#include <itkBSplineControlPointImageFunction.h>
#include <itkImageSeriesReader.h>

#include <itkHessianRecursiveGaussianImageFilter.h>
#include <itkMultiScaleHessianBasedMeasureImageFilter.h>
#include <itkHessianToObjectnessMeasureImageFilter.h>
//#include <itkGradientImageFilter.h>
#include <itkGradientVectorFlowImageFilter.h> // ITK version
#include "itkRecursiveGaussianImageFilter.h"
#include "itkImageRegionIterator.h"
#include <itkSymmetricSecondRankTensor.h>
#include "itkHessian3DToVesselnessMeasureImageFilter.h"
#include <itkStatisticsImageFilter.h>
#include "itkTileImageFilter.h"
#include "itkPermuteAxesImageFilter.h"
#include <itkDiscreteGaussianImageFilter.h>

using namespace std;

// Type definitions
typedef itk::Image< double, 3 >	ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageRegionConstIterator<ImageType> ImageIterator;
typedef itk::RescaleIntensityImageFilter< ImageType, ImageType > RescaleFilterType;
typedef itk::CovariantVector< double, 3 > GradientPixelType;
typedef itk::Image< GradientPixelType, 3 > GradientImageType;
typedef itk::GradientImageFilter< ImageType, float, double, GradientImageType > VectorGradientFilterType;
typedef itk::GradientVectorFlowImageFilter< GradientImageType, GradientImageType >  GradientVectorFlowFilterType;
typedef itk::GradientMagnitudeImageFilter< ImageType, ImageType > GradientMFilterType;

typedef itk::Image< unsigned char, 3 >	BinaryImageType;
typedef itk::ImageFileReader<BinaryImageType> BinaryReaderType;
typedef itk::ImageRegionConstIterator<BinaryImageType> BinaryImageIterator;

typedef itk::ImageFileWriter< ImageType >     WriterType;

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

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

void help()
{
    cout << "sct_propseg - Version 1.0.3 (2014-11-28)" << endl;
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
    cout << StrPad("  -o <outputfolderpath>",30) << StrPad("default is current folder",70,StrPad("",30)) << endl;
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
    cout << StrPad("  -K <number>",30) << StrPad("double, trade-off between GGVF field smoothness and gradient conformity. Range of values from 0.01 to 2000.",70,StrPad("",30)) << endl;
    cout << StrPad("  -iter-GGVF <number>",30) << StrPad("int, Number of iteration for GGVF filter. Default is 2.",70,StrPad("",30)) << endl;
	cout << endl;
}

ImageType::Pointer vesselnessFilter(ImageType::Pointer im, float typeImageFactor_, double alpha, double beta, double gamma, double sigmaMinimum, double sigmaMaximum, unsigned int numberOfSigmaSteps, double sigmaDistance)
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

int main(int argc, char *argv[])
{
    printf("STARTING\n");
    srand (time(NULL));
    
	if (argc == 1)
    {
        help();
        return EXIT_FAILURE;
    }
    
    // Initialization of parameters
    string inputFilename = "", outputPath = "", outputFilenameBinary = "", outputFilenameMesh = "", outputFilenameBinaryCSF = "", outputFilenameMeshCSF = "", outputFilenameAreas = "", outputFilenameAreasCSF = "", outputFilenameCenterline = "", outputFilenameCenterlineBinary = "", inputCenterlineFilename = "", initMaskFilename = "";
    double typeImageFactor = 0.0, initialisation = 0.5;
    int downSlice = -10000, upSlice = 10000;
    string suffix;
	bool input_dicom = false, output_detection = false, output_detection_nii = false, output_mesh = false, output_centerline_binary = false, output_centerline_coord = false, output_cross = false, init_with_centerline = false, init_with_mask = false, verbose = false, output_init_tube = false, completeCenterline = false, init_validation = false, low_res_mesh = false, CSF_segmentation = false;
	int gapInterSlices = 4, nbSlicesInitialisation = 5;
	double radius = 4.0;
    int numberOfPropagationIteration = 200;
    double maxDeformation = 0.0, maxArea = 0.0, minContrast = 50.0, tradeoff_d = 25, tradeoff_K = 200;
	bool tradeoff_d_bool = false;
    int nbiter_GGVF = 2;
    
    // Initialization with Vesselness Filter and Minimal Path
    bool init_with_minimalpath = true;
    double minimalPath_alpha=0.1;
    double minimalPath_beta=1.0;
    double minimalPath_gamma=5.0;
    double minimalPath_sigmaMinimum=1.5;
    double minimalPath_sigmaMaximum=4.5;
    unsigned int minimalPath_numberOfSigmaSteps=10;
    double minimalPath_sigmaDistance=30.0;
    
    // Reading option parameters from user input
    cout << argc;
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
        else if (strcmp(argv[i],"-param-init")==0)
        {
            printf("\nPARAM INIT\n");
            // param structure delimited by commas:
            // minimalPath_alpha,minimalPath_beta,minimalPath_gamma,minimalPath_sigmaMinimum,minimalPath_sigmaMaximum,minimalPath_numberOfSigmaSteps,minimalPath_sigmaDistance
            vector<string> param_init = split(argv[i], ',');
            minimalPath_alpha = atof(param_init[0].c_str());
            printf("%f\n", minimalPath_alpha);
            minimalPath_beta = atof(param_init[1].c_str());
            printf("%f\n", minimalPath_beta);
            minimalPath_gamma = atof(param_init[2].c_str());
            printf("%f\n", minimalPath_gamma);
            minimalPath_sigmaMinimum = atof(param_init[3].c_str());
            printf("%f\n", minimalPath_sigmaMinimum);
            minimalPath_sigmaMaximum = atof(param_init[4].c_str());
            printf("%f\n", minimalPath_sigmaMaximum);
            minimalPath_numberOfSigmaSteps = atoi(param_init[5].c_str());
            printf("%i\n", minimalPath_numberOfSigmaSteps);
            minimalPath_sigmaDistance = atof(param_init[6].c_str());
            printf("%f\n", minimalPath_sigmaDistance);
        }
		else if (strcmp(argv[i],"-verbose")==0) {
            verbose = true;
        }
        else if (strcmp(argv[i],"-help")==0) {
            help();
            return EXIT_FAILURE;
        }
    }
    
    // Checking if user added mandatory arguments
    if (inputFilename == "")
    {
        cerr << "Input filename not provided" << endl;
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
    
    // Extract the input file name
    unsigned found_slash = inputFilename.find_last_of("/\\");
    string inputFilename_nameonly = inputFilename.substr(found_slash+1);
    unsigned found_point = inputFilename_nameonly.find_first_of(".");
    inputFilename_nameonly = inputFilename_nameonly.substr(0,found_point);
    
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
    
    ReaderType::Pointer reader = ReaderType::New();
    itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
    reader->SetImageIO(io);
    reader->SetFileName(inputFilename);
    try {
        reader->Update();
    } catch( itk::ExceptionObject & e ) {
        cerr << "ERROR: Exception caught while reading input image (-i option). Are you sure the image exist?" << endl;
        cerr << e << endl;
        return EXIT_FAILURE;
    }
    initialImage = reader->GetOutput();
	
	ImageType::SizeType desiredSize = initialImage->GetLargestPossibleRegion().GetSize();
    ImageType::SpacingType spacingI = initialImage->GetSpacing();
    
	// Intensity normalization
	RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
	rescaleFilter->SetInput(initialImage);
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
    
    vesselnessFilter(image, typeImageFactor, minimalPath_alpha, minimalPath_beta, minimalPath_gamma, minimalPath_sigmaMinimum, minimalPath_sigmaMaximum, minimalPath_numberOfSigmaSteps, minimalPath_sigmaDistance);

    return EXIT_SUCCESS;
}