/******************************************
 File:          Main.cpp
 Date:          2012-09-24
 Author:        Benjamin De Leener - NeuroPoly
 Description:   This file compute the approximate center of the spinal cord on a MRI volume
*****************************************/
#define _SCL_SECURE_NO_WARNINGS
#include <iostream>
#include <iomanip>

#include "Initialisation.h"
#include "SymmetricalCropping.h"
#include "OrientImage.h"
#include "SCRegion.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkNiftiImageIO.h>
#include <itkRescaleIntensityImageFilter.h>

using namespace std;

typedef itk::Image< double, 3 >	ImageType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::RescaleIntensityImageFilter< ImageType, ImageType > RescaleFilterType;

void help()
{
    cout << "sct_spinalCordDetection - Version 0.3" << endl;
    cout << "Author : Benjamin De Leener - NeuroPoly lab <www.neuropoly.info> " << endl << endl;
    
    cout << "Usage : \t sct_detect_spinalcord -i <inputfilename> -o <outputfilename> -t <imagetype> [options]" << endl << endl;
    
    cout << "Available options : " << endl;
    cout << "\t-i <inputfilename> \t (no default)" << endl;
    cout << "\t-o <outputfilename> \t (no default)" << endl;
    cout << "\t-t <imagetype> {t1,t2} \t (string, type of image contrast, default is t2)" << endl;
    cout << "\t-n <numslice> \t\t (int, number of slices around slice to analyze, default is 5)" << endl;
    cout << "\t-s <startslice> \t (int, slice in axial direction to analyze, default use middle slice)" << endl;
    cout << "\t-g <gap> \t\t (int, gap between slices to analyze, default is 1)" << endl;
    cout << "\t-r <radius> \t\t (double, average radius of spinal cord, default is 4.0 mm)" << endl;
    //cout << "\t-d \t\t\t (output a PNG image of the detection)" << endl;
	cout << "\t-v \t\t\t (display option)" << endl;
    cout << "\t-help" << endl;
}

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        help();
        return EXIT_FAILURE;
    }
    string inputFilename = "", outputFilename = "";
    double typeImageFactor = 1.0, gap = 1.0, startSlice = 0.5, radius = -1.0;
    int numberOfSlices = 5;
	bool verbose = false, output_detection = false;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i],"-i")==0)
        {
            i++;
            inputFilename = argv[i];
        }
        else if (strcmp(argv[i],"-o")==0)
        {
            i++;
            outputFilename = argv[i];
        }
        else if (strcmp(argv[i],"-t")==0)
        {
            i++;
            if (strcmp(argv[i],"t1")==0) {
				typeImageFactor = -1.0;
				cout << endl << "WARNING: be sure your image is a T1-weighted image." << endl << endl;
			}
            else if (strcmp(argv[i],"t2")==0) {
				typeImageFactor = 1.0;
				cout << endl << "WARNING: be sure your image is a T2-weighted image." << endl << endl;
			}
            else {
				cout << "Invalid type or image (need to be \"t1\" or \"t2\")" << endl;
				help();
			}
        }
        else if (strcmp(argv[i],"-g")==0)
        {
            i++;
            gap = atof(argv[i]);
        }
        else if (strcmp(argv[i],"-s")==0)
        {
            i++;
            startSlice = atof(argv[i]);
        }
        else if (strcmp(argv[i],"-n")==0)
        {
            i++;
            numberOfSlices = atoi(argv[i]);
        }
        else if (strcmp(argv[i],"-r")==0)
        {
            i++;
            radius = atof(argv[i]);
        }
        else if (strcmp(argv[i],"-d")==0)
        {
            output_detection = true;
        }
		else if (strcmp(argv[i],"-v")==0)
        {
            verbose = true;
        }
        else if (strcmp(argv[i],"-help")==0)
        {
            help();
            return EXIT_FAILURE;
        }
    }
    if (inputFilename == "")
    {
        cerr << "Input filename not provided" << endl;
		help();
        return EXIT_FAILURE;
    }
    if (inputFilename != "" && outputFilename == "")
    {
		string nii=".nii", niigz=".nii.gz", suffix=niigz;
		size_t pos = inputFilename.find(niigz);
		if (pos == string::npos) {
			pos = inputFilename.find(nii);
			suffix = nii;
		}
        outputFilename = inputFilename;
		outputFilename.erase(pos);
		outputFilename += "center_binary";
    }
    
    ReaderType::Pointer reader = ReaderType::New();
	itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
	reader->SetImageIO(io);
	reader->SetFileName(inputFilename);
	
    try {
        reader->Update();
    } catch( itk::ExceptionObject & e ) {
        std::cerr << "Exception caught while reading input image " << std::endl;
        std::cerr << e << std::endl;
    }
    
    // Original image
	ImageType::Pointer initialImage = reader->GetOutput(), image = ImageType::New();
    
    OrientImage<ImageType> orientationFilter;
    orientationFilter.setInputImage(initialImage);
    orientationFilter.orientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL);
    initialImage = orientationFilter.getOutputImage();
	
	// Cropping image
	ImageType::SizeType desiredSize = initialImage->GetLargestPossibleRegion().GetSize();
	if (desiredSize[2] > 60)
	{
		SymmetricalCropping symCroppingFilter;
        symCroppingFilter.setInputImage(initialImage);
		int croppingSlice = symCroppingFilter.symmetryDetection();
		if (verbose) cout << "Cropping around slice = " << croppingSlice << endl;
        image = symCroppingFilter.cropping();
	}
	else image = initialImage;
    
	// Intensity normalization to equalize all different images
	RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
	rescaleFilter->SetInput(image);
	rescaleFilter->SetOutputMinimum(0);
	rescaleFilter->SetOutputMaximum(1000);
	rescaleFilter->Update();
	image = rescaleFilter->GetOutput();
    
    
    if (startSlice > 1) startSlice /= desiredSize[1]; // transform the starting slice in a [0,1] range
    Initialisation init;
    
    bool isSpinalCordDetected = false;
    int countFailure = 0;
    double step = 0.025; // step of displacement in pourcentage of the image
    do
    {
        if (verbose) cout << "Initialization - spinal cord detection on axial slices" << endl;
        init = Initialisation(image,typeImageFactor);
        init.setVerbose(verbose);
        init.setGap(gap); // gap between slices is necessary to provide good normals
        if (radius!= -1.0) init.setRadius(radius); // approximate radius of spinal cord. This parameter is used to initiate Hough transform
        if (numberOfSlices!= -1.0) init.setNumberOfSlices(numberOfSlices);
        
        // if the initialization fails at the first position in the image, an other spinal cord detection process is launch higher.
        int d = rand() % 2; if (d==0) d = -1;
        isSpinalCordDetected = init.computeInitialParameters(startSlice+(double)countFailure*step*(double)d);
        //if (!isSpinalCordDetected) isSpinalCordDetected = init.computeInitialParameters(0.7);
        if(isSpinalCordDetected)
        {
            //if (output_detection) init.savePointAsAxialImage(image,outputPath+"result_detection.png");
            CVector3 point, normal1, normal2; // normal1 and normal2 and normals in both direction from initial point
            double stretchingFactor = 1.0;
            init.getPoints(point,normal1,normal2,radius,stretchingFactor);
            
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
    else
    {
        init.savePointAsBinaryImage(initialImage, outputFilename, orientationFilter.getInitialImageOrientation());
    }

    return EXIT_SUCCESS;
}