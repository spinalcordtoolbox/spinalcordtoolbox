/******************************************
 File:          Main.cpp
 Date:          2012-09-24
 Author:        Benjamin De Leener - NeuroPoly
 Description:   This file compute the approximate center of the spinal cord on a MRI volume
*****************************************/
#define _SCL_SECURE_NO_WARNINGS
#include <iostream>

#include "OrientImage.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>
using namespace std;

typedef itk::SpatialOrientation::ValidCoordinateOrientationFlags OrientationType;

string FlagToString(OrientationType orientation);
OrientationType StringToFlag(string orientation);
void printAvailableOrientation();

template<typename TPixelType, unsigned int N>
int changeOrientationMethod(string inputFilename, string outputFilename, OrientationType orientation, bool changeOrientation, bool displayInitialOrientation, bool displayAvailableOrientation);

void help()
{
    cout << "sct_orientation - Version 0.2" << endl;
    cout << "Author : Benjamin De Leener - NeuroPoly lab <www.neuropoly.info> " << endl << endl;
	cout << "This program only change header of images and doesn't change the image itself." << endl << endl;
    
    cout << "Usage : \t sct_orientation -i <inputfilename> -o <outputfilename> [options]" << endl << endl;
    
    cout << "Available options : " << endl;
    cout << "\t-i <inputfilename> \t (no default)" << endl;
    cout << "\t-o <outputfilename> \t (no default)" << endl;
	cout << "\t-orientation <orientation> \t (three letter code of the orientation, no default ; if not provided, only the input image orientation is available)" << endl;
    cout << "\t-get \t (display the input image orientation)" << endl;
	cout << "\t-display \t (display available orientation)" << endl;
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
	OrientationType orientation = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_INVALID;
	bool changeOrientation = false, displayInitialOrientation = false, displayAvailableOrientation = false;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i],"-i")==0) {
            i++;
            inputFilename = argv[i];
        }
        else if (strcmp(argv[i],"-o")==0) {
            i++;
            outputFilename = argv[i];
        }
		else if (strcmp(argv[i],"-orientation")==0) {
            i++;
            orientation = StringToFlag(argv[i]);
			changeOrientation = true;
        }
        else if (strcmp(argv[i],"-get")==0) {
            displayInitialOrientation = true;
        }
		else if (strcmp(argv[i],"-display")==0) {
            printAvailableOrientation();
			return EXIT_FAILURE;
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
	if (!changeOrientation && !displayInitialOrientation) {
		cout << "FAILURE: The command as you wrote it doesn't do anything..." << endl << endl;
		help();
		return EXIT_FAILURE;
	}
	if (orientation == itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_INVALID && changeOrientation) {
		cout << "FAILURE: Invalid orientation." << endl;
		printAvailableOrientation();
		return EXIT_FAILURE;
	}
    if (inputFilename != "" && outputFilename == "" && changeOrientation)
    {
		string nii=".nii", niigz=".nii.gz", suffix=niigz;
		size_t pos = inputFilename.find(niigz);
		if (pos == string::npos) {
			pos = inputFilename.find(nii);
			suffix = nii;
		}
        outputFilename = inputFilename;
		outputFilename.erase(pos);
		outputFilename += "_"+FlagToString(orientation)+"_"+suffix;
    }
    
    itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
    io->SetFileName(inputFilename.c_str());
    io->ReadImageInformation();
    typedef itk::ImageIOBase::IOComponentType ScalarPixelType;
    const ScalarPixelType pixelType = io->GetComponentType();
    unsigned int numberOfDimensions = io->GetNumberOfDimensions();
    
        if (io->GetComponentTypeAsString(pixelType)=="char")
            return changeOrientationMethod<char, 3>(inputFilename,outputFilename,orientation,changeOrientation,displayInitialOrientation,displayAvailableOrientation);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_char")
            return changeOrientationMethod<unsigned char, 3>(inputFilename,outputFilename,orientation,changeOrientation,displayInitialOrientation,displayAvailableOrientation);
        else if (io->GetComponentTypeAsString(pixelType)=="short")
            return changeOrientationMethod<short, 3>(inputFilename,outputFilename,orientation,changeOrientation,displayInitialOrientation,displayAvailableOrientation);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_short")
            return changeOrientationMethod<unsigned short, 3>(inputFilename,outputFilename,orientation,changeOrientation,displayInitialOrientation,displayAvailableOrientation);
        else if (io->GetComponentTypeAsString(pixelType)=="int")
            return changeOrientationMethod<int, 3>(inputFilename,outputFilename,orientation,changeOrientation,displayInitialOrientation,displayAvailableOrientation);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_int")
            return changeOrientationMethod<unsigned int, 3>(inputFilename,outputFilename,orientation,changeOrientation,displayInitialOrientation,displayAvailableOrientation);
        else if (io->GetComponentTypeAsString(pixelType)=="long")
            return changeOrientationMethod<long, 3>(inputFilename,outputFilename,orientation,changeOrientation,displayInitialOrientation,displayAvailableOrientation);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_long")
            return changeOrientationMethod<unsigned long, 3>(inputFilename,outputFilename,orientation,changeOrientation,displayInitialOrientation,displayAvailableOrientation);
        else if (io->GetComponentTypeAsString(pixelType)=="float")
            return changeOrientationMethod<float, 3>(inputFilename,outputFilename,orientation,changeOrientation,displayInitialOrientation,displayAvailableOrientation);
        else if (io->GetComponentTypeAsString(pixelType)=="double")
            return changeOrientationMethod<double, 3>(inputFilename,outputFilename,orientation,changeOrientation,displayInitialOrientation,displayAvailableOrientation);
        else {
            cout << "Pixel type " << io->GetComponentTypeAsString(pixelType) << " is not supported" << endl;
            return EXIT_FAILURE;
        }

    return EXIT_SUCCESS;
}

template<typename TPixelType, unsigned int N>
int changeOrientationMethod(string inputFilename, string outputFilename, OrientationType orientation, bool changeOrientation, bool displayInitialOrientation, bool displayAvailableOrientation)
{
    typedef itk::Image< TPixelType, N >	ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;
    typedef itk::ImageFileWriter<ImageType> WriterType;
    
    typename ReaderType::Pointer reader = ReaderType::New();
	itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
	reader->SetImageIO(io);
	reader->SetFileName(inputFilename);
    
    OrientImage<ImageType> orientationFilter;
    orientationFilter.setInputImage(reader->GetOutput());
    
	if (displayInitialOrientation)
		cout << "Input image orientation : " << FlagToString(orientationFilter.getInitialImageOrientation()) << endl;
    
	if (changeOrientation)
	{
		orientationFilter.orientation(orientation);
        
		typename WriterType::Pointer writer = WriterType::New();
		writer->SetImageIO(io);
		writer->SetFileName(outputFilename);
		writer->SetInput(orientationFilter.getOutputImage());
		try {
			writer->Write();
		} catch( itk::ExceptionObject & e ) {
			std::cerr << "Exception caught while writing output image " << std::endl;
			std::cerr << e << std::endl;
		}
	}
    return EXIT_SUCCESS;
}

string FlagToString(itk::SpatialOrientation::ValidCoordinateOrientationFlags orientation)
{
	string result="INVALID";
	switch(orientation) {
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP : result="RIP"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIP : result="LIP"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSP : result="RSP"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSP : result="LSP"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIA : result="RIA"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIA : result="LIA"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSA : result="RSA"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSA : result="LSA"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRP : result="IRP"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILP : result="ILP"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRP : result="SRP"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLP : result="SLP"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRA : result="IRA"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILA : result="ILA"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRA : result="SRA"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLA : result="SLA"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI : result="RPI"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPI : result="LPI"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI : result="RAI"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAI : result="LAI"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPS : result="RPS"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPS : result="LPS"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS : result="RAS"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAS : result="LAS"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRI : result="PRI"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLI : result="PLI"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARI : result="ARI"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALI : result="ALI"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRS : result="PRS"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLS : result="PLS"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARS : result="ARS"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALS : result="ALS"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPR : result="IPR"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPR : result="SPR"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAR : result="IAR"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAR : result="SAR"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPL : result="IPL"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPL : result="SPL"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAL : result="IAL"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAL : result="SAL"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIR : result="PIR"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSR : result="PSR"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIR : result="AIR"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASR : result="ASR"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIL : result="PIL"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSL : result="PSL"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL : result="AIL"; break;
	case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASL : result="ASL"; break;
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_INVALID : result="INVALID"; break;
	}
	return result;
}

itk::SpatialOrientation::ValidCoordinateOrientationFlags StringToFlag(string orientation)
{
	itk::SpatialOrientation::ValidCoordinateOrientationFlags result;
	if (orientation == "RIP") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP;
	else if (orientation == "LIP") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIP;
	else if (orientation == "RSP") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSP;
	else if (orientation == "LSP") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSP;
	else if (orientation == "RIA") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIA;
	else if (orientation == "LIA") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIA;
	else if (orientation == "RSA") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSA;
	else if (orientation == "LSA") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSA;
	else if (orientation == "IRP") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRP;
	else if (orientation == "ILP") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILP;
	else if (orientation == "SRP") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRP;
	else if (orientation == "SLP") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLP;
	else if (orientation == "IRA") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRA;
	else if (orientation == "ILA") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILA;
	else if (orientation == "SRA") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRA;
	else if (orientation == "SLA") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLA;
	else if (orientation == "RPI") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI;
	else if (orientation == "LPI") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPI;
	else if (orientation == "RAI") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI;
	else if (orientation == "LAI") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAI;
	else if (orientation == "RPS") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPS;
	else if (orientation == "LPS") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPS;
	else if (orientation == "RAS") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS;
	else if (orientation == "LAS") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAS;
	else if (orientation == "PRI") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRI;
	else if (orientation == "PLI") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLI;
	else if (orientation == "ARI") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARI;
	else if (orientation == "ALI") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALI;
	else if (orientation == "PRS") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRS;
	else if (orientation == "PLS") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLS;
	else if (orientation == "ARS") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARS;
	else if (orientation == "ALS") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALS;
	else if (orientation == "IPR") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPR;
	else if (orientation == "SPR") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPR;
	else if (orientation == "IAR") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAR;
	else if (orientation == "SAR") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAR;
	else if (orientation == "IPL") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPL;
	else if (orientation == "SPL") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPL;
	else if (orientation == "IAL") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAL;
	else if (orientation == "SAL") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAL;
	else if (orientation == "PIR") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIR;
	else if (orientation == "PSR") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSR;
	else if (orientation == "AIR") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIR;
	else if (orientation == "ASR") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASR;
	else if (orientation == "PIL") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIL;
	else if (orientation == "PSL") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSL;
	else if (orientation == "AIL") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL;
	else if (orientation == "ASL") result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASL;
	else result=itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_INVALID;

	return result;
}

void printAvailableOrientation()
{
	 cout << "Oriantation available : " << endl << "RIP LIP RSP LSP RIA LIA RSA LSA IRP ILP SRP SLP IRA ILA SRA SLA RPI LPI RAI LAI RPS LPS RAS LAS PRI PLI ARI ALI PRS PLS ARS ALS IPR SPR IAR SAR IPL SPL IAL SAL PIR PSR AIR ASR PIL PSL AIL ASL" << endl;
}