//
//  main.cpp
//  sct_change_nifti_pixel_type
//
//  Created by Benjamin De Leener on 2013-11-12.
//  Copyright (c) 2013 Benjamin De Leener. All rights reserved.
//

#define _SCL_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdlib>
#include <string>

#include <itkImage.h>
#include <itkImageIOBase.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>

using namespace std;

void help()
{
    cout << "sct_change_nifti_pixel_type - Version 0.2" << endl;
    cout << "Author : Benjamin De Leener - NeuroPoly lab <www.neuropoly.info>" << endl << endl;
    
    cout << "This program change the pixel type of a nifti image (2D or 3D). Types available: char, unsigned_char, short, unsigned_short, int, unsigned_int, long, unsigned_long, float, double" << endl << endl;
    
    cout << "Usage : \t sct_change_nifti_pixel_type <inputfilename> <pixeltype> [options]" << endl << endl;
    
    cout << "Available options : " << endl;
    cout << "\t-o <outputfilename> \t (default=inputfilename)" << endl;
    cout << "\t-help" << endl;
}

template<typename TPixelType, unsigned int N>
void transform(string inputFilename, string outputFilename);

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        help();
        return EXIT_FAILURE;
    }
    string inputFilename = "", outputFilename = "", pixelType = "";
    if (argc < 3) {
        cerr << "Not enough input arguments..." << endl;
		help();
        return EXIT_FAILURE;
    }
    else {
        inputFilename = argv[1];
        pixelType = argv[2];
        for (int i = 3; i < argc; ++i) {
            if (strcmp(argv[i],"-o")==0) {
                i++;
                outputFilename = argv[i];
            }
            else if (strcmp(argv[i],"-help")==0) {
                help();
                return EXIT_FAILURE;
            }
        }
    }
    if (inputFilename == "") {
        cerr << "Input filename not provided" << endl;
		help();
        return EXIT_FAILURE;
    }
    if (pixelType != "char" && pixelType != "unsigned_char" && pixelType != "short" && pixelType != "unsigned_short" && pixelType != "int" && pixelType != "unsigned_int" && pixelType != "long" && pixelType != "unsigned_long" && pixelType != "float" && pixelType != "double") {
        cerr << "Pixel Type not supported." << endl;
		help();
        return EXIT_FAILURE;
    }
    if (outputFilename == "") {
        outputFilename = inputFilename;
        cout << "WARNING: Output filename not provided. Input image will be overwritten." << endl;
    }
    
    itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
    io->SetFileName(inputFilename.c_str());
    io->ReadImageInformation();
    unsigned int numberOfDimensions = io->GetNumberOfDimensions();
    
    if (pixelType == "char" && numberOfDimensions == 2)
        transform<char,2>(inputFilename,outputFilename);
    else if (pixelType == "unsigned_char" && numberOfDimensions == 2)
        transform<unsigned char,2>(inputFilename,outputFilename);
    else if (pixelType == "short" && numberOfDimensions == 2)
        transform<short,2>(inputFilename,outputFilename);
    else if (pixelType == "unsigned_short" && numberOfDimensions == 2)
        transform<unsigned short,2>(inputFilename, outputFilename);
    else if (pixelType == "int" && numberOfDimensions == 2)
        transform<int,2>(inputFilename,outputFilename);
    else if (pixelType == "unsigned_int" && numberOfDimensions == 2)
        transform<unsigned int,2>(inputFilename,outputFilename);
    else if (pixelType == "long" && numberOfDimensions == 2)
        transform<long,2>(inputFilename,outputFilename);
    else if (pixelType == "unsigned_long" && numberOfDimensions == 2)
        transform<unsigned long,2>(inputFilename,outputFilename);
    else if (pixelType == "float" && numberOfDimensions == 2)
        transform<float,2>(inputFilename,outputFilename);
    else if (pixelType == "double" && numberOfDimensions == 2)
        transform<double,2>(inputFilename,outputFilename);
    else if (pixelType == "char" && numberOfDimensions == 3)
        transform<char,3>(inputFilename,outputFilename);
    else if (pixelType == "unsigned_char" && numberOfDimensions == 3)
        transform<unsigned char,3>(inputFilename,outputFilename);
    else if (pixelType == "short" && numberOfDimensions == 3)
        transform<short,3>(inputFilename,outputFilename);
    else if (pixelType == "unsigned_short" && numberOfDimensions == 3)
        transform<unsigned short,3>(inputFilename, outputFilename);
    else if (pixelType == "int" && numberOfDimensions == 3)
        transform<int,3>(inputFilename,outputFilename);
    else if (pixelType == "unsigned_int" && numberOfDimensions == 3)
        transform<unsigned int,3>(inputFilename,outputFilename);
    else if (pixelType == "long" && numberOfDimensions == 3)
        transform<long,3>(inputFilename,outputFilename);
    else if (pixelType == "unsigned_long" && numberOfDimensions == 3)
        transform<unsigned long,3>(inputFilename,outputFilename);
    else if (pixelType == "float" && numberOfDimensions == 3)
        transform<float,3>(inputFilename,outputFilename);
    else if (pixelType == "double" && numberOfDimensions == 3)
        transform<double,3>(inputFilename,outputFilename);

    return EXIT_SUCCESS;
}

template<typename TPixelType, unsigned int N>
void transform(string inputFilename, string outputFilename)
{
    typedef itk::Image< TPixelType, N > ImageType;
    typedef itk::ImageFileReader< ImageType > ReaderType;
    typedef itk::ImageFileWriter< ImageType > WriterType;
    
	typename ReaderType::Pointer reader = ReaderType::New();
    itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
	reader->SetImageIO(io);
	reader->SetFileName(inputFilename);
    try {
        reader->Update();
    } catch( itk::ExceptionObject & e ) {
		std::cerr << "Exception caught while reading image " << std::endl;
		std::cerr << e << std::endl;
	}
    
    typename WriterType::Pointer writer = WriterType::New();
	writer->SetImageIO(io);
	writer->SetFileName(outputFilename);
	writer->SetInput(reader->GetOutput());
	try {
		writer->Update();
	} catch( itk::ExceptionObject & e ) {
		std::cerr << "Exception caught while writing image " << std::endl;
		std::cerr << e << std::endl;
	}
}
