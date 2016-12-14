//
//  main.cpp
//  sct_modif_header
//
//  Created by Benjamin De Leener on 2014-04-11.
//  Copyright (c) 2014 Benjamin De Leener. All rights reserved.
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

#include "Matrix4x4.h"
#include "Matrix3x3.h"

using namespace std;

void help()
{
    cout << "sct_modif_header - Version 0.1" << endl;
    cout << "Author : Benjamin De Leener - NeuroPoly lab <www.neuropoly.info>" << endl << endl;
    
    cout << "This program modify the header of an image." << endl << endl;
    
    cout << "Usage : \t sct_modif_header <inputfilename> [options]" << endl << endl;
    
    cout << "Available options : " << endl;
    cout << "\t-mat <matrixfilename> \t (txt, apply the transformation matrix (4X4) to the header)" << endl;
    cout << "\t-o <outputfilename> \t (default=inputfilename)" << endl;
    cout << "\t-help" << endl;
}

template<typename TPixelType, unsigned int N>
int transform(string inputFilename, string outputFilename, string matrixFilename);

int main(int argc, const char * argv[])
{
    if (argc == 1)
    {
        help();
        return EXIT_FAILURE;
    }
    string inputFilename = "", outputFilename = "", matrixFilename = "";
    inputFilename = argv[1];
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i],"-o")==0) {
            i++;
            outputFilename = argv[i];
        }
        else if (strcmp(argv[i],"-mat")==0) {
            i++;
            matrixFilename = argv[i];
        }
        else if (strcmp(argv[i],"-help")==0) {
            help();
            return EXIT_FAILURE;
        }
    }
    if (inputFilename == "") {
        cerr << "Input filename not provided" << endl;
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
    typedef itk::ImageIOBase::IOComponentType ScalarPixelType;
    const ScalarPixelType pixelType = io->GetComponentType();
    unsigned int numberOfDimensions = io->GetNumberOfDimensions();
    
    if (numberOfDimensions == 2)
    {
        if (io->GetComponentTypeAsString(pixelType)=="char")
            return transform<char,2>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_char")
            return transform<unsigned char,2>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="short")
            return transform<short,2>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_short")
            return transform<unsigned short,2>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="int")
            return transform<int,2>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_int")
            return transform<unsigned int,2>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="long")
            return transform<long,2>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_long")
            return transform<unsigned long,2>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="float")
            return transform<float,2>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="double")
            return transform<double,2>(inputFilename,outputFilename,matrixFilename);
        else {
            cout << "Pixel type " << io->GetComponentTypeAsString(pixelType) << " is not supported" << endl;
            return EXIT_FAILURE;
        }
    }
    else if (numberOfDimensions == 3)
    {
        if (io->GetComponentTypeAsString(pixelType)=="char")
            return transform<char,3>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_char")
            return transform<unsigned char,3>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="short")
            return transform<short,3>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_short")
            return transform<unsigned short,3>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="int")
            return transform<int,3>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_int")
            return transform<unsigned int,3>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="long")
            return transform<long,3>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_long")
            return transform<unsigned long,3>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="float")
            return transform<float,3>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="double")
            return transform<double,3>(inputFilename,outputFilename,matrixFilename);
        else {
            cout << "Pixel type " << io->GetComponentTypeAsString(pixelType) << " is not supported" << endl;
            return EXIT_FAILURE;
        }
    }
    else if (numberOfDimensions == 4)
    {
        if (io->GetComponentTypeAsString(pixelType)=="char")
            return transform<char,4>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_char")
            return transform<unsigned char,4>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="short")
            return transform<short,4>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_short")
            return transform<unsigned short,4>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="int")
            return transform<int,4>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_int")
            return transform<unsigned int,4>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="long")
            return transform<long,4>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="unsigned_long")
            return transform<unsigned long,4>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="float")
            return transform<float,4>(inputFilename,outputFilename,matrixFilename);
        else if (io->GetComponentTypeAsString(pixelType)=="double")
            return transform<double,4>(inputFilename,outputFilename,matrixFilename);
        else {
            cout << "Pixel type " << io->GetComponentTypeAsString(pixelType) << " is not supported" << endl;
            return EXIT_FAILURE;
        }
    }
    
    return EXIT_SUCCESS;
}

template<typename TPixelType, unsigned int N>
int transform(string inputFilename, string outputFilename, string matrixFilename)
{
    typedef itk::Image< TPixelType, N > ImageType;
    typedef itk::ImageFileReader< ImageType > ReaderType;
    typedef itk::ImageFileWriter< ImageType > WriterType;
    typedef itk::Point<double, N> PointType;
    
	typename ReaderType::Pointer reader = ReaderType::New();
    itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
	reader->SetImageIO(io);
	reader->SetFileName(inputFilename);
	reader->Update();
	typename ImageType::Pointer image = reader->GetOutput();
    
    if (matrixFilename != "")
    {
        CMatrix4x4 matrix_transform;
        
        ifstream matrix_file;
        matrix_file.open(matrixFilename.c_str());
        matrix_file >> matrix_transform;
        matrix_file.close();
        
        CMatrix3x3 rotation = CMatrix3x3(matrix_transform);
        
        typename ImageType::DirectionType direction_input = image->GetDirection();
        PointType origin_input = image->GetOrigin();
        
        CMatrix3x3 direction_input_m;
        direction_input_m[0] = direction_input[0][0];
        direction_input_m[1] = direction_input[0][1];
        direction_input_m[2] = direction_input[0][2];
        direction_input_m[3] = direction_input[1][0];
        direction_input_m[4] = direction_input[1][1];
        direction_input_m[5] = direction_input[1][2];
        direction_input_m[6] = direction_input[2][0];
        direction_input_m[7] = direction_input[2][1];
        direction_input_m[8] = direction_input[2][2];
        
        CMatrix3x3 direction_output_m = rotation*direction_input_m;
        typename ImageType::DirectionType direction_output;
        direction_output[0][0] = direction_output_m[0];
        direction_output[0][1] = direction_output_m[1];
        direction_output[0][2] = direction_output_m[2];
        direction_output[1][0] = direction_output_m[3];
        direction_output[1][1] = direction_output_m[4];
        direction_output[1][2] = direction_output_m[5];
        direction_output[2][0] = direction_output_m[6];
        direction_output[2][1] = direction_output_m[7];
        direction_output[2][2] = direction_output_m[8];
        direction_output[3][3] = 1.0;
        PointType origin_output;
        origin_output[0] = origin_input[0] + matrix_transform[12];
        origin_output[1] = origin_input[1] + matrix_transform[13];
        origin_output[2] = origin_input[2] + matrix_transform[14];
        
        image->SetDirection(direction_output);
        image->SetOrigin(origin_output);
    }
    
    
	    
	typename WriterType::Pointer writer = WriterType::New();
	writer->SetImageIO(io);
	writer->SetFileName(outputFilename);
	writer->SetInput(image);
	try {
		writer->Update();
	} catch( itk::ExceptionObject & e ) {
		std::cerr << "Exception caught while updating writer" << std::endl;
		std::cerr << e << std::endl;
	}
    
    return EXIT_SUCCESS;
}



