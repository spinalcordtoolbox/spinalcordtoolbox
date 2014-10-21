//
//  main.cpp
//  CroppingImage
//
//  Created by Benjamin De Leener on 2013-09-24.
//  Copyright (c) 2013 NeuroPoly. All rights reserved.
//
#define _SCL_SECURE_NO_WARNINGS
#include <iostream>
#include <cstdlib>
#include <string>

#include "OrientImage.h"
#include "SpinalCord.h"
#include "Vector3.h"

#include <itkImage.h>
#include <itkImageIOBase.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNiftiImageIO.h>
#include <itkIndex.h>
#include <itkCropImageFilter.h>

using namespace std;

typedef itk::SpatialOrientation::ValidCoordinateOrientationFlags OrientationType;

void help()
{
    cout << "sct_crop_image - Version 0.3" << endl;
    cout << "Author : Benjamin De Leener - NeuroPoly lab <www.neuropoly.info>" << endl << endl;
	
	cout << "This program crop an image. You can provide either directly the starting and ending slice number around that the image will be cropped or a mask. You can as well crop in any dimension you want. This program supports 2D to 7D images with the following voxel types: char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, float, double." << endl << endl;
    
    cout << "Usage : " << endl << "\t sct_crop_image -i <inputfilename> -o <outputfilename> [options]" << endl;
    cout << "\t sct_crop_image -i <inputfilename> -o <outputfilename> -m <maskfilename> [options]" << endl;
    cout << "\t sct_crop_image -i <inputfilename> -o <outputfilename> -dim 1,3 -start 20,35 -end 70,50" << endl << endl;
    
    cout << "Available options : " << endl;
    cout << "\t-i <inputfilename> \t (no default)" << endl;
    cout << "\t-o <outputfilename> \t (no default)" << endl;
    cout << "\t-m <maskfilename> \t (cropping around the mask)" << endl;
    cout << "\t-start <s0,...,sn> \t (start slices, ]0,1[: percentage, 0 & >1: slice number)" << endl;
    cout << "\t-end <e0,...,en> \t (end slices, ]0,1[: percentage, 0: last slice, >1: slice number, <0: last slice - value)" << endl;
    cout << "\t-dim <d0,...,dn> \t (dimension to crop, from 0 to n-1, default is 1)" << endl;
    cout << "\t-shift <s0,...,sn> \t (adding shift when used with mask, default is 0)" << endl;
    cout << "\t-b <backgroundvalue> \t (replace voxels outside cropping region with background value)" << endl;
    cout << "\t-mesh <meshfilename> \t (mesh to crop)" << endl;
    cout << "\t-help" << endl;
}

template<typename TPixelType, unsigned int N>
int transform(string inputFilename, string outputFilename, string maskFilename, string meshFilename, bool isMask, bool isMesh, float backgroundValue, bool realCrop, vector<float> startSlices, vector<float> endSlices, vector<float> dims, vector<float> shiftSlices);

// This method split the input argument into int with delimiter and add a shift
vector<float> splitString(string s, string delimiter, float shift=0.0)
{
    size_t pos = 0;
    string token;
    vector<float> result;
    if (s.find(delimiter) == string::npos) {
        result.push_back(atof(s.c_str()));
        return result;
    }
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        s.erase(0, pos + delimiter.length());
        result.push_back(atof(token.c_str())+shift);
    }
    return result;
}

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        help();
        return EXIT_FAILURE;
    }
    string inputFilename = "", outputFilename = "", maskFilename = "", meshFilename = "";
    float startSlice = 0.0, endSlice = 0.0, dim = 1.0, backgroundValue = 0.0;
    bool isMask = false, isMesh = false, realCrop = true;
    vector<float> dims, startSlices, endSlices, shiftSlices;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i],"-i")==0) {
            i++;
            inputFilename = argv[i];
        }
        else if (strcmp(argv[i],"-o")==0) {
            i++;
            outputFilename = argv[i];
        }
		else if (strcmp(argv[i],"-m")==0) {
            i++;
            maskFilename = argv[i];
			isMask = true;
        }
        else if (strcmp(argv[i],"-start")==0) {
            i++;
            startSlices = splitString(argv[i],",");
        }
        else if (strcmp(argv[i],"-end")==0) {
            i++;
            endSlices = splitString(argv[i],",");
        }
        else if (strcmp(argv[i],"-dim")==0) {
            i++;
            dims = splitString(argv[i],",");
        }
        else if (strcmp(argv[i],"-shift")==0) {
            i++;
            shiftSlices = splitString(argv[i],",");
        }
        else if (strcmp(argv[i],"-b")==0) {
            i++;
            backgroundValue = atof(argv[i]);
            realCrop = false;
        }
        else if (strcmp(argv[i],"-mesh")==0) {
            i++;
            meshFilename = argv[i];
			isMesh = true;
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
        cout << "Output filename not provided. Input image will be overwritten" << endl;
    }
    if (shiftSlices.size() == 0) {
        for (int i=0; i<dims.size(); i++) shiftSlices.push_back(0);
    }
    if (startSlices.size() != dims.size() && !isMask) {
        cerr << "Start slices must have the same number of elements than dimension (-dim)" << endl;
		help();
        return EXIT_FAILURE;
    }
    if (endSlices.size() != dims.size() && !isMask) {
        cerr << "End slices must have the same number of elements than dimension (-dim)" << endl;
		help();
        return EXIT_FAILURE;
    }
    if (shiftSlices.size() != dims.size()) {
        cerr << "Shift slices must have the same number of elements than dimension (-dim)" << endl;
		help();
        return EXIT_FAILURE;
    }
    
    
    itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
    io->SetFileName(inputFilename.c_str());
    io->ReadImageInformation();
    typedef itk::ImageIOBase::IOComponentType ScalarPixelType;
    const ScalarPixelType pixelType = io->GetComponentType();
    unsigned int numberOfDimensions = io->GetNumberOfDimensions();
    
    if (numberOfDimensions < 2 || numberOfDimensions > 7) {
        cerr << "Error: Image dimensions (" << io->GetComponentTypeAsString(pixelType) << ") are not supported." << endl;
        help();
        return EXIT_FAILURE;
    }
    if (dims.size() == 0) {
        for (int i=0; i<numberOfDimensions; i++) dims.push_back(i);
    }
    if (shiftSlices.size() == 0) {
        for (int i=0; i<dims.size(); i++) shiftSlices.push_back(0);
    }
    
    if (io->GetComponentTypeAsString(pixelType)=="char") {
        if (numberOfDimensions==2) return transform<char,2>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==3) return transform<char,3>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==4) return transform<char,4>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==5) return transform<char,5>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==6) return transform<char,6>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==7) return transform<char,7>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
    }
    else if (io->GetComponentTypeAsString(pixelType)=="unsigned_char") {
        if (numberOfDimensions==2) return transform<unsigned char,2>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==3) return transform<unsigned char,3>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==4) return transform<unsigned char,4>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==5) return transform<unsigned char,5>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==6) return transform<unsigned char,6>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==7) return transform<unsigned char,7>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
    }
    else if (io->GetComponentTypeAsString(pixelType)=="short") {
        if (numberOfDimensions==2) return transform<short,2>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==3) return transform<short,3>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==4) return transform<short,4>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==5) return transform<short,5>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==6) return transform<short,6>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==7) return transform<short,7>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
    }
    else if (io->GetComponentTypeAsString(pixelType)=="unsigned_short") {
        if (numberOfDimensions==2) return transform<unsigned short,2>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==3) return transform<unsigned short,3>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==4) return transform<unsigned short,4>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==5) return transform<unsigned short,5>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==6) return transform<unsigned short,6>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==7) return transform<unsigned short,7>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
    }
    else if (io->GetComponentTypeAsString(pixelType)=="int") {
        if (numberOfDimensions==2) return transform<int,2>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==3) return transform<int,3>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==4) return transform<int,4>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==5) return transform<int,5>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==6) return transform<int,6>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==7) return transform<int,7>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
    }
    else if (io->GetComponentTypeAsString(pixelType)=="unsigned_int") {
        if (numberOfDimensions==2) return transform<unsigned int,2>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==3) return transform<unsigned int,3>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==4) return transform<unsigned int,4>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==5) return transform<unsigned int,5>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==6) return transform<unsigned int,6>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==7) return transform<unsigned int,7>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
    }
    else if (io->GetComponentTypeAsString(pixelType)=="long") {
        if (numberOfDimensions==2) return transform<long,2>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims, shiftSlices);
        else if (numberOfDimensions==3) return transform<long,3>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==4) return transform<long,4>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==5) return transform<long,5>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==6) return transform<long,6>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==7) return transform<long,7>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
    }
    else if (io->GetComponentTypeAsString(pixelType)=="unsigned_long") {
        if (numberOfDimensions==2) return transform<unsigned long,2>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==3) return transform<unsigned long,3>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==4) return transform<unsigned long,4>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==5) return transform<unsigned long,5>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==6) return transform<unsigned long,6>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==7) return transform<unsigned long,7>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
    }
    else if (io->GetComponentTypeAsString(pixelType)=="float") {
        if (numberOfDimensions==2) return transform<float,2>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==3) return transform<float,3>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==4) return transform<float,4>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==5) return transform<float,5>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==6) return transform<float,6>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==7) return transform<float,7>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
    }
    else if (io->GetComponentTypeAsString(pixelType)=="double") {
        if (numberOfDimensions==2) return transform<double,2>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==3) return transform<double,3>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==4) return transform<double,4>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==5) return transform<double,5>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==6) return transform<double,6>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
        else if (numberOfDimensions==7) return transform<double,7>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,realCrop,startSlices,endSlices,dims,shiftSlices);
    }
    else {
        cerr << "ERROR: Pixel type " << io->GetComponentTypeAsString(pixelType) << " is not supported." << endl;
        help();
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

template<typename TPixelType, unsigned int N>
int transform(string inputFilename, string outputFilename, string maskFilename, string meshFilename, bool isMask, bool isMesh, float backgroundValue, bool realCrop, vector<float> startSlices, vector<float> endSlices, vector<float> dims, vector<float> shiftSlices)
{
    typedef itk::Image< TPixelType, N > ImageType;
    typedef itk::ImageFileReader< ImageType > ReaderType;
    typedef itk::ImageFileWriter< ImageType > WriterType;
    typedef itk::ExtractImageFilter< ImageType, ImageType > CropFilterType;
    typedef itk::ImageRegionIterator<ImageType> ImageIterator;
    
	typename ReaderType::Pointer reader = ReaderType::New();
    itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
	reader->SetImageIO(io);
	reader->SetFileName(inputFilename);
	reader->Update();
	typename ImageType::Pointer image = reader->GetOutput();
    
    typename ImageType::SizeType desiredSize1 = image->GetLargestPossibleRegion().GetSize();
	if (isMask) {
        vector<typename ImageType::IndexType> pointsImage;
        typename ReaderType::Pointer readerMask = ReaderType::New();
        itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
        readerMask->SetImageIO(io);
        readerMask->SetFileName(maskFilename);
        readerMask->Update();
        typename ImageType::Pointer mask = readerMask->GetOutput();
        
        ImageIterator it( mask, mask->GetRequestedRegion() );
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            if (it.Get()==true)
                pointsImage.push_back(it.GetIndex());
            ++it;
        }
        
        // search along points the maximum values of the mask
        for (int i=0; i<dims.size(); i++) {
            vector<float> elements;
            for (int j=0; j<pointsImage.size(); j++)
                elements.push_back(pointsImage[j][dims[i]]);
            startSlices.push_back(*min_element(elements.begin(),elements.end()) - shiftSlices[i]);
            endSlices.push_back(*max_element(elements.begin(),elements.end()) + shiftSlices[i]);
        }
	}
    else
    {
        for (int i=0; i<dims.size(); i++) {
            if (startSlices[i] > 0.0 && startSlices[i] < 1.0) startSlices[i] = desiredSize1[dims[i]]*startSlices[i];
            if (endSlices[i] > 0.0 && endSlices[i] < 1.0) endSlices[i] = desiredSize1[dims[i]]*endSlices[i];
            else if (endSlices[i] < 0) endSlices[i] = desiredSize1[dims[i]] + endSlices[i] - 1.0;
            else if (endSlices[i] == 0.0) endSlices[i] = desiredSize1[dims[i]]-1;
        }
        
		
	}
    
	typename ImageType::IndexType desiredStart1;
    desiredStart1.Fill(0);
    for (int i=0; i<dims.size(); i++) {
        desiredStart1[dims[i]] = startSlices[i];
        desiredSize1[dims[i]] = endSlices[i]-startSlices[i]+1;
    }
    
	if (realCrop)
	{
		typename ImageType::RegionType desiredRegion1(desiredStart1, desiredSize1);
		typename CropFilterType::Pointer cropFilter1 = CropFilterType::New();
		cropFilter1->SetExtractionRegion(desiredRegion1);
		cropFilter1->SetInput(image);
        #if ITK_VERSION_MAJOR >= 4
		cropFilter1->SetDirectionCollapseToIdentity(); // This is required.
        #endif
		try {
			cropFilter1->Update();
		} catch( itk::ExceptionObject & e ) {
			std::cerr << "Exception caught while updating cropFilter " << std::endl;
			std::cerr << e << std::endl;
		}
		image = cropFilter1->GetOutput();
	}
	else
	{
		ImageIterator it( image, image->GetRequestedRegion() );
		it.GoToBegin();
		while(!it.IsAtEnd()) {
            bool isOutsideBox = false;
            for (int i=0; i<dims.size(); i++) {
                if (it.GetIndex()[dims[i]] > endSlices[i] || it.GetIndex()[dims[i]] < startSlices[i]) {
                    isOutsideBox = true;
                    break;
                }
            }
			if (isOutsideBox)
				it.Set(backgroundValue);
			++it;
		}
	}
    
	typename WriterType::Pointer writerCrop = WriterType::New();
	writerCrop->SetImageIO(io);
	writerCrop->SetFileName(outputFilename);
	writerCrop->SetInput(image);
	try {
		writerCrop->Update();
	} catch( itk::ExceptionObject & e ) {
		std::cerr << "Exception caught while updating writerCrop " << std::endl;
		std::cerr << e << std::endl;
	}
    
    // Cropping only in Z direction for now in AIL orientation. To Do: integrate all dimensions
	if (isMesh)
	{
		SpinalCord *mesh = new SpinalCord();
		mesh->read(meshFilename);
        
		typename ImageType::IndexType downIndex, downIndex2, upperIndex, upperIndex2;
		typename ImageType::PointType downPoint, downPoint1, upperPoint, upperPoint2;
		downIndex[0] = 0; downIndex[1] = startSlices[1]; downIndex[2] = 0;
		downIndex2[0] = 0; downIndex2[1] = startSlices[1]-1; downIndex2[2] = 0;
		upperIndex[0] = 0; upperIndex[1] = endSlices[1]; upperIndex[2] = 0;
		upperIndex2[0] = 0; upperIndex2[1] = endSlices[1]+1; upperIndex2[2] = 0;
		image->TransformIndexToPhysicalPoint(downIndex,downPoint);
		image->TransformIndexToPhysicalPoint(downIndex2,downPoint1);
		image->TransformIndexToPhysicalPoint(upperIndex,upperPoint);
		image->TransformIndexToPhysicalPoint(upperIndex2,upperPoint2);
		CVector3	downSlice(downPoint[0],downPoint[1],downPoint[2]),
        downNormal(downPoint1[0]-downPoint[0],downPoint1[1]-downPoint[1],downPoint1[2]-downPoint[2]),
        upperSlice(upperPoint[0],upperPoint[1],upperPoint[2]),
        upperNormal(upperPoint2[0]-upperPoint[0],upperPoint2[1]-upperPoint[1],upperPoint2[2]-upperPoint[2]);
		mesh->reduceMeshUpAndDown(downSlice,downNormal,upperSlice,upperNormal,meshFilename);
	}

    return EXIT_SUCCESS;
}
