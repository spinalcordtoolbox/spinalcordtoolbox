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
	
	cout << "This program crop a binary image as a segmentation above and below index values." << endl << "If a mask (-m) is provided, all values above and below down and up slices are replaced by zero values. If not so, the image is really cropped." << endl << endl;
    
    cout << "Usage : " << endl << "\t sct_crop_image -i <inputfilename> -o <outputfilename> [options]" << endl;
    cout << "\t sct_crop_image -i <inputfilename> -o <outputfilename> -m <maskfilename> [options]" << endl<< endl;
    
    cout << "Available options : " << endl;
    cout << "\t-i <inputfilename> \t (no default)" << endl;
    cout << "\t-o <outputfilename> \t (no default)" << endl;
    cout << "\t-m <maskfilename> \t (binary image containing start and end slices, no default ; if mask not provided, the image is really cropped)" << endl;
    cout << "\t-start <startslice> \t (start slice to crop, 0.5 if middle slice)" << endl;
    cout << "\t-end <endslice> \t (end slice to crop, 0.5 if middle slice, -1 if last slice)" << endl;
    cout << "\t-dim <dimension> \t (dimension to crop, default is 1)" << endl;
    cout << "\t-b <backgroundvalue> \t (background value, default is 0.0)" << endl;
    cout << "\t-mesh <meshfilename> \t (mesh to crop)" << endl;
    cout << "\t-help" << endl;
}

template<typename TPixelType>
int transform(string inputFilename, string outputFilename, string maskFilename, string meshFilename, bool isMask, bool isMesh, float backgroundValue, float startSlice, float endSlice, float dim);

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        help();
        return EXIT_FAILURE;
    }
    string inputFilename = "", outputFilename = "", maskFilename = "", meshFilename = "";
    float startSlice = 0.0, endSlice = 0.0, dim = 1.0, backgroundValue = 0.0;
    bool isMask = false, isMesh = false;
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
            startSlice = atof(argv[i]);
        }
        else if (strcmp(argv[i],"-end")==0) {
            i++;
            endSlice = atof(argv[i]);
        }
        else if (strcmp(argv[i],"-dim")==0) {
            i++;
            dim = atof(argv[i]);
        }
        else if (strcmp(argv[i],"-b")==0) {
            i++;
            backgroundValue = atof(argv[i]);
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
    if (!isMask && startSlice == 0.0 && endSlice == 0.0) {
        cerr << "Start and End slice must be provided and != 0 OR mask must be provided" << endl;
		help();
        return EXIT_FAILURE;
    }
    if (outputFilename == "") {
        outputFilename = inputFilename;
        cout << "Output filename not provided. Input image will be overwritten" << endl;
    }
    
    
    
    itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
    io->SetFileName(inputFilename.c_str());
    io->ReadImageInformation();
    typedef itk::ImageIOBase::IOComponentType ScalarPixelType;
    const ScalarPixelType pixelType = io->GetComponentType();
    
    if (io->GetComponentTypeAsString(pixelType)=="char")
        return transform<char>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,startSlice,endSlice,dim);
    else if (io->GetComponentTypeAsString(pixelType)=="unsigned_char")
        return transform<unsigned char>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,startSlice,endSlice,dim);
    else if (io->GetComponentTypeAsString(pixelType)=="short")
        return transform<short>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,startSlice,endSlice,dim);
    else if (io->GetComponentTypeAsString(pixelType)=="unsigned_short")
        return transform<unsigned short>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,startSlice,endSlice,dim);
    else if (io->GetComponentTypeAsString(pixelType)=="int")
        return transform<int>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,startSlice,endSlice,dim);
    else if (io->GetComponentTypeAsString(pixelType)=="unsigned_int")
        return transform<unsigned int>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,startSlice,endSlice,dim);
    else if (io->GetComponentTypeAsString(pixelType)=="long")
        return transform<long>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,startSlice,endSlice,dim);
    else if (io->GetComponentTypeAsString(pixelType)=="unsigned_long")
        return transform<unsigned long>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,startSlice,endSlice,dim);
    else if (io->GetComponentTypeAsString(pixelType)=="float")
        return transform<float>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,startSlice,endSlice,dim);
    else if (io->GetComponentTypeAsString(pixelType)=="double")
        return transform<double>(inputFilename,outputFilename,maskFilename,meshFilename,isMask,isMesh,backgroundValue,startSlice,endSlice,dim);
    else {
        cout << "Pixel type " << io->GetComponentTypeAsString(pixelType) << " is not supported" << endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

template<typename TPixelType>
int transform(string inputFilename, string outputFilename, string maskFilename, string meshFilename, bool isMask, bool isMesh, float backgroundValue, float startSlice, float endSlice, float dim)
{
    typedef itk::Image< TPixelType, 3 > ImageType;
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
	OrientationType initialOrientation;
	if (isMask) {
		dim = 1;
		OrientImage<ImageType> orientationFilter;
		orientationFilter.setInputImage(image);
		orientationFilter.orientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL);
		image = orientationFilter.getOutputImage();
		initialOrientation = orientationFilter.getInitialImageOrientation();
        
        vector<typename ImageType::IndexType> pointsImage;
        typename ReaderType::Pointer readerMask = ReaderType::New();
        itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
        readerMask->SetImageIO(io);
        readerMask->SetFileName(maskFilename);
        readerMask->Update();
        typename ImageType::Pointer mask = readerMask->GetOutput();
        
        OrientImage<ImageType> orientationFilterPoints;
        orientationFilterPoints.setInputImage(mask);
        orientationFilterPoints.orientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL);
        mask = orientationFilterPoints.getOutputImage();
        
        ImageIterator it( mask, mask->GetRequestedRegion() );
        it.GoToBegin();
        while(!it.IsAtEnd())
        {
            if (it.Get()==true)
                pointsImage.push_back(it.GetIndex());
            ++it;
        }
        
        unsigned long nbPoints = pointsImage.size();
		if (nbPoints != 2) {
			cerr << "There are not enough or too much points in the mask image. Need two points. There are " << nbPoints << " points in the mask." << endl;
			return EXIT_FAILURE;
		} else {
			if (pointsImage[0][1] > pointsImage[1][1]) {
				startSlice = pointsImage[1][1];
				endSlice = pointsImage[0][1];
			} else {
                startSlice = pointsImage[0][1];
				endSlice = pointsImage[1][1];
            }
		}
	} else {
		if (startSlice == 0.5) startSlice = desiredSize1[dim]/2;
		if (endSlice == 0.5) endSlice = desiredSize1[dim]/2;
        else if (endSlice < 0) endSlice = desiredSize1[dim] + endSlice - 1.0;
		else if (endSlice == -1.0) endSlice = desiredSize1[dim]-1;
	}
    
	typename ImageType::IndexType desiredStart1;
    desiredStart1.Fill(0);
    desiredStart1[dim] = startSlice;
	desiredSize1[dim] = endSlice-startSlice+1;
    
	if (!isMask)
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
			if (it.GetIndex()[1] > endSlice || it.GetIndex()[1] < startSlice)
				it.Set(backgroundValue);
			++it;
		}
		OrientImage<ImageType> orientationFilter;
		orientationFilter.setInputImage(image);
		orientationFilter.orientation(initialOrientation);
		image = orientationFilter.getOutputImage();
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
    
	if (isMesh)
	{
		SpinalCord *mesh = new SpinalCord();
		mesh->read(meshFilename);
        
		typename ImageType::IndexType downIndex, downIndex2, upperIndex, upperIndex2;
		typename ImageType::PointType downPoint, downPoint1, upperPoint, upperPoint2;
		downIndex[0] = 0; downIndex[1] = startSlice; downIndex[2] = 0;
		downIndex2[0] = 0; downIndex2[1] = startSlice-1; downIndex2[2] = 0;
		upperIndex[0] = 0; upperIndex[1] = endSlice; upperIndex[2] = 0;
		upperIndex2[0] = 0; upperIndex2[1] = endSlice+1; upperIndex2[2] = 0;
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
