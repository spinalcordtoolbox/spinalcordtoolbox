//
//  main.cpp
//  sct_dice_coefficient
//
//  Created by Benjamin De Leener on 2013-11-08.
//  Copyright (c) 2013 Benjamin De Leener. All rights reserved.
//

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkNiftiImageIO.h>
#include <itkExtractImageFilter.h>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkBinaryThresholdImageFilter.h>

#include <iostream>
#include <string>
using namespace std;

typedef itk::Image< double, 3 >	ImageType;
typedef itk::Image< unsigned char, 3 >	BinaryImageType;
typedef itk::Image< unsigned char, 2 >	BinaryImageType2D;
typedef itk::BinaryThresholdImageFilter< ImageType, BinaryImageType > ThresholdType;
typedef itk::ImageFileReader<ImageType> ReaderType;
typedef itk::ImageFileReader<BinaryImageType> BinaryReaderType;
typedef itk::ExtractImageFilter< BinaryImageType, BinaryImageType > FilterType3D;
typedef itk::ExtractImageFilter< BinaryImageType, BinaryImageType2D > FilterType2D;
typedef itk::LabelOverlapMeasuresImageFilter< BinaryImageType > DiceFilter3D;
typedef itk::LabelOverlapMeasuresImageFilter< BinaryImageType2D > DiceFilter2D;
typedef itk::ImageRegionIterator<BinaryImageType> ImageIterator;

void help()
{
    cout << "sct_dice_coefficient - Version 0.3" << endl;
    cout << "Author : Benjamin De Leener - NeuroPoly lab <www.neuropoly.info> " << endl << endl;
    
    cout << "Usage : \t sct_dice_coefficient <filename_target> <filename_source>" << endl;
    cout << "Usage : \t sct_dice_coefficient <filename_target> <filename_source> -b <xindex> <xsize> <yindex> <ysize> <zindex> <zsize>" << endl;
    cout << "Usage : \t sct_dice_coefficient <filename_target> <filename_source> -bmax" << endl << endl;
    
    cout << "Available options : " << endl;
    cout << "\t-b <xindex> <xsize> <yindex> <ysize> <zindex> <zsize> \t (int, bounding box [origin(x,y,z) ; size(x,y,z)])" << endl;
    cout << "\t-bmax \t (use maximum bounding box of of the images union to compute DC)" << endl;
    cout << "\t-bzmax \t (use maximum bounding box of of the images union in the 'z' direction)" << endl;
    cout << "\t-bin \t (binarize the image before computing DC, put non-zero voxels to 1)" << endl;
    cout << "\t-o <filename> \t (output file .txt with Dice coefficient result)" << endl;
    cout << "\t-2d-slices <dim> \t (int, performs Dice coefficient on 2D slices in dimensions 'dim' {0,1,2})" << endl;
    cout << "\t-v \t (hide display)" << endl;
    cout << "\t-help" << endl;
    cout << endl << "Note: indexing (in both time and space) starts with 0 not 1! Inputting -1 for a size will set it to the full image extent for that dimension." << endl;
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

int main(int argc, const char * argv[])
{
    string filename_target, filename_source, filename_output;
    bool boundingBox = false, maxBoundingBox = false, maxZBoundingBox = false, verbose = true, slices = false, need2binarize = false;
    int boundingBoxIndex[3], boundingBoxSize[3], dimension = -1;
    double threshold_target = 0.0001, threshold_source = 0.0001;
    
    if (argc < 3)
    {
        cout << endl << "ERROR: not enough arguments..." << endl << endl;
        help();
        return EXIT_FAILURE;
    }
    filename_target = argv[1];
    filename_source = argv[2];
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i],"-b")==0)
        {
            if (argc != 10) {
                cout << endl << "ERROR: not enough argument for bounding box. b and -bmax can't be called at the same time." << endl << endl;
                help();
                return EXIT_FAILURE;
            } else {
                boundingBox = true;
                boundingBoxIndex[0] = atoi(argv[i+1]);
                boundingBoxSize[0] = atoi(argv[i+2]);
                boundingBoxIndex[1] = atoi(argv[i+3]);
                boundingBoxSize[1] = atoi(argv[i+4]);
                boundingBoxIndex[2] = atoi(argv[i+5]);
                boundingBoxSize[2] = atoi(argv[i+6]);
            }
            i += 6;
        }
        else if (strcmp(argv[i],"-bmax")==0)
        {
            maxBoundingBox = true;
        }
        else if (strcmp(argv[i],"-bzmax")==0)
        {
            maxZBoundingBox = true;
        }
        else if (strcmp(argv[i],"-bin")==0)
        {
            i++;
            need2binarize = true;
            string thr = argv[i];
            vector<string> thresholds = split(thr,',');
            threshold_target = atof(thresholds[0].c_str());
            threshold_source = atof(thresholds[1].c_str());
        }
        else if (strcmp(argv[i],"-o")==0)
        {
            i++;
            filename_output = argv[i];
        }
        else if (strcmp(argv[i],"-v")==0)
        {
            verbose = false;
        }
        else if (strcmp(argv[i],"-2d-slices")==0)
        {
            i++;
            slices = true;
            dimension = atoi(argv[i]);
        }
        else if (strcmp(argv[i],"-help")==0)
        {
            help();
            return EXIT_FAILURE;
        }
    }

    BinaryImageType::Pointer imageTarget;
    BinaryImageType::Pointer imageSource;
    itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
    
    if (need2binarize)
    {
        ReaderType::Pointer reader_target = ReaderType::New();
        reader_target->SetImageIO(io);
        reader_target->SetFileName(filename_target);
        reader_target->Update();
        
        ThresholdType::Pointer thresholdFilter_target = ThresholdType::New();
        thresholdFilter_target->SetInput(reader_target->GetOutput());
        thresholdFilter_target->SetLowerThreshold(threshold_target);
        //thresholdFilter_target->SetUpperThreshold(1);
        thresholdFilter_target->SetInsideValue(1);
        thresholdFilter_target->SetOutsideValue(0);
        thresholdFilter_target->Update();
        
        imageTarget = thresholdFilter_target->GetOutput();
        
        ReaderType::Pointer reader_source = ReaderType::New();
        reader_source->SetImageIO(io);
        reader_source->SetFileName(filename_source);
        reader_source->Update();
        
        ThresholdType::Pointer thresholdFilter_source = ThresholdType::New();
        thresholdFilter_source->SetInput(reader_source->GetOutput());
        thresholdFilter_source->SetLowerThreshold(threshold_source);
        //thresholdFilter_source->SetUpperThreshold(1);
        thresholdFilter_source->SetInsideValue(1);
        thresholdFilter_source->SetOutsideValue(0);
        thresholdFilter_source->Update();
        
        imageSource = thresholdFilter_source->GetOutput();
    }
    else
    {
        BinaryReaderType::Pointer readerRef = BinaryReaderType::New();
        readerRef->SetImageIO(io);
        readerRef->SetFileName(filename_target);
        readerRef->Update();
        imageTarget = readerRef->GetOutput();
        
        BinaryReaderType::Pointer readerSeg = BinaryReaderType::New();
        readerSeg->SetImageIO(io);
        readerSeg->SetFileName(filename_source);
        readerSeg->Update();
        imageSource = readerSeg->GetOutput();
    }
    
	BinaryImageType::PointType origin = imageTarget->GetOrigin();
	BinaryImageType::PointType originSource = imageSource->GetOrigin();
    
	double dist = sqrt((origin[0]-originSource[0])*(origin[0]-originSource[0])+(origin[1]-originSource[1])*(origin[1]-originSource[1])+(origin[2]-originSource[2])*(origin[2]-originSource[2]));
	if (dist >= 0.00001 && dist < 0.001)
		imageSource->SetOrigin(origin);
    
	BinaryImageType::SizeType sizeTarget = imageTarget->GetLargestPossibleRegion().GetSize(), desiredSize = sizeTarget;
	BinaryImageType::SizeType sizeSource = imageSource->GetLargestPossibleRegion().GetSize();
	BinaryImageType::IndexType desiredStart; desiredStart.Fill(0);
	
    if (boundingBox)
    {
        desiredStart[0] = boundingBoxIndex[0];
        desiredStart[1] = boundingBoxIndex[1];
        desiredStart[2] = boundingBoxIndex[2];
    
        if (boundingBoxSize[0] != -1) desiredSize[0] = boundingBoxSize[0];
        if (boundingBoxSize[1] != -1) desiredSize[1] = boundingBoxSize[1];
        if (boundingBoxSize[2] != -1) desiredSize[2] = boundingBoxSize[2];
    }
    else if (maxBoundingBox)
    {
        ImageIterator itTarget( imageTarget, imageTarget->GetLargestPossibleRegion() );
        itTarget.GoToBegin();
        BinaryImageType::PixelType pixelTarget;
        BinaryImageType::IndexType indexTarget, start, end;
        start.Fill(0); end.Fill(0);
        while( !itTarget.IsAtEnd() )
        {
            indexTarget = itTarget.GetIndex();
            pixelTarget = itTarget.Get() && imageSource->GetPixel(indexTarget);
            if (pixelTarget != 0)
            {
                if (start[0]==0 && start[1]==0 && start[2]==0)
                {
                    start[0] = indexTarget[0];
                    start[1] = indexTarget[1];
                    start[2] = indexTarget[2];
                }
                if (end[0]==0 && end[1]==0 && end[2]==0)
                {
                    end[0] = indexTarget[0];
                    end[1] = indexTarget[1];
                    end[2] = indexTarget[2];
                }

                if (indexTarget[0] < start[0]) start[0] = indexTarget[0];
                if (indexTarget[1] < start[1]) start[1] = indexTarget[1];
                if (indexTarget[2] < start[2]) start[2] = indexTarget[2];
                if (indexTarget[0] > end[0]) end[0] = indexTarget[0];
                if (indexTarget[1] > end[1]) end[1] = indexTarget[1];
                if (indexTarget[2] > end[2]) end[2] = indexTarget[2];
            }
            ++itTarget;
        }
        desiredStart[0] = start[0];
        desiredStart[1] = start[1];
        desiredStart[2] = start[2];
        desiredSize[0] = end[0]-start[0]+1;
        desiredSize[1] = end[1]-start[1]+1;
        desiredSize[2] = end[2]-start[2]+1;
        
        if (verbose)
        {
            cout << "WARNING: please check bounding box" << endl;
            cout << "Origin: \t" << desiredStart[0] << "\t" << desiredStart[1] << "\t" << desiredStart[2] << endl;
            cout << "Size: \t\t" << desiredSize[0] << "\t" << desiredSize[1] << "\t" << desiredSize[2] << endl << endl;
        }
    }
    else if (maxZBoundingBox)
    {
	BinaryImageType::RegionType region = imageTarget->GetLargestPossibleRegion();
        ImageIterator itTarget( imageTarget, region );
        itTarget.GoToBegin();
        BinaryImageType::PixelType pixelTarget;
        BinaryImageType::IndexType indexTarget, start, end;
	BinaryImageType::SizeType sizeRegion=region.GetSize();
        start.Fill(0); end.Fill(0);
	start[0] = 0;
	start[2] = 0;
	end[0] = sizeRegion[0]-1;
	end[2] = sizeRegion[2]-1;
        while( !itTarget.IsAtEnd() )
        {
            indexTarget = itTarget.GetIndex();
            pixelTarget = itTarget.Get() && imageSource->GetPixel(indexTarget);
            if (pixelTarget != 0)
            {
                if (start[1]==0)
                    start[1] = indexTarget[1];
                if (end[1]==0)
                    end[1] = indexTarget[1];

                if (indexTarget[1] < start[1]) start[1] = indexTarget[1];
                if (indexTarget[1] > end[1]) end[1] = indexTarget[1];
            }
            ++itTarget;
        }
        desiredStart[0] = start[0];
        desiredStart[1] = start[1];
        desiredStart[2] = start[2];
        desiredSize[0] = end[0]-start[0]+1;
        desiredSize[1] = end[1]-start[1]+1;
        desiredSize[2] = end[2]-start[2]+1;
        
        if (verbose)
        {
            cout << "WARNING: please check bounding box" << endl;
            cout << "Origin: \t" << desiredStart[0] << "\t" << desiredStart[1] << "\t" << desiredStart[2] << endl;
            cout << "Size: \t\t" << desiredSize[0] << "\t" << desiredSize[1] << "\t" << desiredSize[2] << endl << endl;
        }
    }
    
    ofstream myfile;
    if (filename_output != "")
        myfile.open(filename_output.c_str());
    
    if (maxBoundingBox)
    {
        myfile << "WARNING: please check bounding box" << endl;
        myfile << "Origin: \t" << desiredStart[0] << "\t" << desiredStart[1] << "\t" << desiredStart[2] << endl;
        myfile << "Size: \t\t" << desiredSize[0] << "\t" << desiredSize[1] << "\t" << desiredSize[2] << endl << endl;
    }

    
	// Computing 3D Dice Coefficient
	BinaryImageType::RegionType desiredRegion(desiredStart, desiredSize);
	FilterType3D::Pointer filterGlobal1 = FilterType3D::New();
#if ITK_VERSION_MAJOR >= 4
	filterGlobal1->SetDirectionCollapseToIdentity(); // This is required.
#endif
	filterGlobal1->SetExtractionRegion(desiredRegion);
	filterGlobal1->SetInput(imageTarget);
	filterGlobal1->Update();
	BinaryImageType::Pointer croppedImageReference = filterGlobal1->GetOutput();
	FilterType3D::Pointer filterGlobal2 = FilterType3D::New();
#if ITK_VERSION_MAJOR >= 4
	filterGlobal2->SetDirectionCollapseToIdentity(); // This is required.
#endif
	filterGlobal2->SetExtractionRegion(desiredRegion);
	filterGlobal2->SetInput(imageSource);
	filterGlobal2->Update();
	BinaryImageType::Pointer croppedImageSegmentation = filterGlobal2->GetOutput();
	DiceFilter3D::Pointer diceFilter3D = DiceFilter3D::New();
	diceFilter3D->SetSourceImage(croppedImageReference);
	diceFilter3D->SetTargetImage(croppedImageSegmentation);
	try {
		diceFilter3D->Update();
	} catch( itk::ExceptionObject & e ) {
		cerr << "Exception caught while updating diceFilter3D " << endl;
		cerr << e << std::endl;
        return EXIT_FAILURE;
	}
    
	double DiceCoefficient3D = diceFilter3D->GetDiceCoefficient();
    if (verbose) cout << "3D Dice coefficient = " << DiceCoefficient3D << endl << endl;
    
    if (filename_output != "")
        myfile << "3D Dice coefficient = " << DiceCoefficient3D << endl;
    
    // 2D Dice coefficient on slices in the input dimension
    if (slices)
    {
        DiceFilter2D::Pointer diceFilter2D = DiceFilter2D::New();
        vector<double> vectorDiceCoefficients;
        int startslice = desiredStart[dimension], endSlice = desiredStart[dimension]+desiredSize[dimension]-1;
        
        FilterType2D::Pointer filter1 = FilterType2D::New();
        #if ITK_VERSION_MAJOR >= 4
        filter1->SetDirectionCollapseToIdentity(); // This is required.
        #endif
        FilterType2D::Pointer filter2 = FilterType2D::New();
        #if ITK_VERSION_MAJOR >= 4
        filter2->SetDirectionCollapseToIdentity(); // This is required.
        #endif
        
        desiredSize[dimension] = 0;
        
        for (int slice=startslice; slice<=endSlice; slice++)
        {
            //cout << "Slice num " << slice << " / " << endZ << endl;
            desiredStart[dimension] = slice;
            BinaryImageType::RegionType desiredRegionRef(desiredStart, desiredSize), desiredRegionSeg(desiredStart, desiredSize);
            
            filter1->SetExtractionRegion(desiredRegionRef);
            filter1->SetInput(croppedImageReference);
            try {
                filter1->Update();
            } catch( itk::ExceptionObject & e ) {
                cerr << "Exception caught while updating 2D extraction on target on slice " << slice << endl;
                cerr << e << std::endl;
                return EXIT_FAILURE;
            }
            BinaryImageType2D::Pointer image2DReference = filter1->GetOutput();
            
            filter2->SetExtractionRegion(desiredRegionSeg);
            filter2->SetInput(croppedImageSegmentation);
            try {
                filter2->Update();
            } catch( itk::ExceptionObject & e ) {
                cerr << "Exception caught while updating 2D extraction on source on slice " << slice << endl;
                cerr << e << std::endl;
                return EXIT_FAILURE;
            }
            BinaryImageType2D::Pointer image2DSegmentation = filter2->GetOutput();
            
            diceFilter2D->SetSourceImage(image2DReference);
            diceFilter2D->SetTargetImage(image2DSegmentation);
            diceFilter2D->Update();
            
            vectorDiceCoefficients.push_back(diceFilter2D->GetDiceCoefficient());
        }
        
        unsigned int nbrSlices = vectorDiceCoefficients.size();
        // Computing mean dice coefficient
        double meanDiceCoefficient = 0.0;
        for (int i=0; i<nbrSlices; i++)
            meanDiceCoefficient += vectorDiceCoefficients[i];
        meanDiceCoefficient /= nbrSlices;
        
        // Computing median dice coefficient
        double medianDiceCoefficient = 0.0;
        vector<double> temp = vectorDiceCoefficients;
        sort(temp.begin(), temp.end());
        if (nbrSlices%2 == 0)
            medianDiceCoefficient = (temp[nbrSlices / 2 - 1] + temp[nbrSlices / 2]) / 2;
        else
            medianDiceCoefficient = temp[nbrSlices / 2];
        
        // Computing standard deviation and variance
        double standardDeviationDiceCoefficient = 0.0, varianceDiceCoefficient = 0.0;
        for (int i=0; i<nbrSlices; i++)
            varianceDiceCoefficient += (vectorDiceCoefficients[i]-meanDiceCoefficient)*(vectorDiceCoefficients[i]-meanDiceCoefficient);
        varianceDiceCoefficient = varianceDiceCoefficient/nbrSlices;
        standardDeviationDiceCoefficient = sqrt(varianceDiceCoefficient);
        
        
        if (filename_output != "") {
            myfile << endl << "Slice Dice coefficient on dimension " << dimension << endl;
            myfile << "Mean 2D Dice coefficient = " << meanDiceCoefficient << endl;
            myfile << "Median 2D Dice coefficient = " << medianDiceCoefficient << endl;
            myfile << "Standard Deviation = " << standardDeviationDiceCoefficient << endl;
            myfile << "Variance = " << varianceDiceCoefficient << endl << endl;
            myfile << "2D Dice coefficient by slice:" << endl;
            for (int i=0; i<nbrSlices; i++)
                myfile << i << " " << vectorDiceCoefficients[i] << endl;
        }
        
        if (verbose) {
            cout << "Slice Dice coefficient on dimension " << dimension << endl;
            cout << "Mean 2D Dice coefficient = " << meanDiceCoefficient << endl;
            cout << "Median 2D Dice coefficient = " << medianDiceCoefficient << endl;
            cout << "Standard Deviation = " << standardDeviationDiceCoefficient << endl;
            cout << "Variance = " << varianceDiceCoefficient << endl << endl;
            cout << "2D Dice coefficient by slice:" << endl;
            for (int i=0; i<nbrSlices; i++)
                cout << i+startslice << " " << vectorDiceCoefficients[i] << endl;
        }
    }
    
    myfile.close();
    
        
    return EXIT_SUCCESS;
}

