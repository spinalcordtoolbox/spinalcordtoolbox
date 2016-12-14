//
//  main.cpp
//  isct_bsplineapproximator
//
//  Created by Benjamin De Leener on 2015-03-18.
//  Copyright (c) 2015 NeuroPoly. All rights reserved.
//
// std libraries
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>

// local references
#include "BSplineApproximation.h"
#include "util/Vector3.h"
#include "OrientImage.h"

// ITK libraries
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkNiftiImageIO.h>

using namespace std;

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
    cout << "isct_bsplineapproximator - Version 0.1 (2015-03-18)" << endl;
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \nPart of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox> \nAuthor : Benjamin De Leener" << endl << endl;
    
    cout << "DESCRIPTION" << endl;
    cout << "This program smooth a set of point using BSpline approximation." << endl;
    
    cout << "USAGE" << endl;
    cout << "  isct_bsplineapproximator -i <inputfilename> -o <outputfilename> [options]" << endl << endl;
    
    cout << "MANDATORY ARGUMENTS" << endl;
    cout << StrPad("  -i <inputfilename>",30) << StrPad("no default",70,StrPad("",30)) << endl;
    cout << StrPad("  -o <outputfilename>",30) << StrPad("no default",70,StrPad("",30)) << endl;
    cout << endl;
    
    // Output files
    cout << "OPTIONAL ARGUMENTS" << endl;
    cout << StrPad("  -n <numberofpoints>",30) << StrPad("int, number of points, default is the same number as input",70,StrPad("",30)) << endl;
    cout << StrPad("  -l <numberOfLevels>",30) << StrPad("int, number of levels of BSpline approximator, default is 15",70,StrPad("",30)) << endl;
    cout << StrPad("  -help",30) << endl;
    cout << endl;
}

int main(int argc, const char * argv[])
{
    if (argc == 1)
    {
        help();
        return EXIT_FAILURE;
    }
    
    string inputFilename = "", outputFilename = "";
    int numberOfPoints = 0, numberOfLevels = 15;
    bool worldCoordinate = false, numberOfSlices = true;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i],"-i")==0) {
            i++;
            inputFilename = argv[i];
        }
        else if (strcmp(argv[i],"-o")==0) {
            i++;
            outputFilename = argv[i];
        }
        else if (strcmp(argv[i],"-n")==0) {
            i++;
            numberOfPoints = atoi(argv[i]);
            numberOfSlices = false;
        }
        else if (strcmp(argv[i],"-l")==0) {
            i++;
            numberOfLevels = atoi(argv[i]);
        }
        else if (strcmp(argv[i],"-help")==0) {
            help();
            return EXIT_FAILURE;
        }
    }
    
    if (inputFilename == "" || outputFilename == "")
    {
        cerr << "ERROR: Input filename and output filename have to be provided." << endl;
        help();
        return EXIT_FAILURE;
    }
    
    // If the inputfile is a nifti file, extract the centerline and write it into .txt file
    typedef itk::Image< double, 3 >	ImageType;
    ImageType::Pointer image_centerline;
    bool inputIsImage = false;
    
    vector<CVector3> centerline;
    string nii=".nii", niigz=".nii.gz", txt=".txt", suffix="";
    size_t pos_niigz = inputFilename.find(niigz), pos_nii = inputFilename.find(nii), pos_txt = inputFilename.find(txt);
    if (pos_niigz != string::npos || pos_nii != string::npos)
    {
        inputIsImage = true;
        
        typedef itk::ImageFileReader<ImageType> ReaderType;
        ReaderType::Pointer reader = ReaderType::New();
        itk::NiftiImageIO::Pointer io = itk::NiftiImageIO::New();
        reader->SetImageIO(io);
        reader->SetFileName(inputFilename);
        try {
            reader->Update();
        } catch( itk::ExceptionObject & e ) {
            cerr << "Exception caught while reading centerline input image " << endl;
            cerr << e << endl;
        }
        image_centerline = reader->GetOutput();
        
        OrientImage<ImageType> orientationFilter;
        orientationFilter.setInputImage(image_centerline);
        orientationFilter.orientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI);
        image_centerline = orientationFilter.getOutputImage();
        
        ImageType::IndexType ind;
        itk::Point<double,3> point;
        typedef itk::ImageRegionConstIterator<ImageType> ImageIterator;
        ImageIterator it( image_centerline, image_centerline->GetRequestedRegion() );
        it.GoToBegin();
        int dim = 2;
        while(!it.IsAtEnd())
        {
            if (it.Get()!=0)
            {
                ind = it.GetIndex();
                
                if (worldCoordinate) {
                    image_centerline->TransformIndexToPhysicalPoint(ind, point);
                }
                else {
                    point[0] = ind[0];
                    point[1] = ind[1];
                    point[2] = ind[2];
                }
                bool added = false;
                if (centerline.size() == 0) {
                    centerline.push_back(CVector3(point[0],point[1],point[2]));
                    added = true;
                }
                else {
                    for (vector<CVector3>::iterator it=centerline.begin(); it!=centerline.end(); it++) {
                        if (point[dim] < (*it)[dim]) {
                            centerline.insert(it, CVector3(point[0],point[1],point[2]));
                            added = true;
                            break;
                        }
                    }
                }
                if (!added) centerline.push_back(CVector3(point[0],point[1],point[2]));
            }
            ++it;
        }
        
        // check if each slice contains more than one voxel
        double slice = -1000000.0;
        CVector3 temp;
        int countVox = 0;
        vector<CVector3> centerline_temp;
        for (int k=0; k<centerline.size(); k++)
        {
            if (centerline[k][dim] == slice) {
                temp += centerline[k];
                countVox++;
            }
            else if (centerline[k][dim] > slice) {
                if (k != 0)
                    centerline_temp.push_back(temp/countVox);
                countVox = 1;
                temp = centerline[k];
                slice = centerline[k][dim];
            }
        }
        centerline_temp.push_back(temp/countVox);
        centerline = centerline_temp;
    }
    else if (pos_txt != string::npos)
    {
        ifstream myfile;
        string l;
        double x, y, z;
        CVector3 point, pointPrecedent;
        int i = 0;
        myfile.open(inputFilename.c_str());
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
                    centerline.push_back(point);
                }
                i++;
            }
        }
        myfile.close();
    }
    else
    {
        cerr << "Error: Centerline input file not supported" << endl;
        return EXIT_FAILURE;
    }
    
    // Compute BSpline approximation on centerline
    BSplineApproximation bspline = BSplineApproximation(&centerline, numberOfLevels);
    
    // Generate new centerline
    double numberOfPointsSpline = 5000;
    if (!numberOfSlices)
        numberOfPointsSpline = numberOfPoints;
    
    vector<CVector3> centerlineDerivative;
    centerline.clear();
    typedef itk::Vector<double, 3> VectorType;
    CVector3 V, Vd;
    double value;
    for (double i=0; i<numberOfPointsSpline; i++)
    {
        value = i/(numberOfPointsSpline-1);
        centerline.push_back(bspline.EvaluateBSpline(value));
        centerlineDerivative.push_back(bspline.EvaluateGradient(value).Normalize());
    }

    
    if (numberOfSlices && !worldCoordinate && inputIsImage)
    {
        int dim = 2;
        int numberOfSlices = image_centerline->GetLargestPossibleRegion().GetSize()[dim];
        
        int currentPoint = 0;
        vector<CVector3> temp_vector_centerline, temp_vector_derivative;
        for (int slice=0; slice<numberOfSlices; slice++)
        {
            CVector3 temp_centerline, temp_derivative;
            int numberOfPointInSlice = 0;
            while (round(centerline[currentPoint][dim]) == slice)
            {
                temp_centerline += centerline[currentPoint];
                temp_derivative += centerlineDerivative[currentPoint];
                currentPoint++;
                numberOfPointInSlice++;
            }
            if (numberOfPointInSlice != 0)
            {
                temp_centerline /= numberOfPointInSlice;
                temp_centerline[dim] = slice;
                temp_vector_centerline.push_back(temp_centerline);
                temp_vector_derivative.push_back(temp_derivative/numberOfPointInSlice);
            }
        }
        centerline = temp_vector_centerline;
        centerlineDerivative = temp_vector_derivative;
    }
    
    // Write new centerline into text file
    ofstream outputfile;
    outputfile.open(outputFilename.c_str());
    for (double i=0; i<centerline.size(); i++)
        outputfile << centerline[i][0] << " " << centerline[i][1] << " " << centerline[i][2] << " " << centerlineDerivative[i][0] << " " << centerlineDerivative[i][1] << " " << centerlineDerivative[i][2] << endl;
    outputfile.close();
    
    return 0;
}
