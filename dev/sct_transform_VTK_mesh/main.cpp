//
//  main.cpp
//  sct_transform_VTK_mesh
//
//  Created by Benjamin De Leener on 2014-01-14.
//  Copyright (c) 2014 Benjamin De Leener. All rights reserved.
//

#include "Mesh.h"
#include "Matrix3x3.h"
#include <iostream>
using namespace std;

void help()
{
    cout << "sct_transform_VTK_mesh - Version 0.1" << endl;
    cout << "Author : Benjamin De Leener - NeuroPoly lab <www.neuropoly.info>" << endl << endl;
	
	cout << "This program apply a transformation on a VTK mesh." << endl << endl;
    
    cout << "Usage : " << endl << "\t sct_transform_VTK_mesh -i <inputfilename> -o <outputfilename> [options]" << endl;
    
    cout << "Available options : " << endl;
    cout << "\t-i <inputfilename> \t (no default)" << endl;
    cout << "\t-o <outputfilename> \t (no default)" << endl;
    cout << "\t-dim <dimension> \t (dimension to crop, default is 1)" << endl;
    cout << "\t-help" << endl;
}

int main(int argc, const char * argv[])
{
    string filenameInput = "", filenameOutput = "";
    int dim = 1;
    double rotation = 0.0;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i],"-i")==0) {
            i++;
            filenameInput = argv[i];
        }
        else if (strcmp(argv[i],"-o")==0) {
            i++;
            filenameOutput = argv[i];
        }
        else if (strcmp(argv[i],"-help")==0) {
            help();
            return EXIT_FAILURE;
        }
    }
    if (filenameInput == "") {
        cerr << "Input filename not provided" << endl;
		help();
        return EXIT_FAILURE;
    }
    if (filenameOutput == "") {
        filenameOutput = filenameInput;
        cout << "Output filename not provided. Input image will be overwritten" << endl;
    }
    
	Mesh *m = new Mesh();
	m->read(filenameInput);
    
	CMatrix3x3 trZ; trZ[0] = -1; trZ[4] = -1;
    m->transform(trZ);
    m->save(filenameOutput);
    
    return EXIT_SUCCESS;
}

