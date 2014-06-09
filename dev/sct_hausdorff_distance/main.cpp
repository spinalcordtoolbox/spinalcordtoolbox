// Copyright (c) 2011 LTSI INSERM U642
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
//     * Neither name of LTSI, INSERM nor the names
// of any contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// This software was downloaded as a soumission in VTK journal and modified to our purpose.

#include "vtkHausdorffDistancePointSetFilter.h"

#include "vtkXMLPolyDataReader.h"
#include "vtkPolyDataReader.h"
#include "vtkAlgorithm.h"
#include "vtkPolyDataWriter.h"
#include "vtkSmartPointer.h"
#include "vtkPointSet.h"
#include "vtkFieldData.h"
#include <vtkPlane.h>
#include <vtkCutter.h>
#include <vtkClipPolyData.h>
#include <iostream>
#include <string>
using namespace std;
int main(int argc,char** argv)
{
    int inputAParam = 0;
    int inputBParam = 0;
    int outputAParam = 0;
    int outputBParam = 0;
    int targetParam = 0;
    int printOnlyParam = 0;
    int xmlReader = 0;
    string filename_output = "";
    double up = 10000.0, down = -10000.0;
    
    for(int i=1;i<argc;i++)
    {
        if(strcmp(argv[i],"-a") == 0 || strcmp(argv[i],"--inputA") == 0)
        {
            inputAParam = i + 1;
        }
        else if(strcmp(argv[i],"-b") == 0 || strcmp(argv[i],"--inputB") == 0)
        {
            inputBParam = i + 1;
        }
        else if(strcmp(argv[i],"-t") == 0 || strcmp(argv[i],"--target") == 0)
        {
            targetParam = i + 1;
        }
        else if(strcmp(argv[i],"-x") == 0 || strcmp(argv[i],"--xml-reader") == 0)
        {
            xmlReader = 1;
        }
        else if(strcmp(argv[i],"--up") == 0)
        {
            up = atof(argv[i+1]);
        }
        else if(strcmp(argv[i],"--down") == 0)
        {
            down = atof(argv[i+1]);
        }
        else if(strcmp(argv[i],"-o") == 0 || strcmp(argv[i],"--output") == 0)
        {
            filename_output = argv[i+1];
        }
    }
    
    if(argc<5)
    {
        cout<<"Missing Parameters!"<<endl;
        cout<<"\nUsage:"<<argv[0]<<endl;
        cout<<"\n(-a , --inputA) InputAFileName"<<endl;
        cout<<"\n(-b , --inputB) InputBFileName"<<endl;
        cout<<"\n(-t , --target) <0 point-to-point/1 point-to-cell> (Optional)"<<endl;
        cout<<"\n(-x , --xml-reader) use vtkXMLPolyDataReader as input reader"<<endl;
        return EXIT_FAILURE;
    }
    
    
    vtkAlgorithm* readerA;
    vtkAlgorithm* readerB;
    
    if( xmlReader )
    {
        vtkXMLPolyDataReader* xmlReaderA = vtkXMLPolyDataReader::New();
        readerA = xmlReaderA;
        
        vtkXMLPolyDataReader* xmlReaderB = vtkXMLPolyDataReader::New();
        readerB = xmlReaderB;
        
        xmlReaderA->SetFileName(argv[inputAParam]);
        xmlReaderB->SetFileName(argv[inputBParam]);
    }
    else
    {
        vtkPolyDataReader* legacyReaderA = vtkPolyDataReader::New();
        readerA = legacyReaderA;
        
        vtkPolyDataReader* legacyReaderB = vtkPolyDataReader::New();
        readerB = legacyReaderB;
        
        legacyReaderA->SetFileName(argv[inputAParam]);
        legacyReaderB->SetFileName(argv[inputBParam]);
    }
    

    vtkSmartPointer<vtkPlane> downPlane = vtkSmartPointer<vtkPlane>::New(), upperPlane = vtkSmartPointer<vtkPlane>::New();
    downPlane->SetOrigin(0,0,down);
    downPlane->SetNormal(0,0,-1);
    upperPlane->SetOrigin(0,0,up);
    upperPlane->SetNormal(0,0,1);
        
    vtkSmartPointer<vtkClipPolyData> downClipperA = vtkSmartPointer<vtkClipPolyData>::New();
    downClipperA->SetInputConnection(readerA->GetOutputPort());
    downClipperA->SetClipFunction(downPlane);
    downClipperA->InsideOutOn();
    downClipperA->Update();
    vtkSmartPointer<vtkClipPolyData> upperClipperA = vtkSmartPointer<vtkClipPolyData>::New();
    upperClipperA->SetInputConnection(downClipperA->GetOutputPort());
    upperClipperA->SetClipFunction(upperPlane);
    upperClipperA->InsideOutOn();
    upperClipperA->Update();
    //meshA = upperClipperA->GetOutput();
        
    vtkSmartPointer<vtkClipPolyData> downClipperB = vtkSmartPointer<vtkClipPolyData>::New();
    downClipperB->SetInputConnection(readerB->GetOutputPort());
    downClipperB->SetClipFunction(downPlane);
    downClipperB->InsideOutOn();
    downClipperB->Update();
    vtkSmartPointer<vtkClipPolyData> upperClipperB = vtkSmartPointer<vtkClipPolyData>::New();
    upperClipperB->SetInputConnection(downClipperB->GetOutputPort());
    upperClipperB->SetClipFunction(upperPlane);
    upperClipperB->InsideOutOn();
    upperClipperB->Update();
    //meshB = upperClipperB->GetOutput();
        
    
    
    vtkSmartPointer<vtkHausdorffDistancePointSetFilter> filter = vtkSmartPointer<vtkHausdorffDistancePointSetFilter>::New();
    //filter->SetInputData(meshA);
    //filter->SetInputData(1,meshB);
    filter->SetInputConnection(upperClipperA->GetOutputPort());
    filter->SetInputConnection(1,upperClipperB->GetOutputPort());
    
    if( atoi(argv[targetParam]) )
        filter->SetTargetDistanceMethod( 0 );
    try {
        filter->Update();
    }
    catch(exception e) {
        cout << e.what() << endl;
    }
    
    double result = static_cast<vtkPointSet*>(filter->GetOutput(0))->GetFieldData()->GetArray("HausdorffDistance")->GetComponent(0,0);
    cout << "Hausdorff distance [mm] = " << result << endl;
    
    if (filename_output != "") {
        ofstream myfile;
        myfile.open(filename_output.c_str());
        myfile << "Hausdorff distance [mm] = " << result << endl;
        myfile.close();
    }
    
    readerA->Delete();
    readerB->Delete();
    
    return( EXIT_SUCCESS );  
}
