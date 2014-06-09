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

//! \class vtkHausdorffDistancePointSetFilter
//! \brief Compute Hausdorff distance between two point sets
//!
//! This class computes the relative and hausdorff distances from two point 
//! sets (input port 0 and input port 1). If no topology is specified (ie.
//! vtkPointSet or vtkPolyData without vtkPolys), the distances are 
//! computed between point location. If polys exist (ie triangulation),
//! the TargetDistanceMethod allows for an interpolation of the cells to 
//! ensure a better minimal distance exploration.
//!
//! The ouputs (port 0 and 1) have the same geometry and topology as its
//! respective input port. Two FieldData arrays are added : HausdorffDistance
//! and RelativeDistance. The former is equal on both outputs whereas the 
//! latter may differ. A PointData containing the specific point minimal 
//! distance is also added to both outputs.
//!
//! \author Frederic Commandeur
//! \author Jerome Velut
//! \author LTSI
//! \date 2011


#ifndef __vtkHausdorffDistancePointSetFilter_h
#define __vtkHausdorffDistancePointSetFilter_h

#include <vtkPointSetAlgorithm.h>

class VTK_EXPORT vtkHausdorffDistancePointSetFilter : public
vtkPointSetAlgorithm
{
public:
    static vtkHausdorffDistancePointSetFilter *New();
    vtkTypeMacro(vtkHausdorffDistancePointSetFilter, vtkPointSetAlgorithm);

    void PrintSelf(ostream &os, vtkIndent indent);

    vtkSetVector2Macro( RelativeDistance, double );
    vtkGetVector2Macro( RelativeDistance, double );

    vtkSetMacro( HausdorffDistance, double );
    vtkGetMacro( HausdorffDistance, double );
    
    vtkSetMacro( TargetDistanceMethod, int );
    vtkGetMacro( TargetDistanceMethod, int );
 

    enum DistanceMethod {POINT_TO_POINT, POINT_TO_CELL};

protected:
    vtkHausdorffDistancePointSetFilter();
    ~vtkHausdorffDistancePointSetFilter();
    int FillInputPortInformation( int port, vtkInformation* info );
    int FillOutputPortInformation( int port, vtkInformation* info );
    int RequestData(vtkInformation *, vtkInformationVector **,vtkInformationVector *); //the function that makes this class work with the vtk pipeline
private:
  
    int TargetDistanceMethod; //!< point-to-point if 0, point-to-cell if 1
    
    double RelativeDistance[2]; //!< relative distance between inputs
    double HausdorffDistance; //!< hausdorff distance (max(relative distance))
};

#endif
