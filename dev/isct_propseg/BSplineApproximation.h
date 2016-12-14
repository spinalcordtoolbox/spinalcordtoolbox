//
//  BSplineApproximation.h
//  sct_segmentation_propagation
//
//  Created by Benjamin De Leener on 2014-04-16.
//  Copyright (c) 2014 Benjamin De Leener. All rights reserved.
//

#ifndef __sct_segmentation_propagation__BSplineApproximation__
#define __sct_segmentation_propagation__BSplineApproximation__

#include <iostream>
#include <vector>
#include "util/Vector3.h"

#include <itkPointSet.h>
#include <itkBSplineScatteredDataPointSetToImageFilter.h>
#include <itkBSplineControlPointImageFunction.h>

using namespace std;

const unsigned int ParametricDimension = 1;
const unsigned int DataDimension = 3;
typedef double RealType;
typedef itk::Vector<RealType, DataDimension> VectorTypeSpline;
typedef itk::Image<VectorTypeSpline, ParametricDimension> ImageTypeSpline;
typedef itk::PointSet <VectorTypeSpline , ParametricDimension > PointSetType;
typedef itk::BSplineScatteredDataPointSetToImageFilter <PointSetType , ImageTypeSpline > FilterTypeSpline;
typedef itk::BSplineControlPointImageFunction < ImageTypeSpline, double > BSplineType;

class BSplineApproximation
{
public:
    BSplineApproximation() {};
    ~BSplineApproximation() {};
    
    BSplineApproximation(vector<CVector3>* centerline)
    {
        double range = centerline->size();
        
        PointSetType::Pointer pointSet = PointSetType::New();
        // Sample the helix.
        for (unsigned long i=0; i<range; i++) {
            PointSetType::PointType point; point[0] = (double)i/(double)(range-1);
            pointSet ->SetPoint( i, point );
            VectorTypeSpline V;
            V[0] = (*centerline)[i][0]; V[1] = (*centerline)[i][1]; V[2] = (*centerline)[i][2];
            pointSet ->SetPointData( i, V );
        }
        
        // Instantiate the filter and set the parameters
        FilterTypeSpline::Pointer filter = FilterTypeSpline::New();
        // Define the parametric domain
        ImageTypeSpline::SpacingType spacing; spacing.Fill( 1.0 ); ImageTypeSpline::SizeType size; size.Fill( 2.0); ImageTypeSpline::PointType origin; origin.Fill( 0.0 );
        ImageTypeSpline::RegionType region(size); FilterTypeSpline::ArrayType closedim; closedim.Fill(0);
        filter->SetSize( size ); filter->SetOrigin( origin ); filter->SetSpacing( spacing ); filter->SetInput( pointSet );
        int splineOrder = 3; filter->SetSplineOrder( splineOrder ); FilterTypeSpline::ArrayType ncps;
        ncps.Fill( splineOrder + 1 ); filter->SetNumberOfControlPoints( ncps ); filter->SetNumberOfLevels( 5 ); filter->SetGenerateOutputImage( false );
        
        try
        {
            filter->Update();
        } catch( itk::ExceptionObject & e ) {
            std::cerr << "Exception caught while creating spline" << std::endl;
            std::cerr << e << std::endl;
        }
        
        bspline = BSplineType::New();
        bspline->SetSplineOrder(filter->GetSplineOrder());
        bspline->SetOrigin(origin);
        bspline->SetSpacing(spacing);
        bspline->SetSize(size);
        bspline->SetInputImage(filter->GetPhiLattice());
    };
    
    CVector3 EvaluateBSpline(double value)
    {
        PointSetType::PointType point; point[0] = value;
        VectorTypeSpline V = bspline->Evaluate( point );
        return CVector3(V[0],V[1],V[2]);
    };
    
    CVector3 EvaluateGradient(double value)
    {
        PointSetType::PointType point; point[0] = value;
        BSplineType::GradientType Vd = bspline->EvaluateGradient( point );
        return CVector3(Vd[0][0],Vd[1][0],Vd[2][0]);
    };
    
    vector<CVector3> EvaluateBSplinePoints(unsigned int numberOfPoints)
    {
        vector<double> points(numberOfPoints,0.0); // 0.0 is the starting number, numberOfPoints is the range size
        transform(points.begin(),points.end(),++points.begin(),bind2nd(plus<double>(),1.0/(numberOfPoints-1))); // 1.0/(numberOfPoints-1) is the increment
        vector<CVector3> centerline;
        for (unsigned int i=0; i<points.size(); i++)
            centerline.push_back(EvaluateBSpline(points[i]));
        return centerline;
    };
    
    double getNearestPoint(CVector3 point, double range)
    {
        double minIndex = 0;
        double minDistance = 10000.0, distance = 0.0;
        CVector3 point_bspline;
        for (int k=0; k<range; k++)
        {
            point_bspline = EvaluateBSpline((double)k/range);
            distance = sqrt((point[0]-point_bspline[0])*(point[0]-point_bspline[0])+(point[1]-point_bspline[1])*(point[1]-point_bspline[1])+(point[2]-point_bspline[2])*(point[2]-point_bspline[2]));
            if (distance <= minDistance) {
                minDistance = distance;
                minIndex = k;
            }
        }
        return (double)minIndex/range;
    };
    
private:
    BSplineType::Pointer bspline;
};

#endif /* defined(__sct_segmentation_propagation__BSplineApproximation__) */
