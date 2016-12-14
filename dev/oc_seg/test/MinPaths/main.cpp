int main(int argc, char *argv[])
{
	/* % MINIMALPATH Recherche du chemin minimum de Haut vers le bas et de
	% bas vers le haut tel que dÈcrit par Luc Vincent 1998
	% [sR,sC,S] = MinimalPath(I,factx)
	%
	%   I     : Image d'entrÔøΩe dans laquelle on doit trouver le
	%           chemin minimal
	%   factx : Poids de linearite [1 10]
	%
	% Programme par : Ramnada Chav
	% ModifiÈ le 16 novembre 2007 */

	    typedef itk::ImageDuplicator< ImageType > DuplicatorType3D;
	    typedef itk::InvertIntensityImageFilter <ImageType> InvertIntensityImageFilterType;
	    typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;

	    ImageType::Pointer inverted_image = image;

//	    if (invert)
//	    {
//	        StatisticsImageFilterType::Pointer statisticsImageFilterInput = StatisticsImageFilterType::New();
//	        statisticsImageFilterInput->SetInput(image);
//	        statisticsImageFilterInput->Update();
//	        double maxIm = statisticsImageFilterInput->GetMaximum();
//	        InvertIntensityImageFilterType::Pointer invertIntensityFilter = InvertIntensityImageFilterType::New();
//	        invertIntensityFilter->SetInput(image);
//	        invertIntensityFilter->SetMaximum(maxIm);
//	        invertIntensityFilter->Update();
//	        inverted_image = invertIntensityFilter->GetOutput();
//	    }


	    ImageType::SizeType sizeImage = image->GetLargestPossibleRegion().GetSize();
	    int m = sizeImage[0]; // x to change because we are in AIL
	    int n = sizeImage[2]; // y
	    int p = sizeImage[1]; // z

	    // create image with high values J1
	    DuplicatorType3D::Pointer duplicator = DuplicatorType3D::New();
	    duplicator->SetInputImage(inverted_image);
	    duplicator->Update();
	    ImageType::Pointer J1 = duplicator->GetOutput();
	    typedef itk::ImageRegionIterator<ImageType> ImageIterator3D;
	    ImageIterator3D vItJ1( J1, J1->GetBufferedRegion() );
	    vItJ1.GoToBegin();
	    while ( !vItJ1.IsAtEnd() )
	    {
	        vItJ1.Set(100000000);
	        ++vItJ1;
	    }

	    // create image with high values J2
	    DuplicatorType3D::Pointer duplicatorJ2 = DuplicatorType3D::New();
	    duplicatorJ2->SetInputImage(inverted_image);
	    duplicatorJ2->Update();
	    ImageType::Pointer J2 = duplicatorJ2->GetOutput();
	    ImageIterator3D vItJ2( J2, J2->GetBufferedRegion() );
	    vItJ2.GoToBegin();
	    while ( !vItJ2.IsAtEnd() )
	    {
	        vItJ2.Set(100000000);
	        ++vItJ2;
	    }

	    DuplicatorType3D::Pointer duplicatorCPixel = DuplicatorType3D::New();
	    duplicatorCPixel->SetInputImage(inverted_image);
	    duplicatorCPixel->Update();
	    ImageType::Pointer cPixel = duplicatorCPixel->GetOutput();

	    ImageType::IndexType index;

	    // iterate on slice from slice 1 (start=0) to slice p-2. Basically, we avoid first and last slices.
	    // IMPORTANT: first slice of J1 and last slice of J2 must be set to 0...
	    for (int x=0; x<m; x++)
	    {
	        for (int y=0; y<n; y++)
	        {
	            index[0] = x; index[1] = 0; index[2] = y;
	            J1->SetPixel(index, 0.0);
	        }
	    }
	    for (int slice=1; slice<p; slice++)
	    {
	        // 1. extract pJ = the (slice-1)th slice of the image J1
	        Matrice pJ = Matrice(m,n);
	        for (int x=0; x<m; x++)
	        {
	            for (int y=0; y<n; y++)
	            {
	                index[0] = x; index[1] = slice-1; index[2] = y;
	                pJ(x,y) = J1->GetPixel(index);
	            }
	        }

	        // 2. extract cP = the (slice)th slice of the image cPixel
	        Matrice cP = Matrice(m,n);
	        for (int x=0; x<m; x++)
	        {
	            for (int y=0; y<n; y++)
	            {
	                index[0] = x; index[1] = slice; index[2] = y;
	                cP(x,y) = cPixel->GetPixel(index);
	            }
	        }

	        // 2'
	        Matrice cPm = Matrice(m,n);
	        if (homoInt)
	        {
	            for (int x=0; x<m; x++)
	            {
	                for (int y=0; y<n; y++)
	                {
	                    index[0] = x; index[1] = slice-1; index[2] = y;
	                    cP(x,y) = cPixel->GetPixel(index);
	                }
	            }
	        }

	        // 3. Create a matrix VI with 5 slices, that are exactly a repetition of cP without borders
	        // multiply all elements of all slices of VI except the middle one by factx
	        Matrice VI[5];
	        for (int i=0; i<5; i++)
	        {
	            // Create VI
	            Matrice cP_in = Matrice(m-1, n-1);
	            for (int x=0; x<m-2; x++)
	            {
	                for (int y=0; y<n-2; y++)
	                {
	                    cP_in(x,y) = cP(x+1,y+1);
	                    if (i!=2)
	                        cP_in(x,y) *= factx;
	                }
	            }
	            VI[i] = cP_in;
	        }

	        // 3'.
	        Matrice VIm[5];
	        if (homoInt)
	        {
	            for (int i=0; i<5; i++)
	            {
	                // Create VIm
	                Matrice cPm_in = Matrice(m-1, n-1);
	                for (int x=0; x<m-2; x++)
	                {
	                    for (int y=0; y<n-2; y++)
	                    {
	                        cPm_in(x,y) = cPm(x+1,y+1);
	                        if (i!=2)
	                            cPm_in(x,y) *= factx;
	                    }
	                }
	                VIm[i] = cPm_in;
	            }
	        }

	        // 4. create a matrix of 5 slices, containing pJ(vectx-1,vecty),pJ(vectx,vecty-1),pJ(vectx,vecty),pJ(vectx,vecty+1),pJ(vectx+1,vecty) where vectx=2:m-1; and vecty=2:n-1;
	        Matrice Jq[5];
	        int s = 0;
	        Matrice pJ_temp = Matrice(m-1, n-1);
	        for (int x=0; x<m-2; x++)
	        {
	            for (int y=0; y<n-2; y++)
	            {
	                pJ_temp(x,y) = pJ(x+1,y+1);
	            }
	        }
	        Jq[2] = pJ_temp;
	        for (int k=-1; k<=1; k+=2)
	        {
	            for (int l=-1; l<=1; l+=2)
	            {
	                Matrice pJ_temp = Matrice(m-1, n-1);
	                for (int x=0; x<m-2; x++)
	                {
	                    for (int y=0; y<n-2; y++)
	                    {
	                        pJ_temp(x,y) = pJ(x+k+1,y+l+1);
	                    }
	                }
	                Jq[s] = pJ_temp;
	                s++;
	                if (s==2) s++; // we deal with middle slice before
	            }
	        }

	        // 4'. An alternative is to minimize the difference in intensity between slices.
	        if (homoInt)
	        {
	            Matrice VI_temp[5];
	            // compute the difference between VI and VIm
	            for (int i=0; i<5; i++)
	                VI_temp[i] = VI[i] - VIm[i];

	            // compute the minimum value for each element of the matrices
	            for (int i=0; i<5; i++)
	            {
	                for (int x=0; x<m-2; x++)
	                {
	                    for (int y=0; y<n-2; y++)
	                    {
	                        if (VI_temp[i](x,y) > 0)
	                            VI[i](x,y) = abs(VI_temp[i](x,y));///VIm[i](x,y);
	                        else
	                            VI[i](x,y) = abs(VI_temp[i](x,y));///VI[i](x,y);
	                    }
	                }
	            }
	        }

	        // 5. sum Jq and Vi voxel by voxel to produce JV
	        Matrice JV[5];
	        for (int i=0; i<5; i++)
	            JV[i] = VI[i] + Jq[i];

	        // 6. replace each pixel of the (slice)th slice of J1 with the minimum value of the corresponding column in JV
	        for (int x=0; x<m-2; x++)
	        {
	            for (int y=0; y<n-2; y++)
	            {
	                double min_value = 1000000;
	                for (int i=0; i<5; i++)
	                {
	                    if (JV[i](x,y) < min_value)
	                        min_value = JV[i](x,y);
	                }
	                index[0] = x+1; index[1] = slice; index[2] = y+1;
	                J1->SetPixel(index, min_value);
	            }
	        }
	    }

	    // iterate on slice from slice n-1 to slice 1. Basically, we avoid first and last slices.
	    // IMPORTANT: first slice of J1 and last slice of J2 must be set to 0...
	    for (int x=0; x<m; x++)
	    {
	        for (int y=0; y<n; y++)
	        {
	            index[0] = x; index[1] = p-1; index[2] = y;
	            J2->SetPixel(index, 0.0);
	        }
	    }
	    for (int slice=p-2; slice>=0; slice--)
	    {
	        // 1. extract pJ = the (slice-1)th slice of the image J1
	        Matrice pJ = Matrice(m,n);
	        for (int x=0; x<m; x++)
	        {
	            for (int y=0; y<n; y++)
	            {
	                index[0] = x; index[1] = slice+1; index[2] = y;
	                pJ(x,y) = J2->GetPixel(index);
	            }
	        }

	        // 2. extract cP = the (slice)th slice of the image cPixel
	        Matrice cP = Matrice(m,n);
	        for (int x=0; x<m; x++)
	        {
	            for (int y=0; y<n; y++)
	            {
	                index[0] = x; index[1] = slice; index[2] = y;
	                cP(x,y) = cPixel->GetPixel(index);
	            }
	        }

	        // 2'
	        Matrice cPm = Matrice(m,n);
	        if (homoInt)
	        {
	            for (int x=0; x<m; x++)
	            {
	                for (int y=0; y<n; y++)
	                {
	                    index[0] = x; index[1] = slice+1; index[2] = y;
	                    cPm(x,y) = cPixel->GetPixel(index);
	                }
	            }
	        }

	        // 3. Create a matrix VI with 5 slices, that are exactly a repetition of cP without borders
	        // multiply all elements of all slices of VI except the middle one by factx
	        Matrice VI[5];
	        for (int i=0; i<5; i++)
	        {
	            // Create VI
	            Matrice cP_in = Matrice(m-1, n-1);
	            for (int x=0; x<m-2; x++)
	            {
	                for (int y=0; y<n-2; y++)
	                {
	                    cP_in(x,y) = cP(x+1,y+1);
	                    if (i!=2)
	                        cP_in(x,y) *= factx;
	                }
	            }
	            VI[i] = cP_in;
	        }

	        // 3'.
	        Matrice VIm[5];
	        if (homoInt)
	        {
	            for (int i=0; i<5; i++)
	            {
	                // Create VI
	                Matrice cPm_in = Matrice(m-1, n-1);
	                for (int x=0; x<m-2; x++)
	                {
	                    for (int y=0; y<n-2; y++)
	                    {
	                        cPm_in(x,y) = cPm(x+1,y+1);
	                        if (i!=2)
	                            cPm_in(x,y) *= factx;
	                    }
	                }
	                VIm[i] = cPm_in;
	            }
	        }

	        // 4. create a matrix of 5 slices, containing pJ(vectx-1,vecty),pJ(vectx,vecty-1),pJ(vectx,vecty),pJ(vectx,vecty+1),pJ(vectx+1,vecty) where vectx=2:m-1; and vecty=2:n-1;
	        Matrice Jq[5];
	        int s = 0;
	        Matrice pJ_temp = Matrice(m-1, n-1);
	        for (int x=0; x<m-2; x++)
	        {
	            for (int y=0; y<n-2; y++)
	            {
	                pJ_temp(x,y) = pJ(x+1,y+1);
	            }
	        }
	        Jq[2] = pJ_temp;
	        for (int k=-1; k<=1; k+=2)
	        {
	            for (int l=-1; l<=1; l+=2)
	            {
	                Matrice pJ_temp = Matrice(m-1, n-1);
	                for (int x=0; x<m-2; x++)
	                {
	                    for (int y=0; y<n-2; y++)
	                    {
	                        pJ_temp(x,y) = pJ(x+k+1,y+l+1);
	                    }
	                }
	                Jq[s] = pJ_temp;
	                s++;
	                if (s==2) s++; // we deal with middle slice before
	            }
	        }

	        // 4'. An alternative is to minimize the difference in intensity between slices.
	        if (homoInt)
	        {
	            Matrice VI_temp[5];
	            // compute the difference between VI and VIm
	            for (int i=0; i<5; i++)
	                VI_temp[i] = VI[i] - VIm[i];

	            // compute the minimum value for each element of the matrices
	            for (int i=0; i<5; i++)
	            {
	                for (int x=0; x<m-2; x++)
	                {
	                    for (int y=0; y<n-2; y++)
	                    {
	                        if (VI_temp[i](x,y) > 0)
	                            VI[i](x,y) = abs(VI_temp[i](x,y));///VIm[i](x,y);
	                        else
	                            VI[i](x,y) = abs(VI_temp[i](x,y));///VI[i](x,y);
	                    }
	                }
	            }
	        }

	        // 5. sum Jq and Vi voxel by voxel to produce JV
	        Matrice JV[5];
	        for (int i=0; i<5; i++)
	            JV[i] = VI[i] + Jq[i];

	        // 6. replace each pixel of the (slice)th slice of J1 with the minimum value of the corresponding column in JV
	        for (int x=0; x<m-2; x++)
	        {
	            for (int y=0; y<n-2; y++)
	            {
	                double min_value = 10000000;
	                for (int i=0; i<5; i++)
	                {
	                    if (JV[i](x,y) < min_value)
	                        min_value = JV[i](x,y);
	                }
	                index[0] = x+1; index[1] = slice; index[2] = y+1;
	                J2->SetPixel(index, min_value);
	            }
	        }
	    }

	    // add J1 and J2 to produce "S" which is actually J1 here.
	    ImageIterator3D vItS( J1, J1->GetBufferedRegion() );
	    ImageIterator3D vItJ2b( J2, J2->GetBufferedRegion() );
	    vItS.GoToBegin();
	    vItJ2b.GoToBegin();
	    while ( !vItS.IsAtEnd() )
	    {
	        vItS.Set(vItS.Get()+vItJ2b.Get());
	        ++vItS;
	        ++vItJ2b;
	    }

	    // Find the minimal value of S for each slice and create a binary image with all the coordinates
	    // TO DO: the minimal path shouldn't be a pixelar path. It should be a continuous spline that is minimum.
	    double val_temp;
	    vector<CVector3> list_index;
	    for (int slice=1; slice<p-1; slice++)
	    {
	        double min_value_S = 10000000;
	        ImageType::IndexType index_min;
	        for (int x=1; x<m-1; x++)
	        {
	            for (int y=1; y<n-1; y++)
	            {
	                index[0] = x; index[1] = slice; index[2] = y;
	                val_temp = J1->GetPixel(index);
	                if (val_temp < min_value_S)
	                {
	                    min_value_S = val_temp;
	                    index_min = index;
	                }
	            }
	        }
	        list_index.push_back(CVector3(index_min[0], index_min[1], index_min[2]));
	    }

	    //BSplineApproximation centerline_approximator = BSplineApproximation(&list_index);
	    //list_index = centerline_approximator.EvaluateBSplinePoints(list_index.size());

	    /*// create image with high values J1
	    ImageType::Pointer result_bin = J2;
	    ImageIterator3D vItresult( result_bin, result_bin->GetBufferedRegion() );
	    vItresult.GoToBegin();
	    while ( !vItresult.IsAtEnd() )
	    {
	        vItresult.Set(0.0);
	        ++vItresult;
	    }
	    for (int i=0; i<list_index.size(); i++)
	    {
	        index[0] = list_index[i][0]; index[1] = list_index[i][1]; index[2] = list_index[i][2];
	        result_bin->SetPixel(index,1.0);
	    }

	    typedef itk::ImageFileWriter< ImageType > WriterTypeM;
	    WriterTypeM::Pointer writerMin = WriterTypeM::New();
	    itk::NiftiImageIO::Pointer ioV = itk::NiftiImageIO::New();
	    writerMin->SetImageIO(ioV);
	    writerMin->SetInput( J1 ); // result_bin
	    writerMin->SetFileName("minimalPath.nii.gz");
	    try {
	        writerMin->Update();
	    }
	    catch( itk::ExceptionObject & e )
	    {
	        cout << "Exception thrown ! " << endl;
	        cout << "An error ocurred during Writing Min" << endl;
	        cout << "Location    = " << e.GetLocation()    << endl;
	        cout << "Description = " << e.GetDescription() << endl;
	    }*/

	    centerline = list_index;
	    return J1; // return image with minimal path
	}
}
