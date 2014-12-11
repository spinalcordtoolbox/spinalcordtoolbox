#include "mex.h"
#include "string.h"
#include "matrix.h"
#include "math.h"

/*
%
% DESCRIPTION
% -------------------------------------------------------------------------
% Find maxima of the ODF based on a given sampling of the sphere (e.g.,
% 181, 362, ...). The algorithm searches for local maxima on the ODF.
% 
% To improve the robustness of the algorithm, it takes the mean of both
% antipodal local maxima, in a parametric way (i.e., the definition of the 
% principal directions does NOT depend on the sampling of the sphere). Of
% course, the better the original sampling of the sphere, the more accurate
% the definition of the principal direction is.
% 
% 
% INPUTS
% -------------------------------------------------------------------------
% odf					(1xn) float		values in amplitudes on the sphere
% scheme.vert			(nx3) float
% nearest_neighb		cell			neighrest neighbours each sample on the sphere
% output				string			'discreet','parametric'*. Definition of the maxima is either discreet (depends on the sampling of the ODF and defined as a nx1 matrix) or parametric (defined as a 3x1 vector, this option is way better)
% 
% 
% OUTPUTS
% -------------------------------------------------------------------------
% max_odf_param			nx3 float		vector coordinates of each maximum
% nb_maxima				integer			number of maxima
%
% 
% COPYRIGHT
% 2009-10-29
% Julien Cohen-Adad 
% Massachusetts General Hospital
% =========================================================================
*/


/*
USE dbmex!!
		!!!!!
*/





/*=========================================================================
 Find the maximum value of a vector and output it
 Equivalent of Matlab function 'max'

 INPUT
 *x				*double array
 ncols			int
 max			double

 OUTPUT
 index			int			index of max
 
 ========================================================================*/
double max(double *x, mwSize ncols, double *xmax)
{
	int		i,index;

	*xmax = x[0];
	index = 0;
	
	
	for (i=0; i<ncols; i++)
	{

		if (x[i] > *xmax)
		{
			*xmax = x[i];
			index = i;
		}
	}


	
	return index;
}






/*
=========================================================================
Find the non-zero values
Equivalent of Matlab function 'find'

INPUT
*x			array

OUTPUT
*y			array
 
=========================================================================
*/
double find(double *x, int *y, mwSize ncols)
{

	int			i,j;

	j=0;
	
	for (i=0; i<ncols; i++)
	{

		if (x[i] != 0)
		{
			y[j] = i;

			j++;
		}
	}


	return j;
}





/*========================================================================
reshape
--------------------------------------------------------------------------
Reshape from nx1 to n/2x2
 
INPUT
*x			*double array

OUTPUT
*y			*double array
 
=========================================================================*/
double reshape(double *x, double **y, mwSize ncols)
{
	int			i,j;



	
	for (i=0; i<ncols; i++)
	{
		y[i][0]=x[i];
		y[i][1]=x[i+ncols];
		y[i][2]=x[i+2*(ncols)];
		
	}


	return;
}





/*
==========================================================================
 sort
--------------------------------------------------------------------------
 INPUT
 *x			*double array
 ncols		int
 *y			*double array
 
 OUTPUT
 index		*int array
========================================================================*/
void sort(double *x, mwSize n, int *index)
{
	int			i;
	double		xmax;
	double		*y;
	

	y = mxMalloc (n*sizeof(double));
	if (!y)
	{
		printf("Out of memory for allocating *y.\n");
		exit(1);
	}

	for(i=0; i<n; i++)
	{
		y[i] = x[i];
	}



	for(i=0; i<n; i++)
	{



		index[i] = max(y,n,&xmax); 
		x[i] = y[index[i]];
		y[index[i]] = 0;
		

	}
	mxFree(y);
	return;
}
	
	



/*
==========================================================================
 Gateway function
--------------------------------------------------------------------------
 INPUT
 
 OUTPUT
 
========================================================================*/
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
	double		*odf; 
	double		*vert; 
	double		**vert2d;
	mxArray		*nearest_neighb_array;
	double		*maxima_threshold;
	double		*nearest_neighb;
	double		*angular_threshold; 
	double		peak_odf[1];
	double		*max_odf;
	double		u1,u2,u3,v1,v2,v3;
	double		product[20];
	double		max_antipodal[1];
	double		**max_odf_param;
	double		**max_odf_param_tmp;
	double		**max_odf_param_tmp2;
	double		*max_odf_val;
	double		*max_odf_val_sorted;
	double		value1, value2;
	double		angle_max;
	double		angle_dir;
	int			antipodal[20];
	int			ineighb;
	double		*nb_neighbours;
	int			sup_neighb;
	int			index_neighbour;
	int			i_max_odf[20]; 
	int			nb_maxima;
	int			nb_maxc;
	int			size_i_max_odf;
	int			i,j,irow,icol;
	int			*index_sort_max;
	int			*index_max;
	double			*output2;
	const double		pi = 3.141592653589793;
	mwSize mrows,ncols;
	mwIndex		icell;
    mxClassID  category;
	mxArray		*k = NULL;


	/* check proper input and output */
	if(nrhs<3)
	{
		mexErrMsgTxt("Three inputs required.");
	}

/*
	odf = mxMalloc(sizeof(mxGetPr(prhs[0])));
	if (!odf)
	{
		printf("Out of memory for allocating odf.\n");
		exit(1);
	}

	vert = mxMalloc(sizeof(mxGetPr(prhs[1])));
	if (!vert)
	{
		printf("Out of memory for allocating vert.\n");
		exit(1);
	}
*/

	
	/*  create a pointer to the input matrices */
	odf = mxGetPr(prhs[0]);
	vert = mxGetPr(prhs[1]);
	nb_neighbours = mxGetPr(prhs[3]);
	angular_threshold = mxGetPr(prhs[4]);
	maxima_threshold = mxGetPr(prhs[5]);
	
	/*  get the dimensions of the matrix input y */
	mrows = mxGetM(prhs[0]);
	ncols = mxGetN(prhs[0]);
	

	vert2d = mxMalloc(ncols * sizeof(double *));
	if (!vert2d)
	{
		printf("Out of memory for allocating **vert2d.\n");
		exit(1);
	}
	for(i = 0; i < ncols; i++)
	{
		vert2d[i] = mxMalloc(3 * sizeof(double));
		if (!vert2d[i])
		{
			printf("Out of memory for allocating *vert2d.\n");
			exit(1);
		}
	}
	


	
	reshape(vert,vert2d,ncols);


	
	max(odf,ncols,peak_odf);

    /* Find each sample's neighbour on the sphere */
	size_i_max_odf=0;
	for (icol=0; icol<ncols; icol++)	
	{
			if ((odf[icol]/(*peak_odf)) > *maxima_threshold)
			{

				nearest_neighb_array = mxGetCell(prhs[2],icol);
				nearest_neighb = mxGetPr(nearest_neighb_array);
				
				sup_neighb = 0;
				for (ineighb=0; ineighb<*nb_neighbours; ineighb++)
				{ 
					index_neighbour = nearest_neighb[ineighb] - 1;
					if (odf[icol] > odf[index_neighbour])
					{
						sup_neighb++;
					}
				}

				if (sup_neighb == *nb_neighbours)
				{
					i_max_odf[size_i_max_odf] = icol;
					size_i_max_odf++;
				}
			}
	}

	size_i_max_odf = size_i_max_odf;
	nb_maxima = floor(size_i_max_odf/2 );
	
/*    printf("Number of maxima: %d\n",nb_maxima);*/
    if (nb_maxima > 0)
    {
        /* Loop on each sample on the half-sphere to find local maxima */
        for (i=0; i<size_i_max_odf; i++)
        {
            for (j=0; j<size_i_max_odf; j++)
            {
                u1 = vert2d[i_max_odf[i]][0];
                u2 = vert2d[i_max_odf[i]][1];
                u3 = vert2d[i_max_odf[i]][2];
                v1 = vert2d[i_max_odf[j]][0];
                v2 = vert2d[i_max_odf[j]][1];
                v3 = vert2d[i_max_odf[j]][2];
                product[j] = u1*v1+u2*v2+u3*v3;
                if (product[j] > 1)
                {
                    product[j] = 1;
                }
                product[j] = acos(product[j])*180/pi;
            }
            antipodal[i] = max(product,size_i_max_odf,max_antipodal);
        }


        max_odf_param = mxMalloc(nb_maxima * sizeof(double *));
        if (!max_odf_param)
        {
            printf("Out of memory for allocating max_odf_param.\n");
            exit(1);
        }
        for(i=0; i<nb_maxima; i++)
        {
            max_odf_param[i] = mxMalloc(3 * sizeof(double));
            if (!max_odf_param[i])
            {
                printf("Out of memory for allocating max_odf_param[i].\n");
                exit(1);
            }
        }

        max_odf_val = mxMalloc(nb_maxima * sizeof(double));
        if (!max_odf_val)
        {
            printf("Out of memory for allocating max_odf_val.\n");
            exit(1);
        }

        /* compute the mean for each maxima */
        for (i=0; i<nb_maxima; i++)
        {
            u1 = vert2d[i_max_odf[i]][0];
            u2 = vert2d[i_max_odf[i]][1];
            u3 = vert2d[i_max_odf[i]][2];
            v1 = -vert2d[i_max_odf[antipodal[i]]][0];
            v2 = -vert2d[i_max_odf[antipodal[i]]][1];
            v3 = -vert2d[i_max_odf[antipodal[i]]][2];
            max_odf_param[i][0] = (u1+v1)/2;
            max_odf_param[i][1] = (u2+v2)/2;
            max_odf_param[i][2] = (u3+v3)/2;
            value1 = odf[i_max_odf[i]];
            value2 = odf[i_max_odf[antipodal[i]]];
            max_odf_val[i] = (value1+value2)/2;
        }

        /* Order ODF maxima based on the mean value of each antipodal pair */
        max_odf_val_sorted = mxMalloc(nb_maxima * sizeof(double));
        if (!max_odf_val_sorted)
        {
            printf("Out of memory for allocating max_odf_val_sorted.\n");
            exit(1);
        }
        index_sort_max = mxMalloc(nb_maxima * sizeof(int));
        if (!index_sort_max)
        {
            printf("Out of memory for allocating index_sort_max.\n");
            exit(1);
        }
        sort(max_odf_val,nb_maxima,index_sort_max);

        max_odf_param_tmp = mxMalloc(nb_maxima * sizeof(double *));
        for(i=0; i<nb_maxima; i++)
        {
            max_odf_param_tmp[i] = mxMalloc(3 * sizeof(double));
        }
        for (i=0; i<nb_maxima; i++)
        {
            max_odf_param_tmp[i][0] = max_odf_param[index_sort_max[i]][0];
            max_odf_param_tmp[i][1] = max_odf_param[index_sort_max[i]][1];
            max_odf_param_tmp[i][2] = max_odf_param[index_sort_max[i]][2];
        }
        max_odf_param = max_odf_param_tmp;

        index_max = mxMalloc(nb_maxima * sizeof(int));
        if (!index_max)
        {
            printf("Out of memory for allocating index_max.\n");
            exit(1);
        }
        for (i=0; i<nb_maxima; i++)
        {
            index_max[i]=1;
        }

        if (*angular_threshold > 0)
        {
            for (i=1; i<nb_maxima; i++)
            {
                for (j=i; j<nb_maxima; j++)
                {
                    u1 = max_odf_param[i-1][0];
                    u2 = max_odf_param[i-1][1];
                    u3 = max_odf_param[i-1][2];
                    v1 = max_odf_param[j][0];
                    v2 = max_odf_param[j][1];
                    v3 = max_odf_param[j][2];
                    angle_max = u1*v1+u2*v2+u3*v3;
                    if (angle_max > 1)
                    {
                        angle_max = 1;
                    }
                    angle_max = acos(angle_max)*180/pi;
                    if (angle_max < *angular_threshold)
                    {
                        index_max[j] = 0;
                    }
                }
            }

            nb_maxc = 1;
            for (i=1; i<nb_maxima; i++)
            {
                if (index_max[i] == 1)
                {
                    nb_maxc++;
                }
            }


            max_odf_param_tmp2 = mxMalloc(nb_maxc * sizeof(double *));
            for(i=0; i<nb_maxc; i++)
            {
                max_odf_param_tmp2[i] = mxMalloc(3 * sizeof(double));
            }

            j=0;
            for(i=0; i<nb_maxima; i++)
            {
                if (index_max[i] == 1)
                {
                    max_odf_param_tmp2[j][0] = max_odf_param[i][0];
                    max_odf_param_tmp2[j][1] = max_odf_param[i][1];
                    max_odf_param_tmp2[j][2] = max_odf_param[i][2];
                    j++;
                }
            }
            nb_maxima = nb_maxc;

            max_odf_param = mxRealloc(max_odf_param, nb_maxima * sizeof(double *));
            if (!max_odf_param)
            {
                printf("Out of memory for allocating max_odf_param.\n");
                exit(1);
            }
            for(i=0; i<nb_maxima; i++)
            {
                max_odf_param[i] = mxRealloc(max_odf_param[i], 3 * sizeof(double));
                if (!max_odf_param[i])
                {
                    printf("Out of memory for allocating max_odf_param[i].\n");
                    exit(1);
                }
            }
            max_odf_param = max_odf_param_tmp2;
        }

        plhs[0] = mxCreateDoubleMatrix(nb_maxima,3,mxREAL);

        max_odf = mxGetPr(plhs[0]);
        i = 0;
        for (icol=0;icol<3; icol++)
        {
            for (irow=0;irow<nb_maxima; irow++)
            {
                max_odf[i] = max_odf_param[irow][icol];
                i++;
            }
        }

        plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
        output2 = mxGetPr(plhs[1]);

        *output2 = nb_maxima;

        mxFree(*vert2d);
        mxFree(max_odf_val);
        mxFree(*max_odf_param);
        mxFree(max_odf_val_sorted);
        mxFree(index_sort_max);
        mxFree(index_max);
    }
    else
    {
        plhs[0] = mxCreateDoubleMatrix(1,3,mxREAL);
        plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    }
  return;
}
