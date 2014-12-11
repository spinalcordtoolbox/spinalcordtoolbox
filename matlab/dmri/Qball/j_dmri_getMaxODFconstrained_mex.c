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
	
//	printf("max: x[0]=%f, x[1]=%f, x[2]=%f\n",x[0],x[1],x[2]);
//	printf("x[0]=%f\n",x[0]);
	
	for (i=0; i<ncols; i++)
	{
//		printf("i=%d, x[i]=%f, xmax=\n",i,x[i]);
		if (x[i] > *xmax)
		{
			*xmax = x[i];
			index = i;
		}
	}
//	printf("max: x[0]=%f, x[1]=%f, x[2]=%f\n",x[0],x[1],x[2]);
//	printf("max: index=%d, max=%f\n",index,*xmax);
	
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
//	int			y[400];
	int			i,j;

	j=0;
	// allocate memory
//	y = (int*) mxMalloc (ncols+30);
	
	for (i=0; i<ncols; i++)
	{
//		y[i]=i;
		if (x[i] != 0)
		{
			y[j] = i;
//			printf("i=%d\n",y[j]);
			j++;
		}
	}
//	printf("%d, %d", y[0],y[4]);

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

//	printf("mrows=%d\n",mrows);
//	printf("ncols=%d\n",ncols);
	
	for (i=0; i<ncols; i++)
	{
		y[i][0]=x[i];
		y[i][1]=x[i+ncols];
		y[i][2]=x[i+2*(ncols)];
		
	}
//	printf("%d, %d", y[0],y[4]);

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
	
	// copy array x into array y
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
//	printf("y[0]=%f, y[1]=%f, y[2]=%f\n",y[0],y[1],y[2]);

	// descending sorting algorithm (VERY unefficient method - only use it with small n, like here)
	for(i=0; i<n; i++)
	{
//		printf("y[i]=%f, x[i]=%f\n",y[i],x[i]);
//	printf("y[0]=%f, y[1]=%f, y[2]=%f\n",y[0],y[1],y[2]);

		index[i] = max(y,n,&xmax); //JULIEN max is double and not int!!! ACHTUNG!!
		x[i] = y[index[i]];
		y[index[i]] = 0;
		
//		printf("i=%d, index=%d, max=%f\n",i,index[i],xmax);
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
	double		*odf; // input1
	double		*vert; // input2
	double		**vert2d;
	mxArray		*nearest_neighb_array; // input3
	double		*maxima_threshold;
	double		*nearest_neighb; // input4
	double		*angular_threshold; // input5
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
	int			antipodal[20];
	int			ineighb;
	double		*nb_neighbours;
	int			sup_neighb;
	int			index_neighbour;
	int			i_max_odf[20]; // ALLOCATE THAT PROPERLY!!!!
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


	// default parameters
//	nb_neighbours			= 15; // make it depends on the sampling
//	maxima_threshold		= 0.33;
//	angular_threshold		= 45; // in degree. Two identified maxima cannot make an angle below this value. If you don't want to use that, put 0.
	
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
//	printf("odf[0]=%f, odf[1]=%f, odf[2]=%f\n",odf[0],odf[1],odf[2]);
//	printf("01. nb_neighbours=%f\n", *nb_neighbours);
	
	/*  get the dimensions of the matrix input y */
	mrows = mxGetM(prhs[0]);
	ncols = mxGetN(prhs[0]);
//	printf("01b. HOLA\n");
//	printf("mrows=%d, ncols=%d\n",mrows,ncols);
	

	// allocate 2D matrix
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
	

//	vert2d[0][0] = 2;
//	vert2d[5][1] = 2;
//	vert2d[50][0] = 2;
//	vert2d[50][1] = 2;
//	vert2d[50][2] = 2;
//	printf("vert2d=%f\n",vert2d[0][0]);
//	printf("vert2d=%f\n",vert2d[5][1]);
//	printf("vert2d=%f\n",vert2d[50][0]);
//	printf("vert2d=%f\n",vert2d[50][1]);
//	printf("vert2d=%f\n",vert2d[1][0]);
//	printf("01c. HOLA\n");

	
	reshape(vert,vert2d,ncols);

//	nb_neighbours			= round(ncols/30); // make it depends on the sampling

//	for (i=0;i<5;i++)
//	{
//		printf("vert1d=%f\n",vert[i]);
//		printf("vert2d=%f\n\n",vert2d[i][1]);
//	}

	
	// Get the maxima of the ODF (to apply subsequent threshold on the maxima)
	max(odf,ncols,peak_odf);
//	printf("02. max=%f\n",*peak_odf);
//	printf("02. HOLA\n");
	
	// Loop on each sample of the sphere to find local maxima
//	dont_test = [];

	size_i_max_odf=0;
	for (icol=0; icol<ncols; icol++)	
//	for (icol=41; icol<42; icol++)
	{
//		printf("icol=%d, odf[icol]=%f\n",icol,odf[icol]);
		// check if this guy is not inferior to an already found local maxima
//		if isempty(find(dont_test==isample))
		
			// check if this guy is superior to the given threshold
			if ((odf[icol]/(*peak_odf)) > *maxima_threshold)
			{
//				printf("icol=%d\n",icol);

				// Get specific cell from input
				nearest_neighb_array = mxGetCell(prhs[2],icol);
				nearest_neighb = mxGetPr(nearest_neighb_array);
				
				// loop accross neighbours
				sup_neighb = 0;
//				for (ineighb=0; ineighb<5; ineighb++)
				for (ineighb=0; ineighb<*nb_neighbours; ineighb++)
				{ 
					// check if this guy is superior to its neighbour
					index_neighbour = nearest_neighb[ineighb] - 1;
					// N.B. Since the cell is passed from Matlab, indexation
					// starts at 1 instead of 0. This explains the "-1" in 
					// the previous line.
//					printf("icol=%d, ineighb=%d, nearest_neighb=%d\n",icol,ineighb,index_neighbour);
//					printf("icol=%d, ineighb=%d, odf[icol]=%f, odf[index_neighbour]=%f\n",icol,ineighb,odf[icol],odf[index_neighbour]);
					if (odf[icol] > odf[index_neighbour])
					{
//						printf("HOLA! index_neighbour=%d\n",index_neighbour);
						sup_neighb++;
					}
				}
//				printf("sup_neighb=%d, nb_neighbours=%d\n",sup_neighb,*nb_neighbours);

				// if that guy is superior to all its neighbours (how lucky!)
				if (sup_neighb == *nb_neighbours)
				{
					// assign this guy as a superhero
//					max_odf[icol] = 1;
//					printf("GOT IT! icol=%d, odf[icol]=%f\n",icol,odf[icol]);
					i_max_odf[size_i_max_odf] = icol;
					size_i_max_odf++;
//					% prevent neighbours to be tested (for obvious computational reasons)
//					dont_test = cat(1,dont_test,nearest_neighb{isample});
				}
			}
	}

	size_i_max_odf = size_i_max_odf;
//	printf("03. size_i_max_odf=%d\n",size_i_max_odf);
	
	// get maxima indices
//	printf("i_max_odf=%d\n",i_max_odf[0]);
//	printf("i_max_odf=%d\n",i_max_odf[1]);
//	printf("i_max_odf=%d\n",i_max_odf[2]);
//	printf("i_max_odf=%d\n",i_max_odf[3]);

	// Get the number of maxima, assuming symmetry of the ODF. Also, if the
	// number of maxima is odd, then rounds by the inferior number.
	nb_maxima = floor(size_i_max_odf/2);
//	printf("nb_maxima=%d\n",nb_maxima);

	// allocate angle_dir
//	angle_dir = (double*) mxMalloc (2*nb_maxima);
	
	// find the antipod of the first direction by computing the dot
	// product between the first direction and the other ones to get the
	// angle. Two antipodal directions are assumed to form an angle of about
	// 180°, so the maximum angle between each direction serves as a basis
	// to select two antipodal directions.

//	printf("05. vert=%f %f %f\n",vert2d[0][0],vert2d[0][1],vert2d[0][2]);	
	
	for (i=0; i<size_i_max_odf; i++)
	{
		for (j=0; j<size_i_max_odf; j++)
		{
//			iv = i_max_odf[i];
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
//			printf("i=%d, j=%d:  ",i,j);
//			printf("u*v=%f\n",acos(product)*180/pi);
//			printf("i=%d, j=%d\n",i,j);
//			angle_dir[i][j] = abs(acos(u1*v1+u2*v2+u3*v3))*180/pi;
//			printf("angle_dir=%f\n",angle_dir[i][j]);
//			printf("u1=%f\n",u1);
//			printf("u2=%f\n",u2);
//			printf("u3=%f\n",u3);
//			printf("v1=%f\n",v1);
//			printf("v2=%f\n",v2);
//			printf("v3=%f\n",v3);

		}
		// Find the antipod of the selected angle (for i)
//		printf("product = %f %f %f %f\n",product[0],product[1],product[2],product[3]);
		antipodal[i] = max(product,size_i_max_odf,max_antipodal);
//		printf("antipodal[i] = %d\n",antipodal[i]);
	}
	
	
	// allocate 2D matrix
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

	// compute the mean for each maxima
//	max_odf_param = [];
	for (i=0; i<nb_maxima; i++)
	{
		// get both antipodal vectors
		u1 = vert2d[i_max_odf[i]][0];
		u2 = vert2d[i_max_odf[i]][1];
		u3 = vert2d[i_max_odf[i]][2];
		v1 = -vert2d[i_max_odf[antipodal[i]]][0];
		v2 = -vert2d[i_max_odf[antipodal[i]]][1];
		v3 = -vert2d[i_max_odf[antipodal[i]]][2];
		// compute the mean vector
		max_odf_param[i][0] = (u1+v1)/2;
		max_odf_param[i][1] = (u2+v2)/2;
		max_odf_param[i][2] = (u3+v3)/2;
//		printf("index=%d, max_odf_param = %f %f %f\n",i,max_odf_param[i][0],max_odf_param[i][1],max_odf_param[i][2]);
		// compute the mean value on the ODF for each each antipodal pair
		value1 = odf[i_max_odf[i]];
		value2 = odf[i_max_odf[antipodal[i]]];
//		printf("value1=%f, value2=%f\n",value1,value2);
		max_odf_val[i] = (value1+value2)/2;
	}

	// Find the order of ODF maxima based on the closest value with the maxima given as input of this function
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
//	index_sort_max[0]=1;
//	printf("index_sort_max[0]=%d\n",index_sort_max[0]);
	sort(max_odf_val,nb_maxima,index_sort_max);
		
//	printf("10. max_odf_val = %f %f\n",max_odf_val[0],max_odf_val[1]);

	// Allocate memory for 2D matrix
	max_odf_param_tmp = mxMalloc(nb_maxima * sizeof(double *));
	for(i=0; i<nb_maxima; i++)
	{
		max_odf_param_tmp[i] = mxMalloc(3 * sizeof(double));
	}
//	max_odf_param_tmp = max_odf_param;

	// Order ODF maxima
	for (i=0; i<nb_maxima; i++)
	{
		max_odf_param_tmp[i][0] = max_odf_param[index_sort_max[i]][0];
		max_odf_param_tmp[i][1] = max_odf_param[index_sort_max[i]][1];
		max_odf_param_tmp[i][2] = max_odf_param[index_sort_max[i]][2];
//		printf("index_sort_max[%d]=%d\n",i,index_sort_max[i]);
//		printf("index=%d, max_odf_param_tmp = %f %f %f\n",i,max_odf_param_tmp[i][0],max_odf_param_tmp[i][1],max_odf_param_tmp[i][2]);
	}
	max_odf_param = max_odf_param_tmp;

	// Free pointer
//	mxFree(max_odf_param_tmp);

	// allocate
	index_max = mxMalloc(nb_maxima * sizeof(int));
	if (!index_max)
	{
		printf("Out of memory for allocating index_max.\n");
		exit(1);
	}
	// initialize
	for (i=0; i<nb_maxima; i++)
	{
		index_max[i]=1;
//		printf("index_max[i]=%d\n",index_max[i]);
	}
	
	// delete maxima that are similar to other maxima (given an angular threshold)
	if (*angular_threshold > 0)
	{
		// calculate the angle between maxima
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
//				printf("u1=%f, u2=%f, u3=%f\n",u1,u2,u3);
				if (angle_max > 1)
				{
					angle_max = 1;
				}
				angle_max = acos(angle_max)*180/pi;
				if (angle_max < *angular_threshold)
				{
					index_max[j] = 0;
				}
//				printf("product=%f\n",product[i-1]);
		//		printf("index=%d, max_odf_param_tmp = %f %f %f\n",i,max_odf_param_tmp[i][0],max_odf_param_tmp[i][1],max_odf_param_tmp[i][2]);
			}
		}
		
		// Find new number of maxima
		nb_maxc = 1;
		for (i=1; i<nb_maxima; i++)
		{
			if (index_max[i] == 1)
			{
				nb_maxc++;
			}
		}

		
		// Realloc pointer for 2D matrix
		max_odf_param_tmp2 = mxMalloc(nb_maxc * sizeof(double *));
		for(i=0; i<nb_maxc; i++)
		{
			max_odf_param_tmp2[i] = mxMalloc(3 * sizeof(double));
		}

		// Create new max ODF matrix
		j=0;
		for(i=0; i<nb_maxima; i++)
		{
			if (index_max[i] == 1)
			{
				max_odf_param_tmp2[j][0] = max_odf_param[i][0];
				max_odf_param_tmp2[j][1] = max_odf_param[i][1];
				max_odf_param_tmp2[j][2] = max_odf_param[i][2];
//				printf("i=%d, j=%d, nb_maxc=%d\n",i,j,nb_maxc);		
//				printf("j=%d, max_odf_param_tmp2 = %f %f %f\n",j,max_odf_param_tmp2[j][0],max_odf_param_tmp2[j][1],max_odf_param_tmp2[j][2]);
				j++;
			}
		}
		// rename variables
//		printf("nb_maxima=%d, nb_maxc=%d\n",nb_maxima,nb_maxc);
		nb_maxima = nb_maxc;
		
		// free max_odf_param
//		mxFree(*max_odf_param);
		
		// allocate 2D matrix
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
	
	//  set the output pointer to the output matrix 
	plhs[0] = mxCreateDoubleMatrix(nb_maxima,3,mxREAL);

	// create a C pointer to a copy of the output matrix
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
	
	// set the output pointer to the output matrix
	plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
	output2 = mxGetPr(plhs[1]);
	
	*output2 = nb_maxima;

//	printf("15. nb_maxima = %d\n",nb_maxima);

	// free memory
	mxFree(*vert2d);
	mxFree(max_odf_val);
	mxFree(*max_odf_param);
	mxFree(max_odf_val_sorted);
	mxFree(index_sort_max);
	mxFree(index_max);
//	mxFree(*max_odf_param_tmp);
//	mxFree(odf);
//	mxFree(vert);
//	mxFree(nearest_neighb_array);
//	mxFree(max_odf);
	
  return;
}
