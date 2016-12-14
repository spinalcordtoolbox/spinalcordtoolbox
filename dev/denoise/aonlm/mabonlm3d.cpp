/* Jose V. Manjon - jmanjon@fis.upv.es                                     */
/* Pierrick Coupe - pierrick.coupe@gmail.com                               */
/* Brain Imaging Center, Montreal Neurological Institute.                  */
/* Mc Gill University                                                      */
/*                                                                         */
/* Copyright (C) 2010 Jose V. Manjon and Pierrick Coupe                    */

/***************************************************************************
*              Adaptive Non-Local Means Denoising of MR Images            *             

%              With Spatially Varying Noise Levels                        *

%                                                                         * 

%              Jose V. Manjon, Pierrick Coupe, Luis Marti-Bonmati,        *

%              D. Louis Collins and Montserrat Robles                      *
***************************************************************************/

#include "math.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
/* Multithreading stuff*/
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#endif

#ifndef _isnan
int _isnan(double x){
	const long long EXPONENT_MASK64=(2047LL<<52);
	const long long MANTISSA_MASK64=((1LL<<52)-1);
	const long long SIGNALING_BIT64=(1LL<<51);
	if(((*((long long*)(&x)))&EXPONENT_MASK64)!=EXPONENT_MASK64){
		return false;
	}
	if((*((long long*)(&x)))&MANTISSA_MASK64){
		return !((*((long long*)(&x)))&SIGNALING_BIT64);
	}
	return false;
}
#endif

#define pi 3.1415926535

typedef struct{
    int rows;
    int cols;
    int slices;
    double * in_image;   
    double * means_image;
    double * var_image;    
    double * estimate;    
    double * label;    
    double * bias;
    int ini;
    int fin;
    int radioB;
    int radioS;   
}myargument;

/*Returns the modified Bessel function I0(x) for any real x.*/
double bessi0(double x)
{
  double ax,ans,a;
  double y; 
  if ((ax=fabs(x)) < 3.75) 
  { 
    y=x/3.75;
    y*=y;
    ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
  } 
  else 
  {
    y=3.75/ax;
    ans=(exp(ax)/sqrt(ax));
    a=y*(0.916281e-2+y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1+y*0.392377e-2))));
    ans=ans*(0.39894228 + y*(0.1328592e-1 +y*(0.225319e-2+y*(-0.157565e-2+a))));    
  }
  return ans;
}

/*Returns the modified Bessel function I1(x) for any real x.*/
double bessi1(double x)
{
  double ax,ans;
  double y; 
  if ((ax=fabs(x)) < 3.75)
  { 
    y=x/3.75;
    y*=y;
    ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934+y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
  } 
  else 
  {
    y=3.75/ax;
    ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1-y*0.420059e-2));
    ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2+y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
    ans *= (exp(ax)/sqrt(ax));
  }
  return x < 0.0 ? -ans : ans;
}

double Epsi(double snr)
{
    double val;    
    val=2 + snr*snr - (pi/8)*exp(-(snr*snr)/2)*((2+snr*snr)*bessi0((snr*snr)/4) + (snr*snr)*bessi1((snr*snr)/4))*((2+snr*snr)*bessi0((snr*snr)/4) + (snr*snr)*bessi1((snr*snr)/4));     
    if(val<0.001) val=1;
    if(val>10) val=1;    
    return val;
}

/* Function which compute the weighted average for one block */
void Average_block(double *ima,int x,int y,int z,int neighborhoodsize,double *average, double weight, int sx,int sy,int sz, int rician)
{
int x_pos,y_pos,z_pos;
bool is_outside; 
int a,b,c,ns,sxy,count;


ns=2*neighborhoodsize+1;
sxy=sx*sy;

count = 0;

	for (c = 0; c<ns;c++)
	{
		for (b = 0; b<ns;b++)
		{
			for (a = 0; a<ns;a++)
			{	
				is_outside = false;
				x_pos = x+a-neighborhoodsize;
				y_pos = y+b-neighborhoodsize;
				z_pos = z+c-neighborhoodsize;
	
				if ((z_pos < 0) || (z_pos > sz-1)) is_outside = true;
				if ((y_pos < 0) || (y_pos > sy-1)) is_outside = true;
				if ((x_pos < 0) || (x_pos > sx-1)) is_outside = true;
		
                if(rician)
                {
				    if (is_outside)
					average[count] = average[count] + ima[z*(sxy)+(y*sx)+x]*ima[z*(sxy)+(y*sx)+x]*weight;
				    else	
					average[count] = average[count] + ima[z_pos*(sxy)+(y_pos*sx)+x_pos]*ima[z_pos*(sxy)+(y_pos*sx)+x_pos]*weight;
                }
                else
                {
                    if (is_outside)
					average[count] = average[count] + ima[z*(sxy)+(y*sx)+x]*weight;
				    else	
					average[count] = average[count] + ima[z_pos*(sxy)+(y_pos*sx)+x_pos]*weight;
                }
				count++;
			}
		}
	}
}

/* Function which computes the value assigned to each voxel */
void Value_block(double *Estimate, double *Label,int x,int y,int z,int neighborhoodsize,double *average, double global_sum, int sx,int sy,int sz)
{
int x_pos,y_pos,z_pos;
bool is_outside;
double value = 0.0;
double label = 0.0;
double denoised_value =0.0;
int count=0 ;
int a,b,c,ns,sxy;

ns=2*neighborhoodsize+1;
sxy=sx*sy;


	for (c = 0; c<ns;c++)
	{
		for (b = 0; b<ns;b++)
		{
			for (a = 0; a<ns;a++)
			{		
				is_outside = false;
				x_pos = x+a-neighborhoodsize;
				y_pos = y+b-neighborhoodsize;
				z_pos = z+c-neighborhoodsize;
	
				if ((z_pos < 0) || (z_pos > sz-1)) is_outside = true;
				if ((y_pos < 0) || (y_pos > sy-1)) is_outside = true;
				if ((x_pos < 0) || (x_pos > sx-1)) is_outside = true;
				if (!is_outside)
				{		
					value = Estimate[z_pos*(sxy)+(y_pos*sx)+x_pos];                          
                    value = value + (average[count]/global_sum);                    
                    
					label = Label[(x_pos + y_pos*sx + z_pos*sxy)];
					Estimate[z_pos*(sxy)+(y_pos*sx)+x_pos] = value;
					Label[(x_pos + y_pos*sx + z_pos *sxy)] = label +1;		
				}
				count++;
			}
		}
	}
}

double distance(double* ima,int x,int y,int z,int nx,int ny,int nz,int f,int sx,int sy,int sz)
{
double d,acu,distancetotal;
int i,j,k,ni1,nj1,ni2,nj2,nk1,nk2;

distancetotal=0;
for(k=-f;k<=f;k++)
{
 nk1=z+k;
 nk2=nz+k;  
 if(nk1<0) nk1=-nk1;
 if(nk2<0) nk2=-nk2;
 if(nk1>=sz) nk1=2*sz-nk1-1;    
 if(nk2>=sz) nk2=2*sz-nk2-1;

 for(j=-f;j<=f;j++)
 {
  nj1=y+j;
  nj2=ny+j;    
  if(nj1<0) nj1=-nj1;    
  if(nj2<0) nj2=-nj2;
  if(nj1>=sy) nj1=2*sy-nj1-1;
  if(nj2>=sy) nj2=2*sy-nj2-1;

  for(i=-f;i<=f;i++)
  {
    ni1=x+i;
    ni2=nx+i;    
    if(ni1<0) ni1=-ni1;
    if(ni2<0) ni2=-ni2;    
	if(ni1>=sx) ni1=2*sx-ni1-1;
    if(ni2>=sx) ni2=2*sx-ni2-1;
                
    distancetotal = distancetotal + ((ima[nk1*(sx*sy)+(nj1*sx)+ni1]-ima[nk2*(sx*sy)+(nj2*sx)+ni2])*(ima[nk1*(sx*sy)+(nj1*sx)+ni1]-ima[nk2*(sx*sy)+(nj2*sx)+ni2]));   
  }
 }
}

acu=(2*f+1)*(2*f+1)*(2*f+1);
d=distancetotal/acu;

return d;
}

double distance2(double* ima,double * medias,int x,int y,int z,int nx,int ny,int nz,int f,int sx,int sy,int sz)
{
double d,acu,distancetotal;
int i,j,k,ni1,nj1,ni2,nj2,nk1,nk2;

acu=0;
distancetotal=0;
	
for(k=-f;k<=f;k++)
{
 nk1=z+k;
 nk2=nz+k;  
 if(nk1<0) nk1=-nk1;
 if(nk2<0) nk2=-nk2;
 if(nk1>=sz) nk1=2*sz-nk1-1;    
 if(nk2>=sz) nk2=2*sz-nk2-1;
 for(j=-f;j<=f;j++)
 {
  nj1=y+j;
  nj2=ny+j;    
  if(nj1<0) nj1=-nj1;    
  if(nj2<0) nj2=-nj2;
  if(nj1>=sy) nj1=2*sy-nj1-1;
  if(nj2>=sy) nj2=2*sy-nj2-1;
  for(i=-f;i<=f;i++)
  {
    ni1=x+i;
    ni2=nx+i;    
    if(ni1<0) ni1=-ni1;
    if(ni2<0) ni2=-ni2;    
	if(ni1>=sx) ni1=2*sx-ni1-1;
    if(ni2>=sx) ni2=2*sx-ni2-1;
			
    d=(ima[nk1*(sx*sy)+(nj1*sx)+ni1]-medias[nk1*(sx*sy)+(nj1*sx)+ni1])-(ima[nk2*(sx*sy)+(nj2*sx)+ni2]-medias[nk2*(sx*sy)+(nj2*sx)+ni2]);            
	distancetotal = distancetotal + d*d;			   
  }
 }
}

acu=(2*f+1)*(2*f+1)*(2*f+1);
d=distancetotal/acu;

return d;
}

void Regularize(double* in,double * out,int r,int sx,int sy,int sz)
{
double acu;
int ind,i,j,k,ni,nj,nk,ii,jj,kk;

//double * temp=mxCalloc(sx*sy*sz,sizeof(double));
double * temp=(double*)malloc(sx*sy*sz*sizeof(double));

// separable convolution
for(k=0;k<sz;k++)
for(j=0;j<sy;j++)
for(i=0;i<sx;i++)
{
  if(in[k*(sx*sy)+(j*sx)+i]==0) continue;
  
  acu=0;
  ind=0;
  for(ii=-r;ii<=r;ii++)
  {
	ni=i+ii;
    if(ni<0) ni=-ni;
	if(ni>=sx) ni=2*sx-ni-1;
    if(in[k*(sx*sy)+(j*sx)+ni]>0)
    {
	  acu+=in[k*(sx*sy)+(j*sx)+ni];            
      ind++;		
    }
  }
  if(ind==0) ind=1; 
  out[k*(sx*sy)+(j*sx)+i]=acu/ind;
}
for(k=0;k<sz;k++)
for(j=0;j<sy;j++)
for(i=0;i<sx;i++)
{
  if(out[k*(sx*sy)+(j*sx)+i]==0) continue;
  
  acu=0;
  ind=0;
  for(jj=-r;jj<=r;jj++)
  {	
    nj=j+jj;		
	if(nj<0) nj=-nj;		
	if(nj>=sy) nj=2*sy-nj-1;
	
    if(out[k*(sx*sy)+(nj*sx)+i]>0)
    {
      acu+=out[k*(sx*sy)+(nj*sx)+i];            
      ind++;		
    }
  }
  if(ind==0) ind=1;
  temp[k*(sx*sy)+(j*sx)+i]=acu/ind;
}
for(k=0;k<sz;k++)
for(j=0;j<sy;j++)
for(i=0;i<sx;i++)
{
  if(temp[k*(sx*sy)+(j*sx)+i]==0) continue;
  
  acu=0;
  ind=0;
  for(kk=-r;kk<=r;kk++)
  {
	nk=k+kk;			
	if(nk<0) nk=-nk;			
	if(nk>=sz) nk=2*sz-nk-1;		
    if(temp[nk*(sx*sy)+(j*sx)+i]>0)
    {
      acu+=temp[nk*(sx*sy)+(j*sx)+i];            
      ind++;		
    }
  }
  if(ind==0) ind=1;
  out[k*(sx*sy)+(j*sx)+i]=acu/ind;
}
//mxFree(temp);
free(temp);
return;
}

void* ThreadFunc( void* pArguments, int rician, double globalMax )
{
    double *bias,*Estimate,*Label,*ima,*means,*variances,*average,epsilon,mu1,var1,totalweight,wmax,t1,t1i,t2,d,w,distanciaminima;
    int rows,cols,slices,ini,fin,v,f,init,i,j,k,rc,ii,jj,kk,ni,nj,nk,Ndims;
        
    myargument arg;
    arg=*(myargument *) pArguments;

    rows=arg.rows;    
    cols=arg.cols;
    slices=arg.slices;
    ini=arg.ini;    
    fin=arg.fin;
    ima=arg.in_image;    
    means=arg.means_image;  
    variances=arg.var_image;     
    Estimate=arg.estimate;
    bias=arg.bias;
    Label=arg.label;    
    v=arg.radioB;
    f=arg.radioS;    
                      
//filter
epsilon = 0.00001;
mu1 = 0.95;
var1 = 0.5+1e-7;
init = 0;
rc=rows*cols;

Ndims = (2*f+1)*(2*f+1)*(2*f+1);

average=(double*)malloc(Ndims*sizeof(double));

wmax=0.0;
for(k=ini;k<fin;k+=2)
for(j=0;j<rows;j+=2)
for(i=0;i<cols;i+=2)
{ 
    
  // init  
  for (init=0 ; init < Ndims; init++) average[init]=0.0;	 
  totalweight=0.0;												
  distanciaminima=100000000000000;
       
  if(ima[k*rc+(j*cols)+i]>0 && (means[k*rc+(j*cols)+i])>epsilon && (variances[k*rc+(j*cols)+i]>epsilon))
  {       		       
       // calculate minimum distance
        
       for(kk=-v;kk<=v;kk++)
       {
         nk=k+kk;
		 for(jj=-v;jj<=v;jj++)
		 {
           nj=j+jj;
		   for(ii=-v;ii<=v;ii++)
           {
                    
			  ni=i+ii;														
			  if(ii==0 && jj==0 && kk==0) continue; 				
			  if(ni>=0 && nj>=0 && nk>=0 && ni<cols && nj<rows && nk<slices)
			  {			
                    
				if (ima[nk*rc+(nj*cols)+ni]>0 && (means[nk*(rc)+(nj*cols)+ni])> epsilon && (variances[nk*rc+(nj*cols)+ni]>epsilon))
				{				
                        
				  t1 = (means[k*rc+(j*cols)+i])/(means[nk*rc+(nj*cols)+ni]);  
                  t1i= (globalMax-means[k*(rc)+(j*cols)+i])/(globalMax-means[nk*(rc)+(nj*cols)+ni]);  
				  t2 = (variances[k*rc+(j*cols)+i])/(variances[nk*rc+(nj*cols)+ni]);
				  if( (t1>mu1 && t1<(1/mu1)) || (t1i>mu1 && t1i<(1/mu1)) && t2>var1 && t2<(1/var1))
				  {   
					d=distance2(ima,means,i,j,k,ni,nj,nk,f,cols,rows,slices);
                    if(d<distanciaminima) distanciaminima=d;
                  }
                }
               }
            }
          }
        }                
        if(distanciaminima==0) distanciaminima=1; 
        // rician correction
        if(rician)
        {
            for(kk=-f;kk<=f;kk++)
		    {
              nk=k+kk;
		   	  for(ii=-f;ii<=f;ii++)
			  {
                ni=i+ii;
			    for(jj=-f;jj<=f;jj++)
				{
					nj=j+jj;							
					if(ni>=0 && nj>=0 && nk>=0 && ni<cols && nj<rows && nk<slices)
					{   
                       if(distanciaminima==100000000000000) bias[nk*(rc)+(nj*cols)+ni]=0; 
                       else bias[nk*(rc)+(nj*cols)+ni]=(distanciaminima);                                  
                    }
                }
              }
            }
         }
         
         // block filtering
         for(kk=-v;kk<=v;kk++)
	     {
           nk=k+kk;
		   for(jj=-v;jj<=v;jj++)
           {
              nj=j+jj;
			  for(ii=-v;ii<=v;ii++)
              {
				ni=i+ii;														
				if(ii==0 && jj==0 && kk==0) continue; 
				
				if(ni>=0 && nj>=0 && nk>=0 && ni<cols && nj<rows && nk<slices)
				{									
					if (ima[nk*rc+(nj*cols)+ni]>0 && (means[nk*(rc)+(nj*cols)+ni])> epsilon && (variances[nk*rc+(nj*cols)+ni]>epsilon))
					{				
						t1 = (means[k*rc+(j*cols)+i])/(means[nk*rc+(nj*cols)+ni]);  
                        t1i= (globalMax-means[k*(rc)+(j*cols)+i])/(globalMax-means[nk*(rc)+(nj*cols)+ni]);  
						t2 = (variances[k*rc+(j*cols)+i])/(variances[nk*rc+(nj*cols)+ni]);
	
						if( (t1>mu1 && t1<(1/mu1)) || (t1i>mu1 && t1i<(1/mu1)) && t2>var1 && t2<(1/var1))
						{                 										
							d=distance(ima,i,j,k,ni,nj,nk,f,cols,rows,slices);
                                       
                            if(d>3*distanciaminima) w=0;
                            else w = exp(-d/distanciaminima);      											
                                        
							if(w>wmax) wmax = w;
										
							if(w>0)
                            {
                               Average_block(ima,ni,nj,nk,f,average,w,cols,rows,slices, rician);																			
  			 			       totalweight = totalweight + w;
                            }
						}
					}							
				}
			  }
			}							
		}
				
		if(wmax==0.0) wmax=1.0;						
		Average_block(ima,i,j,k,f,average,wmax,cols,rows,slices, rician);					
		totalweight = totalweight + wmax;										 
		Value_block(Estimate,Label,i,j,k,f,average,totalweight,cols,rows,slices);				
	}
    else 
    {      
      wmax=1.0;  
	  Average_block(ima,i,j,k,f,average,wmax,cols,rows,slices, rician);	
      totalweight = totalweight + wmax;
 	  Value_block(Estimate,Label,i,j,k,f,average,totalweight,cols,rows,slices);
    }
}
	#ifdef _WIN32
       //_endthreadex(0);    
    #else
       //pthread_exit(0);    
    #endif

return 0;
}


void mabonlm3d_c(double *ima, int *dims, int v, int f, int r, double *fima)
{
/*Declarations*/
//mxArray *xData;
//double *ima, *fima,*average,*bias;
double  *bias=0;
//mxArray *Mxmeans, *Mxvariances, *MxEstimate, *MxLabel,*MxBias;
//double *means, *variances, *Estimate, *Label;
//mxArray *pv;
//double SNR,h,mean,var,label,estimate;
double SNR,mean,var,label,estimate,globalMax;
//int Ndims,i,j,k,ii,jj,kk,ni,nj,nk,v,f,ndim,indice,Nthreads,ini,fin,r;
int i,j,k,ii,jj,kk,ni,nj,nk,indice,Nthreads,ini,fin;
//const int  *dims;

//----------allocate memory-------
int ndim=3;
int nvox=dims[0]*dims[1]*dims[2];
int Ndims = pow((2*f+1),ndim);
double *means = (double *)malloc(nvox*sizeof(double));
double *variances = (double *)malloc(nvox*sizeof(double));
double *Estimate = (double *)malloc(nvox*sizeof(double));
double *Label = (double *)malloc(nvox*sizeof(double));
double *average=(double*)malloc(Ndims*sizeof(double));
memset(means, 0, nvox*sizeof(double));
memset(variances, 0, nvox*sizeof(double));
memset(Estimate, 0, nvox*sizeof(double));
memset(Label, 0, nvox*sizeof(double));
//memset(average, 0, Ndims*sizeof(double));
//--------------------------------


myargument *ThreadArgs;  
#ifdef _WIN32
HANDLE *ThreadList; /* Handles to the worker threads*/
#else
pthread_t * ThreadList;
#endif

/*Copy input pointer x*/
//xData = prhs[0];

/*Get matrix x*/
//ima = mxGetPr(xData);

//ndim = mxGetNumberOfDimensions(prhs[0]);
//dims= mxGetDimensions(prhs[0]);



/*Copy input parameters*/
//pv = prhs[1];
/*Get the Integer*/
//v = (int)(mxGetScalar(pv));

//pv = prhs[2];
//f = (int)(mxGetScalar(pv));

//pv = prhs[3];
//r = (int)(mxGetScalar(pv));
//if(r>0) rician=true;
//else rician=false;

//Ndims = pow((2*f+1),ndim);

/*Allocate memory and assign output pointer*/

//plhs[0] = mxCreateNumericArray(ndim,dims,mxDOUBLE_CLASS, mxREAL);
//Mxmeans = mxCreateNumericArray(ndim,dims,mxDOUBLE_CLASS, mxREAL); 
//Mxvariances = mxCreateNumericArray(ndim,dims,mxDOUBLE_CLASS, mxREAL); 
//MxEstimate = mxCreateNumericArray(ndim,dims,mxDOUBLE_CLASS, mxREAL); 
//MxLabel = mxCreateNumericArray(ndim,dims,mxDOUBLE_CLASS, mxREAL);

//if(rician) MxBias = mxCreateNumericArray(ndim,dims,mxDOUBLE_CLASS, mxREAL);    
if(r) bias = (double*)malloc(nvox*sizeof(double));    

//average=(double*)malloc(Ndims*sizeof(double));

/*Get a pointer to the data space in our newly allocated memory*/
//fima = mxGetPr(plhs[0]);
//means = mxGetPr(Mxmeans);
//variances = mxGetPr(Mxvariances);
//Estimate = mxGetPr(MxEstimate);
//Label = mxGetPr(MxLabel);
//if(rician) bias = mxGetPr(MxBias);

for (i = 0; i < dims[2] *dims[1] * dims[0];i++)
{
	Estimate[i] = 0.0;
	Label[i] = 0.0;
	fima[i] = 0.0;
    if(r) bias[i]=0.0;
}


globalMax=0;
for(k=0;k<dims[2];k++)
{
	for(j=0;j<dims[1];j++)
	{
		for(i=0;i<dims[0];i++)
		{
            if(ima[k*(dims[0]*dims[1])+(j*dims[0])+i]>globalMax) globalMax=ima[k*(dims[0]*dims[1])+(j*dims[0])+i];
            
			mean=0;
			indice=0;
			for(ii=-1;ii<=1;ii++)
			{
				for(jj=-1;jj<=1;jj++)
				{
					for(kk=-1;kk<=1;kk++)
					{
						ni=i+ii;
						nj=j+jj;		   		  
						nk=k+kk;
						
						if(ni<0) ni=-ni;
						if(nj<0) nj=-nj;
						if(nk<0) nk=-nk;
						if(ni>=dims[0]) ni=2*dims[0]-ni-1;
						if(nj>=dims[1]) nj=2*dims[1]-nj-1;
						if(nk>=dims[2]) nk=2*dims[2]-nk-1;
						
								
						mean = mean + ima[nk*(dims[0]*dims[1])+(nj*dims[0])+ni];
						indice=indice+1;                
					
					}
				}
			}
			mean=mean/indice;
			means[k*(dims[0]*dims[1])+(j*dims[0])+i]=mean;
		}
	}
}

for(k=0;k<dims[2];k++)
{
	for(j=0;j<dims[1];j++)
	{
		for(i=0;i<dims[0];i++)
		{
			var=0;
			indice=0;
			for(ii=-1;ii<=1;ii++)
			{
				for(jj=-1;jj<=1;jj++)
				{
					for(kk=-1;kk<=1;kk++)
					{
						ni=i+ii;
						nj=j+jj;		   		  
						nk=k+kk;		   		  
							if(ni>=0 && nj>=0 && nk>0 && ni<dims[0] && nj<dims[1] && nk<dims[2])
							{
							var = var + (ima[nk*(dims[0]*dims[1])+(nj*dims[0])+ni]-means[k*(dims[0]*dims[1])+(j*dims[0])+i])*(ima[nk*(dims[0]*dims[1])+(nj*dims[0])+ni]-means[k*(dims[0]*dims[1])+(j*dims[0])+i]);
							indice=indice+1;
							}
					}
				}
			}
			var=var/(indice-1);
			variances[k*(dims[0]*dims[1])+(j*dims[0])+i]=var;
		}
	}
}

Nthreads = 1;

#ifdef _WIN32

// Reserve room for handles of threads in ThreadList
ThreadList = (HANDLE*)malloc(Nthreads* sizeof( HANDLE ));
ThreadArgs = (myargument*) malloc( Nthreads*sizeof(myargument));

for (i=0; i<Nthreads; i++)
{         
	// Make Thread Structure   
    ini=(i*dims[2])/Nthreads;
    fin=((i+1)*dims[2])/Nthreads;            
   	ThreadArgs[i].cols=dims[0];
    ThreadArgs[i].rows=dims[1];
   	ThreadArgs[i].slices=dims[2];
    ThreadArgs[i].in_image=ima;	
  	ThreadArgs[i].var_image=variances;
    ThreadArgs[i].means_image=means;  
    ThreadArgs[i].estimate=Estimate;
    ThreadArgs[i].bias=bias;    
    ThreadArgs[i].label=Label;    
    ThreadArgs[i].ini=ini;
    ThreadArgs[i].fin=fin;
    ThreadArgs[i].radioB=v;
    ThreadArgs[i].radioS=f;  
    	
    //ThreadList[i] = (HANDLE)_beginthreadex( NULL, 0, &ThreadFunc, &ThreadArgs[i] , 0, NULL );
        
}
ThreadFunc(&ThreadArgs[0], r, globalMax);
//for (i=0; i<Nthreads; i++) { WaitForSingleObject(ThreadList[i], INFINITE); }
//for (i=0; i<Nthreads; i++) { CloseHandle( ThreadList[i] ); }

#else

/* Reserve room for handles of threads in ThreadList*/
ThreadList = (pthread_t *) calloc(Nthreads,sizeof(pthread_t));
ThreadArgs = (myargument*) calloc( Nthreads,sizeof(myargument));

for (i=0; i<Nthreads; i++)
{         
	// Make Thread Structure   
    ini=(i*dims[2])/Nthreads;
    fin=((i+1)*dims[2])/Nthreads;            
   	ThreadArgs[i].cols=dims[0];
    ThreadArgs[i].rows=dims[1];
   	ThreadArgs[i].slices=dims[2];
    ThreadArgs[i].in_image=ima;	
  	ThreadArgs[i].var_image=variances;
    ThreadArgs[i].means_image=means;  
    ThreadArgs[i].estimate=Estimate;
    ThreadArgs[i].bias=bias;    
    ThreadArgs[i].label=Label;    
    ThreadArgs[i].ini=ini;
    ThreadArgs[i].fin=fin;
    ThreadArgs[i].radioB=v;
    ThreadArgs[i].radioS=f;      	        
}
ThreadFunc(&ThreadArgs[0], r, globalMax);
/*for (i=0; i<Nthreads; i++)
{
    if(pthread_create(&ThreadList[i], NULL, ThreadFunc,&ThreadArgs[i]))
    {
       //printf("Threads cannot be created\n");
       exit(1);
    }        
}
   
for (i=0; i<Nthreads; i++)
{
  pthread_join(ThreadList[i],NULL);
}*/

#endif
free(ThreadArgs); 
free(ThreadList);
if(r)
{
  r=5;
  if(dims[0]<r){
	r=dims[0];
  }
  if(dims[1]<r){
	r=dims[1];
  }
  if(dims[2]<r){
	r=dims[2];
  }
  Regularize(bias,variances,r,dims[0],dims[1],dims[2]);
  for(i=0;i<dims[0]*dims[1]*dims[2];i++)
  {
     if(variances[i]>0) 
     {
       SNR=means[i]/sqrt(variances[i]);      
       bias[i]=2*(variances[i]/Epsi(SNR));      
       if(_isnan(bias[i])) bias[i]=0;     
     }
  }
}

/* Aggregation of the estimators (i.e. means computation) */
label = 0.0;
estimate = 0.0;
for(i=0;i<dims[0]*dims[1]*dims[2];i++)
{
	label = Label[i];
    if (label == 0.0) fima[i] = ima[i];				
    else
	{
	  estimate = Estimate[i];
	  estimate = (estimate/label);
	  if(r)
      {
         estimate = (estimate-bias[i])<0?0:(estimate-bias[i]);                    
         fima[i]=sqrt(estimate);
      }
	  else fima[i]=estimate;	
    }       
}
free(means);
free(variances);
free(Estimate);
free(Label);
free(average);
if(r){
    free(bias);
}
return;

}

