void mabonlm3d_c(double *ima, int *dims, int v, int f, int r, double *fima);
void Average_block(double *ima,int x,int y,int z,int neighborhoodsize,double *average, double weight, int sx,int sy,int sz, int rician);
void Value_block(double *Estimate, double *Label,int x,int y,int z,int neighborhoodsize,double *average, double global_sum, int sx,int sy,int sz);
double distance(double* ima,int x,int y,int z,int nx,int ny,int nz,int f,int sx,int sy,int sz);
double distance2(double* ima,double * medias,int x,int y,int z,int nx,int ny,int nz,int f,int sx,int sy,int sz);