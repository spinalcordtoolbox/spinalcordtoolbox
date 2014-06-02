% =========================================================================
% FUNCTION
% j_detrend
%
% Detrend data of dimension 1, 2 or 4 (fMRI data).
%
% data_d = st_detrend(data,deg,mask)
%
% INPUTS
% data          (nx,nt) or (nx,ny,nz,nt) float. Perform detrending along t.
% (deg)         degree of the polynomial OR number of cosine functions (5*)
% (func_type)   'poly','cos'*: type of function used for the regression
% (mask)       	mask of interest (for 4D data).
%
% OUTPUTS
% data_d        dataset corrected from trends of degree deg.
%
% COMMENTS
% *: default value
% Julien Cohen-Adad 2010-10-10
% Eugénie Ullmann 2013-10-19 add an option not to take account of outliers
% Eugénie Ullmann 2013-08-23 some part of the code  is copied from John Ashburner code (spm_dctmtx)
% =========================================================================
function data_d = j_detrend_new_v2(data,deg,func_type,robust,mask)


% initialization
if (nargin<1), help j_detrend; return; end
if (nargin<2), deg          = 5; return; end
if (nargin<3), func_type    = 'cos'; end % TODO
if (nargin<4), robust         = 0; end
if (nargin<5), mask         = 0; end

% get data size
[nx ny nz nt] = size(data);
if nt==1 % 1 or 2-D data
    data2d = data';
else % 4D data
    data2d = reshape(data,nx*ny*nz,nt)';
end
clear data

% get mask
if length(mask)~=1
    mask1d = reshape(mask,1,nx*ny*nz);
else
    mask1d = ones(1,nx*ny*nz);
end
[index_mask] = int8(find(mask1d));

nb_samples = size(data2d,1);
nb_vectors = size(index_mask,1);

if strcmp(func_type,'linear')
    % Linear trend
    D = (-1:2/(nb_samples-1):1)';
else
    % create DCT basis of regressors, 
  
    
    N=nb_samples;
    K=deg;
    d = 0;    
    n = (0:(N-1))';    
    C = zeros(size(n,1),K);
    
   
        C(:,1)=ones(size(n,1),1)/sqrt(N);
        for k=2:K
            C(:,k) = sqrt(2/N)*cos(pi*(2*n+1)*(k-1)/(2*N));
        end
    
    D=C*sqrt(nb_samples);
    
    
end

% initialize variable
data2d_d = zeros(nt,nx*ny*nz);

% detrend data
j_progress('Detrend data ..................................')
for i_vect=1:nb_vectors
    % build data vector
    data1d = data2d(:,index_mask(i_vect));
    if robust
        l = robustfit(D,data1d,'bisquare',4.685,'off');
    % estimate l (i.e. projection of data2d onto D)
    else
        l = pinv(D'*D)*D'*data1d;
    end
    % calculate residuals
    res_l = data1d - D*l;
    % reconstruct drift signal
    Dl = D*l;
    % reconstruct data2d without low frequency drifts
    data1d_d = data1d-Dl;
    % save in variable
    data2d_d(index_mask(i_vect),:) = data1d_d;
%     data2d_d(:,index_mask(i_vect)) = data1d_d;   TEST 
	% display progress
	j_progress(i_vect/nb_vectors)
end

% reshape data
data2d_d = data2d_d';
data_d = reshape(data2d_d,nx,ny,nz,nt);

