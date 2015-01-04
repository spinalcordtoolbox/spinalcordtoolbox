% =========================================================================
% FUNCTION                                 
% j_morpho.m   
%
% Perform 2d morphology on images.
%
% INPUT
% matrix_in     2d matrix
% operation     'erode'
%               'dilate'
%               'open'
%               'close'
%               'smooth': gaussian 2d convolution.
%
% (size_se)     integer. Width of the structured element (default=3)
% (type_se)     'disk', 'square' (default='disk') 
% 
% OUTPUT
% matrix_out
%
% Julien Cohen-Adad, 2007-07-17
% =========================================================================
function varargout = j_morpho(matrix_in,operation,size_se,type_se)


% default initialization
if nargin<2, error('Not enought arguments. Use help.'); end
if nargin<3 size_se = 3; end
if nargin<4 type_se = 'disk'; end
sigma = 0.5;

% design gaussian filter
h = fspecial('gaussian',[size_se size_se],sigma); 

% build structural element
se = strel(type_se,size_se);

% filter
matrix_out=zeros(size(matrix_in,1),size(matrix_in,2),size(matrix_in,3));
for k=1:size(matrix_in,3)
    matrix_in2d = matrix_in(:,:,k);
    if strcmp(operation,'erode')
        matrix_out(:,:,k)=imerode(matrix_in2d,se);
    elseif strcmp(operation,'dilate')
        matrix_out(:,:,k)=imdilate(matrix_in2d,se);
    elseif strcmp(operation,'open')
        matrix_in2dE=imerode(matrix_in2d,se);
        matrix_out(:,:,k)=imdilate(matrix_in2dE,se);
    elseif strcmp(operation,'close')
        matrix_in2dE=imdilate(matrix_in2d,se);
        matrix_out(:,:,k)=imerode(matrix_in2dE,se);
    elseif strcmp(operation,'smooth')
        matrix_out(:,:,k)=imfilter(matrix_in2d,h);
    end
end 

varargout{1}=matrix_out;

