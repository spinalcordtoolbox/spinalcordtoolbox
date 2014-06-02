% =========================================================================
% FUNCTION                                 
% j_morpho3d.m   
%
% Perform 3d morphology on binary images. It works by (i) doing a 3d
% convolution, (ii) truncating the result to get same size as input matrix,
% (iii) rebinarizing the result.
%
% INPUT
% matrix_in     3d matrix
% operation     'erode', 'dilate', 'open', 'close'
% (size_se)     [x y z] integer. size of the structuring element for the convolution (default = 3x3x3).
% (type_se)     'disk', 'square' (default='square') 
% 
% OUTPUT
% matrix_out
%
% Julien Cohen-Adad, 2007-04-07
% =========================================================================
function varargout = j_morpho3d(matrix_in,operation,size_se,type_se)


% Initializations
if nargin<2, error('Not enought arguments. Use help.'); end
if nargin<3 size_se=[3 3 3]; end
if nargin<4 type_se='square'; end

% se = strel(type_se,size_se);
se = ones(size_se(1),size_se(2),size_se(3));

% matrix_out=zeros(size(matrix_in,1),size(matrix_in,2),size(matrix_in,3));

switch operation
    case 'erode'
        for k=1:size(matrix_in,3)
            matrix_in2d = matrix_in(:,:,k);
            matrix_out(:,:,k)=imerode(matrix_in2d,se);
        end
    case 'dilate'
        matrix_out = convn(matrix_in,se,'same');
            
%         end
    case 'open'
        for k=1:size(matrix_in,3)
            matrix_in2d = matrix_in(:,:,k);
            matrix_in2dE=imerode(matrix_in2d,se);
            matrix_out(:,:,k)=imdilate(matrix_in2dE,se);
        end
    case 'close'
        for k=1:size(matrix_in,3)
            matrix_in2d = matrix_in(:,:,k);
            matrix_in2dE=imdilate(matrix_in2d,se);
            matrix_out(:,:,k)=imerode(matrix_in2dE,se);
        end

end 

% binarize convoluted matrix
matrix_out = matrix_out >0;

varargout{1}=matrix_out;

