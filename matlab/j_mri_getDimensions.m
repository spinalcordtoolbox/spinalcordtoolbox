% =========================================================================
% FUNCTION
% j_mri_getDimensions
%
% Get dimensions of Nifti file using fslsize and some findstr tricks.
%
% INPUTS
% result			1xn char.		string resulting from 'fslsize' command 
%
% OUTPUTS
% dims				structure
%  x
%  y
%  z
%  t
%
% COMMENTS
% Julien Cohen-Adad 2010-04-19
% =========================================================================
function dims = j_mri_getDimensions(result)


[a b]= textscan(result,'%s %d');

dims(1) = double(a{1,2}(1));
dims(2) = double(a{1,2}(2));
dims(3) = double(a{1,2}(3));
dims(4) = double(a{1,2}(4));
