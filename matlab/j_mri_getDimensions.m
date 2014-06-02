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


[a b]=strread(result,'%s %d');

dims(1) = b(1);
dims(2) = b(2);
dims(3) = b(3);
dims(4) = b(4);
