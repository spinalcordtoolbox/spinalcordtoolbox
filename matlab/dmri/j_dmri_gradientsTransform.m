
% =========================================================================
% FUNCTION
% j_dmri_gradientsTransform()
%
% Transform gradient file regarding to the orientation of the patient.
%
% INPUT
% gradient_list			nx3 double
% hdr					structure
%
% OUTPUT
% gradient_list_t		nx3 double
%
% COMMENTS
% julien cohen-adad 2009-08-31
% =========================================================================
function gradient_list_t = j_dmri_gradientsTransform(gradient_list,hdr)


% get patient orientation
R = reshape(hdr.ImageOrientationPatient,[3 2]);
R(:,3) = null(R'); % compute the orthonormal basis based on the 2 vectors orient(:,1) and orient(:,2)

% Apply transformation matrix
gradient_list_t = gradient_list*R;
    

