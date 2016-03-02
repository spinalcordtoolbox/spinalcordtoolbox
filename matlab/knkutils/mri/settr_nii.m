function nii = settr_nii(nii,tr)

% function nii = settr_nii(nii,tr)
%
% <nii> is the output of make_nii.m
% <tr> is the TR in seconds
%
% set nii.hdr.dime.pixdim(5) to <tr>.
% the reason for the existence of this function
% is that for some reason, make_nii.m does not
% allow you to set the TR.

nii.hdr.dime.pixdim(5) = tr;
