function slicethickness=sct_slicethickness(fname)
% sct_slicethickness(fname)
nii=load_nii(fname);
slicethickness=nii.hdr.dime.pixdim(4);
