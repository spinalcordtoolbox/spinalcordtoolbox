function data=load_nii_data(fname,slice)
% data=load_nii_data(fname)
data=load_nii(fname); 
if exist('slice','var')
    data=data.img(:,:,slice,:);
else
    data=data.img;
end