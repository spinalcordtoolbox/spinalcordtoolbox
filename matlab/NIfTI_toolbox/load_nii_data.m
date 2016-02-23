function data=load_nii_data(fname,slice)
% data=load_nii_data(fname)
data=load_nii(fname); 
if exist('slice','var')
    data=data.img(:,:,min(slice,end),:);
else
    data=data.img;
end