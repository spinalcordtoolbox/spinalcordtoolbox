function sct_seg_minimalpath(img_fname,centerline_fname)
% sct_seg_minimalpath(img_fname(,centerline_fname))
[basename, path, ext]=sct_tool_remove_extension(img_fname,0);
if nargin<2
    sct_centerline_t2(img_fname);
    centerline_fname = [basename '_centerline' ext];
end
t2=load_nii(img_fname);
dims=size(t2.img);
mask=load_nii(centerline_fname);
for level=1:size(t2.img,3);
    se = strel('disk',2);
    BW=imdilate(mask.img(:,:,level),se);
    
    [initialArray(:,:,level), SC(:,:,level)] = myelinInitialSegmention(255*(double(t2.img(:,:,level))/double(max(max(t2.img(:,:,level))))), BW, false(dims(1:2)));
end

mask.img=initialArray;
save_nii(mask,[basename '_csf' ext])
save_nii(t2,[basename '_orient' ext])