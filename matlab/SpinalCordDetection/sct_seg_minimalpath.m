function sct_seg_minimalpath(img_fname,centerline_fname,t1)
% sct_seg_minimalpath(img_fname(,centerline_fname,t1?))
% If centerline failed.. call sct_get_centerline
[basename, path, ext]=sct_tool_remove_extension(img_fname,0);
if nargin<2
    sct_centerline_t2(img_fname,0);
    centerline_fname = [basename '_centerline' ext];
end
t2=load_nii(img_fname);
if exist('t1','var'),
    t1=t2;
    t2.img=max(max(max(t2.img)))-t2.img;
end
dims=size(t2.img);
mask=load_nii(centerline_fname);
for level=1:size(t2.img,3);
    se = strel('disk',2);
    BW=imdilate(mask.img(:,:,level),se);
    
    [initialArray(:,:,level), SC(:,:,level)] = myelinInitialSegmention(255*(double(t2.img(:,:,level))/double(max(max(t2.img(:,:,level)))))+1, BW, false(dims(1:2)),0,1,1/5,1);
end


for level=1:size(t2.img,3);
    se = strel('disk',1);
    BW=imerode(SC(:,:,level),se);
    if length(find(BW))<10, BW=SC(:,:,level); end
    BW=bwconvhull(BW);
    [initialArray(:,:,level), SC(:,:,level)] = myelinInitialSegmention(255*(double(t2.img(:,:,level))/double(max(max(t2.img(:,:,level)))))+1, BW, false(dims(1:2)),0,1,1/5,1);
end

mask.img=initialArray;

save_nii(mask,[basename '_csf' ext])
if exist('t1','var'),
    save_nii(t1,[basename '_orient' ext])
else
    save_nii(t2,[basename '_orient' ext])
end
save_nii_v2(SC,[basename '_cord' ext],img_fname)