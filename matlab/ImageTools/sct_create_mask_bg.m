function mask = sct_create_mask_bg(A,threshold)
% mask = sct_create_mask_bg(A,threshold)

if ~exist('threshold','var'), threshold=0.3; end

mask=false(size(A));
for iz=1:size(A,3)
    mask(:,:,iz)=imfill(A(:,:,iz)>threshold*max(max(A(:,:,iz))),'holes');
end

