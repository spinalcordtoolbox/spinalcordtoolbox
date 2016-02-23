function pos=sct_centerline_createRoi(mat)
% nii=sct_centerline_createRoi(nii)
% call manual selection
figure
imagesc(squeeze(mat(round(end/2),:,:,1))); axis image; colormap gray;
h=imrect;
BW=zeros(1,size(mat,2),size(mat,3));
BW(1,:,:,1)=createMask(h);
pos1=getPosition(h);
delete(h)
close
BW1=logical(repmat(BW,[size(mat,1) 1 1 size(mat,4)]));

figure
imagesc(squeeze(mat(:,round(pos1(2)+pos1(4)/2),:,1))); axis image; colormap gray;
h=imrect;
BW=zeros(size(mat,1),1,size(mat,3));
BW(:,1,:,1)=createMask(h);
pos2=getPosition(h);
delete(h)
close
BW2=logical(repmat(BW,[1 size(mat,2) 1 size(mat,4)]));

BW=BW1 & BW2;
pos=max(round([pos2(2) min(pos2(2)+pos2(4),size(mat,1)) pos1(2) min(pos1(2)+pos1(4),size(mat,2)) pos1(1) min(pos1(1)+pos1(3),size(mat,3))]),1);
