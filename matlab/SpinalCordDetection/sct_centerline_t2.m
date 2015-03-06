function sct_centerline_t2(fname)
% sct_centerline_t2(fname)
[fname,path,ext]=sct_tool_remove_extension(fname,1);
prout=load_nii([fname ext]);
v=double(prout.img);
[~,~,S]=minimalPath3d(max(max(max(v)))-v,sqrt(2),1);
binaire2=false(size(S));
for level=1:size(prout.img,3)
    k=v(:,:,level);
    u=S(:,:,level);
    t=sort(u(:));
%     figure(3)
%     imagesc(u<t(400)); drawnow
%     figure(4)
%     imagesc(k); drawnow
    BW=bwconvhull(u<t(400));
    center=regionprops(BW,'Centroid');
    binaire2(round(center.Centroid(2)),round(center.Centroid(1)),level)=true;
end
prout2=prout;
prout2.img=binaire2;
save_nii(prout2,[fname '_centerline' ext])
save_nii(prout,[fname '_orient' ext])


