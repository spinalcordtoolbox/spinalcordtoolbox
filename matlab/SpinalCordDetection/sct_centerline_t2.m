function sct_centerline_t2(fname,verbose)
% sct_centerline_t2(fname)

if nargin<2
    verbose=false;
end
[fname,path,ext]=sct_tool_remove_extension(fname,1);
nii=load_nii([fname ext]);
v=double(nii.img);
[~,~,S]=minimalPath3d(v,sqrt(2),1,verbose);
S=int8(S); S(S==0)=inf; S=smooth3(S);
[~,~,S]=minimalPath3d(S,1,0,verbose);
[~,~,S]=minimalPath3d(S,1,0,verbose);
binaire2=false(size(S));
N=floor(15*15/(nii.scales(1)*nii.scales(2)));
for level=1:size(nii.img,3)
    k=v(:,:,level);
    u=S(:,:,level);
    t=sort(u(:));
    BW=bwconvhull(u<t(N));
    
    if verbose
        figure(3)
        imagesc(u<t(N)); drawnow
        figure(4)
        imagesc(k); drawnow
        imagesc(BW); drawnow
    end
    center=regionprops(BW,'Centroid');
    binaire2(round(center.Centroid(2)),round(center.Centroid(1)),level)=true;
end
niibin=nii;
niibin.img=binaire2;
save_nii(niibin,[fname '_centerline' ext])
save_nii(nii,[fname '_orient' ext])
