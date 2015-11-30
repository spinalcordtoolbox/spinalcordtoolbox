warpfield=load_nii_data('warp_forward.nii');
warpfield(isinf(warpfield))=0; warpfield(abs(warpfield)>1e20)=0;
warpx=warpfield(:,:,:,:,1); warpx=warpx-mean(warpx(:));
warpy=warpfield(:,:,:,:,2); warpy=warpy-mean(warpy(:));
warpz=warpfield(:,:,:,:,3); warpz=warpz-mean(warpz(:));

warpfield=cat(5,warpx,warpy,warpz);
save_nii_v2(warpfield,'warp_forward_wthoutTr.nii','warp_forward.nii')

sct_unix('CreateWarpedGridImage 3  warp_forward_wthoutTr.nii grid.nii 1x1x0 2x2x1 0.1x0.1x1')
