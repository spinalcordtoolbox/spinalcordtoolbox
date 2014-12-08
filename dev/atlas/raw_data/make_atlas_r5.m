
fname_atlas = 'atlas_grays_cerv_sym_correc_r5raw.png';
csf_value = 255;
value_gm = 238;

% open png file
atlas = imread(fname_atlas);
figure, imagesc(atlas), axis image, colormap jet

% pad
atlas_pad = padarray(atlas,[50 50]);
figure, imagesc(atlas_pad), axis image, colormap jet

% binarize
atlas_pad_bin = im2bw(atlas_pad,0.05);
figure, imagesc(atlas_pad_bin), axis image, colormap jet

% dilate
se = strel('disk',30);
atlas_pad_bin_dil = imdilate(atlas_pad_bin, se);
figure, imagesc(atlas_pad_bin_dil), axis image, colormap jet

% substract to get contour
atlas_pad_bin_dil_sub = atlas_pad_bin_dil - atlas_pad_bin;
figure, imagesc(atlas_pad_bin_dil_sub), axis image, colormap jet

% add to atlas
atlas_csf = atlas_pad + uint8(atlas_pad_bin_dil_sub).*csf_value;
figure, imagesc(atlas_csf), axis image, colormap jet

% remove GM from binarized atlas (for registration to AMU)
atlas_pad_without_gm = atlas_pad;
atlas_pad_without_gm(find(atlas_pad == value_gm)) = 0;
atlas_pad_without_gm_bin = im2bw(atlas_pad_without_gm, 0.05);
figure, imagesc(atlas_pad_without_gm_bin), axis image, colormap jet

% save
imwrite(atlas_csf,'atlas_grays_cerv_sym_correc_r5.png')
imwrite(atlas_pad_without_gm_bin,'mask_grays_cerv_sym_correc_r5.png')

