function [ ] = scs_straightening(input, output)
% scs_straightening
%   Straightens the volume of the image along the center-line 
%   i.e. the central is at the center of the volume for each axial slice.
%   To be performed after completion of the spinal cord segmentation by spinal_cord_segmentation.m
% 
% SYNTAX
% [] = scs_straightening(INPUT, OUTPUT, PARAM)
% 
% INPUTS:
% INPUT
%  Path and name of the file that contains the output of the segmentation.
%  The program works with the same syntax as given by the output of spinal_cord_segmentation.m
%
% OUTPUT
%   path and name of the file in which the function will save the result of
%   the straightening
%
% OUTPUTS:
%   None 
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The core of the function starts here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load(input);

% Creation of m_volume, the 3D-image in the zone specified by slices(1) and slices(2).
m_volume = m_nifti;
m_volume = imresize(m_volume,resampling);
m_volume(:,:,slices(2)+1:size(m_volume,3)) = [];  % Suppression of the last slices
m_volume(:,:,1:slices(1)) = [];                   % Suppression of the first slices

% Definition of the boundaries of the image
x_max = size(m_volume,1);
y_max = size(m_volume,2);
z_max = size(m_volume,3);

% Creation of m_coor, the matrix corresponding to the planes normal to the
% cord.
vect1 = zeros(3,z_max);
vect2 = zeros(3,z_max);
point = zeros(3,z_max);
svec = -x_max/6:1:x_max/6;
tvec = -y_max/2:1:y_max/2;
m_coor = zeros(3,size(svec,2),size(tvec,2),z_max);
for k = 1:z_max
    vect1(:,k) = norm_cord(:,k,ceil(size(norm_cord,3)/4));
    vect2(:,k) = cross(tangent_cord(:,k),vect1(:,k));
    point = squeeze(centerline(end, :, :));
    %point = point1(:,k);
    for i = 1:size(svec,2)
        for j = 1:size(tvec,2)
        s = svec(i);
        t = tvec(j);
        m_coor(:,i,j,k) = point(:,k)+s*vect1(:,k)+t*vect2(:,k);
        end
    end
end

% vect1s(:,:) = norm_cord(:,1:z_max,ceil(size(norm_cord,3)/4));
% vect2s(:,:) = cross(tangent_cord(:,1:z_max),vect1(:,1:z_max));
% points(:,:) = squeeze(centerline(end, :, :))(:,1:z_max);
% for k = 1:z_max
% for i = 1:size(svec,2)
%     for j = 1:size(tvec,2)
%         s = svec(i);
%         t = tvec(j);
%         m_coor(:,i,j,k) = points(:,k)+s*vect1s(:,k)+t*vect2s(:,k);
%     end
% end
% end

% Interpolation of the intensities and reconstruction to the points defined
% by m_coor
% - Creation of the grid
coord_xs = [floor(m_coor(1,:,:,:)) ; floor(m_coor(1,:,:,:))+1];
coord_ys = [floor(m_coor(2,:,:,:)) ; floor(m_coor(2,:,:,:))+1];
coord_zs = [floor(m_coor(3,:,:,:)) ; floor(m_coor(3,:,:,:))+1];
x_grid = min(min(min(min(coord_xs(coord_xs>0))))) : max(max(max(max(coord_xs(coord_xs<=x_max))))); % Correction for out-boundaries values
y_grid = min(min(min(min(coord_ys(coord_ys>0))))) : max(max(max(max(coord_ys(coord_ys<=y_max)))));
z_grid = min(min(min(min(coord_zs(coord_zs>0))))) : max(max(max(max(coord_zs(coord_zs<=z_max)))));
[xi,yi,zi] = meshgrid(x_grid,y_grid,z_grid);

% - Interpolation
%m_straight = zeros(size(m_volume,1),size(m_volume,2),size(m_volume,3));
m_volume = permute(m_volume,[2 1 3]);
m_straight(:,:,:) = interp3(xi,yi,zi,m_volume(y_grid,x_grid,z_grid),m_coor(1,:,:,:),m_coor(2,:,:,:),m_coor(3,:,:,:));  

save(output,'m_straight')

end

