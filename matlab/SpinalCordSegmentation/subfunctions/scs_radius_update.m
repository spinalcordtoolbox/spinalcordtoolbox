function [m_radius m_norm_cord] = scs_radius_update(m_volume_sobel, m_center_line, m_tangent_cord, coord_radius, m_previous_radius, step, nom_radius, scales, angles, param)
% scs_radius_update 
%   This function update the radius of the spinal cord by an amount
%   determined by the gradient of the image.
%
% SYNTAX:
% [M_RADIUS M_NORM_CORD] = scs_radius_update(M_VOLUME_SOBEL,M_CENTER_LINE,M_TANGENT_CORD,COORD_RADIUS,M_PREVIOUS_RADIUS,STEP,NOM_RADIUS,SCALES,ANGLES,PARAM)
%
% _________________________________________________________________________
% INPUTS:  
%
% M_VOLUME_SOBEL
%   (XxYxZ array) Voxels intensity of desired NIfTI image after a
%   transformation with a Sobel filter
%
% M_CENTER_LINE
%   (Nx3 array) Coordinates of the spinal cord for each slice (N) of
%
% M_TANGENT_CORD
%   Coordinates of the unit tangent vector
%
% COORD_RADIUS
%   Direction of the different radius for each slice
%
% M_PREVIOUS_RADIUS
%   Radius from the previous iteration
%
% STEP
%   Buffer parameter that insures the script doesn't go out of the image at
%   the extremities
%
% NOM_RADIUS
%   Nominal radius in pixel
%
% SCALES 
%   Pixel to mm ratio in all directions
%
% ANGLES
%   Angles used to compute the radius
%
% PARAM
%   Struct containing an assortment of user defined parameters
% _________________________________________________________________________
% OUTPUTS:
%
% M_RADIUS    
%   (2D matrix) Value of the radius for each angles and each slice
%   of the splinal cord 
%
% M_NORM_CORD
%   Normal to the centerline for each slices and each angle

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The core of the function starts here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% convert in mm
m_tangent_cord(:,1)=m_tangent_cord(:,1)*scales(1);
m_tangent_cord(:,2)=m_tangent_cord(:,2)*scales(2);
m_tangent_cord(:,3)=m_tangent_cord(:,3)*scales(3);

num_angles=size(angles,2);
height = size(m_volume_sobel,3)-step;

% coordinate of the radius
x_radius = repmat(coord_radius(1,:), [ 1 1 size(m_volume_sobel,3)-step]); 
x_radius = permute(x_radius,[1 3 2]);
y_radius = repmat(coord_radius(2,:), [ 1 1 size(m_volume_sobel,3)-step]); 
y_radius = permute(y_radius,[1 3 2]);
z_radius = repmat(coord_radius(3,:), [ 1 1 size(m_volume_sobel,3)-step]);
z_radius = permute(z_radius,[1 3 2]);

% coordinate of the tangent
x_tangent = repmat(m_tangent_cord(1,:), [1 1 num_angles]);
y_tangent = repmat(m_tangent_cord(2,:), [1 1 num_angles]);
z_tangent = repmat(m_tangent_cord(3,:), [1 1 num_angles]);

% Computation of the normal to the center line and radius vectors
x_auxiliary = y_tangent.*z_radius - z_tangent.*y_radius;
y_auxiliary = -(x_tangent.*z_radius - z_tangent.*x_radius);
z_auxiliary = x_tangent.*y_radius - y_tangent.*x_radius;

% Compute the plane of the radius vectors and the center line
x_plane = y_auxiliary.*z_tangent - z_auxiliary.*y_tangent;
y_plane = -(x_auxiliary.*z_tangent - z_auxiliary.*x_tangent);
z_plane = x_auxiliary.*y_tangent - y_auxiliary.*x_tangent;
normalize=sqrt(x_plane(:).^2+y_plane(:).^2+z_plane(:).^2);
x_plane(:) = x_plane(:)./normalize;
y_plane(:) = y_plane(:)./normalize;
z_plane(:) = z_plane(:)./normalize;

m_norm_cord = [x_plane;y_plane;z_plane];

% Compute the position that will be needed
positions=[-0.2*nom_radius -0.1*nom_radius 0 0.1*nom_radius 0.2*nom_radius];

% Definition of the angles used in the scs_radius_update function
radius_position = zeros(3,num_angles,size(m_volume_sobel,3)-step,length(positions));
for j = 1 : length(positions)
    for i = 1 : size(m_previous_radius,1)
    [radius_x radius_y]=pol2cart(angles, m_previous_radius(i,:)+positions(j)); % m_previous_radius(i,:)+m_previous_radius(i,:)*positions(j))
    radius_position(:,:,i,j) = [radius_x; radius_y; zeros(size(radius_x,2),1)'*m_center_line(end,i)];%ones(size(radius_x,2),1)'*m_center_line(end,i)
    end
end

% Computation of the points of interests where the gradient value is needed
points = repmat( permute(m_center_line,[1 3 2]), [1,num_angles,1,length(positions)]) + radius_position;  %définition de tous les points a interpoler

% Verification that those points are within m_volume_sobel
if min(min(min(min(points)))) < 1 % no negative coordinates
    scs_warning_error(220, param)
elseif max(max(max(points(1,:,:,:)))) > size(m_volume_sobel,1) || max(max(max(points(2,:,:,:)))) > size(m_volume_sobel,2)...
        || max(max(max(points(3,:,:,:)))) > size(m_volume_sobel,3)   % no coordinates out of the volume
	scs_warning_error(220, param)
end

% Definition of the volume of interest and its corresponding meshgrid
coord_xs = [floor(points(1,:,:,:)) ; floor(points(1,:,:,:))+1];
coord_ys = [floor(points(2,:,:,:)) ; floor(points(2,:,:,:))+1];
coord_zs = [floor(points(3,:,:,:)) ; floor(points(3,:,:,:))+1] + step/2;
x_grid = min(min(min(min(coord_xs(coord_xs>0))))) : max(max(max(max(coord_xs(coord_xs<=size(m_volume_sobel,1))))));
y_grid = min(min(min(min(coord_ys(coord_ys>0))))) : max(max(max(max(coord_ys(coord_ys<=size(m_volume_sobel,2))))));
z_grid = min(min(min(min(coord_zs(coord_zs>0))))) : max(max(max(max(coord_zs(coord_zs<=size(m_volume_sobel,3))))));
[xi,yi,zi] = meshgrid(x_grid,y_grid,z_grid);
m_volume_sobel_permute = permute(m_volume_sobel,[2 1 3 4]);

% Computation of the gradient for each point
gradient_points(1,:,:,:) = interp3(xi,yi,zi,m_volume_sobel_permute(y_grid,x_grid,z_grid,1),points(1,:,:,:),points(2,:,:,:),points(3,:,:,:)+step/2);    % x gradient
gradient_points(2,:,:,:) = interp3(xi,yi,zi,m_volume_sobel_permute(y_grid,x_grid,z_grid,2),points(1,:,:,:),points(2,:,:,:),points(3,:,:,:)+step/2);    % y gradient
gradient_points(3,:,:,:) = interp3(xi,yi,zi,m_volume_sobel_permute(y_grid,x_grid,z_grid,3),points(1,:,:,:),points(2,:,:,:),points(3,:,:,:)+step/2);    % z gradient

vect_sign = permute([-1 -1 0 1 1], [1 3 2]);
m_sign = repmat(vect_sign,[num_angles height 1]);
f = squeeze(sum(gradient_points .* repmat(permute(m_norm_cord, [1 3 2]),[1 1 1 length(positions)]),1)) .*m_sign;

m_vect_force = sum(f,3)';

% Conversion of the radius in mm
[x,y]=pol2cart(repmat(angles,[size(m_previous_radius,1) 1]),m_previous_radius);
x=x*scales(1); y=y*scales(2);
[new_angles, new_previous_radius] = cart2pol(x,y);

% Update of the radius in mm 
m_radius = new_previous_radius+((-1)^param.image_type).*param.update_multiplier.*m_vect_force; %Éq 3. 

% Conversion in pixel
[x,y]=pol2cart(new_angles,m_radius);
x=x/scales(1); y=y/scales(2);
[~, m_radius] = cart2pol(x,y);

end % END OF SCS_RADIUS_UPDATE
